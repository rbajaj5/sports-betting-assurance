"""Canonical historical records and provenance-preserving user-data imports."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from betguard.artifact import sha256_file, write_json

SCHEMA_VERSION = "betguard-historical-v1"
RECORD_TYPES = {
    "game",
    "market_snapshot",
    "player_availability",
    "formation_evidence",
    "identity_reconciliation",
}
MARKET_TYPES = {"moneyline", "spread", "total"}
MARKET_DESIGNATIONS = {"opening", "closing", "current"}
RIGHTS_BASES = {"user_supplied", "official_public", "licensed", "synthetic", "redacted"}
_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{1,127}$")


class HistoricalProvider(Protocol):
    """Interface for licensed or user-controlled historical-data providers."""

    provider_name: str

    def read(self, path: Path) -> list[dict[str, Any]]:
        """Read provider records without mutating the source export."""


@dataclass(frozen=True)
class GameRecord:
    schema_version: str
    league: str
    season: str
    game_id: str
    scheduled_start_time: str
    scheduled_timezone: str
    home_team_id: str
    away_team_id: str
    home_final_score: int | None
    away_final_score: int | None
    result: str | None
    official_source_url: str
    retrieval_timestamp: str

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        if self.league != "WNBA":
            raise ValueError("the initial historical pipeline is intentionally WNBA-only")
        _stable("game_id", self.game_id)
        _stable("home_team_id", self.home_team_id)
        _stable("away_team_id", self.away_team_id)
        if self.home_team_id == self.away_team_id:
            raise ValueError("home and away team IDs must differ")
        _aware_timestamp("scheduled_start_time", self.scheduled_start_time)
        _aware_timestamp("retrieval_timestamp", self.retrieval_timestamp)
        if not self.scheduled_timezone.strip():
            raise ValueError("scheduled_timezone is required")
        _url("official_source_url", self.official_source_url)
        scores = (self.home_final_score, self.away_final_score)
        if any(score is not None and score < 0 for score in scores):
            raise ValueError("final scores cannot be negative")
        if (self.home_final_score is None) != (self.away_final_score is None):
            raise ValueError("final scores must be both present or both absent")
        if self.home_final_score is None and self.result is not None:
            raise ValueError("result cannot be known without final scores")


@dataclass(frozen=True)
class MarketSnapshotRecord:
    schema_version: str
    game_id: str
    operator: str
    operator_jurisdiction: str
    market_type: str
    side: str
    line: float | None
    american_odds: int | None
    decimal_odds: float | None
    observation_timestamp: str
    designation: str
    source_url: str | None
    licensed_provider_record_id: str | None
    rights_basis: str

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _stable("game_id", self.game_id)
        if not self.operator.strip() or not self.operator_jurisdiction.strip():
            raise ValueError("operator and jurisdiction are required")
        if self.market_type not in MARKET_TYPES:
            raise ValueError(f"market_type must be one of {sorted(MARKET_TYPES)}")
        if not self.side.strip():
            raise ValueError("market side is required")
        if self.market_type == "moneyline" and self.line is not None:
            raise ValueError("moneyline records cannot carry a spread/total line")
        if self.market_type in {"spread", "total"} and self.line is None:
            raise ValueError("spread and total records require a line")
        if self.american_odds is None and self.decimal_odds is None:
            raise ValueError("at least one odds representation is required")
        if self.american_odds is not None and -100 < self.american_odds < 100:
            raise ValueError("American odds must be <= -100 or >= 100")
        if self.decimal_odds is not None and self.decimal_odds <= 1:
            raise ValueError("decimal odds must exceed one")
        _aware_timestamp("observation_timestamp", self.observation_timestamp)
        if self.designation not in MARKET_DESIGNATIONS:
            raise ValueError(
                f"designation must be one of {sorted(MARKET_DESIGNATIONS)}"
            )
        _rights(self.rights_basis)
        if self.source_url is None and self.licensed_provider_record_id is None:
            raise ValueError("a source URL or licensed provider record ID is required")
        if self.source_url is not None:
            _url("source_url", self.source_url)
        if self.licensed_provider_record_id is not None:
            _stable("licensed_provider_record_id", self.licensed_provider_record_id)


@dataclass(frozen=True)
class PlayerAvailabilityRecord:
    schema_version: str
    game_id: str
    team_id: str
    player_id: str
    status: str
    reason: str | None
    report_timestamp: str
    official_source_url: str
    retrieval_timestamp: str

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for label in ("game_id", "team_id", "player_id"):
            _stable(label, getattr(self, label))
        if not self.status.strip():
            raise ValueError("availability status is required")
        _aware_timestamp("report_timestamp", self.report_timestamp)
        _aware_timestamp("retrieval_timestamp", self.retrieval_timestamp)
        _url("official_source_url", self.official_source_url)


@dataclass(frozen=True)
class FormationEvidenceRecord:
    schema_version: str
    game_id: str
    clip_source_id: str
    source_uri: str
    rights_basis: str
    clip_sha256: str | None
    clip_start_timestamp: str
    clip_end_timestamp: str
    calibration_method: str
    calibration_rmse_feet: float
    identity_tracking_method: str
    projected_lineup_coverage: float
    possession_count: int
    usable_frame_count: int
    player_ids: tuple[str, ...]
    coordinates: tuple[dict[str, Any], ...]
    derived_features: dict[str, float]

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _stable("game_id", self.game_id)
        _stable("clip_source_id", self.clip_source_id)
        if not self.source_uri.strip():
            raise ValueError("source_uri is required")
        _rights(self.rights_basis)
        if self.clip_sha256 is not None:
            _digest(self.clip_sha256)
        _aware_timestamp("clip_start_timestamp", self.clip_start_timestamp)
        _aware_timestamp("clip_end_timestamp", self.clip_end_timestamp)
        if _parse_time(self.clip_end_timestamp) <= _parse_time(self.clip_start_timestamp):
            raise ValueError("clip end must follow clip start")
        if not self.calibration_method.strip() or not self.identity_tracking_method.strip():
            raise ValueError("calibration and identity methods are required")
        if self.calibration_rmse_feet < 0:
            raise ValueError("calibration RMSE cannot be negative")
        if not 0 <= self.projected_lineup_coverage <= 1:
            raise ValueError("projected-lineup coverage must be in [0, 1]")
        if self.possession_count < 0 or self.usable_frame_count < 0:
            raise ValueError("possession and usable-frame counts cannot be negative")
        if len(self.player_ids) < 5 or len(set(self.player_ids)) != len(self.player_ids):
            raise ValueError("formation evidence requires unique fixed player IDs")
        for player_id in self.player_ids:
            _stable("player_id", player_id)
        expected = set(self.player_ids)
        for coordinate in self.coordinates:
            if coordinate.get("player_id") not in expected:
                raise ValueError("coordinate row uses an unknown player ID")
            for field in ("timestamp_seconds", "x_feet", "y_feet"):
                value = float(coordinate[field])
                if not math_is_finite(value):
                    raise ValueError(f"coordinate {field} must be finite")
        if any(not math_is_finite(float(value)) for value in self.derived_features.values()):
            raise ValueError("derived formation features must be finite")


@dataclass(frozen=True)
class IdentityReconciliationRecord:
    schema_version: str
    entity_type: str
    canonical_id: str
    source_system: str
    source_id: str
    display_name: str
    effective_start: str
    effective_end: str | None
    evidence_url: str

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        if self.entity_type not in {"game", "team", "player", "operator"}:
            raise ValueError("unsupported reconciliation entity_type")
        _stable("canonical_id", self.canonical_id)
        _stable("source_id", self.source_id)
        if not self.source_system.strip() or not self.display_name.strip():
            raise ValueError("source system and display name are required")
        _aware_timestamp("effective_start", self.effective_start)
        if self.effective_end is not None:
            _aware_timestamp("effective_end", self.effective_end)
        _url("evidence_url", self.evidence_url)


RECORD_CLASSES = {
    "game": GameRecord,
    "market_snapshot": MarketSnapshotRecord,
    "player_availability": PlayerAvailabilityRecord,
    "formation_evidence": FormationEvidenceRecord,
    "identity_reconciliation": IdentityReconciliationRecord,
}


class UserExportProvider:
    """Read explicit CSV, JSON, JSONL, or optional Parquet exports."""

    def __init__(self, provider_name: str = "user_export") -> None:
        if not provider_name.strip():
            raise ValueError("provider_name is required")
        self.provider_name = provider_name

    def read(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                return [_decode_mapping(dict(row)) for row in csv.DictReader(handle)]
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("records") if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                raise ValueError("JSON export must be a list or contain a records list")
            return [_mapping(row) for row in rows]
        if suffix in {".jsonl", ".ndjson"}:
            return [
                _mapping(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as parquet  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet import requires optional pyarrow; install it separately"
                ) from exc
            return [_mapping(row) for row in parquet.read_table(path).to_pylist()]
        raise ValueError("supported imports are CSV, JSON, JSONL, NDJSON, and Parquet")


def import_historical_export(
    source_path: Path,
    output_path: Path,
    *,
    record_type: str,
    provider: HistoricalProvider | None = None,
) -> dict[str, Any]:
    """Validate records and write canonical JSONL plus a provenance sidecar."""

    if record_type not in RECORD_TYPES:
        raise ValueError(f"record_type must be one of {sorted(RECORD_TYPES)}")
    reader = provider or UserExportProvider()
    rows = reader.read(source_path)
    record_class = RECORD_CLASSES[record_type]
    canonical: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        normalized = _coerce_record(record_type, row)
        normalized.setdefault("schema_version", SCHEMA_VERSION)
        for field in ("player_ids", "coordinates"):
            if field in normalized and isinstance(normalized[field], list):
                normalized[field] = tuple(normalized[field])
        try:
            canonical.append(asdict(record_class(**normalized)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {record_type} record {index}: {exc}") from exc
    if not canonical:
        raise ValueError("historical export contains no records")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in canonical),
        encoding="utf-8",
    )
    report = {
        "schema_name": "betguard-import-provenance",
        "schema_version": "1.0.0",
        "record_type": record_type,
        "provider": reader.provider_name,
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "record_count": len(canonical),
        "imported_at": datetime.now().astimezone().isoformat(),
        "source_mutated": False,
        "qualification_granted": False,
        "note": "Import validation alone does not establish source rights or calibration.",
    }
    sidecar = output_path.with_suffix(output_path.suffix + ".provenance.json")
    write_json(sidecar, report)
    return report


def data_gap_report() -> dict[str, Any]:
    """Machine-readable statement of why historical calibration remains absent."""

    return {
        "schema_name": "betguard-historical-data-gap",
        "schema_version": "1.0.0",
        "league_scope": "WNBA",
        "status": "UNQUALIFIED",
        "real_games_imported": 0,
        "qualifying_formation_games": 0,
        "minimum_required_games": 200,
        "licensed_market_data_available": False,
        "permitted_tracking_or_footage_available": False,
        "formation_coefficients_present": False,
        "formation_coefficients_validated": False,
        "missing_inputs": [
            "licensed timestamped two-sided historical market prices",
            "official pregame availability and projected-lineup records",
            "permitted real-game coordinates or auditable footage annotations",
            "at least 200 chronologically evaluable games passing all evidence gates",
        ],
        "prohibited_substitutions": [
            "synthetic affine episodes as outcome-training data",
            "fabricated closing lines or player coordinates",
            "display-name-only identity joins",
            "data obtained by bypassing access controls or licenses",
        ],
        "verdict": "PASS — no qualified trade.",
    }


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("each imported record must be an object")
    return value


def _decode_mapping(row: dict[str, str]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            decoded[key] = None
            continue
        candidate = value.strip()
        if candidate.startswith(("[", "{")):
            try:
                decoded[key] = json.loads(candidate)
                continue
            except json.JSONDecodeError:
                pass
        if candidate.lower() in {"true", "false"}:
            decoded[key] = candidate.lower() == "true"
        else:
            decoded[key] = value
    return decoded


def _coerce_record(record_type: str, row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    integer_fields = {
        "game": {"home_final_score", "away_final_score"},
        "market_snapshot": {"american_odds"},
        "formation_evidence": {"possession_count", "usable_frame_count"},
    }.get(record_type, set())
    float_fields = {
        "market_snapshot": {"line", "decimal_odds"},
        "formation_evidence": {"calibration_rmse_feet", "projected_lineup_coverage"},
    }.get(record_type, set())
    for field in integer_fields:
        if normalized.get(field) is not None:
            normalized[field] = int(normalized[field])
    for field in float_fields:
        if normalized.get(field) is not None:
            normalized[field] = float(normalized[field])
    return normalized


def _schema(value: str) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")


def _stable(label: str, value: str) -> None:
    if not _STABLE_ID.fullmatch(value):
        raise ValueError(f"{label} must be a stable machine ID, not a display-name join")


def _aware_timestamp(label: str, value: str) -> None:
    parsed = _parse_time(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone offset")


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc


def _url(label: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an HTTP(S) source URL")


def _rights(value: str) -> None:
    if value not in RIGHTS_BASES:
        raise ValueError(f"rights_basis must be one of {sorted(RIGHTS_BASES)}")


def _digest(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError("SHA-256 must be a 64-character hexadecimal digest")


def math_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
