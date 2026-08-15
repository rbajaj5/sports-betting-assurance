from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from betguard.historical import (
    SCHEMA_VERSION,
    FormationEvidenceRecord,
    GameRecord,
    MarketSnapshotRecord,
    UserExportProvider,
    data_gap_report,
    import_historical_export,
)


def _game() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "league": "WNBA",
        "season": "2026",
        "game_id": "WNBA:2026:001",
        "scheduled_start_time": "2026-06-01T19:00:00-04:00",
        "scheduled_timezone": "America/New_York",
        "home_team_id": "WNBA:NYL",
        "away_team_id": "WNBA:MIN",
        "home_final_score": 82,
        "away_final_score": 78,
        "result": "home_win",
        "official_source_url": "https://example.org/official/game/001",
        "retrieval_timestamp": "2026-06-02T10:00:00Z",
    }


def test_game_schema_requires_stable_ids_and_wnba_scope() -> None:
    record = GameRecord(**_game())
    assert record.game_id == "WNBA:2026:001"

    invalid = _game()
    invalid["home_team_id"] = "New York Liberty"
    with pytest.raises(ValueError, match="stable machine ID"):
        GameRecord(**invalid)


def test_market_schema_requires_traceable_two_sided_source_record() -> None:
    record = MarketSnapshotRecord(
        schema_version=SCHEMA_VERSION,
        game_id="WNBA:2026:001",
        operator="Licensed Exchange",
        operator_jurisdiction="US-NJ",
        market_type="moneyline",
        side="home",
        line=None,
        american_odds=-110,
        decimal_odds=1.91,
        observation_timestamp="2026-06-01T17:00:00-04:00",
        designation="current",
        source_url=None,
        licensed_provider_record_id="provider:row:001",
        rights_basis="licensed",
    )
    assert record.rights_basis == "licensed"


def test_formation_evidence_rejects_unknown_coordinate_identity() -> None:
    with pytest.raises(ValueError, match="unknown player ID"):
        FormationEvidenceRecord(
            schema_version=SCHEMA_VERSION,
            game_id="WNBA:2026:001",
            clip_source_id="clip:001",
            source_uri="user://clip-001",
            rights_basis="user_supplied",
            clip_sha256="a" * 64,
            clip_start_timestamp="2026-06-01T18:00:00Z",
            clip_end_timestamp="2026-06-01T18:00:10Z",
            calibration_method="four court landmarks",
            calibration_rmse_feet=0.8,
            identity_tracking_method="manual review",
            projected_lineup_coverage=1.0,
            possession_count=5,
            usable_frame_count=20,
            player_ids=("p1", "p2", "p3", "p4", "p5"),
            coordinates=(
                {"timestamp_seconds": 0.0, "player_id": "p6", "x_feet": 1, "y_feet": 2},
            ),
            derived_features={"conditioning": 0.8},
        )


def test_user_csv_import_writes_canonical_jsonl_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "games.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_game()))
        writer.writeheader()
        writer.writerow(_game())
    output = tmp_path / "games.jsonl"

    report = import_historical_export(
        source,
        output,
        record_type="game",
        provider=UserExportProvider("licensed-user-export"),
    )

    assert report["record_count"] == 1
    assert report["source_mutated"] is False
    assert report["qualification_granted"] is False
    assert json.loads(output.read_text().strip())["game_id"] == "WNBA:2026:001"
    assert output.with_suffix(".jsonl.provenance.json").is_file()


def test_data_gap_report_does_not_create_coefficients() -> None:
    report = data_gap_report()

    assert report["real_games_imported"] == 0
    assert report["formation_coefficients_present"] is False
    assert report["verdict"] == "PASS — no qualified trade."


def test_versioned_json_schemas_are_valid_json() -> None:
    schema_root = Path(__file__).parents[1] / "schemas" / "v1"
    names = {
        "game.schema.json",
        "market_snapshot.schema.json",
        "player_availability.schema.json",
        "formation_evidence.schema.json",
        "identity_reconciliation.schema.json",
    }

    assert {path.name for path in schema_root.glob("*.json")} == names
    assert all(json.loads((schema_root / name).read_text())["$schema"] for name in names)
