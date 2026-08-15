"""Native reader for the basketball affine-conditioning artifact bundle.

The compact pricing file contains realized observations and the source
controller's decision.  It does not contain every proposed coordinate or the
full held-map telemetry.  Consequently, source decision counts are preserved
as authoritative while realized geometry is recomputed as a separate audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_NAME = "basketball_affine_config.json"
SUMMARY_NAME = "basketball_affine_summary.json"
VERIFICATION_NAME = "basketball_affine_csv_verification.json"
PRICING_NAME = "basketball_affine_pricing_example.json"
MANIFEST_NAME = "BASKETBALL_AFFINE_V2_MANIFEST.sha256"
TELEMETRY_NAME = "basketball_affine_telemetry.csv"
README_NAME = "README.md"
SUPPORTED_SIMULATION_VERSION = "2.0.0"
SUPPORTED_PRICING_SCHEMA_VERSION = "1.0"
SOURCE_DECISIONS = {"ACCEPT", "REJECT", "HOLD LAST SAFE MAP"}

COMPACT_REQUIRED = (
    CONFIG_NAME,
    SUMMARY_NAME,
    VERIFICATION_NAME,
    PRICING_NAME,
    MANIFEST_NAME,
)

TELEMETRY_COLUMNS = (
    "simulation_version",
    "random_seed",
    "episode_seed",
    "episode_id",
    "controller_arm",
    "frame_index",
    "timestamp_seconds",
    "possession_id",
    "possession_state",
    "phase_id",
    "team_id",
    "lineup_id",
    "player_id",
    "player_role",
    "impact_rating",
    "leader_or_follower",
    "reference_x",
    "reference_y",
    "proposed_x",
    "proposed_y",
    "observed_x",
    "observed_y",
    "controlled_x",
    "controlled_y",
    "follower_weight_1",
    "follower_weight_2",
    "follower_weight_3",
    "ball_x",
    "ball_y",
    "ball_possessor_id",
    "sigma_min",
    "sigma_max",
    "normalized_conditioning",
    "determinant",
    "area_scale",
    "minimum_pairwise_spacing_feet",
    "formation_tracking_rmse_feet",
    "follower_rmse_feet",
    "tracking_gate_passed",
    "separation_gate_passed",
    "conditioning_gate_passed",
    "controller_decision",
    "held_last_safe_map",
    "last_safe_map_frame",
    "hold_start_frame",
    "hold_end_frame",
    "last_safe_map_a11",
    "last_safe_map_a12",
    "last_safe_map_a21",
    "last_safe_map_a22",
    "last_safe_translation_x",
    "last_safe_translation_y",
    "rank_loss",
    "hidden_collapse",
)


@dataclass(frozen=True)
class ArtifactVerificationReport:
    artifact_directory: str
    compact_mode: bool
    telemetry_streamed: bool
    passed: bool
    simulation_version: str | None
    pricing_schema_version: str | None
    hash_results: dict[str, str]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    telemetry: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactArmReport:
    controller_arm: str
    managed_team_id: str
    leader_ids: tuple[str, str, str]
    follower_ids: tuple[str, str]
    n_frames: int
    accepted_frames: int
    held_frames: int
    rejected_frames: int
    rejected_or_held_frames: int
    usable_fraction: float
    realized_well_conditioned_fraction: float
    source_median_controlled_conditioning: float
    observed_median_recomputed_conditioning: float
    median_follower_rmse_feet: float | None
    follower_rmse_semantics: str
    median_area_scale: float
    minimum_area_scale: float
    median_spacing_feet: float
    minimum_spacing_feet: float
    production_evidence_qualified: bool
    production_gate_reasons: tuple[str, ...]
    source_semantics: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactImportReport:
    schema_name: str
    schema_version: str
    source_artifact: str
    source_simulation_version: str
    source_rights_basis: str
    source_is_real_game_evidence: bool
    random_seed: int
    frame_count: int
    possession_ids: tuple[str, ...]
    managed_team_id: str
    context_team_ids: tuple[str, ...]
    player_rows: tuple[dict[str, Any], ...]
    fixed_follower_weights: dict[str, tuple[float, float, float]]
    comparison_arms: tuple[ArtifactArmReport, ...]
    calibration: dict[str, Any]
    paper_candidate_qualified: bool
    verdict: str
    semantic_notes: tuple[str, ...]
    verification: ArtifactVerificationReport

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def verify_artifact_directory(
    artifact_directory: Path,
    *,
    stream_telemetry: bool = False,
    verify_large_hashes: bool = False,
) -> ArtifactVerificationReport:
    """Verify compact files and optionally stream the full telemetry CSV."""

    root = artifact_directory.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    hashes: dict[str, str] = {}
    for name in COMPACT_REQUIRED:
        if not (root / name).is_file():
            errors.append(f"missing required compact artifact: {name}")
    if errors:
        return ArtifactVerificationReport(
            artifact_directory=str(root),
            compact_mode=not stream_telemetry,
            telemetry_streamed=False,
            passed=False,
            simulation_version=None,
            pricing_schema_version=None,
            hash_results=hashes,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    config = _load_json(root / CONFIG_NAME)
    summary = _load_json(root / SUMMARY_NAME)
    verification = _load_json(root / VERIFICATION_NAME)
    pricing = _load_json(root / PRICING_NAME)
    simulation_version = str(config.get("simulation_version", ""))
    pricing_version = str(pricing.get("schema_version", ""))
    if simulation_version != SUPPORTED_SIMULATION_VERSION:
        errors.append(
            f"unsupported simulation version {simulation_version!r}; "
            f"expected {SUPPORTED_SIMULATION_VERSION!r}"
        )
    for label, payload in (("summary", summary), ("pricing", pricing)):
        if payload.get("simulation_version") != simulation_version:
            errors.append(f"{label} simulation_version does not match config")
    if pricing_version != SUPPORTED_PRICING_SCHEMA_VERSION:
        errors.append(
            f"unsupported pricing schema version {pricing_version!r}; "
            f"expected {SUPPORTED_PRICING_SCHEMA_VERSION!r}"
        )

    _validate_dimensions(config, summary, pricing, verification, errors)
    manifest = _read_manifest(root / MANIFEST_NAME, errors)
    for relative_name, expected in manifest.items():
        candidate = _manifest_candidate(root, relative_name)
        if candidate is None:
            warnings.append(f"manifest entry is unavailable in this directory: {relative_name}")
            hashes[relative_name] = "unavailable"
            continue
        is_large = relative_name in {TELEMETRY_NAME, "basketball_affine_conditioning_v2.mp4"}
        if is_large and not verify_large_hashes and not (
            stream_telemetry and relative_name == TELEMETRY_NAME
        ):
            hashes[relative_name] = "declared-not-recomputed"
            continue
        actual = sha256_file(candidate)
        if actual != expected:
            errors.append(f"SHA-256 mismatch for {relative_name}")
            hashes[relative_name] = f"mismatch:{actual}"
        else:
            hashes[relative_name] = "verified"

    telemetry_report: dict[str, Any] | None = None
    if stream_telemetry:
        telemetry_path = root / TELEMETRY_NAME
        if not telemetry_path.is_file():
            errors.append(f"streaming requested but {TELEMETRY_NAME} is unavailable")
        else:
            telemetry_report = stream_verify_telemetry(
                telemetry_path,
                config=config,
                summary=summary,
            )
            errors.extend(telemetry_report.pop("errors"))
            warnings.extend(telemetry_report.pop("warnings"))

    return ArtifactVerificationReport(
        artifact_directory=str(root),
        compact_mode=not stream_telemetry,
        telemetry_streamed=stream_telemetry and telemetry_report is not None,
        passed=not errors,
        simulation_version=simulation_version,
        pricing_schema_version=pricing_version,
        hash_results=hashes,
        errors=tuple(errors),
        warnings=tuple(warnings),
        telemetry=telemetry_report,
    )


def import_artifact(
    artifact_directory: Path,
    *,
    demo_two_possessions: bool = False,
) -> ArtifactImportReport:
    """Create a canonical compact report without reinterpreting arms as teams."""

    root = artifact_directory.resolve()
    verified = verify_artifact_directory(root)
    if not verified.passed:
        raise ValueError("artifact verification failed: " + "; ".join(verified.errors))
    config = _load_json(root / CONFIG_NAME)
    pricing = _load_json(root / PRICING_NAME)
    players = tuple(config["players"])
    managed = tuple(
        player for player in players if player["leader_or_follower"] != "context_defender"
    )
    if len(managed) != 5:
        raise ValueError("artifact must contain exactly five managed formation players")
    managed_team_ids = {str(player["team_id"]) for player in managed}
    if len(managed_team_ids) != 1:
        raise ValueError("managed formation rows must belong to one stable team ID")
    managed_team_id = next(iter(managed_team_ids))
    context_team_ids = tuple(
        sorted({str(player["team_id"]) for player in players if player not in managed})
    )
    impact_order = sorted(
        managed,
        key=lambda player: (-float(player["impact_rating"]), str(player["player_id"])),
    )
    leader_ids = tuple(str(player["player_id"]) for player in impact_order[:3])
    if len(set(leader_ids)) != 3:
        raise ValueError("top-three leader identities are not unique")
    follower_ids = tuple(
        str(player["player_id"])
        for player in managed
        if str(player["player_id"]) not in leader_ids
    )
    configured_leaders = tuple(config["leader_selection"]["leader_ids"])
    if leader_ids != configured_leaders:
        raise ValueError(
            "top-three impact leaders disagree with the frozen source leader identities"
        )
    fixed_weights: dict[str, tuple[float, float, float]] = {}
    for player in managed:
        raw = player["follower_weights_by_leader"]
        if str(player["player_id"]) in follower_ids:
            if not isinstance(raw, dict) or tuple(raw) != leader_ids:
                raise ValueError("each follower must retain exactly three fixed weights")
            fixed_weights[str(player["player_id"])] = tuple(
                float(raw[leader_id]) for leader_id in leader_ids
            )
        elif raw is not None:
            raise ValueError("leader rows cannot carry follower weights")

    frames = pricing["frames"]
    _validate_compact_frame_identity(frames, config, errors=[])
    possession_ids = tuple(dict.fromkeys(str(frame["possession_id"]) for frame in frames))
    minimum_possessions = 2 if demo_two_possessions else 5
    reports = tuple(
        _analyze_compact_arm(
            arm,
            frames,
            managed,
            leader_ids=leader_ids,  # type: ignore[arg-type]
            follower_ids=follower_ids,  # type: ignore[arg-type]
            minimum_possessions=minimum_possessions,
            possession_count=len(possession_ids),
            demo_override=demo_two_possessions,
            config=config,
        )
        for arm in config["controller_arms"]
    )
    calibration = {
        "version": "illustrative-v1",
        "historical_games": int(pricing.get("historical_games", 0)),
        "validated": bool(pricing.get("validated", False)),
        "source_qualified_paper_candidate": bool(
            pricing.get("qualified_paper_candidate", False)
        ),
    }
    notes = (
        "The artifact compares two controller arms over one synthetic offense; "
        "the second team is contextual defense, not an independently weighted away formation.",
        "Source decisions evaluate proposed affine maps. Compact coordinates are realized "
        "observations after control, so source decision counts and realized-geometry audits "
        "are intentionally reported separately.",
        "The compact artifact has two possessions. A demo override may exercise software, "
        "but it cannot satisfy production qualification.",
        "No player-row permutations are used.",
    )
    paper_qualified = False
    return ArtifactImportReport(
        schema_name="betguard-native-affine-artifact",
        schema_version="1.0.0",
        source_artifact=str(root),
        source_simulation_version=str(config["simulation_version"]),
        source_rights_basis=str(pricing["evidence"]["rights_basis"]),
        source_is_real_game_evidence=False,
        random_seed=int(config["random_seed"]),
        frame_count=len(frames),
        possession_ids=possession_ids,
        managed_team_id=managed_team_id,
        context_team_ids=context_team_ids,
        player_rows=tuple(_canonical_player_row(player) for player in players),
        fixed_follower_weights=fixed_weights,
        comparison_arms=reports,
        calibration=calibration,
        paper_candidate_qualified=paper_qualified,
        verdict="PASS — no qualified trade.",
        semantic_notes=notes,
        verification=verified,
    )


def artifact_price_report(
    artifact_directory: Path,
    *,
    home_decimal_odds: float | None = None,
    away_decimal_odds: float | None = None,
    demo_two_possessions: bool = False,
) -> dict[str, Any]:
    """Return a clearly separated scenario audit and a fail-closed verdict."""

    report = import_artifact(
        artifact_directory,
        demo_two_possessions=demo_two_possessions,
    )
    source = _load_json(artifact_directory / PRICING_NAME)
    market = source["synthetic_market"]
    home_odds = float(home_decimal_odds or market["home_decimal_odds"])
    away_odds = float(away_decimal_odds or market["away_decimal_odds"])
    if home_odds <= 1 or away_odds <= 1:
        raise ValueError("two-sided decimal odds must both exceed one")
    home_implied = 1 / home_odds
    away_implied = 1 / away_odds
    total = home_implied + away_implied
    return {
        "mode": "synthetic_controller_arm_scenario_only",
        "source_is_real_game_evidence": False,
        "formation_adjustment_applied": False,
        "reason": (
            "Controller arms are not opposing teams, and the source calibration is "
            "illustrative rather than historically validated."
        ),
        "market": {
            "home_decimal_odds": home_odds,
            "away_decimal_odds": away_odds,
            "no_vig_home_probability": home_implied / total,
            "no_vig_away_probability": away_implied / total,
            "rights_basis": "synthetic",
        },
        "artifact": report.to_dict(),
        "qualified_paper_candidate": False,
        "verdict": "PASS — no qualified trade.",
    }


def stream_verify_telemetry(
    telemetry_path: Path,
    *,
    config: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify telemetry incrementally while retaining only one frame group."""

    expected_players = {str(player["player_id"]): player for player in config["players"]}
    expected_arms = tuple(str(arm) for arm in config["controller_arms"])
    errors: list[str] = []
    warnings: list[str] = []
    row_count = 0
    group_count = 0
    current_group: tuple[str, str, int] | None = None
    group_players: set[str] = set()
    decision_counts = {arm: Counter() for arm in expected_arms}
    rank_loss_frames = Counter()
    hidden_collapse_frames = Counter()

    def finish_group() -> None:
        nonlocal group_count
        if current_group is None:
            return
        if group_players != set(expected_players):
            errors.append(f"identity mismatch in telemetry frame group {current_group}")
        group_count += 1

    with telemetry_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TELEMETRY_COLUMNS:
            errors.append("telemetry header does not match the v2 schema")
            return {
                "row_count": 0,
                "frame_groups": 0,
                "decision_counts": {},
                "errors": errors,
                "warnings": warnings,
            }
        for row in reader:
            row_count += 1
            arm = row["controller_arm"]
            player_id = row["player_id"]
            if arm not in decision_counts:
                errors.append(f"unknown controller arm at row {row_count}: {arm}")
                break
            if player_id not in expected_players:
                errors.append(f"unknown player identity at row {row_count}: {player_id}")
                break
            player = expected_players[player_id]
            if row["team_id"] != str(player["team_id"]):
                errors.append(f"team identity changed for {player_id} at row {row_count}")
                break
            if row["player_role"] != str(player["role"]):
                errors.append(f"player role changed for {player_id} at row {row_count}")
                break
            if row["leader_or_follower"] != str(player["leader_or_follower"]):
                errors.append(f"leader/follower identity changed at row {row_count}")
                break
            if not math.isclose(
                float(row["impact_rating"]),
                float(player["impact_rating"]),
                rel_tol=0,
                abs_tol=1e-9,
            ):
                errors.append(f"impact rating changed for {player_id} at row {row_count}")
                break
            _validate_weight_row(row, player, row_count, errors)
            if errors:
                break
            for field in (
                "reference_x",
                "reference_y",
                "proposed_x",
                "proposed_y",
                "observed_x",
                "observed_y",
                "controlled_x",
                "controlled_y",
                "normalized_conditioning",
                "area_scale",
                "minimum_pairwise_spacing_feet",
                "formation_tracking_rmse_feet",
                "follower_rmse_feet",
            ):
                if not math.isfinite(float(row[field])):
                    errors.append(f"non-finite {field} at row {row_count}")
                    break
            if errors:
                break
            group = (row["episode_id"], arm, int(row["frame_index"]))
            if group != current_group:
                finish_group()
                current_group = group
                group_players = set()
            if player_id in group_players:
                errors.append(f"duplicate player {player_id} in frame group {group}")
                break
            group_players.add(player_id)
            if len(group_players) == 1:
                decision = row["controller_decision"]
                if decision not in SOURCE_DECISIONS:
                    errors.append(f"unknown controller decision at row {row_count}: {decision}")
                    break
                decision_counts[arm][decision] += 1
                if _as_bool(row["rank_loss"]):
                    rank_loss_frames[arm] += 1
                if _as_bool(row["hidden_collapse"]):
                    hidden_collapse_frames[arm] += 1
    finish_group()

    expected_rows = int(config["episodes_per_arm"]) * len(expected_arms) * int(
        config["video"]["frames"]
    ) * len(expected_players)
    if summary is not None and row_count != expected_rows:
        errors.append(f"telemetry row count {row_count} does not match {expected_rows}")
    compact_counts = {
        arm: {
            "accepted": decision_counts[arm]["ACCEPT"],
            "rejected": decision_counts[arm]["REJECT"],
            "held": decision_counts[arm]["HOLD LAST SAFE MAP"],
        }
        for arm in expected_arms
    }
    if summary is not None and not errors:
        for arm in expected_arms:
            expected = summary["results_by_controller_arm"][arm]["decision_frame_counts"]
            if compact_counts[arm] != expected:
                errors.append(f"decision counts disagree with summary for {arm}")
            if rank_loss_frames[arm] != int(
                summary["results_by_controller_arm"][arm]["rank_loss"]["affected_frames"]
            ):
                errors.append(f"rank-loss frame count disagrees with summary for {arm}")
            if hidden_collapse_frames[arm] != int(
                summary["results_by_controller_arm"][arm]["hidden_collapse"][
                    "affected_frames"
                ]
            ):
                errors.append(f"hidden-collapse frame count disagrees with summary for {arm}")
    return {
        "row_count": row_count,
        "frame_groups": group_count,
        "decision_counts": compact_counts,
        "rank_loss_frames": dict(rank_loss_frames),
        "hidden_collapse_frames": dict(hidden_collapse_frames),
        "bounded_state": {
            "maximum_player_ids_retained": len(expected_players),
            "episode_rows_retained": 0,
            "all_rows_retained": False,
        },
        "errors": errors,
        "warnings": warnings,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _analyze_compact_arm(
    arm: str,
    frames: list[dict[str, Any]],
    managed: tuple[dict[str, Any], ...],
    *,
    leader_ids: tuple[str, str, str],
    follower_ids: tuple[str, str],
    minimum_possessions: int,
    possession_count: int,
    demo_override: bool,
    config: dict[str, Any],
) -> ArtifactArmReport:
    player_ids = tuple(str(player["player_id"]) for player in managed)
    reference = {
        str(player["player_id"]): np.asarray(
            player["design_reference_position_feet"], dtype=float
        )
        for player in managed
    }
    leader_design = np.column_stack(
        [np.stack([reference[player_id] for player_id in leader_ids]), np.ones(3)]
    )
    if np.linalg.matrix_rank(leader_design) < 3:
        raise ValueError("frozen leader reference is affinely rank deficient")
    source_counts = Counter()
    conditioning: list[float] = []
    source_conditioning: list[float] = []
    areas: list[float] = []
    spacings: list[float] = []
    follower_errors: list[float] = []
    usable = 0
    realized_well_conditioned = 0
    last_well_conditioned: np.ndarray | None = None
    threshold = float(config["thresholds"]["normalized_conditioning_reject_below"])
    spacing_limit = float(config["thresholds"]["minimum_spacing_pass_at_or_above_feet"])
    area_limit = 0.15
    for frame in frames:
        arm_payload = frame["controller_arms"][arm]
        source_conditioning.append(float(arm_payload["normalized_conditioning"]))
        observed_by_id = {
            str(row["player_id"]): np.asarray(
                [row["observed_x"], row["observed_y"]], dtype=float
            )
            for row in arm_payload["players"]
        }
        positions = np.stack([observed_by_id[player_id] for player_id in player_ids])
        coefficients, _, _, _ = np.linalg.lstsq(
            leader_design,
            np.stack([observed_by_id[player_id] for player_id in leader_ids]),
            rcond=None,
        )
        singular = np.linalg.svd(coefficients[:2, :], compute_uv=False)
        normalized = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
        area = abs(float(np.linalg.det(coefficients[:2, :])))
        spacing = _minimum_spacing(positions)
        conditioning.append(normalized)
        areas.append(area)
        spacings.append(spacing)
        realized_safe = normalized >= threshold and area >= area_limit and spacing >= spacing_limit
        if realized_safe:
            realized_well_conditioned += 1
        decision = str(arm_payload["controller_decision"])
        if decision not in SOURCE_DECISIONS:
            raise ValueError(f"unknown source decision {decision!r}")
        source_counts[decision] += 1
        if decision == "ACCEPT":
            control_map = coefficients
            if realized_safe:
                last_well_conditioned = coefficients.copy()
        else:
            control_map = last_well_conditioned
        follower_rmse: float | None = None
        if control_map is not None:
            design = np.column_stack(
                [np.stack([reference[player_id] for player_id in follower_ids]), np.ones(2)]
            )
            predicted = design @ control_map
            actual = np.stack([observed_by_id[player_id] for player_id in follower_ids])
            follower_rmse = float(
                np.sqrt(np.mean(np.sum(np.square(predicted - actual), axis=1)))
            )
            follower_errors.append(follower_rmse)
            if follower_rmse <= 5.0:
                usable += 1

    reasons: list[str] = []
    if possession_count < minimum_possessions:
        reasons.append(
            f"too few possessions: {possession_count} < {minimum_possessions}"
        )
    if demo_override:
        reasons.append("demo two-possession override cannot qualify a paper candidate")
    reasons.append("synthetic artifact is not verified real-game evidence")
    return ArtifactArmReport(
        controller_arm=arm,
        managed_team_id=str(managed[0]["team_id"]),
        leader_ids=leader_ids,
        follower_ids=follower_ids,
        n_frames=len(frames),
        accepted_frames=source_counts["ACCEPT"],
        held_frames=source_counts["HOLD LAST SAFE MAP"],
        rejected_frames=source_counts["REJECT"],
        rejected_or_held_frames=(
            source_counts["REJECT"] + source_counts["HOLD LAST SAFE MAP"]
        ),
        usable_fraction=usable / len(frames),
        realized_well_conditioned_fraction=realized_well_conditioned / len(frames),
        source_median_controlled_conditioning=float(np.median(source_conditioning)),
        observed_median_recomputed_conditioning=float(np.median(conditioning)),
        median_follower_rmse_feet=(
            float(np.median(follower_errors)) if follower_errors else None
        ),
        follower_rmse_semantics=(
            "compact reconstruction from observed fixed-identity rows and the last "
            "observed well-conditioned accepted map; full telemetry retains the exact "
            "source last-safe matrix and source follower RMSE"
        ),
        median_area_scale=float(np.median(areas)),
        minimum_area_scale=min(areas),
        median_spacing_feet=float(np.median(spacings)),
        minimum_spacing_feet=min(spacings),
        production_evidence_qualified=False,
        production_gate_reasons=tuple(reasons),
        source_semantics=(
            "accepted/held/rejected use stored proposed-map controller decisions; "
            "geometry statistics are recomputed from realized observed coordinates"
        ),
    )


def _validate_dimensions(
    config: dict[str, Any],
    summary: dict[str, Any],
    pricing: dict[str, Any],
    verification: dict[str, Any],
    errors: list[str],
) -> None:
    try:
        frames = int(config["video"]["frames"])
        fps = float(config["video"]["fps"])
        episodes = int(config["episodes_per_arm"])
        arms = tuple(config["controller_arms"])
        players = tuple(config["players"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid config dimensions: {exc}")
        return
    if frames < 1 or fps <= 0 or episodes < 1 or len(arms) != 2 or len(players) != 10:
        errors.append("unexpected video, episode, arm, or player dimensions")
    if int(summary.get("video_frames", -1)) != frames:
        errors.append("summary video frame count does not match config")
    if int(summary.get("episodes_per_arm", -1)) != episodes:
        errors.append("summary episode count does not match config")
    expected_rows = episodes * len(arms) * frames * len(players)
    if int(summary.get("telemetry_rows", -1)) != expected_rows:
        errors.append("summary telemetry row count does not match dimensions")
    if int(verification.get("all_csv_rows", -1)) != expected_rows:
        errors.append("CSV verification row count does not match dimensions")
    if len(pricing.get("frames", [])) != frames:
        errors.append("pricing example frame count does not match config")
    _validate_compact_frame_identity(pricing.get("frames", []), config, errors)
    for arm in arms:
        counts = summary.get("results_by_controller_arm", {}).get(arm, {}).get(
            "decision_frame_counts", {}
        )
        if sum(int(counts.get(name, -10**12)) for name in ("accepted", "rejected", "held")) != (
            episodes * frames
        ):
            errors.append(f"summary decision counts do not reconcile for {arm}")


def _validate_compact_frame_identity(
    frames: list[dict[str, Any]],
    config: dict[str, Any],
    errors: list[str],
) -> None:
    expected_ids = tuple(str(player["player_id"]) for player in config["players"])
    arms = tuple(str(arm) for arm in config["controller_arms"])
    fps = float(config["video"]["fps"])
    for expected_index, frame in enumerate(frames):
        if int(frame.get("frame_index", -1)) != expected_index:
            errors.append(f"compact frame order changed at index {expected_index}")
            return
        if not math.isclose(
            float(frame.get("timestamp_seconds", -1)),
            expected_index / fps,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            errors.append(f"compact timestamp mismatch at frame {expected_index}")
            return
        if not str(frame.get("possession_id", "")):
            errors.append(f"missing possession ID at frame {expected_index}")
            return
        if tuple(frame.get("controller_arms", {})) != arms:
            errors.append(f"controller-arm ordering changed at frame {expected_index}")
            return
        for arm in arms:
            rows = frame["controller_arms"][arm].get("players", [])
            ids = tuple(str(row.get("player_id", "")) for row in rows)
            if ids != expected_ids:
                errors.append(
                    f"player identity/order mismatch at frame {expected_index}, arm {arm}"
                )
                return


def _read_manifest(path: Path, errors: list[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"malformed manifest line {line_number}")
            continue
        digest, relative_name = parts
        if any(character not in "0123456789abcdef" for character in digest.lower()):
            errors.append(f"invalid manifest digest on line {line_number}")
            continue
        candidate = Path(relative_name.strip())
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"unsafe manifest path on line {line_number}")
            continue
        entries[candidate.as_posix()] = digest.lower()
    return entries


def _manifest_candidate(root: Path, relative_name: str) -> Path | None:
    local = root / relative_name
    if local.is_file():
        return local
    relative = Path(relative_name)
    if relative.parts and relative.parts[0] == "analysis" and len(root.parents) >= 3:
        lab_relative = root.parents[2] / relative
        if lab_relative.is_file():
            return lab_relative
    return None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _canonical_player_row(player: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_id": str(player["player_id"]),
        "team_id": str(player["team_id"]),
        "lineup_id": str(player["lineup_id"]),
        "role": str(player["role"]),
        "impact_rating": float(player["impact_rating"]),
        "leader_or_follower": str(player["leader_or_follower"]),
        "design_reference_position_feet": tuple(
            float(value) for value in player["design_reference_position_feet"]
        ),
    }


def _validate_weight_row(
    row: dict[str, str],
    player: dict[str, Any],
    row_count: int,
    errors: list[str],
) -> None:
    expected = player["follower_weights_by_leader"]
    actual = [row[f"follower_weight_{index}"] for index in range(1, 4)]
    if expected is None:
        if any(value.strip() for value in actual):
            errors.append(f"unexpected follower weight at row {row_count}")
        return
    if any(not value.strip() for value in actual):
        errors.append(f"missing follower weight at row {row_count}")
        return
    expected_values = tuple(float(value) for value in expected.values())
    if any(
        not math.isclose(float(value), target, rel_tol=0, abs_tol=1e-6)
        for value, target in zip(actual, expected_values, strict=True)
    ):
        errors.append(f"follower weights changed at row {row_count}")


def _minimum_spacing(positions: np.ndarray) -> float:
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(np.square(differences), axis=2))
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.min(distances))


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid Boolean value {value!r}")
