from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from betguard.artifact import (
    TELEMETRY_COLUMNS,
    artifact_price_report,
    import_artifact,
    stream_verify_telemetry,
    verify_artifact_directory,
)
from betguard.cli import build_parser

FIXTURE = Path(__file__).parent / "fixtures" / "basketball_affine_conditioning_v2_compact"


def test_compact_artifact_preserves_native_controller_semantics() -> None:
    report = import_artifact(FIXTURE, demo_two_possessions=True)
    arms = {arm.controller_arm: arm for arm in report.comparison_arms}

    assert report.managed_team_id == "SYN_HOME_AURORAS"
    assert report.context_team_ids == ("SYN_AWAY_SENTINELS",)
    assert tuple(row["player_id"] for row in report.player_rows[:3]) == (
        "PG01",
        "WG01",
        "C01",
    )
    assert arms["tracking_and_separation"].accepted_frames == 275
    assert arms["tracking_and_separation"].held_frames == 17
    assert arms["tracking_and_separation"].rejected_frames == 8
    assert arms["conditioning_aware_rta"].accepted_frames == 209
    assert arms["conditioning_aware_rta"].held_frames == 76
    assert arms["conditioning_aware_rta"].rejected_frames == 15
    assert all(not arm.production_evidence_qualified for arm in arms.values())
    assert not report.paper_candidate_qualified
    assert report.verdict == "PASS — no qualified trade."
    assert any("proposed" in note and "realized" in note for note in report.semantic_notes)


def test_normal_production_gate_rejects_two_possession_artifact() -> None:
    report = import_artifact(FIXTURE)

    assert all(
        "too few possessions: 2 < 5" in arm.production_gate_reasons
        for arm in report.comparison_arms
    )


def test_compact_verification_checks_available_hashes() -> None:
    report = verify_artifact_directory(FIXTURE)

    assert report.passed
    assert report.hash_results["basketball_affine_config.json"] == "verified"
    assert report.hash_results["basketball_affine_telemetry.csv"] == "unavailable"
    assert any("telemetry" in warning for warning in report.warnings)


def test_hash_mismatch_fails_verification(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(FIXTURE, copied)
    config_path = copied / "basketball_affine_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["random_seed"] += 1
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_artifact_directory(copied)

    assert not report.passed
    assert any("SHA-256 mismatch" in error for error in report.errors)


def test_artifact_price_never_relabels_arms_as_opposing_teams() -> None:
    output = artifact_price_report(FIXTURE, demo_two_possessions=True)

    assert output["mode"] == "synthetic_controller_arm_scenario_only"
    assert not output["formation_adjustment_applied"]
    assert not output["qualified_paper_candidate"]
    assert output["verdict"] == "PASS — no qualified trade."


def test_streaming_verifier_retains_only_one_frame_group(tmp_path: Path) -> None:
    config = json.loads((FIXTURE / "basketball_affine_config.json").read_text())
    config["episodes_per_arm"] = 1
    config["video"]["frames"] = 2
    path = tmp_path / "small_telemetry.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TELEMETRY_COLUMNS)
        writer.writeheader()
        for arm in config["controller_arms"]:
            for frame_index in range(2):
                for player in config["players"]:
                    row = {field: "0" for field in TELEMETRY_COLUMNS}
                    row.update(
                        {
                            "simulation_version": "2.0.0",
                            "random_seed": "20260813",
                            "episode_seed": "1",
                            "episode_id": "BBAFF-E0000",
                            "controller_arm": arm,
                            "frame_index": str(frame_index),
                            "timestamp_seconds": str(frame_index / 15),
                            "possession_id": "BBAFF-E0000-POS01",
                            "possession_state": "guard_control",
                            "phase_id": "PHASE_TEST",
                            "team_id": player["team_id"],
                            "lineup_id": player["lineup_id"],
                            "player_id": player["player_id"],
                            "player_role": player["role"],
                            "impact_rating": str(player["impact_rating"]),
                            "leader_or_follower": player["leader_or_follower"],
                            "controller_decision": "ACCEPT",
                            "tracking_gate_passed": "true",
                            "separation_gate_passed": "true",
                            "conditioning_gate_passed": "true",
                            "held_last_safe_map": "false",
                            "rank_loss": "false",
                            "hidden_collapse": "false",
                        }
                    )
                    weights = player["follower_weights_by_leader"]
                    weight_values = None if weights is None else tuple(weights.values())
                    for index in range(1, 4):
                        row[f"follower_weight_{index}"] = (
                            "" if weight_values is None else str(weight_values[index - 1])
                        )
                    writer.writerow(row)

    report = stream_verify_telemetry(path, config=config)

    assert report["errors"] == []
    assert report["row_count"] == 40
    assert report["frame_groups"] == 4
    assert report["bounded_state"]["maximum_player_ids_retained"] == 10
    assert report["bounded_state"]["all_rows_retained"] is False


def test_native_cli_commands_are_exposed() -> None:
    help_text = build_parser().format_help()

    assert "artifact-import" in help_text
    assert "artifact-verify" in help_text
    assert "artifact-price" in help_text
    assert "historical-import" in help_text
