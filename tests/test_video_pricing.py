from __future__ import annotations

import numpy as np

from betguard.video_pricing import (
    ControlledFormationThresholds,
    VideoEvidence,
    VideoPricingCalibration,
    analyze_controlled_formation,
    price_game_from_video,
)

REFERENCE = np.array(
    [
        [0.0, 10.0],
        [-8.0, 5.0],
        [8.0, 5.0],
        [-10.0, 8.0],
        [10.0, 8.0],
    ]
)
PLAYER_IDS = ("p1", "p2", "p3", "p4", "p5")


def _evidence(*, identities_verified: bool = True) -> VideoEvidence:
    return VideoEvidence(
        source_uri="synthetic://unit-test",
        captured_at="2026-08-12T12:00:00Z",
        clip_sha256="a" * 64,
        rights_basis="synthetic",
        annotation_method="deterministic unit-test coordinates",
        frame_timestamps=(0.0, 1.0, 2.0),
        possession_count=2,
        lineup_coverage=1.0,
        court_calibration_rmse_feet=0.1,
        identities_verified=identities_verified,
    )


def _thresholds() -> ControlledFormationThresholds:
    return ControlledFormationThresholds(
        min_area_scale=0.01,
        min_pairwise_spacing_feet=0.0,
        max_follower_rmse_feet=5.0,
        min_usable_frames=3,
        min_possessions=2,
        min_well_conditioned_fraction=0.5,
        min_usable_fraction=0.5,
    )


def _transform(matrix: np.ndarray, translation: tuple[float, float]) -> np.ndarray:
    return REFERENCE @ matrix + np.asarray(translation)


def _report(
    matrix: np.ndarray,
    ratings: dict[str, float],
    *,
    identities_verified: bool = True,
):
    frames = np.stack(
        [
            _transform(matrix, (0.0, 0.0)),
            _transform(matrix, (0.5, 0.2)),
            _transform(matrix, (1.0, 0.4)),
        ]
    )
    return analyze_controlled_formation(
        REFERENCE,
        frames,
        player_ids=PLAYER_IDS,
        impact_ratings=ratings,
        evidence=_evidence(identities_verified=identities_verified),
        thresholds=_thresholds(),
    )


def test_holds_last_well_conditioned_map_during_collapse() -> None:
    healthy = _transform(np.eye(2), (0.0, 0.0))
    translated = _transform(np.eye(2), (1.0, 0.0))
    collapsed = _transform(np.diag([1.0, 0.05]), (1.0, 0.0))
    report = analyze_controlled_formation(
        REFERENCE,
        np.stack([healthy, translated, collapsed]),
        player_ids=PLAYER_IDS,
        impact_ratings={"p1": 10, "p2": 9, "p3": 8, "p4": 2, "p5": 1},
        evidence=_evidence(),
        thresholds=_thresholds(),
    )

    assert report.leader_ids == ("p1", "p2", "p3")
    assert report.accepted_frames == 2
    assert report.held_frames == 1
    assert report.frames[2].normalized_leader_conditioning < 0.12
    assert report.frames[2].held_last_well_conditioned_map
    assert report.frames[2].follower_rmse_feet is not None
    assert report.frames[2].follower_rmse_feet > 5.0


def test_unverified_player_identities_fail_closed() -> None:
    report = _report(
        np.eye(2),
        {"p1": 10, "p2": 9, "p3": 8, "p4": 2, "p5": 1},
        identities_verified=False,
    )

    assert not report.quality_qualified
    assert "player identities were not verified" in report.quality_reasons


def test_validated_calibration_can_create_paper_candidate() -> None:
    home = _report(
        np.eye(2),
        {"p1": 10, "p2": 9, "p3": 8, "p4": 2, "p5": 1},
    )
    away = _report(
        np.diag([1.0, 0.4]),
        {"p1": 6, "p2": 5, "p3": 4, "p4": 2, "p5": 1},
    )
    calibration = VideoPricingCalibration(
        version="frozen-test-v1",
        historical_games=300,
        minimum_historical_games=200,
        validated=True,
        probability_uncertainty=0.02,
    )

    price = price_game_from_video(
        home,
        away,
        baseline_home_probability=0.52,
        market_home_decimal_odds=2.0,
        market_away_decimal_odds=2.0,
        calibration=calibration,
    )

    assert price.formation_adjustment_applied
    assert price.adjusted_home_probability > price.baseline_home_probability
    assert price.conservative_home_edge >= 0.03
    assert price.paper_candidate == "home"
    assert price.qualified


def test_illustrative_calibration_remains_scenario_only() -> None:
    home = _report(
        np.eye(2),
        {"p1": 10, "p2": 9, "p3": 8, "p4": 2, "p5": 1},
    )
    away = _report(
        np.diag([1.0, 0.4]),
        {"p1": 6, "p2": 5, "p3": 4, "p4": 2, "p5": 1},
    )

    price = price_game_from_video(
        home,
        away,
        baseline_home_probability=0.52,
        market_home_decimal_odds=2.0,
        market_away_decimal_odds=2.0,
    )

    assert price.scenario_preference == "home"
    assert price.paper_candidate is None
    assert not price.qualified
    assert any("illustrative" in reason for reason in price.reasons)
