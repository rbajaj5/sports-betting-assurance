from __future__ import annotations

import numpy as np
import pytest

from betguard.evaluation import (
    NO_TRADE_VERDICT,
    QualificationInputs,
    assert_pregame_row,
    bootstrap_metric_interval,
    chronological_split,
    devig_two_sided,
    evaluate_qualification,
    fit_formation_increment,
    fit_pregame_baseline,
    flat_stake_paper_metrics,
    probability_metrics,
)


def test_devigging_normalizes_two_sided_market() -> None:
    home, away = devig_two_sided(1.86, 2.08)
    assert home + away == pytest.approx(1.0)
    assert home == pytest.approx((1 / 1.86) / ((1 / 1.86) + (1 / 2.08)))


def test_chronological_split_is_sorted_and_nonoverlapping() -> None:
    records = [
        {"game_id": f"g{index}", "time": f"2026-06-{index:02d}T12:00:00Z"}
        for index in range(1, 11)
    ]
    records.reverse()
    split = chronological_split(records, timestamp_field="time")

    assert [row["game_id"] for row in split.train] == [f"g{i}" for i in range(1, 7)]
    assert [row["game_id"] for row in split.validation] == ["g7", "g8"]
    assert [row["game_id"] for row in split.test] == ["g9", "g10"]


def test_pregame_feature_gate_rejects_postmarket_information() -> None:
    row = {
        "scheduled_start_time": "2026-06-01T19:00:00Z",
        "market_observation_timestamp": "2026-06-01T17:00:00Z",
        "rest_days": 2,
        "feature_available_at": {"rest_days": "2026-06-01T18:00:00Z"},
    }
    with pytest.raises(ValueError, match="after the quoted market"):
        assert_pregame_row(row, feature_fields=["rest_days"])

    with pytest.raises(ValueError, match="forbidden"):
        assert_pregame_row(row | {"home_final_score": 80}, feature_fields=["home_final_score"])


def test_qualification_fails_closed_for_demo_artifact() -> None:
    result = evaluate_qualification(
        QualificationInputs(
            verified_real_game_provenance=False,
            stable_player_identities=True,
            calibration_rmse_feet=0.0,
            usable_frames=300,
            possessions=2,
            projected_lineup_coverage=1.0,
            pregame_impact_ratings_present=True,
            availability_assumptions_present=False,
            coefficients_version=None,
            coefficients_frozen=False,
            qualifying_historical_games=0,
            regulated_two_sided_prices=False,
            conservative_post_uncertainty_edge=0.20,
            synthetic_or_demo_override=True,
        )
    )

    assert not result.qualified
    assert result.verdict == NO_TRADE_VERDICT
    assert any("fewer than five possessions" in reason for reason in result.reasons)


def test_qualification_requires_every_gate() -> None:
    result = evaluate_qualification(
        QualificationInputs(
            verified_real_game_provenance=True,
            stable_player_identities=True,
            calibration_rmse_feet=1.0,
            usable_frames=12,
            possessions=5,
            projected_lineup_coverage=0.8,
            pregame_impact_ratings_present=True,
            availability_assumptions_present=True,
            coefficients_version="frozen-v1",
            coefficients_frozen=True,
            qualifying_historical_games=200,
            regulated_two_sided_prices=True,
            conservative_post_uncertainty_edge=0.03,
        )
    )

    assert result.qualified


def test_probability_and_paper_metrics_are_computed_not_hardcoded() -> None:
    probabilities = [0.2, 0.4, 0.6, 0.8]
    outcomes = [0, 0, 1, 1]
    report = probability_metrics(probabilities, outcomes, bins=2)
    interval = bootstrap_metric_interval(
        probabilities,
        outcomes,
        lambda predicted, observed: float(np.mean(np.square(predicted - observed))),
        samples=200,
    )
    paper = flat_stake_paper_metrics([0.2, -0.25, 0.2], closing_line_values=[0.01, 0.0, 0.02])

    assert report.brier_score == pytest.approx(0.1)
    assert interval[0] <= report.brier_score <= interval[1]
    assert paper["bets"] == 3
    assert paper["turnover_units"] == pytest.approx(0.75)


def test_baseline_and_formation_layers_are_independently_frozen() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(220, 2))
    outcomes = (features[:, 0] + rng.normal(size=220) > 0).astype(int)
    game_ids = [f"g{index:03d}" for index in range(220)]
    baseline = fit_pregame_baseline(
        features,
        outcomes,
        feature_names=["team_strength", "rest_days"],
        training_game_ids=game_ids,
        version="baseline-frozen-v1",
    )
    baseline_probabilities = baseline.predict(features)
    formation = rng.normal(size=(220, 1))
    increment = fit_formation_increment(
        baseline_probabilities,
        formation,
        outcomes,
        feature_names=["leader_conditioning"],
        training_game_ids=game_ids,
        version="formation-frozen-v1",
        baseline_version=baseline.version,
        verified_real_game_evidence=True,
    )

    assert baseline.layer == "pregame_baseline"
    assert increment.baseline_version == baseline.version
    assert increment.adjust(baseline_probabilities, formation).shape == (220,)


def test_synthetic_evidence_cannot_fit_formation_coefficients() -> None:
    with pytest.raises(ValueError, match="synthetic formation evidence"):
        fit_formation_increment(
            [0.5] * 200,
            np.arange(200, dtype=float).reshape(-1, 1),
            [0, 1] * 100,
            feature_names=["conditioning"],
            training_game_ids=[f"g{index}" for index in range(200)],
            version="forbidden-v1",
            baseline_version="baseline-v1",
            verified_real_game_evidence=False,
        )
