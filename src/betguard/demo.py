"""Synthetic basketball demonstration of conditioning-aware assurance."""

from __future__ import annotations

import json

import numpy as np

from betguard.assurance import BetProposal, GateConfig, evaluate_proposal
from betguard.conditioning import analyze_design_matrix
from betguard.formation import analyze_formation_sequence, basketball_templates
from betguard.portfolio import analyze_covariance, shrink_covariance


def run_demo() -> None:
    """Print deterministic design, portfolio, and gate diagnostics."""

    rng = np.random.default_rng(17)
    observations = 240
    pace = rng.normal(78.0, 3.5, observations)
    offensive_rating = rng.normal(108.0, 7.0, observations)
    turnover_rate = rng.normal(0.16, 0.02, observations)
    three_point_rate = rng.normal(0.34, 0.04, observations)
    points_per_game = pace * offensive_rating / 100 + rng.normal(0, 1.5, observations)
    points_per_game_copy = points_per_game + rng.normal(0, 1e-8, observations)

    healthy_features = np.column_stack([pace, offensive_rating, turnover_rate, three_point_rate])
    collapsed_features = np.column_stack([points_per_game, points_per_game_copy, offensive_rating])

    healthy_report = analyze_design_matrix(healthy_features)
    collapsed_report = analyze_design_matrix(collapsed_features)

    correlation = np.array(
        [
            [1.00, 0.97, 0.89, 0.08],
            [0.97, 1.00, 0.86, 0.05],
            [0.89, 0.86, 1.00, 0.12],
            [0.08, 0.05, 0.12, 1.00],
        ]
    )
    volatility = np.diag([1.0, 1.0, 0.9, 1.1])
    covariance = volatility @ correlation @ volatility
    portfolio_report = analyze_covariance(covariance, min_effective_bets=2.5)
    shrunk_report = analyze_covariance(shrink_covariance(covariance, 0.30), min_effective_bets=2.5)

    proposal = BetProposal(
        name="Synthetic Indiana team-total over",
        model_probability=0.57,
        decimal_odds=1.91,
        probability_uncertainty=0.015,
        requested_stake_fraction=0.008,
        factor_loadings={"indiana_offense": 1.0, "game_pace": 0.7},
    )
    config = GateConfig(max_factor_exposure=0.012)
    current_exposure = {"indiana_offense": 0.007, "game_pace": 0.004}
    healthy_gate = evaluate_proposal(
        proposal,
        config=config,
        current_factor_exposure=current_exposure,
        design_report=healthy_report,
    )
    collapsed_gate = evaluate_proposal(
        proposal,
        config=config,
        current_factor_exposure=current_exposure,
        design_report=collapsed_report,
    )

    reference = basketball_templates()["five_out"]
    compressions = np.concatenate([np.linspace(1.0, 0.05, 8), np.linspace(0.05, 1.0, 8)[1:]])
    formation_frames = np.stack(
        [
            reference @ np.array([[1.0, 0.0], [0.0, compression]])
            + np.array([3.0, 8.0])
            + rng.normal(0, 0.03, reference.shape)
            for compression in compressions
        ]
    )
    formation_report = analyze_formation_sequence(
        reference,
        formation_frames,
        template_name="five_out",
    )

    output = {
        "healthy_design": healthy_report.to_dict(),
        "collapsed_design": collapsed_report.to_dict(),
        "correlated_portfolio": portfolio_report.to_dict(),
        "shrunk_portfolio": shrunk_report.to_dict(),
        "gate_with_healthy_design": healthy_gate.to_dict(),
        "gate_with_collapsed_design": collapsed_gate.to_dict(),
        "basketball_formation_sequence": formation_report.to_dict(include_fits=False),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    run_demo()
