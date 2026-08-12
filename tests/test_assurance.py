import numpy as np
import pytest

from betguard.assurance import (
    BetProposal,
    GateConfig,
    GateDecision,
    evaluate_proposal,
)
from betguard.conditioning import analyze_design_matrix


def _healthy_report():
    return analyze_design_matrix(np.random.default_rng(9).normal(size=(100, 4)))


def test_accepts_healthy_positive_edge() -> None:
    proposal = BetProposal(
        name="synthetic side",
        model_probability=0.58,
        decimal_odds=1.91,
        probability_uncertainty=0.01,
        requested_stake_fraction=0.005,
    )

    result = evaluate_proposal(proposal, design_report=_healthy_report())

    assert result.decision is GateDecision.ACCEPT
    assert result.approved_stake_fraction == pytest.approx(0.005)


def test_rejects_when_uncertainty_consumes_edge() -> None:
    proposal = BetProposal(
        name="fragile total",
        model_probability=0.55,
        decimal_odds=1.91,
        probability_uncertainty=0.03,
    )

    result = evaluate_proposal(proposal)

    assert result.decision is GateDecision.REJECT
    assert result.approved_stake_fraction == 0
    assert any("conservative edge" in reason for reason in result.reasons)


def test_rejects_bad_design_conditioning() -> None:
    x = np.arange(100, dtype=float)
    collapsed_report = analyze_design_matrix(np.column_stack([x, x]))
    proposal = BetProposal(name="side", model_probability=0.60, decimal_odds=1.91)

    result = evaluate_proposal(proposal, design_report=collapsed_report)

    assert result.decision is GateDecision.REJECT
    assert any("conditioning" in reason for reason in result.reasons)


def test_reduces_correlated_factor_exposure() -> None:
    proposal = BetProposal(
        name="team total",
        model_probability=0.60,
        decimal_odds=1.91,
        requested_stake_fraction=0.01,
        factor_loadings={"team_offense": 1.0},
    )
    config = GateConfig(max_factor_exposure=0.012)

    result = evaluate_proposal(
        proposal,
        config=config,
        current_factor_exposure={"team_offense": 0.007},
    )

    assert result.decision is GateDecision.REDUCE
    assert result.approved_stake_fraction == pytest.approx(0.005)


@pytest.mark.parametrize("flag", ["out_of_distribution", "unresolved_lineup"])
def test_rejects_runtime_uncertainty_flags(flag: str) -> None:
    kwargs = {flag: True}
    proposal = BetProposal(
        name="side",
        model_probability=0.60,
        decimal_odds=1.91,
        **kwargs,
    )

    result = evaluate_proposal(proposal)

    assert result.decision is GateDecision.REJECT
