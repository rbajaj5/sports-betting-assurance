"""Independent accept/reduce/reject gate for proposed wagers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from betguard.conditioning import DesignConditioningReport
from betguard.portfolio import fractional_kelly_binary


class GateDecision(StrEnum):
    """Possible runtime-assurance outcomes."""

    ACCEPT = "accept"
    REDUCE = "reduce"
    REJECT = "reject"


@dataclass(frozen=True)
class BetProposal:
    """A wager proposed by an upstream probability model."""

    name: str
    model_probability: float
    decimal_odds: float
    probability_uncertainty: float = 0.0
    requested_stake_fraction: float | None = None
    factor_loadings: Mapping[str, float] = field(default_factory=dict)
    out_of_distribution: bool = False
    unresolved_lineup: bool = False


@dataclass(frozen=True)
class GateConfig:
    """Safety limits applied independently of the predictive model."""

    min_conservative_edge: float = 0.01
    max_probability_uncertainty: float = 0.05
    max_stake_fraction: float = 0.01
    max_factor_exposure: float = 0.015
    kelly_fraction: float = 0.25
    reject_bad_conditioning: bool = True


@dataclass(frozen=True)
class GateResult:
    """Decision and audit trail returned by the assurance gate."""

    decision: GateDecision
    approved_stake_fraction: float
    reasons: tuple[str, ...]
    metrics: Mapping[str, float | str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload


def evaluate_proposal(
    proposal: BetProposal,
    *,
    config: GateConfig | None = None,
    current_factor_exposure: Mapping[str, float] | None = None,
    design_report: DesignConditioningReport | None = None,
) -> GateResult:
    """Apply a fail-safe runtime-assurance gate to a proposed wager."""

    settings = config or GateConfig()
    current_exposure = current_factor_exposure or {}
    _validate_proposal(proposal)
    _validate_config(settings)

    break_even_probability = 1.0 / proposal.decimal_odds
    conservative_probability = max(
        0.0, proposal.model_probability - proposal.probability_uncertainty
    )
    raw_edge = proposal.model_probability - break_even_probability
    conservative_edge = conservative_probability - break_even_probability
    metrics: dict[str, float | str] = {
        "break_even_probability": break_even_probability,
        "raw_edge": raw_edge,
        "conservative_probability": conservative_probability,
        "conservative_edge": conservative_edge,
    }
    rejection_reasons: list[str] = []

    if proposal.out_of_distribution:
        rejection_reasons.append("matchup is outside the model's supported distribution")
    if proposal.unresolved_lineup:
        rejection_reasons.append("important lineup or injury information is unresolved")
    if proposal.probability_uncertainty > settings.max_probability_uncertainty:
        rejection_reasons.append(
            "probability uncertainty above limit: "
            f"{proposal.probability_uncertainty:.3g} > "
            f"{settings.max_probability_uncertainty:.3g}"
        )
    if settings.reject_bad_conditioning and design_report and not design_report.accepted:
        rejection_reasons.append("predictive feature matrix failed conditioning checks")
    if conservative_edge < settings.min_conservative_edge:
        rejection_reasons.append(
            "conservative edge below limit: "
            f"{conservative_edge:.3g} < {settings.min_conservative_edge:.3g}"
        )

    if rejection_reasons:
        return GateResult(
            decision=GateDecision.REJECT,
            approved_stake_fraction=0.0,
            reasons=tuple(rejection_reasons),
            metrics=metrics,
        )

    suggested_stake = fractional_kelly_binary(
        conservative_probability,
        proposal.decimal_odds,
        fraction=settings.kelly_fraction,
        cap=settings.max_stake_fraction,
    )
    requested_stake = (
        proposal.requested_stake_fraction
        if proposal.requested_stake_fraction is not None
        else suggested_stake
    )
    metrics["fractional_kelly_stake"] = suggested_stake
    metrics["requested_stake_fraction"] = requested_stake

    reduction_reasons: list[str] = []
    approved_stake = requested_stake
    if approved_stake > settings.max_stake_fraction:
        approved_stake = settings.max_stake_fraction
        reduction_reasons.append("stake capped by maximum bankroll fraction")

    for factor, loading in proposal.factor_loadings.items():
        if not loading:
            continue
        existing = abs(float(current_exposure.get(factor, 0.0)))
        capacity = settings.max_factor_exposure - existing
        allowed_stake = max(0.0, capacity / abs(float(loading)))
        if allowed_stake < approved_stake:
            approved_stake = allowed_stake
            reduction_reasons.append(f"stake reduced by '{factor}' exposure cap")

    if approved_stake <= 0:
        return GateResult(
            decision=GateDecision.REJECT,
            approved_stake_fraction=0.0,
            reasons=tuple(reduction_reasons or ["no stake remains after exposure limits"]),
            metrics=metrics,
        )

    metrics["approved_stake_fraction"] = approved_stake
    return GateResult(
        decision=GateDecision.REDUCE if reduction_reasons else GateDecision.ACCEPT,
        approved_stake_fraction=approved_stake,
        reasons=tuple(reduction_reasons or ["all configured assurance checks passed"]),
        metrics=metrics,
    )


def _validate_proposal(proposal: BetProposal) -> None:
    if not proposal.name.strip():
        raise ValueError("proposal name cannot be empty")
    if not 0 <= proposal.model_probability <= 1:
        raise ValueError("model_probability must be between 0 and 1")
    if proposal.decimal_odds <= 1:
        raise ValueError("decimal_odds must be greater than 1")
    if not 0 <= proposal.probability_uncertainty <= 1:
        raise ValueError("probability_uncertainty must be between 0 and 1")
    if proposal.requested_stake_fraction is not None and proposal.requested_stake_fraction < 0:
        raise ValueError("requested_stake_fraction cannot be negative")
    if any(not factor.strip() for factor in proposal.factor_loadings):
        raise ValueError("factor names cannot be empty")


def _validate_config(config: GateConfig) -> None:
    if config.min_conservative_edge < 0:
        raise ValueError("min_conservative_edge cannot be negative")
    if not 0 <= config.max_probability_uncertainty <= 1:
        raise ValueError("max_probability_uncertainty must be between 0 and 1")
    if config.max_stake_fraction < 0 or config.max_factor_exposure < 0:
        raise ValueError("stake and exposure limits cannot be negative")
    if not 0 <= config.kelly_fraction <= 1:
        raise ValueError("kelly_fraction must be between 0 and 1")
