"""Conditioning-aware assurance checks for sports-betting systems."""

from betguard.assurance import (
    BetProposal,
    GateConfig,
    GateDecision,
    GateResult,
    evaluate_proposal,
)
from betguard.conditioning import DesignConditioningReport, analyze_design_matrix
from betguard.formation import (
    FormationFit,
    FormationSequenceReport,
    FormationThresholds,
    analyze_formation_sequence,
    basketball_templates,
    fit_affine_formation,
    fit_best_template,
)
from betguard.portfolio import (
    PortfolioConditioningReport,
    analyze_covariance,
    effective_number_of_bets,
    fractional_kelly_binary,
    shrink_covariance,
)

__all__ = [
    "BetProposal",
    "DesignConditioningReport",
    "FormationFit",
    "FormationSequenceReport",
    "FormationThresholds",
    "GateConfig",
    "GateDecision",
    "GateResult",
    "PortfolioConditioningReport",
    "analyze_covariance",
    "analyze_design_matrix",
    "analyze_formation_sequence",
    "basketball_templates",
    "effective_number_of_bets",
    "evaluate_proposal",
    "fit_affine_formation",
    "fit_best_template",
    "fractional_kelly_binary",
    "shrink_covariance",
]

__version__ = "0.1.0"
