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
from betguard.simulation import (
    PaperLedgerReport,
    PaperOutcome,
    PaperTrade,
    add_paper_trade,
    analyze_paper_trades,
    load_paper_ledger,
    paper_trade_from_gate,
    save_paper_ledger,
    settle_paper_trade,
)
from betguard.video_pricing import (
    ControlledFormationFrame,
    ControlledFormationReport,
    ControlledFormationThresholds,
    VideoEvidence,
    VideoGamePrice,
    VideoPricingCalibration,
    analyze_controlled_formation,
    analyze_team_payload,
    price_game_from_video,
)

__all__ = [
    "BetProposal",
    "ControlledFormationFrame",
    "ControlledFormationReport",
    "ControlledFormationThresholds",
    "DesignConditioningReport",
    "FormationFit",
    "FormationSequenceReport",
    "FormationThresholds",
    "GateConfig",
    "GateDecision",
    "GateResult",
    "PaperLedgerReport",
    "PaperOutcome",
    "PaperTrade",
    "PortfolioConditioningReport",
    "VideoEvidence",
    "VideoGamePrice",
    "VideoPricingCalibration",
    "analyze_covariance",
    "analyze_design_matrix",
    "analyze_controlled_formation",
    "analyze_formation_sequence",
    "analyze_paper_trades",
    "analyze_team_payload",
    "add_paper_trade",
    "basketball_templates",
    "effective_number_of_bets",
    "evaluate_proposal",
    "fit_affine_formation",
    "fit_best_template",
    "fractional_kelly_binary",
    "load_paper_ledger",
    "paper_trade_from_gate",
    "price_game_from_video",
    "save_paper_ledger",
    "settle_paper_trade",
    "shrink_covariance",
]

__version__ = "0.3.0"
