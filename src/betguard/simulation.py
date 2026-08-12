"""Paper-trading ledger and performance metrics.

This module records hypothetical positions only. It has no broker, sportsbook,
wallet, payment, or account integration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from betguard.assurance import BetProposal, GateResult


class PaperOutcome(StrEnum):
    """Settlement states supported by the paper ledger."""

    OPEN = "open"
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"


@dataclass(frozen=True)
class PaperTrade:
    """One hypothetical wager admitted by the assurance gate."""

    trade_id: str
    event: str
    market: str
    selection: str
    event_start: str
    quoted_at: str
    source: str
    decimal_odds: float
    model_probability: float
    probability_uncertainty: float
    stake_units: float
    assurance_decision: str
    approved_stake_fraction: float
    entry_threshold: str = ""
    outcome: PaperOutcome = PaperOutcome.OPEN
    closing_decimal_odds: float | None = None
    settled_at: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "trade_id",
            "event",
            "market",
            "selection",
            "event_start",
            "quoted_at",
            "source",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be empty")
        if self.decimal_odds <= 1:
            raise ValueError("decimal_odds must be greater than 1")
        if not 0 <= self.model_probability <= 1:
            raise ValueError("model_probability must be between 0 and 1")
        if not 0 <= self.probability_uncertainty <= 1:
            raise ValueError("probability_uncertainty must be between 0 and 1")
        if self.stake_units <= 0:
            raise ValueError("stake_units must be positive")
        if self.approved_stake_fraction <= 0:
            raise ValueError("approved_stake_fraction must be positive")
        if self.closing_decimal_odds is not None and self.closing_decimal_odds <= 1:
            raise ValueError("closing_decimal_odds must be greater than 1")
        if self.outcome is PaperOutcome.OPEN and self.settled_at is not None:
            raise ValueError("an open trade cannot have settled_at")
        if self.outcome is not PaperOutcome.OPEN and not self.settled_at:
            raise ValueError("a settled trade requires settled_at")

    @property
    def profit_units(self) -> float:
        """Return realized paper profit, with open and void positions at zero."""

        if self.outcome is PaperOutcome.WIN:
            return self.stake_units * (self.decimal_odds - 1)
        if self.outcome is PaperOutcome.LOSS:
            return -self.stake_units
        return 0.0

    @property
    def closing_line_value(self) -> float | None:
        """Return odds-based CLV; positive means the entry beat the close."""

        if self.closing_decimal_odds is None:
            return None
        return self.decimal_odds / self.closing_decimal_odds - 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaperTrade:
        """Create a paper trade from serialized ledger data."""

        values = dict(payload)
        values["outcome"] = PaperOutcome(values.get("outcome", PaperOutcome.OPEN))
        return cls(**values)


@dataclass(frozen=True)
class PaperLedgerReport:
    """Aggregate performance for a hypothetical ledger."""

    total_trades: int
    open_trades: int
    graded_trades: int
    wins: int
    losses: int
    pushes: int
    voids: int
    risked_units: float
    net_units: float
    roi: float | None
    hit_rate: float | None
    average_closing_line_value: float | None
    brier_score: float | None
    max_drawdown_units: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def paper_trade_from_gate(
    proposal: BetProposal,
    result: GateResult,
    metadata: dict[str, Any],
) -> PaperTrade:
    """Create a paper trade after an assurance decision approves a stake."""

    if result.approved_stake_fraction <= 0:
        raise ValueError("rejected proposals cannot be recorded as paper trades")
    return PaperTrade(
        trade_id=str(metadata["trade_id"]),
        event=str(metadata["event"]),
        market=str(metadata["market"]),
        selection=str(metadata["selection"]),
        event_start=str(metadata["event_start"]),
        quoted_at=str(metadata["quoted_at"]),
        source=str(metadata["source"]),
        decimal_odds=proposal.decimal_odds,
        model_probability=proposal.model_probability,
        probability_uncertainty=proposal.probability_uncertainty,
        stake_units=float(metadata.get("stake_units", 0.25)),
        assurance_decision=result.decision.value,
        approved_stake_fraction=result.approved_stake_fraction,
        entry_threshold=str(metadata.get("entry_threshold", "")),
    )


def add_paper_trade(trades: list[PaperTrade], trade: PaperTrade) -> list[PaperTrade]:
    """Return a new ledger with one uniquely identified trade appended."""

    if any(existing.trade_id == trade.trade_id for existing in trades):
        raise ValueError(f"duplicate trade_id: {trade.trade_id}")
    return [*trades, trade]


def settle_paper_trade(
    trades: list[PaperTrade],
    trade_id: str,
    outcome: PaperOutcome,
    *,
    settled_at: str,
    closing_decimal_odds: float | None = None,
) -> list[PaperTrade]:
    """Return a new ledger with one open paper trade settled."""

    if outcome is PaperOutcome.OPEN:
        raise ValueError("settlement outcome cannot be open")
    updated: list[PaperTrade] = []
    found = False
    for trade in trades:
        if trade.trade_id != trade_id:
            updated.append(trade)
            continue
        if trade.outcome is not PaperOutcome.OPEN:
            raise ValueError(f"trade is already settled: {trade_id}")
        found = True
        updated.append(
            replace(
                trade,
                outcome=outcome,
                settled_at=settled_at,
                closing_decimal_odds=closing_decimal_odds,
            )
        )
    if not found:
        raise ValueError(f"unknown trade_id: {trade_id}")
    return updated


def analyze_paper_trades(trades: list[PaperTrade]) -> PaperLedgerReport:
    """Calculate realized paper performance and forecast calibration."""

    wins = sum(trade.outcome is PaperOutcome.WIN for trade in trades)
    losses = sum(trade.outcome is PaperOutcome.LOSS for trade in trades)
    pushes = sum(trade.outcome is PaperOutcome.PUSH for trade in trades)
    voids = sum(trade.outcome is PaperOutcome.VOID for trade in trades)
    open_trades = sum(trade.outcome is PaperOutcome.OPEN for trade in trades)
    risked = sum(
        trade.stake_units
        for trade in trades
        if trade.outcome in {PaperOutcome.WIN, PaperOutcome.LOSS, PaperOutcome.PUSH}
    )
    net = sum(trade.profit_units for trade in trades)
    win_loss_count = wins + losses
    graded = wins + losses + pushes

    clv_values = [
        value
        for trade in trades
        if (value := trade.closing_line_value) is not None
    ]
    brier_values = [
        (trade.model_probability - (1.0 if trade.outcome is PaperOutcome.WIN else 0.0))
        ** 2
        for trade in trades
        if trade.outcome in {PaperOutcome.WIN, PaperOutcome.LOSS}
    ]

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in trades:
        if trade.outcome is PaperOutcome.OPEN:
            continue
        equity += trade.profit_units
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return PaperLedgerReport(
        total_trades=len(trades),
        open_trades=open_trades,
        graded_trades=graded,
        wins=wins,
        losses=losses,
        pushes=pushes,
        voids=voids,
        risked_units=risked,
        net_units=net,
        roi=net / risked if risked else None,
        hit_rate=wins / win_loss_count if win_loss_count else None,
        average_closing_line_value=(
            sum(clv_values) / len(clv_values) if clv_values else None
        ),
        brier_score=sum(brier_values) / len(brier_values) if brier_values else None,
        max_drawdown_units=max_drawdown,
    )


def load_paper_ledger(path: Path) -> list[PaperTrade]:
    """Load a ledger, treating a missing path as an empty ledger."""

    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported paper ledger schema_version")
    return [PaperTrade.from_dict(item) for item in payload.get("trades", [])]


def save_paper_ledger(path: Path, trades: list[PaperTrade]) -> None:
    """Atomically write the complete paper ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "trades": [trade.to_dict() for trade in trades],
    }
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
