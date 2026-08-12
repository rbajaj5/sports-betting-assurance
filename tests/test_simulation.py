import json
from pathlib import Path

import pytest

from betguard.assurance import BetProposal, evaluate_proposal
from betguard.cli import main
from betguard.simulation import (
    PaperOutcome,
    add_paper_trade,
    analyze_paper_trades,
    load_paper_ledger,
    paper_trade_from_gate,
    save_paper_ledger,
    settle_paper_trade,
)


def _paper_trade(trade_id: str, probability: float = 0.60, odds: float = 2.0):
    proposal = BetProposal(
        name=f"synthetic {trade_id}",
        model_probability=probability,
        decimal_odds=odds,
        requested_stake_fraction=0.005,
    )
    result = evaluate_proposal(proposal)
    return paper_trade_from_gate(
        proposal,
        result,
        {
            "trade_id": trade_id,
            "event": "Alpha at Beta",
            "market": "spread",
            "selection": "Alpha +4.5",
            "event_start": "2026-08-12T19:00:00-04:00",
            "quoted_at": "2026-08-12T11:00:00-04:00",
            "source": "synthetic-regulated-book",
            "stake_units": 0.25,
            "entry_threshold": "+4.5 or better",
        },
    )


def test_rejected_proposal_cannot_enter_paper_ledger() -> None:
    proposal = BetProposal(name="no edge", model_probability=0.40, decimal_odds=1.91)
    result = evaluate_proposal(proposal)

    with pytest.raises(ValueError, match="rejected proposals"):
        paper_trade_from_gate(proposal, result, {})


def test_duplicate_trade_id_is_rejected() -> None:
    trade = _paper_trade("same-id")

    with pytest.raises(ValueError, match="duplicate trade_id"):
        add_paper_trade([trade], trade)


def test_settlement_and_performance_metrics() -> None:
    first = _paper_trade("win", probability=0.60, odds=2.0)
    second = _paper_trade("loss", probability=0.60, odds=1.8)
    trades = settle_paper_trade(
        [first, second],
        "win",
        PaperOutcome.WIN,
        settled_at="2026-08-13T00:00:00Z",
        closing_decimal_odds=1.9,
    )
    trades = settle_paper_trade(
        trades,
        "loss",
        PaperOutcome.LOSS,
        settled_at="2026-08-13T00:05:00Z",
        closing_decimal_odds=1.85,
    )

    report = analyze_paper_trades(trades)

    assert report.wins == 1
    assert report.losses == 1
    assert report.net_units == pytest.approx(0.0)
    assert report.roi == pytest.approx(0.0)
    assert report.hit_rate == pytest.approx(0.5)
    assert report.brier_score == pytest.approx((0.16 + 0.36) / 2)
    assert report.average_closing_line_value == pytest.approx(
        ((2.0 / 1.9 - 1) + (1.8 / 1.85 - 1)) / 2
    )
    assert report.max_drawdown_units == pytest.approx(0.25)


def test_ledger_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "paper.json"
    trade = _paper_trade("round-trip")

    save_paper_ledger(path, [trade])

    assert load_paper_ledger(path) == [trade]


def test_cli_places_only_an_approved_paper_trade(tmp_path: Path, capsys) -> None:
    proposal_path = tmp_path / "proposal.json"
    ledger_path = tmp_path / "ledger.json"
    proposal_path.write_text(
        json.dumps(
            {
                "proposal": {
                    "name": "Synthetic basketball side",
                    "model_probability": 0.62,
                    "decimal_odds": 1.91,
                    "probability_uncertainty": 0.01,
                    "requested_stake_fraction": 0.005,
                },
                "simulation": {
                    "trade_id": "cli-paper-1",
                    "event": "Alpha at Beta",
                    "market": "spread",
                    "selection": "Alpha +4.5",
                    "event_start": "2026-08-12T19:00:00-04:00",
                    "quoted_at": "2026-08-12T11:00:00-04:00",
                    "source": "synthetic-regulated-book",
                    "stake_units": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["paper-place", str(proposal_path), str(ledger_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["mode"] == "paper_only"
    assert output["recorded"] is True
    assert load_paper_ledger(ledger_path)[0].trade_id == "cli-paper-1"
