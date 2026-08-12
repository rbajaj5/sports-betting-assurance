# Paper trading

BetGuard's simulation layer creates an auditable record of decisions without
placing real wagers. It accepts explicit market observations, applies the same
assurance gate used by `betguard evaluate`, and writes an approved hypothetical
position to a local JSON ledger.

## Safety boundary

The package has no account, sportsbook, exchange, wallet, deposit, or order
execution integration. Stakes are abstract units. A default entry risks 0.25
units, and changing that number changes only simulated accounting.

## Proposal format

`paper-place` expects the regular `proposal` and optional gate `config` fields,
plus simulation metadata:

```json
{
  "proposal": {
    "name": "Synthetic basketball side",
    "model_probability": 0.62,
    "decimal_odds": 1.91,
    "probability_uncertainty": 0.02,
    "requested_stake_fraction": 0.005
  },
  "simulation": {
    "trade_id": "synthetic-alpha-beta-20260812",
    "event": "Alpha at Beta",
    "market": "spread",
    "selection": "Alpha +4.5",
    "event_start": "2026-08-12T19:00:00-04:00",
    "quoted_at": "2026-08-12T11:00:00-04:00",
    "source": "synthetic-regulated-book",
    "stake_units": 0.25,
    "entry_threshold": "+4.5 or better at decimal odds 1.91 or better"
  }
}
```

Rejected proposals are printed with the gate's reasons and are not added to the
ledger. Trade identifiers must be unique, so re-running the same input cannot
silently duplicate a position.

## Settlement and measurement

Use `paper-settle` with `win`, `loss`, `push`, or `void`. Supplying the closing
decimal odds enables odds-based closing-line value:

`entry decimal odds / closing decimal odds - 1`

Positive values mean the simulated entry obtained a better price than the
close. `paper-report` also calculates:

- net and risked units;
- return on simulated risked units;
- win/loss hit rate;
- Brier score for probability calibration;
- maximum drawdown over ledger order; and
- counts of open, graded, pushed, and void positions.

Pushes count as graded and risked but add zero profit. Voids add neither risk
nor profit. Open positions do not affect realized performance.

## Reproducible monitoring loop

1. Capture a timestamped market price and its source.
2. Create a proposal with a unique ID and an exact entry threshold.
3. Run `paper-place`; do not record a trade that the gate rejects.
4. After the event, capture the close and official result.
5. Run `paper-settle`, then `paper-report`.
6. Commit the JSON ledger so every change is reviewable.

This process evaluates whether the model beats prices over time. It does not
prove that the same performance would survive limits, latency, market impact,
price movement, or other real-world execution effects.
