# BetGuard

BetGuard is a reference implementation of **conditioning-aware runtime
assurance for sports-betting models**. Its central idea is simple:

> Low prediction error does not prove that a model or betting portfolio is
> structurally healthy.

A model can continue to track outcomes or closing prices while its features
become redundant, its coefficients become unstable, or its open wagers collapse
onto one hidden source of risk. BetGuard places an independent assurance gate
between a predictive model and the decision to stake money.

```text
odds + data -> predictive model -> proposed probability/stake
                                      |
                                      v
                              independent assurance gate
                                /                   \
                       accept or reduce           reject
                                                     |
                                                   no bet
```

## What it checks

1. **Basketball formation conditioning** — fits affine maps to player-tracking
   coordinates and detects directional compression, area collapse, and crowding.
2. **Design conditioning** — detects rank loss, near-singular feature matrices,
   constant columns, and excessive condition numbers.
3. **Portfolio conditioning** — estimates how many independent bets a portfolio
   really contains and identifies unstable covariance matrices.
4. **Decision assurance** — accepts, reduces, or rejects a proposed wager using
   conservative edge, uncertainty, lineup status, distribution shift, feature
   conditioning, fractional Kelly, and factor-exposure caps.

BetGuard does **not** scrape odds, predict games, recommend sportsbooks, or
promise profitability. The included basketball example is synthetic.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Run the demonstration

```bash
betguard demo
```

The demonstration creates a healthy basketball feature matrix and a collapsed
one with nearly duplicated scoring features. It then shows the assurance gate
reducing a correlated position and rejecting the same position when feature
rank is lost.

## Check your own feature CSV

Every selected column must be numeric.

```bash
betguard check-design examples/basketball_features.csv \
  --columns pace,offensive_rating,turnover_rate,three_point_rate
```

## Analyze an actual basketball formation

With five role-ordered `(x, y)` player coordinates in feet:

```bash
betguard check-formation \
  examples/five_out_reference.json \
  examples/compressed_observed.json \
  --template-name five_out
```

The fit reports the affine map, principal singular values, condition number,
area scale, minimum player spacing, and residual error. The supplied observed
frame is an exact low-residual affine transformation that nevertheless collapses
toward one line—the failure illustrated by the source animation.

See [docs/basketball-formations.md](docs/basketball-formations.md) for the
coordinate model and a careful path from possession-level spatial features to
betting inputs.

To see the redundant-feature failure:

```bash
betguard check-design examples/basketball_features.csv \
  --columns points_per_game,points_per_game_copy,offensive_rating
```

## Evaluate a proposal

```bash
betguard evaluate examples/proposal.json
```

The JSON format is intentionally explicit:

```json
{
  "proposal": {
    "name": "Synthetic basketball total over",
    "model_probability": 0.56,
    "decimal_odds": 1.91,
    "probability_uncertainty": 0.02,
    "requested_stake_fraction": 0.008,
    "factor_loadings": {"team_offense": 1.0, "game_pace": 0.7}
  },
  "current_factor_exposure": {"team_offense": 0.007, "game_pace": 0.004},
  "config": {
    "min_conservative_edge": 0.01,
    "max_factor_exposure": 0.012,
    "max_stake_fraction": 0.01
  }
}
```

Stake fractions are proportions of bankroll: `0.005` means 0.5%.

## Framework

The assurance gate follows a fail-safe pattern:

- A predictive model may propose a probability and stake.
- A separate monitor checks numerical conditioning and decision robustness.
- A well-conditioned proposal may pass.
- A proposal with excessive shared exposure is reduced.
- A proposal with rank loss, unresolved lineup information, distribution shift,
  excessive uncertainty, or no conservative edge is rejected.

See [docs/framework.md](docs/framework.md) for the mathematical interpretation
and implementation notes.

## Development

```bash
python -m pytest
ruff check .
```

## Responsible use

This project is educational software, not financial advice or a betting system.
No diagnostic can turn an inaccurate probability model into a profitable one.
If you choose to wager, comply with local laws, use predetermined loss limits,
and never stake money needed for living expenses.
