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
5. **Video formation pricing** — treats the top three pregame impact players as
   affine leaders, holds the last well-conditioned map during collapse, and
   produces auditable simulation-only moneyline prices from annotated footage.
6. **Native artifact verification** — imports the retained affine experiment,
   checks compact hashes and identities, and can stream its telemetry with
   bounded memory.
7. **Historical data contracts** — validates stable-ID WNBA game, price,
   availability, formation, and reconciliation records without bundling
   licensed raw data.

BetGuard does **not** scrape odds, predict games, recommend sportsbooks, place
real wagers, or promise profitability. The included basketball examples are
synthetic. Its trading workflow is deliberately paper-only.

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

## Price a game from annotated video

```bash
betguard video-price examples/video_price_synthetic.json
```

This command does not infer trustworthy player positions directly from an
arbitrary broadcast. It requires stable player IDs, court-calibrated
coordinates, permitted clip provenance, and explicit player impact ratings. The
three highest-rated players become formation leaders. If their normalized
affine conditioning falls below `0.12`, the monitor holds the last safe map and
measures follower deviation.

The supplied 4,096-episode GPU comparison motivates this assurance endpoint but
is not basketball evidence and is not independently reproduced here. BetGuard
transfers the structural failure test, not the experiment's outcome counts.

The bundled calibration is deliberately illustrative, so its output is a
scenario rather than a qualified paper candidate. See
[docs/video-formation-pricing.md](docs/video-formation-pricing.md) for the input
schema, evidence gates, pricing equation, and footage-use restrictions.

## Import the retained affine artifact

```bash
betguard artifact-verify /path/to/basketball_affine_conditioning_v2
betguard artifact-import /path/to/basketball_affine_conditioning_v2 \
  --output artifact_report.json
betguard artifact-price /path/to/basketball_affine_conditioning_v2
```

Compact mode does not read the 16 GB telemetry CSV. Add `--stream-telemetry`
only when a full bounded-memory reconciliation is intended. The adapter keeps
source proposed-map decisions separate from geometry recomputed on realized
coordinates and does not relabel controller arms as opposing teams. See
[docs/artifact-adapter.md](docs/artifact-adapter.md).

## Import historical exports

```bash
betguard historical-import game games.csv --output private/games.jsonl
betguard data-gap-report --output historical_data_gap.json
```

The initial scope is WNBA-only. Raw licensed data stays outside Git. There are
currently no real games or validated formation coefficients in this repository;
the machine-readable gap is
[reports/historical_data_gap.json](reports/historical_data_gap.json). Schema and
evaluation details are in [docs/historical-data.md](docs/historical-data.md) and
[docs/calibration-and-evaluation.md](docs/calibration-and-evaluation.md).

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

## Run a paper-trading ledger

Gate a hypothetical position and record it only when approved:

```bash
betguard paper-place examples/paper_proposal.json data/paper_trades.json
```

Settle it after the event, including a closing price when available:

```bash
betguard paper-settle data/paper_trades.json synthetic-alpha-beta-20260812 \
  win --settled-at 2026-08-13T01:30:00Z --closing-decimal-odds 1.84
```

Summarize record, paper ROI, closing-line value, Brier score, and drawdown:

```bash
betguard paper-report data/paper_trades.json
```

The ledger uses abstract units rather than currency and cannot connect to a
sportsbook or execute a bet. See [docs/paper-trading.md](docs/paper-trading.md)
for the schema, workflow, and metric definitions.

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
