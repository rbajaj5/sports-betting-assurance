# Historical basketball data

The first historical scope is the WNBA. BetGuard does not mix leagues whose
rules, schedules, lineups, prices, and tracking distributions differ.

Versioned JSON Schemas live in `schemas/v1/` for games, market snapshots,
player availability, formation evidence, and ID reconciliation. Runtime
validation implements the same contracts. Joins use stable game, team, player,
operator, and provider record IDs; display names are metadata, never join keys.

## Permitted inputs

- Official league or team sources for schedules, results, rosters, and
  availability.
- Licensed or user-supplied exports from regulated operators, exchanges, or
  reputable data vendors for historical prices.
- User-supplied or otherwise permitted tracking coordinates or footage with a
  documented rights basis.

BetGuard does not bypass logins, paywalls, geolocation, robots rules, DRM, API
authentication, or contractual restrictions. It does not use offshore books.
Raw licensed data stays outside Git.

## User exports

CSV, JSON, JSONL, and NDJSON work without optional dependencies. Parquet is
supported when `pyarrow` is installed by the data owner.

```bash
betguard historical-import game games.csv --output private/games.jsonl \
  --provider-name licensed-user-export
betguard historical-import market_snapshot markets.parquet \
  --output private/markets.jsonl --provider-name vendor-export
```

The command validates every row, writes canonical JSONL, and creates a
`.provenance.json` sidecar containing source and output hashes. Import success
does not establish a data license or grant model qualification.

## Current gap

No licensed historical odds, real-game tracking coordinates, or 200-game
formation sample was supplied. BetGuard therefore contains no fitted formation
coefficient file and makes no backtest or profitability claim. The exact gap is
recorded in `reports/historical_data_gap.json`.
