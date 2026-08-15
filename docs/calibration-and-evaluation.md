# Calibration and out-of-sample evaluation

BetGuard separates two layers.

1. A pregame statistical baseline uses only information available by the
   quoted market timestamp: team strength, rest, travel, pace, recent form,
   home court, projected lineups, and official availability when sourced.
2. A formation increment is evaluated only after the baseline is frozen and
   only for real games that pass every tracking and provenance gate.

Synthetic affine episodes test software behavior. They cannot train or
validate outcome-pricing coefficients.

## Frozen chronology

Games are split chronologically into train, validation, and untouched test
periods. Stable IDs must be unique, and no period overlaps a later one.
Feature-level availability timestamps must be at or before the market snapshot;
final scores, postgame status, and future prices are rejected as pregame
features.

The future evaluation will report exclusions at every gate and separately
evaluate moneyline, spread, and total. Two-sided prices are de-vigged. Required
outputs are Brier score, log loss, calibration intercept and slope, reliability
bins, game-level bootstrap intervals, flat 0.25-unit paper ROI, closing-line
value, maximum drawdown, turnover, and sensitivity to vig removal, stale prices,
uncertainty, missing players, and line movement. Comparators are baseline-only,
market-implied, and a simple rating model.

## Qualification

A paper candidate requires all of the following:

- verified real-game provenance and fixed player identities;
- court calibration RMSE no worse than 1.5 feet;
- at least 12 usable frames across at least five possessions;
- at least 80% projected-lineup coverage;
- explicit pregame impact ratings and availability assumptions;
- frozen, versioned coefficients based on at least 200 qualifying real games;
- current regulated two-sided prices; and
- a conservative post-uncertainty edge of at least three percentage points.

Any missing condition yields `PASS — no qualified trade.` A scenario result is
structurally separate from a qualified paper candidate. No command in this
repository can place a wager or transfer funds.
