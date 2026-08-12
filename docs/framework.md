# Conditioning-aware betting assurance

## 1. Why tracking error is insufficient

Suppose a model predicts a probability using a feature matrix \(X\). Ordinary
validation asks whether predictions are accurate or calibrated. Those checks
are necessary, but they do not ask whether the representation is becoming
singular.

For singular values \(\sigma_1 \geq \dots \geq \sigma_p\), the spectral
condition number is

\[
\kappa(X) = \frac{\sigma_1}{\sigma_p}.
\]

When \(\sigma_p\) approaches zero, small perturbations to data can cause large
changes to fitted coefficients. A model can retain low short-run error because
redundant features still span roughly the same predictive direction. The model
is nevertheless fragile.

Sports examples include simultaneously using points per game, pace-adjusted
points, offensive rating, and a nearly duplicated third-party power rating.

BetGuard standardizes numeric columns, calculates singular values and effective
rank, and rejects matrices with rank loss, constant features, a minimum singular
value below the configured floor, or a condition number above the configured
limit.

## 2. Hidden concentration in a betting portfolio

Several wagers may be different contracts but the same economic exposure. A
team moneyline, team spread, team-total over, game over, and a primary scorer's
points over can all load on the same offensive-performance factor.

For a return covariance matrix \(\Sigma\) with non-negative eigenvalues
\(\lambda_i\), BetGuard reports the participation-ratio effective number of bets:

\[
N_{\mathrm{eff}} =
\frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}.
\]

If five equal-risk bets are independent, \(N_{\mathrm{eff}}\) is near five. If
they move together, it approaches one. This is a diagnostic, not a guarantee of
future correlation.

The covariance helper also supports diagonal shrinkage:

\[
\Sigma_{\alpha} = (1-\alpha)\Sigma + \alpha\,\mathrm{diag}(\Sigma),
\]

which reduces the instability of small-sample covariance inversion.

## 3. Runtime decision gate

For decimal odds \(o\), break-even probability is \(p_0=1/o\). BetGuard uses a
conservative probability

\[
p_c = \hat p - u,
\]

where \(u\) is an explicit probability-uncertainty allowance. A proposal must
retain a configured minimum edge after this deduction.

When a stake is not supplied, the package starts with fractional Kelly:

\[
f^* = \frac{(o-1)p_c - (1-p_c)}{o-1}
\]

and then multiplies it by a conservative fraction and applies a hard bankroll
cap. Factor-exposure limits can reduce the result further.

The gate rejects rather than extrapolates when:

- selected model features are ill-conditioned;
- the matchup is outside the model's supported distribution;
- important lineup information is unresolved;
- probability uncertainty exceeds the configured limit; or
- conservative edge is insufficient.

## 4. Separation of responsibilities

The predictive model and assurance layer should be tested separately. Passing
the assurance gate only means that configured structural checks succeeded. It
does not validate the original probability estimate.

Production systems should additionally monitor calibration, closing-line value,
data freshness, leakage, bookmaker limits, transaction costs, market movement,
and jurisdiction-specific compliance.

