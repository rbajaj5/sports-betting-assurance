# Affine conditioning of basketball formations

The affine-conditioning concept can be applied directly to optical player
tracking. It is not limited to an analogy about model features.

## Coordinate model

Represent the five offensive players in one tracking frame as a matrix
\(Y_t\in\mathbb{R}^{5\times2}\). Let \(X\in\mathbb{R}^{5\times2}\) be a
reference formation such as five-out, horns, or four-out-one-in. With stable
role ordering, fit

\[
Y_t \approx X A_t + \mathbf{1}b_t^\top,
\]

where \(A_t\in\mathbb{R}^{2\times2}\) describes rotation, reflection, scale,
shear, and directional compression, while \(b_t\) is translation.

For singular values \(s_1\ge s_2\) of \(A_t\):

- \(s_2\) measures the weakest retained spatial direction;
- \(s_1/s_2\) measures anisotropic deformation;
- \(|\det A_t|=s_1s_2\) measures formation-area scale; and
- affine residual measures how well the players still resemble the template.

The minimum pairwise player distance is monitored separately. This is necessary
because uniform crowding can make both singular values small while leaving the
condition number near one.

## What a collapse can mean

A low \(s_2\), small determinant, or very small pairwise spacing can identify:

- five players flattening toward one line;
- weak-side spacing disappearing during a drive;
- two offensive roles occupying the same functional space;
- a nominal five-out set compressing into the paint; or
- a defensive shell overloading one axis.

These events are descriptive, not automatically bad. Intentional cuts, screens,
rebounds, and end-of-clock actions can create momentary compression. Models
should therefore use temporal context, ball position, possession phase, and the
opponent's geometry.

## From formation features to betting inputs

Formation metrics are most defensible at the possession level. Potential targets
include expected points, shot quality, turnover probability, foul probability,
and offensive-rebound probability. Aggregated and calibrated possession models
can then contribute to team-strength and game-distribution estimates.

Examples of temporal features:

- minimum and median \(s_2\) before a shot;
- duration below an area-scale threshold;
- compression and recovery rates;
- template residual for several offensive sets;
- offensive and defensive formation interaction;
- ball-to-centroid and ball-to-collapse-axis distances; and
- formation metrics conditioned on lineup and play phase.

Avoid jumping directly from one collapsed frame to a game bet. The full chain
requires tracking-data quality checks, role matching, possession segmentation,
leakage-safe validation, calibration, uncertainty estimation, and an independent
betting assurance gate.

## Role matching

The package supports fixed row ordering and brute-force permutation search.
Permutation search is feasible for five players, but it can erase meaningful
role identity. Production pipelines should prefer stable player IDs and infer
roles using lineup context, ball handling, and historical locations.

