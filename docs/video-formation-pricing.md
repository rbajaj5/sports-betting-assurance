# Video formation pricing

This layer adapts the supplied affine-conditioning animation to basketball
without treating the animation as game footage. It consumes player-identified,
court-calibrated coordinates annotated from footage that is user-supplied,
officially public, licensed, or synthetic. Clips are referenced by source,
timestamp, rights basis, and SHA-256 digest; the repository stores coordinates,
not third-party video.

## What the reference animation establishes

The supplied description uses three leaders to define a two-dimensional affine
map. Followers keep fixed design-time affine weights. Its comparison is between:

- a controller that watches tracking error and pairwise separation; and
- a runtime-assurance controller that separately watches normalized leader
  conditioning and holds the last well-conditioned map.

In the described 4,096 matched illustrative episodes per arm, the first
controller had 2,304 affine-rank-loss episodes, including 1,101 hidden
collapses. The conditioning-aware arm had none under the matched command and
disturbance parameters. These counts are supplied properties of the illustrative
GPU experiment; this repository does not independently reproduce them, and they
are not empirical basketball results.

The narrower transferable claim is structural: low formation error and nonzero
pairwise separation do not certify affine span, localizability, or
recoverability. Preserving recoverability also has a cost because the monitor
must reject some otherwise trackable affine maneuvers.

## Control model

For each team, the three players with the highest explicit pregame impact
ratings become affine leaders. Their reference positions define an affine map
for each frame. Applying that map to every follower's fixed reference position
is algebraically equivalent to retaining fixed design-time affine weights with
respect to the three non-collinear leaders. The normalized conditioning
statistic is:

```text
smallest singular value / largest singular value
```

The animation's `0.12` warning threshold is the default. A frame also fails if
formation area or player spacing is too small. When a leader map fails, the
monitor holds the last well-conditioned map and measures how far the remaining
players (the followers) deviate from that safe command. This catches the key
failure in the animation: tiny tracking residual can coexist with a rank-lost
formation.

Player identities stay fixed. Permutation matching is deliberately disallowed
because swapping rows could make the three selected leaders change identity.

## Evidence gates

A formation report fails closed when any of these are unresolved:

- stable player identity;
- usable court calibration;
- enough frames and possessions;
- adequate projected-lineup coverage;
- enough well-conditioned and follower-coherent frames.

Broadcast pixels alone are not coordinates. A real workflow still needs camera
calibration, player tracking or manual annotation, possession segmentation, and
identity review. Never bypass a sign-in, paywall, or stream restriction to get
footage, and do not redistribute footage without permission.

## Pricing model

`video-price` starts with a separate baseline home win probability, then applies
an explicit log-odds adjustment from four home-minus-away features:

1. well-conditioned frame fraction;
2. median leader conditioning;
3. follower-residual advantage;
4. top-three player impact advantage.

It converts both sides of a moneyline market to implied probabilities and
normalizes them to remove the displayed overround. The conservative edge is the
adjusted probability minus stated probability uncertainty minus the de-vigged
market probability.

The included coefficients are illustrative. They can produce a scenario price,
but never a paper candidate. A paper candidate requires coefficients explicitly
marked as validated, a sufficient out-of-sample historical game count, two
qualified video reports, and at least a three-percentage-point conservative
edge. This remains simulation-only and cannot connect to a sportsbook.

## Example

```bash
betguard video-price examples/video_price_synthetic.json
```

The JSON contains the complete provenance, player ratings, coordinates,
thresholds, baseline model probability, two-sided market price, and calibration
metadata. Replace the synthetic values only with auditable inputs.
