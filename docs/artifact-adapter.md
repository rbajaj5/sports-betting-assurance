# Native affine-artifact adapter

BetGuard can read the `basketball_affine_conditioning_v2` directory directly.
It does not infer coordinates from the MP4 and does not require the 16 GB CSV
for its default compact audit.

```bash
betguard artifact-verify /path/to/basketball_affine_conditioning_v2
betguard artifact-import /path/to/basketball_affine_conditioning_v2 \
  --output artifact_report.json
betguard artifact-price /path/to/basketball_affine_conditioning_v2
```

Compact mode requires the config, summary, CSV-verification, pricing-example,
and manifest files. It checks versions, dimensions, frame order, stable player
identities, top-three impact leaders, fixed follower weights, count
reconciliation, and all available small-file hashes. Large files and an
external generator entry are reported as unverified when they are absent.

`--stream-telemetry` reads the CSV incrementally. It retains one ten-player
frame group, fixed identity metadata, and aggregate counters, never episode or
row tables. The full source CSV and MP4 remain outside Git.

## Proposed decisions versus realized geometry

The source controller evaluates the **proposed** affine map. The compact JSON
exports **realized observed** coordinates after control. In the
conditioning-aware arm, held coordinates can therefore be well conditioned
even though the proposed map was rejected. Re-running a geometry gate only on
realized positions changes the meaning of the experiment.

The native adapter preserves the source decisions as the authoritative
accept/hold/reject census and separately recomputes realized conditioning,
area, spacing, and follower residuals. For representative episode 516, the
source counts are:

| Controller arm | Accept | Hold | Reject | Hold or reject |
|---|---:|---:|---:|---:|
| Tracking and separation | 275 | 17 | 8 | 25 |
| Conditioning-aware RTA | 209 | 76 | 15 | 91 |

These differ from an earlier handoff expectation of `290/10` and `230/70`.
They are not forced to match it. The supplied files support the counts above,
and the distinction is regression-tested.

## Identity and team semantics

The bundle compares two controller arms over the same synthetic five-player
offense. The five defensive players are context. They are not an independently
weighted away-team formation. The adapter therefore emits `comparison_arms`
and refuses to turn the arm comparison into a home-versus-away pricing signal.
Player rows are never permuted.

The representative replay contains two possessions. Production requires at
least five. `--demo-two-possessions` exists only to test the adapter and is
itself a disqualifying condition. The calibration remains `illustrative-v1`,
with zero historical games and `validated=false`.

The fixture under `tests/fixtures/` contains the legal compact JSON and text
files only. Its manifest deliberately reports the omitted CSV, MP4, and
generator source as unavailable.
