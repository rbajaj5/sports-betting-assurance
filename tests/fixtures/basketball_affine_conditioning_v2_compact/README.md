# Basketball Affine Conditioning v2

Deterministic synthetic formation-control data for BetGuard ingestion. No value is extracted from MP4 pixels. All coordinates are exported directly from the simulator at twelve decimal places.

## Reproduce

```powershell
py -3.12 analysis\run_basketball_affine_conditioning_v2_gpu.py
```

Default seed: `20260813`. Episodes per arm: `4096`. Expected telemetry rows: `24,576,000`.

## Deliverables

- `basketball_affine_conditioning_v2.mp4`: 1920x1080, 15 fps, 300 frames.
- `basketball_affine_telemetry.csv`: one row per episode x arm x frame x player, including all five offensive and five defensive identities.
- `basketball_affine_config.json`: frozen identities, coordinates, thresholds, phases, weights, and provenance.
- `basketball_affine_summary.json`: frame-level and episode-level results, confidence intervals, assertions, and hashes.
- `basketball_affine_pricing_example.json`: unvalidated synthetic BetGuard video-price example.
- `basketball_affine_csv_verification.json`: independent full-CSV reconciliation of counts and numeric summaries.
- `BASKETBALL_AFFINE_V2_MANIFEST.sha256`: SHA-256 manifest for the deliverables and generator source.

## Coordinates and events

Basket center is `(0, 0)`. X spans `[-25, 25]` feet from sideline to sideline; Y spans `[0, 47]` feet from baseline to half court.

`rank_loss = realized controlled normalized_conditioning < 0.12`. The conditioning gate evaluates the proposed affine map; keeping these quantities separate exposes residual post-decision disturbance risk rather than silently redefining it away.

`hidden_collapse = rank_loss AND tracking_gate_passed AND separation_gate_passed AND controller_decision == ACCEPT`.

Counts in the summary are separated into affected frames and affected episodes. Wilson 95% intervals are reported only for episode-level rates because frames within an episode are dependent.

## Fields

| Field | Definition |
|---|---|
| `simulation_version` | Version of the deterministic generator. |
| `random_seed` | Global recorded seed shared by the experiment. |
| `episode_seed` | Deterministic per-episode seed; identical between arms. |
| `episode_id` | Stable synthetic episode identifier. |
| `controller_arm` | tracking_and_separation or conditioning_aware_rta. |
| `frame_index` | Zero-based video/simulation frame. |
| `timestamp_seconds` | frame_index divided by frame rate. |
| `possession_id` | Stable possession identifier; changes at the recovery phase. |
| `possession_state` | Scripted ball-control or pass state. |
| `phase_id` | Registered phase identifier for the frame. |
| `team_id` | Stable synthetic offensive or defensive team identifier. |
| `lineup_id` | Stable synthetic lineup identifier. |
| `player_id` | Persistent synthetic player identity. |
| `player_role` | Offensive or defensive role. |
| `impact_rating` | Explicit pre-simulation synthetic impact rating. |
| `leader_or_follower` | L1/L2/L3, F1/F2, or context_defender. |
| `reference_x, reference_y` | Active pre-decision controller reference in feet; defensive rows use fixed matchup references. |
| `proposed_x, proposed_y` | Current phase-command position before controller filtering. |
| `observed_x, observed_y` | Noisy pre-actuation observation exported directly from simulation state. |
| `controlled_x, controlled_y` | Post-actuation simulated position. |
| `follower_weight_1..3` | Fixed design-time weights relative to PG01, WG01, C01; blank for leaders and defenders. |
| `ball_x, ball_y` | Scripted matched ball coordinates in feet. |
| `ball_possessor_id` | Stable possessor ID or NONE while a pass is in flight. |
| `sigma_min, sigma_max` | Singular values of the realized controlled offensive leader map. |
| `normalized_conditioning` | Realized sigma_min divided by sigma_max. |
| `determinant` | Signed determinant of the realized controlled leader map. |
| `area_scale` | Absolute determinant of the realized controlled leader map. |
| `minimum_pairwise_spacing_feet` | Minimum observed offensive-player distance used by the gate. |
| `formation_tracking_rmse_feet` | Observed offensive RMSE against the active pre-decision reference. |
| `follower_rmse_feet` | Observed follower RMSE against the active/held map. |
| `tracking_gate_passed` | Whether formation_tracking_rmse_feet is at or below its threshold. |
| `separation_gate_passed` | Whether minimum spacing is at or above its threshold. |
| `conditioning_gate_passed` | Whether proposed normalized conditioning is at least 0.12. |
| `controller_decision` | ACCEPT, REJECT, or HOLD LAST SAFE MAP. |
| `held_last_safe_map` | True while the arm is in a reject/hold interval. |
| `last_safe_map_frame` | Frame at which the exported fallback map was recorded. |
| `hold_start_frame, hold_end_frame` | Inclusive bounds of the current hold interval; -1 outside holds. |
| `last_safe_map_a11..a22` | Exact 2x2 fallback affine matrix retained by the controller. |
| `last_safe_translation_x, last_safe_translation_y` | Exact fallback affine translation in court feet. |
| `rank_loss` | True when realized controlled normalized_conditioning is below 0.12. |
| `hidden_collapse` | rank_loss AND both ordinary gates pass AND the arm accepts. |

## Scope

Illustrative synthetic simulation only. Not actual basketball evidence, a validated betting model, or proof of profitability. The pricing example sets `validated=false`, `historical_games=0`, and `qualified_paper_candidate=false`.
