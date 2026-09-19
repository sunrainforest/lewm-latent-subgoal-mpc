# Independent-seed evaluation v0.3

100 complete episodes per method; planner/inference RNGs reseeded at every episode; Actor frozen.
Physical success: OGBench cube-to-goal distance ≤ 0.04 m for two consecutive post-action frames.

| Method | Physical success | System success | Entered Transfer | Mean total steps |
|---|---:|---:|---:|---:|
| Baseline | 69/100 | 53/100 | 97/100 | 120.51 |
| Gated Oracle H=25 | 61/100 | 55/100 | 95/100 | 118.15 |

Physical success difference: -8.0 percentage points; paired bootstrap 95% interval [-16.0, +0.0] points.
Paired outcomes: both=56, baseline-only=13, gated-only=5, neither=26; exact McNemar p=0.0962524.
Gate plans: 1055 accepted and 7586 rejected (12.2% accepted).

Caveat: same-seed resets are used, but methods do not share a saved Transfer state and small renderer/model numerical differences can perturb the trajectories.
