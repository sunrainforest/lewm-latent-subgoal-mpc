# Confirmatory evaluation v0.2

Actor frozen; seeds 100–109; 10 episodes per seed and method; H=25; gate threshold 0.0031.
Physical success is OGBench cube-to-goal distance ≤ 0.04 m for two consecutive post-action frames.

| Method | Physical success | System success | Mean steps |
|---|---:|---:|---:|
| Baseline | 55/100 (55.0%) | 46/100 (46.0%) | 130.76 |
| Gated Oracle H=25 | 59/100 (59.0%) | 52/100 (52.0%) | 121.13 |

Physical success difference: +4.0 percentage points.
Exploratory episode-index pairing: baseline-only=6, gated-only=10; exact McNemar p=0.454498.
Strict same-seed first episodes (10 pairs): baseline-only=0, gated-only=0; exact McNemar p=1.
Later episode indices are not strict pairs because different rollout lengths desynchronize the run-level RNG streams.
Gate usage: 1064 accepted, 8640 rejected (11.0% accepted).
