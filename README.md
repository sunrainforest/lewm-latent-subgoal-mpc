# LeWM Latent Subgoal MPC

An experimental extension of a frozen LeWorldModel (LeWM) controller for long-horizon
cube manipulation. This v0.1 release studies whether intermediate latent targets improve
the `TRANSFER` phase while keeping the encoder, dynamics model, actor, MPC planner,
value ensemble, and three-phase finite-state machine unchanged.

## v0.1 results

All methods use the same five seeds (42--46), ten episodes per seed, and a frozen actor.

| Method | Success rate | Mean steps | Mean minimum transfer MSE |
| --- | ---: | ---: | ---: |
| Final-goal baseline | 20/50 (40%) | 139.78 | 0.1466 |
| Ungated trajectory retrieval | 26/50 (52%) | 123.52 | 0.0905 |
| Confidence-gated trajectory retrieval | **30/50 (60%)** | **110.18** | **0.0864** |
| Learned MLP with two gates | 24/50 (48%) | 128.80 | 0.0940 |

The strongest v0.1 result is confidence-gated retrieval. The learned planner has low
offline latent prediction error but does not yet provide a statistically stable control
improvement, which suggests that latent MSE alone is not a sufficient reachability
objective.

## Repository contents

- `source/`: planner, training script, experiment integration, configuration, and result summarizer;
- `docs/V0_1_RELEASE.md`: method, commands, results, limitations, and roadmap;
- `docs/DATA.md`: datasets, checkpoints, sizes, and SHA-256 checksums;
- `results/v0.1/summary.json`: machine-readable five-seed results;
- `CONTENTS.md`: internal bundle index.

The files in `source/` are an experiment change set, not a standalone copy of LeWM.
They are intended to be overlaid on the corresponding cube-robot experiment directory
in the upstream project.

## Design

```text
current latent + final goal latent + phase + proprioception
                         |
                         v
             single-step latent subgoal
                         |
                         v
                  Actor + MPC + LeWM
```

The first version retains `ALIGN -> GRASP -> TRANSFER`. Subgoals are used only during
`TRANSFER`, with confidence gates that fall back to the original final latent target.

## Reproduction

See [`docs/V0_1_RELEASE.md`](docs/V0_1_RELEASE.md) for commands and
[`docs/DATA.md`](docs/DATA.md) for required artifacts. Large data, pretrained weights,
videos, and generated logs are intentionally excluded from Git history.

## Limitations

- MuJoCo simulation only;
- one fixed cube start/goal task;
- 50 evaluation episodes per method;
- the learned planner uses only 64 successful fixed-task trajectories;
- trajectory retrieval depends on an offline success library.

## License

The included code is distributed under the repository's MIT license. Third-party models,
datasets, and upstream components remain subject to their own licenses and are not
redistributed in the source repository.
