# Held-out evaluation protocol v0.5 (fixed before outcomes)

- Question: Does ungated successful-trajectory latent retrieval with Transfer lookahead H=25 improve `proprio_v2` latent task completion over the unchanged Baseline?
- Methods: Baseline (`oracle_subgoal.enabled=false`) versus ungated retrieval (`oracle_subgoal.enabled=true`, `confidence_gate.enabled=false`, `enabled_states=[transfer]`, Transfer H=25). ALIGN and GRASP use existing control.
- Evaluation: 100 full episodes per method, independently reseeded episode seeds 11200–11299 (`--seed 200`, offset 11000), max 300 steps, same `proprio_v2_offline.pt` checkpoint and repository code, Actor frozen with no online updates. Run methods sequentially to avoid GPU contention.
- Primary outcome: `system_success` / `machine.complete` under the existing `proprio_v2` latent criterion. Physical cube position is not used to relabel outcomes.
- Primary analysis: completion counts and paired difference (ungated minus Baseline), discordant-pair table, two-sided exact McNemar p-value. A positive difference with p<0.05 is the threshold for a confirmatory success claim; otherwise report as inconclusive or negative. Also report a paired 95% bootstrap interval (20,000 resamples, seed 20260919), and descriptive steps and Transfer-entry counts.
- Validate before interpreting: exactly 100 rows and intended episode seeds in each summary, no Actor updates, checkpoint paths identical, correct oracle/gate configuration, and no duplicate/overlapping output names.
- Scope: This tests this fixed H=25 trajectory-retrieval implementation, not a learned LatentSubgoalPlanner. Same-seed separate full runs do not restore an identical Transfer state; numerical nondeterminism remains a limitation.
