#!/usr/bin/env python3
"""Validate and summarize one Baseline / ungated-H25 episode-seed batch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def load(path: Path, start_seed: int, episodes: int) -> tuple[dict, list[dict]]:
    online = json.loads(path.read_text(encoding="utf-8"))["online_training"]
    rows = online["episode_results"]
    if online["episodes"] != episodes or len(rows) != episodes:
        raise ValueError(f"Unexpected episode count: {path}")
    if not online["independent_episode_seeds"]:
        raise ValueError(f"Independent episode reseeding disabled: {path}")
    if online["oracle_subgoal"]["online_updates_enabled"]:
        raise ValueError(f"Actor updates enabled: {path}")
    for index, row in enumerate(rows):
        if row["episode"] != index or row["episode_seed"] != start_seed + 11000 + index:
            raise ValueError(f"Unexpected seed at row {index}: {path}")
        if int(row["actor_updates"]) != 0:
            raise ValueError(f"Actor updated at row {index}: {path}")
    if online["system_successes"] != sum(bool(row["system_success"]) for row in rows):
        raise ValueError(f"Success total disagrees with episode rows: {path}")
    return online, rows


def describe(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "latent_completions": sum(bool(row["system_success"]) for row in rows),
        "entered_transfer": sum(bool(row["entered_transfer"]) for row in rows),
        "mean_episode_steps": mean(row["steps"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ungated", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.start_seed < 0:
        parser.error("start-seed must be nonnegative and episodes positive")

    baseline, left = load(args.baseline, args.start_seed, args.episodes)
    ungated, right = load(args.ungated, args.start_seed, args.episodes)
    if baseline["source_checkpoint"] != ungated["source_checkpoint"]:
        raise ValueError("Methods used different checkpoint paths")
    base_oracle = baseline["oracle_subgoal"]
    ours_oracle = ungated["oracle_subgoal"]
    if base_oracle["enabled"] or not ours_oracle["enabled"]:
        raise ValueError("Incorrect Baseline / retrieval configuration")
    if ours_oracle["enabled_states"] != ["transfer"]:
        raise ValueError("Retrieval must be TRANSFER-only")
    if ours_oracle["lookahead_steps_by_state"][2] != 25:
        raise ValueError("Retrieval lookahead is not H=25")
    if ours_oracle["confidence_gate"]["enabled"]:
        raise ValueError("Confidence gate unexpectedly enabled")

    both = left_only = right_only = neither = 0
    differences = []
    for a, b in zip(left, right):
        a_success = bool(a["system_success"])
        b_success = bool(b["system_success"])
        both += a_success and b_success
        left_only += a_success and not b_success
        right_only += not a_success and b_success
        neither += not a_success and not b_success
        differences.append(int(b_success) - int(a_success))
    rng = np.random.default_rng(20260919)
    sampled = rng.choice(np.asarray(differences, dtype=np.float64),
                         size=(20_000, args.episodes), replace=True).mean(axis=1)
    lower, upper = (float(x) for x in np.quantile(sampled, [0.025, 0.975]))
    result = {
        "primary_metric": "proprio_v2 machine.complete (latent completion)",
        "episode_seeds": [args.start_seed + 11000, args.start_seed + 11000 + args.episodes - 1],
        "actor_frozen": True,
        "same_checkpoint_path": True,
        "baseline": describe(left),
        "ungated_h25": describe(right),
        "paired": {
            "both": both,
            "baseline_only": left_only,
            "ungated_only": right_only,
            "neither": neither,
            "rate_difference": float(mean(differences)),
            "paired_percentile_bootstrap_95_interval": [lower, upper],
            "exact_two_sided_mcnemar_p": exact_mcnemar(left_only, right_only),
        },
        "retrieval_plans": {
            "accepted": ours_oracle["confidence_gate"]["accepted_plans"],
            "rejected": ours_oracle["confidence_gate"]["rejected_plans"],
        },
        "limitation": "Separate same-seed runs do not restore identical Transfer states; exact trajectories may diverge.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n = args.episodes
    lines = [
        "# Frozen ungated H=25 evaluation",
        "",
        f"Episode seeds: {result['episode_seeds'][0]}–{result['episode_seeds'][1]}; Actor frozen.",
        "Primary metric: `proprio_v2` latent completion (`machine.complete`).",
        "",
        "| Method | Completion | Entered Transfer | Mean steps |",
        "|---|---:|---:|---:|",
        f"| Baseline | {result['baseline']['latent_completions']}/{n} | {result['baseline']['entered_transfer']}/{n} | {result['baseline']['mean_episode_steps']:.2f} |",
        f"| Ungated H=25 | {result['ungated_h25']['latent_completions']}/{n} | {result['ungated_h25']['entered_transfer']}/{n} | {result['ungated_h25']['mean_episode_steps']:.2f} |",
        "",
        f"Difference: {100 * result['paired']['rate_difference']:+.1f} percentage points; paired 95% bootstrap interval [{100 * lower:+.1f}, {100 * upper:+.1f}] points.",
        f"Discordant pairs: Baseline-only {left_only}, ungated-only {right_only}; exact two-sided McNemar p={result['paired']['exact_two_sided_mcnemar_p']:.6g}.",
        "",
        "Same-seed full runs need not reach identical Transfer states.",
        "",
    ]
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
