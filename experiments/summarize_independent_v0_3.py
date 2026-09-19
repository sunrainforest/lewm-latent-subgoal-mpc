#!/usr/bin/env python3
"""Analyze independent-seed Baseline vs gated-Oracle evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_confirmatory_v0_2 import aggregate, exact_mcnemar


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_run(path: Path) -> tuple[dict, list[dict]]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    online = summary["online_training"]
    if not online["independent_episode_seeds"]:
        raise RuntimeError(f"Independent episode reseeding disabled in {path}")
    if online["oracle_subgoal"]["online_updates_enabled"]:
        raise RuntimeError(f"Actor was not frozen in {path}")
    episodes = online["episode_results"]
    if online["episodes"] != 100 or len(episodes) != 100:
        raise RuntimeError(f"Expected 100 episodes in {path}")
    if any(int(row["actor_updates"]) != 0 for row in episodes):
        raise RuntimeError(f"Actor updates found in {path}")
    for index, row in enumerate(episodes):
        if row["episode"] != index or row["episode_seed"] != 11100 + index:
            raise RuntimeError(f"Unexpected episode/seed mapping in {path}")
    return online, episodes


def pairs(rows_a: list[dict], rows_b: list[dict], field: str) -> dict:
    both = baseline_only = gated_only = neither = 0
    for baseline, gated in zip(rows_a, rows_b):
        if baseline["episode_seed"] != gated["episode_seed"]:
            raise RuntimeError("Seed pairing mismatch")
        left, right = bool(baseline[field]), bool(gated[field])
        both += left and right
        baseline_only += left and not right
        gated_only += not left and right
        neither += not left and not right
    return {
        "both": both,
        "baseline_only": baseline_only,
        "gated_only": gated_only,
        "neither": neither,
        "exact_mcnemar_two_sided_p": exact_mcnemar(baseline_only, gated_only),
    }


def main() -> None:
    args = arguments()
    baseline_online, baseline_rows = load_run(
        args.input_dir / "independent_v0_3_baseline_seed100_199_summary.json"
    )
    gated_online, gated_rows = load_run(
        args.input_dir / "independent_v0_3_gated_h25_seed100_199_summary.json"
    )
    physical_differences = np.asarray(
        [int(g["physical_success"]) - int(b["physical_success"])
         for b, g in zip(baseline_rows, gated_rows)],
        dtype=np.float64,
    )
    bootstrap_rng = np.random.default_rng(20260919)
    samples = bootstrap_rng.choice(
        physical_differences,
        size=(20000, len(physical_differences)),
        replace=True,
    ).mean(axis=1)
    gate = gated_online["oracle_subgoal"]["confidence_gate"]
    result = {
        "design": {
            "method": "100 independently reseeded complete episodes per method",
            "episode_seeds": [11100, 11199],
            "actor_frozen": True,
            "transfer_horizon_steps": 25,
            "gate_match_mse_threshold": 0.0031,
            "physical_success_definition": "OGBench cube-to-goal distance <= 0.04 m for two consecutive post-action frames",
            "limitation": "Methods start from same-seed resets, not identical saved Transfer states; renderer/model numerical differences can still perturb trajectories.",
        },
        "baseline": aggregate(baseline_rows),
        "gated_h25": aggregate(gated_rows),
        "paired_physical_success": pairs(baseline_rows, gated_rows, "physical_success"),
        "paired_system_success": pairs(baseline_rows, gated_rows, "system_success"),
        "physical_success_rate_difference": float(physical_differences.mean()),
        "paired_bootstrap_95_interval": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "gate": {
            "accepted_plans": gate["accepted_plans"],
            "rejected_plans": gate["rejected_plans"],
            "acceptance_rate": gate["accepted_plans"] / (gate["accepted_plans"] + gate["rejected_plans"]),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    base = result["baseline"]
    gated = result["gated_h25"]
    pair = result["paired_physical_success"]
    interval = result["paired_bootstrap_95_interval"]
    lines = [
        "# Independent-seed evaluation v0.3",
        "",
        "100 complete episodes per method; planner/inference RNGs reseeded at every episode; Actor frozen.",
        "Physical success: OGBench cube-to-goal distance ≤ 0.04 m for two consecutive post-action frames.",
        "",
        "| Method | Physical success | System success | Entered Transfer | Mean total steps |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline | {base['physical_successes']}/100 | {base['system_successes']}/100 | {base['entered_transfer']}/100 | {base['mean_steps_all']:.2f} |",
        f"| Gated Oracle H=25 | {gated['physical_successes']}/100 | {gated['system_successes']}/100 | {gated['entered_transfer']}/100 | {gated['mean_steps_all']:.2f} |",
        "",
        f"Physical success difference: {100 * result['physical_success_rate_difference']:+.1f} percentage points; paired bootstrap 95% interval [{100 * interval[0]:+.1f}, {100 * interval[1]:+.1f}] points.",
        f"Paired outcomes: both={pair['both']}, baseline-only={pair['baseline_only']}, gated-only={pair['gated_only']}, neither={pair['neither']}; exact McNemar p={pair['exact_mcnemar_two_sided_p']:.6g}.",
        f"Gate plans: {gate['accepted_plans']} accepted and {gate['rejected_plans']} rejected ({100 * result['gate']['acceptance_rate']:.1f}% accepted).",
        "",
        "Caveat: same-seed resets are used, but methods do not share a saved Transfer state and small renderer/model numerical differences can perturb the trajectories.",
        "",
    ]
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
