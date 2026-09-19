#!/usr/bin/env python3
"""Aggregate paired Baseline vs confidence-gated Oracle evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


METHODS = ("baseline", "gated_h25")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--last-seed", type=int, default=109)
    return parser.parse_args()


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_mcnemar(baseline_only: int, gated_only: int) -> float:
    discordant = baseline_only + gated_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, gated_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def aggregate(rows: list[dict]) -> dict:
    total = len(rows)
    physical = sum(bool(row["physical_success"]) for row in rows)
    system = sum(bool(row["system_success"]) for row in rows)
    entered_transfer = sum(bool(row["entered_transfer"]) for row in rows)
    physical_ever = sum(bool(row["physical_success_ever"]) for row in rows)
    physical_steps = [row["steps"] for row in rows if row["physical_success"]]
    return {
        "episodes": total,
        "physical_successes": physical,
        "physical_success_rate": physical / total,
        "physical_success_wilson_95": wilson(physical, total),
        "system_successes": system,
        "system_success_rate": system / total,
        "system_success_wilson_95": wilson(system, total),
        "physical_success_ever": physical_ever,
        "entered_transfer": entered_transfer,
        "mean_steps_all": mean(row["steps"] for row in rows),
        "mean_steps_physical_success": mean(physical_steps) if physical_steps else None,
        "mean_max_physical_success_streak": mean(
            row["max_physical_success_streak"] for row in rows
        ),
        "mean_transfer_grasp_loss_frames": mean(
            row["transfer_grasp_loss_frames"] for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    by_method: dict[str, list[dict]] = {method: [] for method in METHODS}
    gate_accepted = 0
    gate_rejected = 0
    for seed in range(args.first_seed, args.last_seed + 1):
        for method in METHODS:
            name = f"confirm_v0_2_{method}_seed{seed}_summary.json"
            path = args.input_dir / name
            with path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            online = summary["online_training"]
            for episode in online["episode_results"]:
                by_method[method].append({"seed": seed, **episode})
            if method == "gated_h25":
                gate = online["oracle_subgoal"]["confidence_gate"]
                gate_accepted += int(gate["accepted_plans"])
                gate_rejected += int(gate["rejected_plans"])

    baseline = by_method["baseline"]
    gated = by_method["gated_h25"]
    if len(baseline) != len(gated):
        raise RuntimeError("Methods have different episode counts.")

    paired = []
    for baseline_row, gated_row in zip(baseline, gated):
        if (baseline_row["seed"], baseline_row["episode"]) != (
            gated_row["seed"], gated_row["episode"]
        ):
            raise RuntimeError("Paired episode order mismatch.")
        paired.append((baseline_row, gated_row))

    def comparison(field: str, pairs: list[tuple[dict, dict]]) -> dict:
        both = sum(bool(a[field]) and bool(b[field]) for a, b in pairs)
        baseline_only = sum(bool(a[field]) and not bool(b[field]) for a, b in pairs)
        gated_only = sum(not bool(a[field]) and bool(b[field]) for a, b in pairs)
        neither = len(pairs) - both - baseline_only - gated_only
        return {
            "pairs": len(pairs),
            "both": both,
            "baseline_only": baseline_only,
            "gated_only": gated_only,
            "neither": neither,
            "exact_mcnemar_two_sided_p": exact_mcnemar(baseline_only, gated_only),
        }

    result = {
        "design": {
            "seeds": [args.first_seed, args.last_seed],
            "episodes_per_seed_per_method": len(baseline) // (args.last_seed - args.first_seed + 1),
            "actor_frozen": True,
            "gated_oracle_horizon_steps": 25,
            "confidence_match_mse_threshold": 0.0031,
            "physical_success_definition": "OGBench cube-to-goal distance <= 0.04 m for 2 consecutive post-action frames",
            "pairing_limitation": (
                "Episode 0 is a strict same-seed pair. Later episodes share a "
                "run-level RNG stream that can desynchronize after methods take "
                "different numbers of steps; all-episode McNemar results are exploratory."
            ),
        },
        "baseline": aggregate(baseline),
        "gated_h25": aggregate(gated),
        "exploratory_index_paired_physical_success": comparison(
            "physical_success", paired
        ),
        "exploratory_index_paired_system_success": comparison(
            "system_success", paired
        ),
        "strict_first_episode_physical_success": comparison(
            "physical_success", [pair for pair in paired if pair[0]["episode"] == 0]
        ),
        "strict_first_episode_system_success": comparison(
            "system_success", [pair for pair in paired if pair[0]["episode"] == 0]
        ),
        "gate": {
            "accepted_plans": gate_accepted,
            "rejected_plans": gate_rejected,
            "acceptance_rate": gate_accepted / (gate_accepted + gate_rejected),
        },
    }
    result["effect"] = {
        "physical_success_rate_difference": (
            result["gated_h25"]["physical_success_rate"]
            - result["baseline"]["physical_success_rate"]
        ),
        "system_success_rate_difference": (
            result["gated_h25"]["system_success_rate"]
            - result["baseline"]["system_success_rate"]
        ),
        "mean_steps_all_difference": (
            result["gated_h25"]["mean_steps_all"]
            - result["baseline"]["mean_steps_all"]
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def percent(value: float) -> str:
        return f"{100 * value:.1f}%"

    base = result["baseline"]
    gate = result["gated_h25"]
    physical_pair = result["exploratory_index_paired_physical_success"]
    strict_pair = result["strict_first_episode_physical_success"]
    lines = [
        "# Confirmatory evaluation v0.2",
        "",
        "Actor frozen; seeds 100–109; 10 episodes per seed and method; H=25; gate threshold 0.0031.",
        "Physical success is OGBench cube-to-goal distance ≤ 0.04 m for two consecutive post-action frames.",
        "",
        "| Method | Physical success | System success | Mean steps |",
        "|---|---:|---:|---:|",
        f"| Baseline | {base['physical_successes']}/{base['episodes']} ({percent(base['physical_success_rate'])}) | {base['system_successes']}/{base['episodes']} ({percent(base['system_success_rate'])}) | {base['mean_steps_all']:.2f} |",
        f"| Gated Oracle H=25 | {gate['physical_successes']}/{gate['episodes']} ({percent(gate['physical_success_rate'])}) | {gate['system_successes']}/{gate['episodes']} ({percent(gate['system_success_rate'])}) | {gate['mean_steps_all']:.2f} |",
        "",
        f"Physical success difference: {100 * result['effect']['physical_success_rate_difference']:+.1f} percentage points.",
        f"Exploratory episode-index pairing: baseline-only={physical_pair['baseline_only']}, gated-only={physical_pair['gated_only']}; exact McNemar p={physical_pair['exact_mcnemar_two_sided_p']:.6g}.",
        f"Strict same-seed first episodes ({strict_pair['pairs']} pairs): baseline-only={strict_pair['baseline_only']}, gated-only={strict_pair['gated_only']}; exact McNemar p={strict_pair['exact_mcnemar_two_sided_p']:.6g}.",
        "Later episode indices are not strict pairs because different rollout lengths desynchronize the run-level RNG streams.",
        f"Gate usage: {gate_accepted} accepted, {gate_rejected} rejected ({percent(result['gate']['acceptance_rate'])} accepted).",
        "",
    ]
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
