from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MODES = {
    "baseline": "oracle_control_frozen_seed",
    "retrieval": "oracle_transfer_h25_frozen_seed",
    "gated_retrieval": "oracle_gated_h25_frozen_seed",
    "learned_double_gate": "learned_subgoal_h25_gated_seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v0.1 results.")
    parser.add_argument(
        "--outputs-root", type=Path, default=HERE / "outputs" / "proprio_v2"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "v0.1" / "summary.json"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 47)))
    return parser.parse_args()


def paired_exact(first: list[bool], second: list[bool]) -> dict[str, Any]:
    gain = sum(new and not old for old, new in zip(first, second))
    loss = sum(old and not new for old, new in zip(first, second))
    discordant = gain + loss
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(gain, loss) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "paired_gain": gain,
        "paired_loss": loss,
        "paired_ties": len(first) - discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def load_mode(
    root: Path, prefix: str, seeds: list[int]
) -> tuple[dict[str, Any], list[bool]]:
    statuses: list[bool] = []
    steps: list[int] = []
    minimum_transfer_mse: list[float] = []
    successes_by_seed: dict[str, int] = {}
    for seed in seeds:
        run_name = f"{prefix}{seed}"
        with (root / f"{run_name}_summary.json").open(encoding="utf-8") as source:
            summary = json.load(source)["online_training"]
        episode_statuses = [
            bool(value["success"]) for value in summary["episode_results"]
        ]
        statuses.extend(episode_statuses)
        steps.extend(int(value["steps"]) for value in summary["episode_results"])
        successes_by_seed[str(seed)] = sum(episode_statuses)

        transfer_rows: dict[int, list[float]] = {}
        log_path = root / "logs" / run_name / "online_training.jsonl"
        with log_path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("state_before") != "transfer":
                    continue
                transfer_rows.setdefault(int(record["episode"]), []).append(
                    float(record["transfer_goal_mse"])
                )
        minimum_transfer_mse.extend(
            min(values) for values in transfer_rows.values() if values
        )

    episodes = len(statuses)
    return (
        {
            "episodes": episodes,
            "successes": sum(statuses),
            "success_rate": sum(statuses) / episodes,
            "successes_by_seed": successes_by_seed,
            "mean_steps": statistics.mean(steps),
            "mean_min_transfer_goal_mse": statistics.mean(minimum_transfer_mse),
        },
        statuses,
    )


def main() -> None:
    args = parse_args()
    root = args.outputs_root.expanduser().resolve()
    results: dict[str, Any] = {}
    statuses: dict[str, list[bool]] = {}
    for name, prefix in MODES.items():
        results[name], statuses[name] = load_mode(root, prefix, args.seeds)

    comparisons = {
        f"{name}_vs_baseline": paired_exact(statuses["baseline"], values)
        for name, values in statuses.items()
        if name != "baseline"
    }
    training_summary_path = root / "latent_subgoal_planner_h25_summary.json"
    with training_summary_path.open(encoding="utf-8") as source:
        training = json.load(source)
    output = {
        "version": "v0.1",
        "seeds": args.seeds,
        "episodes_per_seed": 10,
        "actor_frozen": True,
        "methods": results,
        "comparisons": comparisons,
        "latent_subgoal_training": {
            "best_epoch": training["best_epoch"],
            "train_samples": training["samples"]["train"],
            "validation_samples": training["samples"]["validation"],
            "validation_latent_mse": training["validation_metrics"]["latent_mse"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
