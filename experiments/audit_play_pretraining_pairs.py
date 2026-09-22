#!/usr/bin/env python3
"""Audit exact same-episode H=25 pairs in the OGBench play latent cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ALIGN, GRASP, TRANSFER = range(3)


def robot_proprio(observations: np.ndarray) -> np.ndarray:
    closure = np.clip(observations[:, 17:18] / 3.0, 0.0, 1.0)
    return np.concatenate(
        [observations[:, 12:15], observations[:, 15:17], closure], axis=-1
    ).astype(np.float32, copy=False)


def state_ids_from_predicates(
    predicates: np.ndarray, proprios: np.ndarray, closed_threshold: float
) -> np.ndarray:
    aligned = predicates[:, 0] >= 0.5
    grasped = (predicates[:, 1] >= 0.5) & (proprios[:, 5] >= closed_threshold)
    return np.where(grasped, TRANSFER, np.where(aligned, GRASP, ALIGN)).astype(np.int64)


def exact_rows(bank_indices: np.ndarray, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.searchsorted(bank_indices, requested)
    valid = rows < len(bank_indices)
    safe = np.minimum(rows, len(bank_indices) - 1)
    valid &= bank_indices[safe] == requested
    return safe, valid


def counts(values: np.ndarray) -> dict[str, int]:
    unique, frequencies = np.unique(values, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, frequencies)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lookahead", type=int, default=25)
    parser.add_argument("--closed-threshold", type=float, default=0.45)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cache:
        bank_indices = np.asarray(cache["unique_source_indices"], dtype=np.int64)
        latents = np.asarray(cache["latents"], dtype=np.float32)
        current_rows = np.asarray(cache["current_rows"], dtype=np.int64)
        goal_rows = np.asarray(cache["goal_rows"], dtype=np.int64)
        source_indices = np.asarray(cache["source_indices"], dtype=np.int64)
        goal_indices = np.asarray(cache["goal_indices"], dtype=np.int64)
        episode_ids = np.asarray(cache["episode_ids"], dtype=np.int64)
        predicates = np.asarray(cache["predicate_targets"][:, :2], dtype=np.float32)
    with np.load(args.source, allow_pickle=False) as source:
        observations = np.asarray(source["observations"], dtype=np.float32)
        terminals = np.asarray(source["terminals"], dtype=bool)

    if not np.all(np.diff(bank_indices) > 0):
        raise ValueError("Latent bank indices must be sorted and unique")
    if not np.array_equal(bank_indices[current_rows], source_indices):
        raise ValueError("current_rows do not map to source_indices")
    if not np.array_equal(bank_indices[goal_rows], goal_indices):
        raise ValueError("goal_rows do not map to goal_indices")

    proprios = robot_proprio(observations[source_indices])
    states = state_ids_from_predicates(predicates, proprios, args.closed_threshold)
    current_episode_from_index = np.searchsorted(np.flatnonzero(terminals), source_indices)
    goal_episode_from_index = np.searchsorted(np.flatnonzero(terminals), goal_indices)
    if not np.array_equal(current_episode_from_index, episode_ids):
        raise ValueError("Cached episode_ids disagree with terminal boundaries")

    future_indices = source_indices + args.lookahead
    future_rows, future_is_encoded = exact_rows(bank_indices, future_indices)
    same_episode_goal = goal_episode_from_index == episode_ids
    goal_is_future = goal_indices > source_indices
    future_same_episode = (
        np.searchsorted(np.flatnonzero(terminals), future_indices) == episode_ids
    )
    transfer = states == TRANSFER
    exact_h = transfer & same_episode_goal & goal_is_future & future_same_episode & future_is_encoded

    capped_indices = np.minimum(future_indices, goal_indices)
    capped_rows, capped_is_encoded = exact_rows(bank_indices, capped_indices)
    capped_valid = transfer & same_episode_goal & goal_is_future & capped_is_encoded

    def mse(a: np.ndarray, b: np.ndarray) -> float | None:
        return float(np.mean((a - b) ** 2)) if len(a) else None

    h_rows = np.flatnonzero(exact_h)
    c_rows = np.flatnonzero(capped_valid)
    summary = {
        "cache_samples": int(len(source_indices)),
        "source_episodes": int(len(np.unique(episode_ids))),
        "unique_current_source_indices": int(len(np.unique(source_indices))),
        "duplicate_current_rows": int(len(source_indices) - len(np.unique(source_indices))),
        "state_counts": {"align": int(np.sum(states == ALIGN)), "grasp": int(np.sum(states == GRASP)), "transfer": int(np.sum(states == TRANSFER))},
        "same_episode_goal": int(np.sum(same_episode_goal)),
        "future_goal": int(np.sum(goal_is_future)),
        "transfer_future_goal_same_episode": int(np.sum(transfer & same_episode_goal & goal_is_future)),
        "exact_h25_transfer_pairs": int(len(h_rows)),
        "exact_h25_episodes": int(len(np.unique(episode_ids[h_rows]))),
        "capped_at_goal_transfer_pairs": int(len(c_rows)),
        "capped_at_goal_episodes": int(len(np.unique(episode_ids[c_rows]))),
        "capped_full_h25_fraction": float(np.mean(capped_indices[c_rows] == future_indices[c_rows])) if len(c_rows) else None,
        "goal_horizon": {
            "min": int(np.min(goal_indices[c_rows] - source_indices[c_rows])) if len(c_rows) else None,
            "median": float(np.median(goal_indices[c_rows] - source_indices[c_rows])) if len(c_rows) else None,
            "mean": float(np.mean(goal_indices[c_rows] - source_indices[c_rows])) if len(c_rows) else None,
            "max": int(np.max(goal_indices[c_rows] - source_indices[c_rows])) if len(c_rows) else None,
        },
        "capped_baselines": {
            "current_to_target_mse": mse(latents[current_rows[c_rows]], latents[capped_rows[c_rows]]),
            "goal_to_target_mse": mse(latents[goal_rows[c_rows]], latents[capped_rows[c_rows]]),
        },
        "samples_per_episode_quantiles": {},
    }
    per_episode = np.asarray(list(counts(episode_ids[c_rows]).values()), dtype=np.int64)
    if len(per_episode):
        summary["samples_per_episode_quantiles"] = {
            str(q): float(np.quantile(per_episode, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        }
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
