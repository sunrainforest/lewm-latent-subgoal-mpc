#!/usr/bin/env python3
"""Read-only schema and trajectory audit for stage-two planner datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_array(value: np.ndarray) -> dict:
    result = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if value.size and np.issubdtype(value.dtype, np.number):
        result["min"] = float(np.nanmin(value))
        result["max"] = float(np.nanmax(value))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    output = {}
    for path in args.paths:
        with np.load(path, allow_pickle=False) as dataset:
            output[str(path)] = {
                name: summarize_array(np.asarray(dataset[name]))
                for name in dataset.files
            }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
