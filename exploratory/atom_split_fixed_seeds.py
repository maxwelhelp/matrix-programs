#!/usr/bin/env python3
"""Rerun only the held-out atom split with independent random-control seeds.

The full v4.1 experiment driver contains the seed fix. This wrapper skips the
other expensive atlas experiments and executes only atom selection/evaluation.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default="outputs/atom_split_fixed_seeds")
    args, extra = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    driver = repo_root / "experiments" / "qwen_matrix_program_attention_mlp_v4_1_github_fixed.py"
    if not driver.exists():
        raise FileNotFoundError(f"Experiment driver not found: {driver}")

    command = [
        sys.executable,
        str(driver),
        "--model", args.model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--attn-implementation", "eager",
        "--final-article-suite",
        "--final-prompts", "64",
        "--final-batch-size", str(args.batch_size),
        "--max-length", "192",
        "--final-skip-global",
        "--final-skip-layer-groups",
        "--final-skip-head-atlas",
        "--final-skip-mlp-atlas",
        "--final-atom-selection-prompts", "24",
        "--final-atom-eval-prompts", "32",
        "--final-atom-random-controls", "3",
        "--out-dir", args.out_dir,
    ]
    command.extend(extra)

    print("Running:", " ".join(command), flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
