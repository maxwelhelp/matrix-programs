#!/usr/bin/env python3
"""Run the exploratory gate/read/write lens used for the qualitative example.

This is not part of the paper's primary quantitative claims. It launches the
main no-movement extractor with lens-oriented defaults while allowing extra
extractor arguments to be appended on the command line.
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
    parser.add_argument("--out-dir", default="outputs/gate_lens_demo")
    parser.add_argument("--verify-ablation", action="store_true")
    args, extra = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    extractor = repo_root / "extractor" / "qwen_matrix_program_attention_mlp_full_nomovement.py"
    if not extractor.exists():
        raise FileNotFoundError(f"Extractor not found: {extractor}")

    command = [
        sys.executable,
        str(extractor),
        "--model", args.model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--attn-implementation", "eager",
        "--heads", "20:5,21:1,21:6",
        "--prompt-suites", "all",
        "--prompts-per-suite", "8",
        "--same-text-repeats", "1",
        "--max-length", "192",
        "--mlp-layers", "18,20,21,22,23",
        "--mlp-top-neurons", "16",
        "--mlp-example-prompts", "8",
        "--lens-pool", "1000",
        "--lens-topk", "8",
        "--out-dir", args.out_dir,
    ]
    if args.verify_ablation:
        command.append("--mlp-verify-ablation")
    command.extend(extra)

    print("Running:", " ".join(command), flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
