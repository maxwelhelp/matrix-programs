#!/usr/bin/env python3
"""Validate the released paper, code, figures, and machine-readable result tables."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(msg: str) -> None:
    raise AssertionError(msg)


def read_csv(name: str) -> list[dict[str, str]]:
    path = ROOT / "results" / name
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lower_median(values: list[float]) -> float:
    values = sorted(values)
    return values[(len(values) - 1) // 2]


def conventional_median(values: list[float]) -> float:
    return float(statistics.median(values))


def approx(actual: float, expected: float, tol: float, label: str) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tol:
        fail(f"{label}: expected {expected} ± {tol}, got {actual}")


def validate_files() -> None:
    required = [
        "README.md",
        "environment.txt",
        "extractor/qwen_matrix_program_attention_mlp_full_nomovement.py",
        "experiments/qwen_matrix_program_attention_mlp_v4_1_github_fixed.py",
        "exploratory/gate_lens_demo.py",
        "exploratory/atom_split_fixed_seeds.py",
        "figures/generate_matrix_program_figures_v06_1_fixed.py",
        "figures/fig1_method_overview.pdf",
        "figures/fig2_head_heatmaps.pdf",
        "figures/fig3_propagation_profiles.pdf",
        "figures/fig4_global_replacement_by_suite.pdf",
        "paper/paper_en.md",
        "paper/paper_ru.md",
        "paper/main.pdf",
        "paper/latex/main.tex",
        "paper/latex/main.bbl",
        "paper/latex/references.bib",
        "results/all_experiment_summaries.csv",
        "results/all_prompt_metrics.csv",
        "results/head_causal_atlas.csv",
        "results/mlp_causal_atlas.csv",
        "results/layer_group_atlas.csv",
        "results/propagation_profiles.csv",
        "results/experiment_manifest.json",
        "results/final_prompts.json",
        "results/summary.json",
        "results/final_article_results.md",
        "results/AGGREGATION_CONVENTIONS.md",
    ]
    for rel in required:
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size == 0:
            fail(f"missing or empty required file: {rel}")


def validate_text() -> None:
    text_files = [
        ROOT / "README.md",
        ROOT / "environment.txt",
        ROOT / "paper/paper_en.md",
        ROOT / "paper/paper_ru.md",
        ROOT / "paper/latex/main.tex",
        ROOT / "paper/latex/references.bib",
        ROOT / "results/final_article_results.md",
    ]
    forbidden = ["RECORD BEFORE RELEASE", "[GitHub URL", "[имя автора]", "TODO", "PLACEHOLDER"]
    for p in text_files:
        s = p.read_text(encoding="utf-8")
        for token in forbidden:
            if token in s:
                fail(f"placeholder {token!r} in {p.relative_to(ROOT)}")
    tex = (ROOT / "paper/latex/main.tex").read_text(encoding="utf-8")
    if "\\begin{longtable}" in tex:
        fail("longtable remains in main.tex")
    if tex.count("Component & Median relative error") != 1:
        fail("companion table header does not occur exactly once")
    if "Maxim Vladimirovich Zhivotok" not in tex:
        fail("author name missing from main.tex")
    if "https://github.com/maxwelhelp/matrix-programs" not in tex:
        fail("GitHub URL missing from main.tex")
    if "replacement-noise" in tex or "noise floor" in tex:
        fail("deprecated noise terminology in main.tex")
    for core in ["0.258\\%", "0.566\\%", "0.923\\%", "24.89", "20.62", "152.05", "105\\%", "0.0526\\%"]:
        if core not in tex:
            fail(f"core number missing from main.tex: {core}")


def validate_bibliography() -> None:
    tex = (ROOT / "paper/latex/main.tex").read_text(encoding="utf-8")
    bib = (ROOT / "paper/latex/references.bib").read_text(encoding="utf-8")
    cited: set[str] = set()
    for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", tex):
        cited.update(x.strip() for x in group.split(","))
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = sorted(cited - keys)
    if missing:
        fail(f"missing BibTeX entries: {missing}")
    if len(keys) != 15:
        fail(f"expected 15 bibliography entries, got {len(keys)}")
    if "Equivalent Linear Mappings of Large Language Models" not in bib:
        fail("current Golden title missing")


def validate_code() -> None:
    for p in ROOT.rglob("*.py"):
        source = p.read_text(encoding="utf-8")
        compile(source, str(p), "exec")


def validate_json() -> None:
    for p in (ROOT / "results").glob("*.json"):
        with p.open(encoding="utf-8") as f:
            json.load(f)


def validate_results() -> None:
    summaries = read_csv("all_experiment_summaries.csv")
    prompt_rows = read_csv("all_prompt_metrics.csv")
    heads = read_csv("head_causal_atlas.csv")
    mlps = read_csv("mlp_causal_atlas.csv")
    layers = read_csv("layer_group_atlas.csv")
    propagation = read_csv("propagation_profiles.csv")

    if len(summaries) != 1662:
        fail(f"expected 1662 experiment summaries, got {len(summaries)}")
    if len(prompt_rows) != 106368:
        fail(f"expected 106368 prompt metrics, got {len(prompt_rows)}")
    if len(heads) != 336:
        fail(f"expected 336 head rows, got {len(heads)}")
    if len(mlps) != 24:
        fail(f"expected 24 MLP rows, got {len(mlps)}")
    if len(layers) != 24:
        fail(f"expected 24 layer-group rows, got {len(layers)}")
    if not propagation:
        fail("propagation_profiles.csv is empty")

    head_cols = set(heads[0])
    if "qk_effect_to_replacement_discrepancy_ratio" not in head_cols:
        fail("renamed QK ratio column missing")
    if "vo_effect_to_replacement_discrepancy_ratio" not in head_cols:
        fail("renamed VO ratio column missing")
    if "qk_effect_to_noise_ratio" in head_cols or "vo_effect_to_noise_ratio" in head_cols:
        fail("legacy noise column remains in head atlas")
    if "off_effect_to_replacement_discrepancy_ratio" not in set(mlps[0]):
        fail("renamed MLP ratio column missing")

    by_name = {row["name"]: row for row in summaries}
    for name, expected in {
        "global_attention_program": 0.2540059387683868,
        "global_mlp_program": 0.2405132632702589,
        "global_backbone_program": 0.2583828289061785,
    }.items():
        row = by_name[name]
        approx(float(row["logit_rel_percent_median"]), expected, 1e-12, name)
        if float(row["top1_preservation_rate"]) != 1.0:
            fail(f"top-1 not fully preserved for {name}")

    backbone = [float(r["logit_rel_percent"]) for r in prompt_rows if r["experiment"] == "global_backbone_program"]
    if len(backbone) != 64:
        fail(f"expected 64 global backbone prompt rows, got {len(backbone)}")
    approx(lower_median(backbone), 0.2583828289061785, 1e-12, "global backbone lower median")
    approx(max(backbone), 0.92344731092453, 1e-12, "global backbone max")

    approx(conventional_median([float(r["qk_effect_to_replacement_discrepancy_ratio"]) for r in heads]), 24.886930421793377, 1e-10, "QK atlas ratio")
    approx(conventional_median([float(r["vo_effect_to_replacement_discrepancy_ratio"]) for r in heads]), 20.620893034571978, 1e-10, "VO atlas ratio")
    approx(conventional_median([float(r["off_effect_to_replacement_discrepancy_ratio"]) for r in mlps]), 152.04662712224564, 1e-10, "MLP atlas ratio")
    approx(conventional_median([float(r["qk_delayed_peak_layers"]) for r in heads]), 10.0, 0.0, "QK delay")
    approx(conventional_median([float(r["vo_delayed_peak_layers"]) for r in heads]), 9.0, 0.0, "VO delay")


def write_hashes() -> None:
    lines: list[str] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.name == "SHA256SUMS.txt" or "__pycache__" in p.parts:
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {p.relative_to(ROOT).as_posix()}\n")
    (ROOT / "SHA256SUMS.txt").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    validate_files()
    validate_text()
    validate_bibliography()
    validate_code()
    validate_json()
    validate_results()
    write_hashes()
    print("VALIDATION PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise
