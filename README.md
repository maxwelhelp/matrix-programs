# Executable Matrix Programs

**Author:** Maxim Vladimirovich Zhivotok — Independent Researcher, Ukraine  
**Contact:** maxwelhelp@gmail.com

Weight-derived executable programs for faithful replacement and factorized intervention in RoPE attention and SwiGLU MLPs.

## Main result

On `Qwen/Qwen2.5-0.5B-Instruct`, simultaneously replacing all 336 attention heads and all 24 MLP sublayers preserves the top-1 prediction on 64/64 prompts with a median full-logit-vector discrepancy of **0.258%**.

Paper: [`paper/main.pdf`](paper/main.pdf)  
English Markdown: [`paper/paper_en.md`](paper/paper_en.md)  
Russian source: [`paper/paper_ru.md`](paper/paper_ru.md)  
arXiv: to be added after submission.

## Repository layout

```text
extractor/      component reconstruction and exploratory lens extraction
experiments/    final full-model replacement and causal-atlas driver
figures/        reproducible figure-generation script and rendered figures
results/        CSV/JSON outputs used in the paper
exploratory/    gate-lens demonstration and corrected held-out atom rerun
paper/          English/Russian text, LaTeX source, bibliography, and PDF
```

## Full article experiment

```bash
python experiments/qwen_matrix_program_attention_mlp_v4_1_github_fixed.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --final-article-suite \
  --final-prompts 64 \
  --final-batch-size 64 \
  --max-length 192 \
  --final-atom-selection-prompts 24 \
  --final-atom-eval-prompts 32 \
  --final-atom-random-controls 3 \
  --out-dir outputs/qwen_matrix_program_final_article_v4_1
```

Use `--final-batch-size 32` if batch 64 does not fit in GPU memory. Completed experiment JSON files are resumable in the same output directory.

## Component reconstruction

```bash
python extractor/qwen_matrix_program_attention_mlp_full_nomovement.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --heads all \
  --prompt-suites all \
  --prompts-per-suite 8 \
  --same-text-repeats 2 \
  --max-length 192 \
  --max-delta 64 \
  --mlp-layers all \
  --out-dir outputs/qwen_matrix_program_full
```

## Regenerate the figures

```bash
python figures/generate_matrix_program_figures_v06_1_fixed.py \
  --results-dir results \
  --out-dir figures
```

The script writes both vector PDF and PNG versions and validates the propagation examples against the atlas CSV.

## Exploratory analyses

The gate-lens and atom-split analyses are not part of the paper's primary quantitative claims.

```bash
python exploratory/gate_lens_demo.py --device cuda
python exploratory/atom_split_fixed_seeds.py --device cuda --batch-size 32
```

The original archived atom run contained duplicated random controls because the same seed was reused. The v4.1 driver and the wrapper above use independent random-control seeds.

## Hardware and reproducibility

The reported full run used a Tesla P40 24GB, fp16 model weights, eager attention, and seed 0. The exact package versions and model revision were not captured in the original run and are stated as such in [`environment.txt`](environment.txt). To capture a reproduction environment without installing or changing packages, run:

```bash
python scripts/capture_environment.py --output environment_current.json
```

## Validate the release

```bash
python scripts/validate_release.py
```

This checks required files, Python syntax, JSON/CSV integrity, headline metrics, bibliography keys, terminology, and SHA256 hashes. The detailed pre-submission audit is in [`AUDIT_REPORT.md`](AUDIT_REPORT.md).

## Citation

A BibTeX entry will be added after the arXiv identifier is assigned.
