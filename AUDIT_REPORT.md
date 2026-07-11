# Final pre-submission audit - v08

## Scope

This audit covers the English and Russian manuscripts, LaTeX source, BibTeX database, compiled PDF, all four figures, released Python scripts, machine-readable CSV/JSON results, README commands, and the GitHub/arXiv package layouts.

## Blocking layout bug fixed

The empty duplicate header of the Companion reconstruction table was caused by a `longtable` continuation header at a page break, not by a second data table in the source. All short `longtable` environments were replaced with non-floating `tabular` environments. The final PDF contains exactly one `Component / Median relative error` header and all five rows:

- QK scores with RoPE+bias: 0.0526%
- Attention probabilities: 0.0599%
- VO head output: 0.0277%
- MLP output: 0.0426%
- QK without bias folding: 105%

The no-bias paragraph follows the populated table in the source and renders directly after Figure 1 without an empty table skeleton.

## LaTeX and PDF checks

- Clean build sequence completed: `pdflatex`, BibTeX, `pdflatex`, `pdflatex`.
- 15 pages.
- No undefined citations or references.
- No missing figures.
- No LaTeX errors, overfull boxes, or underfull-box warnings in the final build log.
- PDF opens successfully in PyMuPDF and renders in both PDFium and Poppler.
- All four figures are present and visible.
- All fonts are embedded. Matplotlib figure fonts were regenerated as embedded CID TrueType rather than Type 3.
- PDF metadata contains the full author name: Maxim Vladimirovich Zhivotok.
- The GitHub URL in the manuscript is `https://github.com/maxwelhelp/matrix-programs`.

## Figure reproducibility

The four released figures were regenerated from the released result CSVs using `figures/generate_matrix_program_figures_v06_1_fixed.py`. The regenerated PDF/PNG files replace the older copies in both the GitHub and arXiv packages. Figure 3 uses `valid_tensor_rel_percent`, aligned to the first post-source residual, and its example values match the atlas CSV.

## Result-table integrity

Validated counts:

- `all_experiment_summaries.csv`: 1,662 rows
- `all_prompt_metrics.csv`: 106,368 rows
- `head_causal_atlas.csv`: 336 rows
- `mlp_causal_atlas.csv`: 24 rows
- `layer_group_atlas.csv`: 24 rows

Validated headline values:

- global attention program discrepancy: 0.254006% released per-experiment median
- global MLP program discrepancy: 0.240513% released per-experiment median
- global backbone program discrepancy: 0.258383%, reported as 0.258%
- global backbone maximum: 0.923447%, reported as 0.923%
- top-1 preservation: 64/64 prompts
- conventional median QK effect/replacement-discrepancy ratio: 24.8869x
- conventional median VO ratio: 20.6209x
- conventional median MLP ratio: 152.0466x
- median delays to peak: 10 layers for QK and 9 for VO

## Aggregation convention clarified

The released experiment driver uses `torch.median` for per-experiment prompt summaries. For even N this returns the lower of the two central sorted values. Across rows of the released head and MLP atlas tables, the paper uses the conventional median, averaging the two central values. This is now stated in the English manuscript, Russian manuscript, LaTeX source, result README, and `results/AGGREGATION_CONVENTIONS.md`. No experiment values were changed.

## Terminology cleanup

Released atlas column names and the experiment script now use `replacement_discrepancy` rather than legacy internal names containing `noise`. Numeric values are unchanged. The paper does not claim that the native identity control is an independent numerical lower-bound estimate.

## Bibliography audit

- 15 BibTeX entries.
- Every citation key used by `main.tex` has a matching BibTeX entry.
- Author lists were expanded for Beyond Components, RASP decompilation, and White-Box Transformers.
- The Golden reference uses the current title `Equivalent Linear Mappings of Large Language Models` for arXiv:2505.24293.
- The bibliography includes the stated arXiv identifiers for the cited preprints.

## Code and artifact checks

- Every released Python file passes `py_compile`.
- Every released JSON file parses.
- The figure generator reruns successfully against `results/`.
- README commands point to files that exist in the repository.
- GitHub and arXiv copies of `main.tex`, `main.bbl`, `references.bib`, the PDF, and all figures are synchronized byte-for-byte where applicable.
- Auxiliary LaTeX build files are excluded from the GitHub release.

## Historical information that cannot be reconstructed

The exact original Python, PyTorch, Transformers, CUDA package versions, model commit hash, and wall-clock duration were not captured in the archived run. They are marked `not recorded in the original run`; no values were invented. `scripts/capture_environment.py` records the current reproduction environment without installing or modifying packages.

The raw terminal stdout and thousands of resumable per-experiment JSON files were not present in the supplied result archive. The repository includes the complete structured CSV/JSON artifacts used by every table and figure, plus the generated result summary.

## Remaining declared limitation

The original held-out atom run reused a random-control seed. Those files remain available for transparency, the corrected code is released, and the atom result remains explicitly exploratory rather than a main quantitative claim.

## Final status

All automated validation checks pass. The final PDF was visually inspected page by page after the last rebuild. No empty tables, clipped text, overlapping elements, broken glyphs, missing figures, or unresolved references were found.
