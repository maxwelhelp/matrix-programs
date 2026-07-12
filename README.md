# Executable Matrix Programs

**Author:** Maxim Vladimirovich Zhivotok — Independent Researcher, Ukraine  
**Contact:** maxwelhelp@gmail.com

Weight-derived executable programs for faithful replacement, explicit execution tracing, and factorized intervention in RoPE attention and SwiGLU MLPs.

## Main result

On `Qwen/Qwen2.5-0.5B-Instruct`, simultaneously replacing all 336 attention heads and all 24 MLP sublayers preserves the top-1 prediction on 64/64 prompts with a median full-logit-vector discrepancy of **0.258%**.

Paper: [`paper/main.pdf`](paper/main.pdf)  
English Markdown: [`paper/paper_en.md`](paper/paper_en.md)  
Russian source: [`paper/paper_ru.md`](paper/paper_ru.md)  
Zenodo DOI: <https://doi.org/10.5281/zenodo.21312311>  
arXiv: to be added after submission.

## What the method decomposes the model into

The method does not train a surrogate network. It rewrites the original pretrained computation into explicit weight-derived objects that can be executed again inside the native forward pass.

### Attention: routing objects plus a payload object

For a selected attention head, the method extracts:

1. a family of relative-position-conditioned QK routing matrices `M_qk[Δ]`;
2. one VO payload map `C_vo`.

Using homogeneous coordinates `x_aug = [x, 1]`:

```text
score(i,j) = x_aug_i^T M_qk[Δ] x_aug_j
Δ = i - j
```

The homogeneous matrix is not treated as an opaque object. It can be unpacked as:

```text
M_qk[Δ] =
[ B_Δ    q_Δ  ]
[ k_Δ^T  c_Δ  ]
```

which gives the explicit score program:

```text
score(i,j) =
      c_Δ                 constant relative-position route
    + x_i^T q_Δ           query-affine term
    + k_Δ^T x_j           key-affine term
    + x_i^T B_Δ x_j       query-key bilinear interaction
```

After softmax, the separate VO program computes what the head writes:

```text
payload(j) = C_vo x_aug_j
A(i,j) = softmax_j(score(i,j))
head_output(i) = sum_j A(i,j) payload(j)
```

Functionally:

- QK determines **where the head reads from and why a source token is selected**;
- VO determines **what transformed payload is written back to the residual stream**.

For Qwen2.5-0.5B, `H=896` and head dimension `D=64`:

```text
M_qk[Δ]: 897 x 897, rank <= 64
C_vo:     896 x 897, rank <= 64
```

The external matrices are large, but their functional rank is bounded by the head dimension.

### SwiGLU MLP: gated rank-1 read/write atoms

Each intermediate MLP neuron becomes one native executable atom:

```text
gate_j(x) = SiLU(Wg_j x + bg_j)
read_j(x) = Wu_j x + bu_j
coefficient_j(x) = gate_j(x) * read_j(x)
contribution_j(x) = coefficient_j(x) * Wd[:,j]
MLP(x) = sum_j contribution_j(x) + bd
```

Each atom therefore contains:

- a gate direction;
- a read direction;
- a write direction;
- an input-dependent scalar coefficient.

Equivalently, its linear read/write part is a rank-1 object:

```text
atom_j = Wd[:,j] outer Wu_j
```

For Qwen2.5-0.5B, each MLP contains 4864 such gated atoms. The full `896 x 896` matrix for each atom does not need to be materialized; it is stored compactly as read and write vectors.

## Concrete single-head / single-MLP execution trace

A focused demonstration is included to make the structure visible on actual tokens rather than only reporting aggregate replacement metrics.

Script:

[`experiments/matrix_program_single_demo_v3_affine_trace_fixed.py`](experiments/matrix_program_single_demo_v3_affine_trace_fixed.py)

Results:

[`results/demo_affine_trace_L20H5/`](results/demo_affine_trace_L20H5/)

Configuration:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
prompt: "The capital of France is"
attention: layer 20, head 5
MLP: layer 20
selected token: " is"
baseline next token: " Paris"
```

For the edge `" is" <- " France"`, the QK score is explicitly decomposed into:

```text
constant_route   = -1.839059
query_affine     = +0.691513
key_affine       = +0.838055
content_bilinear = +1.865252
--------------------------------
total_score      = +1.555761
attention        = 0.517321
VO contribution norm = 2.584062
```

In this token pair, the largest positive score term is the query-key bilinear interaction. The trace also shows a different mechanism for another source token: `"The"` receives a strong positive key-affine term, but its much smaller VO payload leads to a smaller final contribution.

For the same selected token, the MLP trace exposes active gate/read/write atoms. The strongest causally tested example was:

```text
atom 4520:
gate            = +5.3034
read            = -3.8175
gate * read     = -20.2459
causal delta on logit(" Paris") after removal = -0.8750
```

Faithful reinsertion checks for this demonstration:

```text
selected attention head:
    program/native head-output relative error = 1.061e-3
    full-logit relative discrepancy           = 0.0934%
    top-1 preserved                           = yes

selected MLP:
    program/native MLP-output relative error  = 3.989e-4
    full-logit relative discrepancy           = 0.1014%
    top-1 preserved                           = yes
```

The same report then performs factorized native-forward interventions:

- replace the selected head with its exact QK/VO program;
- change only QK routing to uniform causal routing;
- amplify only QK scores while leaving VO fixed;
- zero only the selected head's VO output;
- replace the selected MLP with its explicit atom program;
- remove individual MLP atoms at the selected token.

This demonstration is illustrative and is separate from the paper's primary aggregate 64-prompt replacement result.

### Run the focused trace

```bash
python experiments/matrix_program_single_demo_v3_affine_trace_fixed.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "The capital of France is" \
  --attention-layer 20 \
  --head 5 \
  --mlp-layer 20 \
  --token -1 \
  --top-k 8 \
  --causal-atoms 3 \
  --device cuda \
  --dtype fp16 \
  --out-dir outputs/demo_affine_trace_L20H5
```

The demo writes:

```text
trace.txt
report.md
report.html
report.json
attention_score_terms.csv
attention_token_flow.csv
mlp_atoms.csv
interventions.csv
precomputed_objects.csv
opaque_vs_explicit.csv
```

## Repository layout

```text
extractor/      component reconstruction and exploratory lens extraction
experiments/    full-model benchmark and focused execution-trace scripts
figures/        reproducible figure-generation script and rendered figures
results/        paper outputs and focused execution-trace results
exploratory/    gate-lens demonstration and corrected held-out atom rerun
paper/          English/Russian text, LaTeX source, bibliography, and PDF
scripts/        release validation and environment capture
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
