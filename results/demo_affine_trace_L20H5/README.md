# Single-Component Affine Execution Trace

This directory contains a concrete token-level demonstration of the executable matrix-program representation.

It is intended to answer four questions directly:

1. What objects are extracted from an attention head and an MLP?
2. What is precomputed from the pretrained weights?
3. What becomes visible for specific token pairs and MLP atoms?
4. Can these objects be reinserted and intervened on inside the native forward pass?

## Configuration

```text
model: Qwen/Qwen2.5-0.5B-Instruct
prompt: "The capital of France is"
tokens:
  [0] "The"
  [1] " capital"
  [2] " of"
  [3] " France"
  [4] " is"

attention: layer 20, head 5
MLP: layer 20
selected token: [4] " is"
baseline next token: " Paris"
dtype: fp16 model, float32 explicit trace arithmetic
```

## Attention object structure

For each relative token distance `Δ=i-j`, the method constructs:

```text
M_qk[Δ] =
[ B_Δ    q_Δ  ]
[ k_Δ^T  c_Δ  ]
```

and evaluates:

```text
score(i,j) =
      c_Δ
    + x_i^T q_Δ
    + k_Δ^T x_j
    + x_i^T B_Δ x_j
```

The four terms are reported separately for every displayed source token.

The payload is computed independently:

```text
payload(j) = C_vo [x_j, 1]
head_output(i) = sum_j softmax(score(i,*))[j] payload(j)
```

Therefore the trace separates:

- why a source token is selected;
- how much attention it receives;
- what payload it carries;
- how large its final contribution is.

## Example token-level attention trace

Destination token: `" is"`.

| source | Δ | constant | query affine | key affine | bilinear | total score | attention | VO contribution norm | dominant score term |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `" France"` | 1 | -1.839059 | +0.691513 | +0.838055 | +1.865252 | +1.555761 | 0.517321 | 2.584062 | content bilinear |
| `"The"` | 4 | -1.831136 | +0.497565 | +3.019386 | -0.425673 | +1.260142 | 0.384924 | 0.231238 | key affine |
| `" capital"` | 3 | -1.818052 | +0.567655 | -0.029973 | +0.470111 | -0.810260 | 0.048552 | 0.277777 | constant route |
| `" is"` | 0 | -1.892749 | +0.736525 | -0.189413 | +0.206183 | -1.139453 | 0.034934 | 0.133321 | constant route |
| `" of"` | 2 | -1.815044 | +0.631276 | -0.581058 | -0.269957 | -2.034783 | 0.014269 | 0.048417 | constant route |

Two different mechanisms are visible:

- `" France"` is selected mainly through the positive bilinear interaction between the destination and source states;
- `"The"` has the strongest key-affine term, but its smaller VO payload gives it a much smaller final residual contribution.

This distinction is not visible from the attention weight alone.

## MLP object structure

Each intermediate neuron is represented as a gated read/write atom:

```text
gate_j(x) = SiLU(Wg_j x)
read_j(x) = Wu_j x
coefficient_j(x) = gate_j(x) * read_j(x)
contribution_j(x) = coefficient_j(x) * Wd[:,j]
MLP(x) = sum_j contribution_j(x)
```

## Top MLP atoms at the selected token

| atom | gate | read | coefficient | causal Δlogit(`" Paris"`) |
|---:|---:|---:|---:|---:|
| 4520 | +5.3034 | -3.8175 | -20.2459 | -0.8750 |
| 3433 | +2.0452 | -3.1237 | -6.3886 | -0.1406 |
| 93 | +2.6902 | +3.6949 | +9.9402 | +0.1406 |

The causal delta is measured by removing that exact atom only at the selected token and rerunning the full model. It is not inferred only from a projection or token lens.

## Exact replacement checks

| replacement | local component error | full-logit discrepancy | target-logit delta | top-1 |
|---|---:|---:|---:|---|
| selected attention head | 1.061e-3 | 0.0934% | +0.0000 | `" Paris"` |
| selected MLP | 3.989e-4 | 0.1014% | +0.0000 | `" Paris"` |

The small nonzero discrepancies are consistent with the mixed fp16/native and float32 explicit calculation paths used in this demo.

## Factorized interventions

| intervention | full-logit discrepancy | Δlogit(`" Paris"`) | top-1 preserved |
|---|---:|---:|---|
| QK uniform, VO unchanged | 3.2920% | +0.0156 | yes |
| QK amplified, VO unchanged | 0.7017% | +0.0000 | yes |
| selected VO output zeroed | 4.8191% | -0.0469 | yes |
| MLP atom 4520 removed | 16.8520% | -0.8750 | yes |
| MLP atom 3433 removed | 5.8590% | -0.1406 | yes |
| MLP atom 93 removed | 9.5456% | +0.1406 | yes |

Top-1 preservation does not mean that the intervention had no effect. The full-logit vector and target logit can change substantially while the same token remains highest.

## Files

```text
trace.txt                  compact plain-text execution trace
report.md                  complete Markdown report
report.html                browser-readable report
report.json                machine-readable combined report
attention_score_terms.csv  unpacked QK score terms by token pair
attention_token_flow.csv   attention, payload, and contribution values
mlp_atoms.csv              gate/read/write atom table
interventions.csv          native-forward replacement and intervention metrics
precomputed_objects.csv    shapes and meanings of weight-derived objects
opaque_vs_explicit.csv     ordinary-forward vs executable-program comparison
```

## Reproduce

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

## Interpretation boundary

The numerical decomposition of the QK score, the VO payload, the MLP coefficients, and the replacement/intervention measurements are directly computed.

Labels such as “content interaction,” “query-affine,” “key-affine,” and “constant route” identify the algebraic role of each term. They do not by themselves establish a complete semantic explanation of the model's final answer.

The demonstration is a component-level execution trace, not a claim that one head or one MLP atom alone fully explains why the model predicts `" Paris"`.
