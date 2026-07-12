# Executable Matrix Program — Single-Component Demo

## Configuration

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Prompt: `The capital of France is`
- Tokens: [0] `The` [1] ` capital` [2] ` of` [3] ` France` [4] ` is`
- Attention: layer `20`, head `5`
- MLP: layer `20`
- Selected token: `4`
- Baseline next-token prediction: ` Paris`

## What the ordinary forward pass hides vs. what the program exposes

| component | ordinary forward exposes | matrix program exposes | separate intervention handle |
| --- | --- | --- | --- |
| Attention L20H5 | One mixed head/module output tensor | Per-token QK scores, routing weights, VO payloads, token-to-token contributions | Change QK routing separately from VO payload |
| MLP L20 | One nonlinear output tensor for the whole MLP | Native atom gate, read, coefficient, write direction, and contribution | Remove or scale individual gate/read/write atoms |


The replacement is **not a trained surrogate**. It is a weight-derived executable refactorization of the original component.

## Static objects precomputed from weights

| component | object | meaning | shape | Frobenius norm | rank <= | input-dependent |
| --- | --- | --- | --- | --- | --- | --- |
| attention | M_qk[0] | QK routing at relative distance d=0 | 897x897 | 2.164815 | 64 | no |
| attention | M_qk[1] | QK routing at relative distance d=1 | 897x897 | 2.115027 | 64 | no |
| attention | M_qk[2] | QK routing at relative distance d=2 | 897x897 | 2.092736 | 64 | no |
| attention | M_qk[3] | QK routing at relative distance d=3 | 897x897 | 2.095429 | 64 | no |
| attention | M_qk[4] | QK routing at relative distance d=4 | 897x897 | 2.106747 | 64 | no |
| attention | C_vo | VO payload/read-write map | 896x897 | 3.618902 | 64 | no |
| mlp | Wg | Gate directions | 4864x896 | 43.409763 | 896 | no |
| mlp | Wu | Read directions | 4864x896 | 42.589981 | 896 | no |
| mlp | Wd | Write directions | 896x4864 | 40.708153 | 896 | no |
| mlp | rank-1 atoms | atom_j = Wd[:,j] outer Wu[j,:], gated at runtime | 4864 atoms of 896x896 |  | 1 | no |


## One attention head: explicit score trace and VO payload

Destination token: ` is`.

For every source token, the total QK score is shown as the exact sum
`c_delta + q_affine(x_i) + k_affine(x_j) + x_i.T B_delta x_j`.
The table then shows the attention weight and the split between the
content-dependent VO payload and the constant VO bias payload.

| src | source token | d | c_delta | q_affine | k_affine | content bilinear | dominant score term | total score | attention | VO content norm | VO bias norm | final contribution norm | direct target proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 |  France | 1 | -1.839059 | 0.691513 | 0.838055 | 1.865252 | content_bilinear | 1.555761 | 0.517321 | 4.976098 | 0.389668 | 2.584062 | 0.184699 |
| 1 |  capital | 3 | -1.818052 | 0.567655 | -0.029973 | 0.470111 | constant_route | -0.810260 | 0.048552 | 5.727304 | 0.389668 | 0.277777 | 0.013581 |
| 0 | The | 4 | -1.831136 | 0.497565 | 3.019386 | -0.425673 | key_affine | 1.260142 | 0.384924 | 0.882959 | 0.389668 | 0.231238 | -0.002858 |
| 4 |  is | 0 | -1.892749 | 0.736525 | -0.189413 | 0.206183 | constant_route | -1.139453 | 0.034934 | 3.785011 | 0.389668 | 0.133321 | 0.007479 |
| 2 |  of | 2 | -1.815044 | 0.631276 | -0.581058 | -0.269957 | constant_route | -2.034783 | 0.014269 | 3.367632 | 0.389668 | 0.048417 | 0.001118 |


### Attention faithfulness

```json
{
  "layer": 20,
  "head": 5,
  "kv_head": 0,
  "hidden_size": 896,
  "head_dim": 64,
  "selected_destination_token": 4,
  "native_vs_program_qk_score_rel": 0.001251577865332365,
  "native_vs_program_attention_rel": 0.0005681165494024754,
  "native_vs_program_payload_rel": 0.0002925188164226711,
  "native_vs_program_head_output_rel": 0.0010610510362312198,
  "qk_amplify": 1.5,
  "captured_input_dtype": "torch.float16",
  "projection_weight_dtype": "torch.float16",
  "program_compute_dtype": "torch.float32",
  "four_term_score_max_abs_error": 2.384185791015625e-07
}
```

## One MLP layer: native gate/read/write atoms

Selected token: ` is`.

The direct target proxy is descriptive. The causal delta is measured with a full native forward pass after removing that exact atom.

| atom | gate | read | gate*read | write norm | atom output norm | direct target proxy | causal delta target | top-1 changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4520 | 5.303450 | -3.817500 | -20.245918 | 0.566649 | 11.472340 | 0.599000 | -0.875000 | no |
| 3433 | 2.045197 | -3.123717 | -6.388614 | 0.627197 | 4.006920 | 0.145363 | -0.140625 | no |
| 93 | 2.690228 | 3.694934 | 9.940214 | 0.546614 | 5.433459 | -0.117681 | 0.140625 | no |
| 4492 | 1.665709 | -1.762724 | -2.936185 | 0.562254 | 1.650881 | 0.085919 |  |  |
| 1043 | 3.945675 | -1.836678 | -7.246933 | 0.507208 | 3.675699 | 0.062917 |  |  |
| 3722 | 3.768771 | 1.135553 | 4.279638 | 0.451744 | 1.933301 | 0.043443 |  |  |
| 3676 | 1.493087 | 1.963298 | 2.931376 | 0.656238 | 1.923679 | 0.040857 |  |  |
| 1935 | 2.900367 | 1.272545 | 3.690848 | 0.432449 | 1.596103 | 0.040333 |  |  |


### MLP faithfulness

```json
{
  "layer": 20,
  "hidden_size": 896,
  "intermediate_size": 4864,
  "selected_token": 4,
  "native_vs_program_mlp_output_rel": 0.0003989161632489413,
  "native_vs_program_mlp_output_max_abs": 0.0032215118408203125
}
```

## Replacement and interventions inside the native forward pass

| run | what changed | logit( Paris) | delta target | full-logit rel % | KL | top-1 | top-1 preserved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Original unmodified model | 17.234375 | 0.000000 | 0.000000 | 1.869e-10 |  Paris | yes |
| attention_exact_program | Replace selected native head with explicit QK/VO program | 17.234375 | 0.000000 | 0.093428 | 7.287e-06 |  Paris | yes |
| attention_qk_uniform | Change only QK routing to uniform causal routing; keep VO payload | 17.250000 | 0.015625 | 3.292048 | 0.001358 |  Paris | yes |
| attention_qk_amplified | Scale only QK routing scores by 1.5; keep VO payload | 17.234375 | 0.000000 | 0.701711 | 8.799e-05 |  Paris | yes |
| attention_vo_zero | Zero only the selected head's VO payload/output | 17.187500 | -0.046875 | 4.819112 | 0.003244 |  Paris | yes |
| mlp_exact_program | Replace selected native MLP with explicit gate/read/write program | 17.234375 | 0.000000 | 0.101392 | 4.637e-06 |  Paris | yes |
| mlp_atom_zero_4520 | Remove native MLP atom 4520 only at token position 4 | 16.359375 | -0.875000 | 16.852038 | 0.158753 |  Paris | yes |
| mlp_atom_zero_3433 | Remove native MLP atom 3433 only at token position 4 | 17.093750 | -0.140625 | 5.859044 | 0.008079 |  Paris | yes |
| mlp_atom_zero_93 | Remove native MLP atom 93 only at token position 4 | 17.375000 | 0.140625 | 9.545588 | 0.022206 |  Paris | yes |


## Conceptual summary

```text
Ordinary attention head:
    hidden tensor -> opaque head output

Executable attention program:
    M_qk[d] = [[B_d, q_d], [k_d^T, c_d]]

    score(i,j) =
        c_d
      + x_i^T q_d
      + k_d^T x_j
      + x_i^T B_d x_j

    score -> softmax routing weights
    C_vo -> source payloads
    routing * payload -> token-to-token contributions
    sum contributions -> head output

Ordinary SwiGLU MLP:
    hidden tensor -> opaque MLP output

Executable MLP program:
    for each native atom j:
      gate_j(x) * read_j(x) * write_j
    sum all atoms -> MLP output
```
