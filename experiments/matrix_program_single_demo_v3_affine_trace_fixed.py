#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matrix_program_single_demo_v3_affine_trace_fixed.py

Readable single-prompt demonstration of one executable Qwen attention head and
one SwiGLU MLP layer.

It does not train a surrogate. It derives the same weight-based objects used by
the research script, reinserts them into the native forward pass, and exports:

  report.html
  report.md
  report.json
  opaque_vs_explicit.csv
  precomputed_objects.csv
  attention_token_flow.csv
  attention_score_terms.csv
  mlp_atoms.csv
  interventions.csv

Example
-------
python matrix_program_single_demo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "The capital of France is" \
  --attention-layer 20 --head 5 --mlp-layer 20 --token -1 \
  --top-k 8 --causal-atoms 3 --device cuda --dtype fp16 \
  --out-dir outputs/demo_L20H5
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def rel_err(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    pred = pred.detach().float()
    target = target.detach().float()
    return float(torch.linalg.norm(pred - target) / torch.linalg.norm(target).clamp_min(eps))


def first_output_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported module output type: {type(output)!r}")


def replace_first_output(output: Any, new_first: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return new_first
    if isinstance(output, tuple):
        return (new_first,) + tuple(output[1:])
    if isinstance(output, list):
        return [new_first] + list(output[1:])
    raise TypeError(f"Unsupported module output type: {type(output)!r}")


def input_tensor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> torch.Tensor:
    if args and torch.is_tensor(args[0]):
        return args[0]
    for key in ("hidden_states", "x"):
        value = kwargs.get(key)
        if torch.is_tensor(value):
            return value
    tensor_keys = [str(k) for k, v in kwargs.items() if torch.is_tensor(v)]
    raise RuntimeError(
        "Could not locate module input tensor. "
        f"positional={len(args)}, tensor kwargs={tensor_keys}"
    )


def register_forward_hook_kwargs(module: Any, fn: Callable[..., Any]):
    try:
        return module.register_forward_hook(fn, with_kwargs=True)
    except TypeError:
        def old_hook(mod: Any, args: Tuple[Any, ...], output: Any):
            return fn(mod, args, {}, output)
        return module.register_forward_hook(old_hook)


def json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() <= 32:
            return value.detach().cpu().tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(json_value(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_value(row.get(k, "")) for k in keys})


def normalize_token_index(index: int, length: int) -> int:
    resolved = index if index >= 0 else length + index
    if not 0 <= resolved < length:
        raise ValueError(f"Token index {index} resolves to {resolved}, length={length}")
    return resolved


def token_text(tokenizer: Any, token_id: int) -> str:
    try:
        text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        text = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    return str(text).replace("\n", "\\n")


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000 or (value != 0 and abs(value) < 1e-4):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(title for _, title in columns) + " |\n"
    sep = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = []
    for row in rows:
        vals = []
        for key, _ in columns:
            vals.append(format_float(row.get(key, "")).replace("|", "\\|"))
        body.append("| " + " | ".join(vals) + " |")
    return header + sep + "\n".join(body) + "\n"


def html_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    if not rows:
        return "<p><em>No rows.</em></p>"
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(format_float(row.get(key, '')))}</td>"
            for key, _ in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def get_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Expected a Qwen2-style model with model.layers")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotate_half_matrix(dim: int, device: torch.device) -> torch.Tensor:
    matrix = torch.zeros(dim, dim, dtype=torch.float32, device=device)
    half = dim // 2
    idx = torch.arange(half, device=device)
    matrix[idx, half + idx] = -1.0
    matrix[half + idx, idx] = 1.0
    return matrix


def rope_matrix(cos_row: torch.Tensor, sin_row: torch.Tensor) -> torch.Tensor:
    cos_row = cos_row.detach().float()
    sin_row = sin_row.detach().float()
    p = rotate_half_matrix(cos_row.numel(), cos_row.device)
    return torch.diag(cos_row) + torch.diag(sin_row) @ p


def apply_rope(q_or_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return q_or_k * cos.unsqueeze(1) + rotate_half(q_or_k) * sin.unsqueeze(1)


def position_cos_sin(
    model: Any,
    hidden_states: torch.Tensor,
    kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    supplied = kwargs.get("position_embeddings")
    if isinstance(supplied, (tuple, list)) and len(supplied) >= 2:
        cos, sin = supplied[0], supplied[1]
    else:
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1], device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0)
        rotary = getattr(model.model, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError("model.model.rotary_emb was not found")
        try:
            cos, sin = rotary(hidden_states, position_ids)
        except TypeError:
            cos, sin = rotary(position_ids)
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    return cos.to(hidden_states.device), sin.to(hidden_states.device)


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    t = scores.shape[-1]
    future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=scores.device), diagonal=1)
    masked = scores.masked_fill(future.unsqueeze(0), torch.finfo(scores.dtype).min)
    return torch.softmax(masked.float(), dim=-1).to(scores.dtype)


def module_bias(module: Any, start: Optional[int] = None, end: Optional[int] = None) -> torch.Tensor:
    bias = getattr(module, "bias", None)
    if bias is None:
        size = int(module.weight.shape[0]) if start is None else int(end - start)
        return torch.zeros(size, dtype=torch.float32, device=module.weight.device)
    out = bias.detach().float()
    if start is not None and end is not None:
        out = out[start:end]
    return out


@dataclass
class Capture:
    attention_input: torch.Tensor
    attention_kwargs: Dict[str, Any]
    attention_output: torch.Tensor
    mlp_input: torch.Tensor
    mlp_output: torch.Tensor
    logits: torch.Tensor


@torch.no_grad()
def capture_native(
    model: Any,
    encoded: Dict[str, torch.Tensor],
    attention_layer: int,
    mlp_layer: int,
) -> Capture:
    layers = get_layers(model)
    attn_module = layers[attention_layer].self_attn
    mlp_module = layers[mlp_layer].mlp
    cap: Dict[str, Any] = {}

    def attn_hook(mod: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any):
        cap["attention_input"] = input_tensor(args, kwargs).detach()
        cap["attention_kwargs"] = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in kwargs.items()
            if key in {"position_ids", "position_embeddings", "attention_mask"}
        }
        cap["attention_output"] = first_output_tensor(output).detach()

    def mlp_hook(mod: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any):
        cap["mlp_input"] = input_tensor(args, kwargs).detach()
        cap["mlp_output"] = output.detach()

    ha = register_forward_hook_kwargs(attn_module, attn_hook)
    hm = register_forward_hook_kwargs(mlp_module, mlp_hook)
    try:
        output = model(**encoded, use_cache=False)
    finally:
        ha.remove()
        hm.remove()

    required = {"attention_input", "attention_kwargs", "attention_output", "mlp_input", "mlp_output"}
    missing = required.difference(cap)
    if missing:
        raise RuntimeError(f"Capture failed; missing {sorted(missing)}")
    return Capture(
        attention_input=cap["attention_input"],
        attention_kwargs=cap["attention_kwargs"],
        attention_output=cap["attention_output"],
        mlp_input=cap["mlp_input"],
        mlp_output=cap["mlp_output"],
        logits=output.logits.detach(),
    )


@dataclass
class AttentionResult:
    native_head_output: torch.Tensor
    program_head_output: torch.Tensor
    qk_uniform_head_output: torch.Tensor
    qk_amplified_head_output: torch.Tensor
    vo_zero_head_output: torch.Tensor
    precomputed_rows: List[Dict[str, Any]]
    token_flow_rows: List[Dict[str, Any]]
    metrics: Dict[str, Any]


@torch.no_grad()
def build_attention_result(
    model: Any,
    tokenizer: Any,
    capture: Capture,
    token_ids: Sequence[int],
    layer_idx: int,
    head_idx: int,
    destination_token: int,
    target_id: int,
    top_k: int,
    qk_amplify: float,
) -> AttentionResult:
    layer = get_layers(model)[layer_idx]
    attn = layer.self_attn
    cfg = model.config
    # Keep two views of the captured input:
    #   * x_native uses the projection module dtype (fp16/bf16/fp32), so
    #     q_proj/k_proj/v_proj can be called without dtype mismatch.
    #   * x is float32 for stable explicit matrix-program arithmetic.
    x_captured = capture.attention_input.detach()
    projection_dtype = attn.q_proj.weight.dtype
    x_native = x_captured.to(device=attn.q_proj.weight.device, dtype=projection_dtype)
    x = x_captured.to(device=attn.q_proj.weight.device, dtype=torch.float32)
    batch, seq_len, hidden = x.shape
    if batch != 1:
        raise ValueError("This demo expects one prompt")

    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden // n_heads))
    if not 0 <= head_idx < n_heads:
        raise ValueError(f"Head {head_idx} outside [0,{n_heads - 1}]")
    kv_idx = head_idx // (n_heads // n_kv)
    scale = float(getattr(attn, "scaling", head_dim ** -0.5))

    q_all = attn.q_proj(x_native).view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    k_all = attn.k_proj(x_native).view(batch, seq_len, n_kv, head_dim).transpose(1, 2)
    v_all = attn.v_proj(x_native).view(batch, seq_len, n_kv, head_dim).transpose(1, 2)
    cos, sin = position_cos_sin(model, x_native, capture.attention_kwargs)
    q_rot = apply_rope(q_all, cos, sin)
    k_rot = apply_rope(k_all, cos, sin)
    qh = q_rot[:, head_idx].float()
    kh = k_rot[:, kv_idx].float()
    vh = v_all[:, kv_idx].float()

    native_scores = torch.bmm(qh, kh.transpose(1, 2)) * scale
    native_attention = causal_softmax(native_scores)
    wo = attn.o_proj.weight.detach().float()[:, head_idx * head_dim:(head_idx + 1) * head_dim]
    native_head = torch.matmul(torch.bmm(native_attention, vh), wo.T)

    q0, q1 = head_idx * head_dim, (head_idx + 1) * head_dim
    k0, k1 = kv_idx * head_dim, (kv_idx + 1) * head_dim
    wq = attn.q_proj.weight.detach().float()[q0:q1]
    wk = attn.k_proj.weight.detach().float()[k0:k1]
    wv = attn.v_proj.weight.detach().float()[k0:k1]
    bq = module_bias(attn.q_proj, q0, q1)
    bk = module_bias(attn.k_proj, k0, k1)
    bv = module_bias(attn.v_proj, k0, k1)
    wq_aug = torch.cat([wq, bq[:, None]], dim=1)
    wk_aug = torch.cat([wk, bk[:, None]], dim=1)
    wv_aug = torch.cat([wv, bv[:, None]], dim=1)
    c_vo = wo @ wv_aug
    x_aug = torch.cat([x, torch.ones(batch, seq_len, 1, device=x.device)], dim=-1)

    rope = [rope_matrix(cos[0, p], sin[0, p]) for p in range(seq_len)]
    r0 = rope[0]
    m_delta: Dict[int, torch.Tensor] = {}
    precomputed: List[Dict[str, Any]] = []
    for d in range(seq_len):
        matrix = wq_aug.T @ (rope[d].T @ r0) @ wk_aug * scale
        m_delta[d] = matrix
        precomputed.append({
            "component": "attention",
            "object": f"M_qk[{d}]",
            "meaning": f"QK routing at relative distance d={d}",
            "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
            "frobenius_norm": float(torch.linalg.norm(matrix).item()),
            "rank_upper_bound": head_dim,
            "input_dependent": False,
        })
    precomputed.append({
        "component": "attention",
        "object": "C_vo",
        "meaning": "VO payload/read-write map",
        "shape": f"{c_vo.shape[0]}x{c_vo.shape[1]}",
        "frobenius_norm": float(torch.linalg.norm(c_vo).item()),
        "rank_upper_bound": head_dim,
        "input_dependent": False,
    })

    program_scores = torch.zeros_like(native_scores)
    for i in range(seq_len):
        for j in range(i + 1):
            program_scores[0, i, j] = x_aug[0, i] @ m_delta[i - j] @ x_aug[0, j]
    program_attention = causal_softmax(program_scores)
    payload = torch.matmul(x_aug, c_vo.T)
    program_head = torch.bmm(program_attention, payload)
    qk_uniform_head = torch.bmm(causal_softmax(program_scores * 0.0), payload)
    qk_amplified_head = torch.bmm(causal_softmax(program_scores * float(qk_amplify)), payload)
    vo_zero_head = torch.zeros_like(program_head)

    # Expand the homogeneous affine-bilinear score into human-readable terms:
    #
    #   score(i,j) = c_delta
    #              + q_affine_delta(x_i)
    #              + k_affine_delta(x_j)
    #              + x_i.T @ B_delta @ x_j
    #
    # This information is mathematically present inside M_qk[d], but the split is
    # much easier to read in a token-level trace.  Likewise C_vo is split into
    # its content-dependent linear payload and constant bias payload.
    c_vo_linear = c_vo[:, :hidden]
    b_vo = c_vo[:, hidden]
    payload_content = torch.matmul(x, c_vo_linear.T)
    payload_bias = b_vo.view(1, 1, hidden).expand_as(payload_content)

    target_vec = model.lm_head.weight[int(target_id)].detach().float()
    rows: List[Dict[str, Any]] = []
    x_i = x[0, destination_token]
    for source in range(destination_token + 1):
        d = destination_token - source
        matrix = m_delta[d]
        b_delta = matrix[:hidden, :hidden]
        q_delta = matrix[:hidden, hidden]
        k_delta = matrix[hidden, :hidden]
        c_delta = matrix[hidden, hidden]
        x_j = x[0, source]

        constant_term = c_delta
        query_affine_term = torch.dot(x_i, q_delta)
        key_affine_term = torch.dot(k_delta, x_j)
        content_bilinear_term = x_i @ b_delta @ x_j
        reconstructed_score = (
            constant_term + query_affine_term + key_affine_term + content_bilinear_term
        )
        term_values = {
            "constant_route": float(constant_term.item()),
            "query_affine": float(query_affine_term.item()),
            "key_affine": float(key_affine_term.item()),
            "content_bilinear": float(content_bilinear_term.item()),
        }
        dominant_term = max(term_values, key=lambda name: abs(term_values[name]))

        attn_weight = program_attention[0, destination_token, source]
        content_contribution = attn_weight * payload_content[0, source]
        bias_contribution = attn_weight * payload_bias[0, source]
        contribution = content_contribution + bias_contribution
        rows.append({
            "destination_index": destination_token,
            "destination_token": token_text(tokenizer, int(token_ids[destination_token])),
            "source_index": source,
            "source_token": token_text(tokenizer, int(token_ids[source])),
            "relative_distance": d,
            "constant_route": term_values["constant_route"],
            "query_affine": term_values["query_affine"],
            "key_affine": term_values["key_affine"],
            "content_bilinear": term_values["content_bilinear"],
            "dominant_score_term": dominant_term,
            "score_from_four_terms": float(reconstructed_score.item()),
            "native_qk_score": float(native_scores[0, destination_token, source].item()),
            "program_qk_score": float(program_scores[0, destination_token, source].item()),
            "score_four_term_abs_error": float(
                abs(reconstructed_score.item() - program_scores[0, destination_token, source].item())
            ),
            "native_attention": float(native_attention[0, destination_token, source].item()),
            "program_attention": float(attn_weight.item()),
            "payload_content_norm": float(torch.linalg.norm(payload_content[0, source]).item()),
            "payload_bias_norm": float(torch.linalg.norm(payload_bias[0, source]).item()),
            "payload_norm": float(torch.linalg.norm(payload[0, source]).item()),
            "content_contribution_norm": float(torch.linalg.norm(content_contribution).item()),
            "bias_contribution_norm": float(torch.linalg.norm(bias_contribution).item()),
            "contribution_norm": float(torch.linalg.norm(contribution).item()),
            "direct_target_logit_proxy": float(torch.dot(contribution, target_vec).item()),
        })
    rows.sort(key=lambda r: float(r["contribution_norm"]), reverse=True)
    rows = rows[: min(top_k, len(rows))]

    native_payload = torch.matmul(vh, wo.T)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
    metrics = {
        "layer": layer_idx,
        "head": head_idx,
        "kv_head": kv_idx,
        "hidden_size": hidden,
        "head_dim": head_dim,
        "selected_destination_token": destination_token,
        "native_vs_program_qk_score_rel": rel_err(program_scores[0][causal_mask], native_scores[0][causal_mask]),
        "native_vs_program_attention_rel": rel_err(program_attention, native_attention),
        "native_vs_program_payload_rel": rel_err(payload, native_payload),
        "native_vs_program_head_output_rel": rel_err(program_head, native_head),
        "qk_amplify": float(qk_amplify),
        "captured_input_dtype": str(x_captured.dtype),
        "projection_weight_dtype": str(projection_dtype),
        "program_compute_dtype": str(x.dtype),
        "four_term_score_max_abs_error": max(
            (float(r["score_four_term_abs_error"]) for r in rows),
            default=0.0,
        ),
    }
    return AttentionResult(
        native_head_output=native_head,
        program_head_output=program_head,
        qk_uniform_head_output=qk_uniform_head,
        qk_amplified_head_output=qk_amplified_head,
        vo_zero_head_output=vo_zero_head,
        precomputed_rows=precomputed,
        token_flow_rows=rows,
        metrics=metrics,
    )


@dataclass
class MLPResult:
    native_output: torch.Tensor
    program_output: torch.Tensor
    selected_atom_vectors: Dict[int, torch.Tensor]
    precomputed_rows: List[Dict[str, Any]]
    atom_rows: List[Dict[str, Any]]
    metrics: Dict[str, Any]


@torch.no_grad()
def build_mlp_result(
    model: Any,
    capture: Capture,
    layer_idx: int,
    selected_token: int,
    target_id: int,
    top_k: int,
) -> MLPResult:
    mlp = get_layers(model)[layer_idx].mlp
    x = capture.mlp_input.detach().float()
    native = capture.mlp_output.detach().float()
    wg = mlp.gate_proj.weight.detach().float()
    wu = mlp.up_proj.weight.detach().float()
    wd = mlp.down_proj.weight.detach().float()
    bg = module_bias(mlp.gate_proj)
    bu = module_bias(mlp.up_proj)
    bd = module_bias(mlp.down_proj)

    gate_pre = torch.matmul(x, wg.T) + bg
    gate = F.silu(gate_pre)
    read = torch.matmul(x, wu.T) + bu
    coeff = gate * read
    program = torch.matmul(coeff, wd.T) + bd

    target_vec = model.lm_head.weight[int(target_id)].detach().float()
    token_coeff = coeff[0, selected_token]
    atom_vectors = token_coeff[:, None] * wd.T
    direct_proxy = torch.matmul(atom_vectors, target_vec)
    write_norms = torch.linalg.norm(wd.T, dim=1)
    atom_norms = torch.linalg.norm(atom_vectors, dim=1)
    indices = torch.topk(direct_proxy.abs(), k=min(top_k, atom_vectors.shape[0])).indices.tolist()

    selected: Dict[int, torch.Tensor] = {}
    rows: List[Dict[str, Any]] = []
    for neuron in indices:
        neuron = int(neuron)
        selected[neuron] = atom_vectors[neuron].detach()
        rows.append({
            "neuron": neuron,
            "gate_pre": float(gate_pre[0, selected_token, neuron].item()),
            "gate_after_silu": float(gate[0, selected_token, neuron].item()),
            "read": float(read[0, selected_token, neuron].item()),
            "coefficient_gate_x_read": float(token_coeff[neuron].item()),
            "write_norm": float(write_norms[neuron].item()),
            "atom_output_norm": float(atom_norms[neuron].item()),
            "direct_target_logit_proxy": float(direct_proxy[neuron].item()),
            "causal_target_logit_delta": None,
            "causal_top1_changed": None,
        })

    precomputed = [
        {"component": "mlp", "object": "Wg", "meaning": "Gate directions", "shape": f"{wg.shape[0]}x{wg.shape[1]}", "frobenius_norm": float(torch.linalg.norm(wg).item()), "rank_upper_bound": min(wg.shape), "input_dependent": False},
        {"component": "mlp", "object": "Wu", "meaning": "Read directions", "shape": f"{wu.shape[0]}x{wu.shape[1]}", "frobenius_norm": float(torch.linalg.norm(wu).item()), "rank_upper_bound": min(wu.shape), "input_dependent": False},
        {"component": "mlp", "object": "Wd", "meaning": "Write directions", "shape": f"{wd.shape[0]}x{wd.shape[1]}", "frobenius_norm": float(torch.linalg.norm(wd).item()), "rank_upper_bound": min(wd.shape), "input_dependent": False},
        {"component": "mlp", "object": "rank-1 atoms", "meaning": "atom_j = Wd[:,j] outer Wu[j,:], gated at runtime", "shape": f"{wd.shape[1]} atoms of {wd.shape[0]}x{wu.shape[1]}", "frobenius_norm": "", "rank_upper_bound": 1, "input_dependent": False},
    ]
    metrics = {
        "layer": layer_idx,
        "hidden_size": int(wd.shape[0]),
        "intermediate_size": int(wd.shape[1]),
        "selected_token": selected_token,
        "native_vs_program_mlp_output_rel": rel_err(program, native),
        "native_vs_program_mlp_output_max_abs": float((program - native).abs().max().item()),
    }
    return MLPResult(native, program, selected, precomputed, rows, metrics)


@dataclass
class InterventionMetrics:
    name: str
    description: str
    target_logit: float
    target_logit_delta: float
    logit_rel_percent: float
    kl_base_to_intervention: float
    top1_id: int
    top1_token: str
    top1_preserved: bool


def distribution_metrics(
    baseline_logits: torch.Tensor,
    changed_logits: torch.Tensor,
    tokenizer: Any,
    target_id: int,
    name: str,
    description: str,
) -> InterventionMetrics:
    base = baseline_logits.detach().float()[0, -1]
    changed = changed_logits.detach().float()[0, -1]
    base_probs = torch.softmax(base, dim=-1)
    changed_log_probs = torch.log_softmax(changed, dim=-1)
    kl = float(torch.sum(base_probs * (torch.log(base_probs.clamp_min(1e-12)) - changed_log_probs)).item())
    top1 = int(changed.argmax().item())
    base_top1 = int(base.argmax().item())
    return InterventionMetrics(
        name=name,
        description=description,
        target_logit=float(changed[target_id].item()),
        target_logit_delta=float(changed[target_id].item() - base[target_id].item()),
        logit_rel_percent=100.0 * rel_err(changed, base),
        kl_base_to_intervention=kl,
        top1_id=top1,
        top1_token=token_text(tokenizer, top1),
        top1_preserved=(top1 == base_top1),
    )


@torch.no_grad()
def run_with_output_delta(
    model: Any,
    encoded: Dict[str, torch.Tensor],
    module: Any,
    delta: torch.Tensor,
) -> torch.Tensor:
    def hook(mod: Any, args: Tuple[Any, ...], output: Any):
        first = first_output_tensor(output)
        patched = first + delta.to(device=first.device, dtype=first.dtype)
        return replace_first_output(output, patched)
    handle = module.register_forward_hook(hook)
    try:
        return model(**encoded, use_cache=False).logits.detach()
    finally:
        handle.remove()


def build_markdown(
    args: argparse.Namespace,
    tokens: Sequence[str],
    target_token: str,
    attention: AttentionResult,
    mlp: MLPResult,
    opaque_rows: Sequence[Dict[str, Any]],
    interventions: Sequence[Dict[str, Any]],
) -> str:
    opaque_cols = [("component", "component"), ("ordinary_forward", "ordinary forward exposes"), ("matrix_program", "matrix program exposes"), ("intervention_handle", "separate intervention handle")]
    pre_cols = [("component", "component"), ("object", "object"), ("meaning", "meaning"), ("shape", "shape"), ("frobenius_norm", "Frobenius norm"), ("rank_upper_bound", "rank <="), ("input_dependent", "input-dependent")]
    attn_cols = [("source_index", "src"), ("source_token", "source token"), ("relative_distance", "d"), ("constant_route", "c_delta"), ("query_affine", "q_affine"), ("key_affine", "k_affine"), ("content_bilinear", "content bilinear"), ("dominant_score_term", "dominant score term"), ("program_qk_score", "total score"), ("program_attention", "attention"), ("payload_content_norm", "VO content norm"), ("payload_bias_norm", "VO bias norm"), ("contribution_norm", "final contribution norm"), ("direct_target_logit_proxy", "direct target proxy")]
    mlp_cols = [("neuron", "atom"), ("gate_after_silu", "gate"), ("read", "read"), ("coefficient_gate_x_read", "gate*read"), ("write_norm", "write norm"), ("atom_output_norm", "atom output norm"), ("direct_target_logit_proxy", "direct target proxy"), ("causal_target_logit_delta", "causal delta target"), ("causal_top1_changed", "top-1 changed")]
    int_cols = [("name", "run"), ("description", "what changed"), ("target_logit", f"logit({target_token})"), ("target_logit_delta", "delta target"), ("logit_rel_percent", "full-logit rel %"), ("kl_base_to_intervention", "KL"), ("top1_token", "top-1"), ("top1_preserved", "top-1 preserved")]
    token_line = " ".join(f"[{i}] `{tok}`" for i, tok in enumerate(tokens))
    return f"""# Executable Matrix Program — Single-Component Demo

## Configuration

- Model: `{args.model}`
- Prompt: `{args.prompt}`
- Tokens: {token_line}
- Attention: layer `{args.attention_layer}`, head `{args.head}`
- MLP: layer `{args.mlp_layer}`
- Selected token: `{attention.metrics['selected_destination_token']}`
- Baseline next-token prediction: `{target_token}`

## What the ordinary forward pass hides vs. what the program exposes

{markdown_table(opaque_rows, opaque_cols)}

The replacement is **not a trained surrogate**. It is a weight-derived executable refactorization of the original component.

## Static objects precomputed from weights

{markdown_table(attention.precomputed_rows + mlp.precomputed_rows, pre_cols)}

## One attention head: explicit score trace and VO payload

Destination token: `{tokens[attention.metrics['selected_destination_token']]}`.

For every source token, the total QK score is shown as the exact sum
`c_delta + q_affine(x_i) + k_affine(x_j) + x_i.T B_delta x_j`.
The table then shows the attention weight and the split between the
content-dependent VO payload and the constant VO bias payload.

{markdown_table(attention.token_flow_rows, attn_cols)}

### Attention faithfulness

```json
{json.dumps(attention.metrics, indent=2)}
```

## One MLP layer: native gate/read/write atoms

Selected token: `{tokens[mlp.metrics['selected_token']]}`.

The direct target proxy is descriptive. The causal delta is measured with a full native forward pass after removing that exact atom.

{markdown_table(mlp.atom_rows, mlp_cols)}

### MLP faithfulness

```json
{json.dumps(mlp.metrics, indent=2)}
```

## Replacement and interventions inside the native forward pass

{markdown_table(interventions, int_cols)}

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
"""


def build_html(
    args: argparse.Namespace,
    tokens: Sequence[str],
    target_token: str,
    attention: AttentionResult,
    mlp: MLPResult,
    opaque_rows: Sequence[Dict[str, Any]],
    interventions: Sequence[Dict[str, Any]],
) -> str:
    opaque_cols = [("component", "component"), ("ordinary_forward", "ordinary forward exposes"), ("matrix_program", "matrix program exposes"), ("intervention_handle", "intervention handle")]
    pre_cols = [("component", "component"), ("object", "object"), ("meaning", "meaning"), ("shape", "shape"), ("frobenius_norm", "Frobenius norm"), ("rank_upper_bound", "rank <="), ("input_dependent", "input-dependent")]
    attn_cols = [("source_index", "src"), ("source_token", "source token"), ("relative_distance", "d"), ("constant_route", "c_delta"), ("query_affine", "q_affine"), ("key_affine", "k_affine"), ("content_bilinear", "content bilinear"), ("dominant_score_term", "dominant score term"), ("program_qk_score", "total score"), ("program_attention", "attention"), ("payload_content_norm", "VO content norm"), ("payload_bias_norm", "VO bias norm"), ("contribution_norm", "final contribution norm"), ("direct_target_logit_proxy", "direct target proxy")]
    mlp_cols = [("neuron", "atom"), ("gate_after_silu", "gate"), ("read", "read"), ("coefficient_gate_x_read", "gate*read"), ("write_norm", "write norm"), ("atom_output_norm", "atom output norm"), ("direct_target_logit_proxy", "direct target proxy"), ("causal_target_logit_delta", "causal delta target"), ("causal_top1_changed", "top-1 changed")]
    int_cols = [("name", "run"), ("description", "what changed"), ("target_logit", f"logit({target_token})"), ("target_logit_delta", "delta target"), ("logit_rel_percent", "full-logit rel %"), ("kl_base_to_intervention", "KL"), ("top1_token", "top-1"), ("top1_preserved", "top-1 preserved")]
    token_line = " ".join(f"<span class='token'>[{i}] {html.escape(tok)}</span>" for i, tok in enumerate(tokens))
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>Executable Matrix Program Demo</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1400px;margin:30px auto;padding:0 22px;color:#17212b;line-height:1.45}}
h1,h2{{color:#0f4c5c}} .card{{border:1px solid #d7e1e5;border-radius:12px;padding:18px;background:#fbfdfe}}
table{{border-collapse:collapse;width:100%;margin:12px 0 25px;font-size:13px}} th{{background:#0f766e;color:white;text-align:left;padding:8px}}
td{{border:1px solid #d8e0e3;padding:7px;vertical-align:top}} tr:nth-child(even){{background:#f4f8f8}}
pre{{background:#f1f5f5;border-radius:7px;padding:14px;overflow:auto}} .token{{display:inline-block;background:#e6f4f1;padding:4px 7px;border-radius:6px;margin:2px}}
</style></head><body>
<h1>Executable Matrix Program — Single-Component Demo</h1>
<div class='card'><b>Model:</b> {html.escape(args.model)}<br><b>Prompt:</b> {html.escape(args.prompt)}<br><b>Tokens:</b> {token_line}<br><b>Attention:</b> L{args.attention_layer}H{args.head}<br><b>MLP:</b> L{args.mlp_layer}<br><b>Baseline next token:</b> {html.escape(target_token)}</div>
<h2>What was opaque, and what is explicit now?</h2>{html_table(opaque_rows, opaque_cols)}
<p>No surrogate was trained. The program is derived from pretrained weights and reinserted into the native forward pass.</p>
<h2>Static objects precomputed from weights</h2>{html_table(attention.precomputed_rows + mlp.precomputed_rows, pre_cols)}
<h2>Attention: four-term score trace and VO payload</h2><p>Each total score is exactly c_delta + q_affine(x_i) + k_affine(x_j) + x_i.T B_delta x_j. The payload columns split content-dependent VO output from the constant VO bias.</p>{html_table(attention.token_flow_rows, attn_cols)}<pre>{html.escape(json.dumps(attention.metrics, indent=2))}</pre>
<h2>MLP gate/read/write atoms</h2>{html_table(mlp.atom_rows, mlp_cols)}<pre>{html.escape(json.dumps(mlp.metrics, indent=2))}</pre>
<h2>Native-forward replacement and interventions</h2>{html_table(interventions, int_cols)}
<h2>Conceptual summary</h2><pre>Attention: token states -> M_qk[d] routing -> C_vo payload -> contributions -> head output
MLP: sum_j gate_j(x) * read_j(x) * write_j</pre>
</body></html>"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Explain one Qwen attention head and one MLP layer with executable matrix programs.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--attention-layer", type=int, default=20)
    ap.add_argument("--head", type=int, default=5)
    ap.add_argument("--mlp-layer", type=int, default=20)
    ap.add_argument("--token", type=int, default=-1, help="Token to explain; negative indices count from the end")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--causal-atoms", type=int, default=3)
    ap.add_argument("--qk-amplify", type=float, default=1.5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/matrix_program_single_demo")
    return ap.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Install torch and transformers before running") from exc

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU")
        device = "cpu"
    dtype = get_dtype(args.dtype)
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    print(f"Loading {args.model} on {device} ({dtype})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "attn_implementation": "eager", "dtype": dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except TypeError:
        model_kwargs.pop("dtype", None)
        model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model = model.to(device).eval()

    layers = get_layers(model)
    if not 0 <= args.attention_layer < len(layers):
        raise ValueError("Bad attention layer")
    if not 0 <= args.mlp_layer < len(layers):
        raise ValueError("Bad MLP layer")

    encoded = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    token_ids = encoded["input_ids"][0].detach().cpu().tolist()
    tokens = [token_text(tokenizer, tid) for tid in token_ids]
    token_index = normalize_token_index(args.token, len(token_ids))

    print("Capturing native forward...", flush=True)
    capture = capture_native(model, encoded, args.attention_layer, args.mlp_layer)
    baseline_logits = capture.logits
    target_id = int(baseline_logits[0, -1].argmax().item())
    target_token = token_text(tokenizer, target_id)

    print("Building attention QK/VO program...", flush=True)
    attention = build_attention_result(
        model, tokenizer, capture, token_ids, args.attention_layer, args.head,
        token_index, target_id, args.top_k, args.qk_amplify,
    )
    print("Building MLP gate/read/write program...", flush=True)
    mlp = build_mlp_result(model, capture, args.mlp_layer, token_index, target_id, args.top_k)

    opaque_rows = [
        {"component": f"Attention L{args.attention_layer}H{args.head}", "ordinary_forward": "One mixed head/module output tensor", "matrix_program": "Per-token QK scores, routing weights, VO payloads, token-to-token contributions", "intervention_handle": "Change QK routing separately from VO payload"},
        {"component": f"MLP L{args.mlp_layer}", "ordinary_forward": "One nonlinear output tensor for the whole MLP", "matrix_program": "Native atom gate, read, coefficient, write direction, and contribution", "intervention_handle": "Remove or scale individual gate/read/write atoms"},
    ]

    interventions: List[Dict[str, Any]] = []
    interventions.append(asdict(distribution_metrics(baseline_logits, baseline_logits, tokenizer, target_id, "baseline", "Original unmodified model")))

    attn_module = layers[args.attention_layer].self_attn
    attention_runs = [
        ("attention_exact_program", "Replace selected native head with explicit QK/VO program", attention.program_head_output - attention.native_head_output),
        ("attention_qk_uniform", "Change only QK routing to uniform causal routing; keep VO payload", attention.qk_uniform_head_output - attention.native_head_output),
        ("attention_qk_amplified", f"Scale only QK routing scores by {args.qk_amplify}; keep VO payload", attention.qk_amplified_head_output - attention.native_head_output),
        ("attention_vo_zero", "Zero only the selected head's VO payload/output", attention.vo_zero_head_output - attention.native_head_output),
    ]
    for name, desc, delta in attention_runs:
        changed = run_with_output_delta(model, encoded, attn_module, delta)
        interventions.append(asdict(distribution_metrics(baseline_logits, changed, tokenizer, target_id, name, desc)))

    mlp_module = layers[args.mlp_layer].mlp
    changed = run_with_output_delta(model, encoded, mlp_module, mlp.program_output - mlp.native_output)
    interventions.append(asdict(distribution_metrics(baseline_logits, changed, tokenizer, target_id, "mlp_exact_program", "Replace selected native MLP with explicit gate/read/write program")))

    atom_rows = {int(row["neuron"]): row for row in mlp.atom_rows}
    for neuron in list(mlp.selected_atom_vectors)[: max(0, args.causal_atoms)]:
        delta = torch.zeros_like(mlp.native_output)
        delta[0, token_index] = -mlp.selected_atom_vectors[neuron]
        changed = run_with_output_delta(model, encoded, mlp_module, delta)
        metric = distribution_metrics(baseline_logits, changed, tokenizer, target_id, f"mlp_atom_zero_{neuron}", f"Remove native MLP atom {neuron} only at token position {token_index}")
        interventions.append(asdict(metric))
        atom_rows[neuron]["causal_target_logit_delta"] = metric.target_logit_delta
        atom_rows[neuron]["causal_top1_changed"] = not metric.top1_preserved

    precomputed = attention.precomputed_rows + mlp.precomputed_rows
    write_csv(out_dir / "opaque_vs_explicit.csv", opaque_rows)
    write_csv(out_dir / "precomputed_objects.csv", precomputed)
    write_csv(out_dir / "attention_token_flow.csv", attention.token_flow_rows)
    write_csv(out_dir / "attention_score_terms.csv", attention.token_flow_rows)
    write_csv(out_dir / "mlp_atoms.csv", mlp.atom_rows)
    write_csv(out_dir / "interventions.csv", interventions)

    report = {
        "configuration": vars(args),
        "tokens": [{"index": i, "token_id": tid, "token": tok} for i, (tid, tok) in enumerate(zip(token_ids, tokens))],
        "baseline": {"target_id": target_id, "target_token": target_token},
        "opaque_vs_explicit": opaque_rows,
        "precomputed_objects": precomputed,
        "attention": {
            "exact_score_formula": "score_ij = c_delta + q_affine_delta(x_i) + k_affine_delta(x_j) + x_i.T @ B_delta @ x_j",
            "exact_payload_formula": "payload_j = C_vo_linear @ x_j + b_vo",
            "metrics": attention.metrics,
            "token_flow": attention.token_flow_rows,
        },
        "mlp": {"metrics": mlp.metrics, "atoms": mlp.atom_rows},
        "interventions": interventions,
    }
    write_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(build_markdown(args, tokens, target_token, attention, mlp, opaque_rows, interventions), encoding="utf-8")
    (out_dir / "report.html").write_text(build_html(args, tokens, target_token, attention, mlp, opaque_rows, interventions), encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXECUTABLE MATRIX PROGRAM DEMO")
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    print("Tokens: " + " ".join(f"[{i}]={tok!r}" for i, tok in enumerate(tokens)))
    print(f"Baseline next token: {target_token!r}")
    print(f"Attention L{args.attention_layer}H{args.head} program/native rel error: {attention.metrics['native_vs_program_head_output_rel']:.3e}")
    print(f"MLP L{args.mlp_layer} program/native rel error: {mlp.metrics['native_vs_program_mlp_output_rel']:.3e}")
    print("\nTop attention token flows with homogeneous matrix unpacked:")
    print("  score = constant + query_affine + key_affine + content_bilinear")
    for row in attention.token_flow_rows:
        print(
            f"  {tokens[token_index]!r} <- {row['source_token']!r} (d={row['relative_distance']}):\n"
            f"      constant_route   = {row['constant_route']:+.6f}\n"
            f"      query_affine     = {row['query_affine']:+.6f}\n"
            f"      key_affine       = {row['key_affine']:+.6f}\n"
            f"      content_bilinear = {row['content_bilinear']:+.6f}\n"
            f"      --------------------------------\n"
            f"      total_score      = {row['program_qk_score']:+.6f} "
            f"(dominant={row['dominant_score_term']})\n"
            f"      attention        = {row['program_attention']:.6f}\n"
            f"      VO_content_norm  = {row['payload_content_norm']:.6f}\n"
            f"      VO_bias_norm     = {row['payload_bias_norm']:.6f}\n"
            f"      contribution_norm= {row['contribution_norm']:.6f}\n"
        )
    print("\nTop MLP atoms:")
    for row in mlp.atom_rows:
        causal = row.get("causal_target_logit_delta")
        causal_text = "not run" if causal is None else f"{causal:+.4f}"
        print(f"  atom {row['neuron']}: gate={row['gate_after_silu']:+.4f}, read={row['read']:+.4f}, coeff={row['coefficient_gate_x_read']:+.4f}, causal_delta={causal_text}")
    print("\nNative-forward interventions:")
    for row in interventions:
        print(f"  {row['name']}: rel={row['logit_rel_percent']:.4f}%, delta_target={row['target_logit_delta']:+.4f}, top1={row['top1_token']!r}, preserved={row['top1_preserved']}")

    trace_lines: List[str] = []
    trace_lines.append("EXECUTABLE MATRIX PROGRAM — READABLE TRACE\n")
    trace_lines.append(f"Prompt: {args.prompt}\n")
    trace_lines.append("Tokens: " + " ".join(f"[{i}]={tok!r}" for i, tok in enumerate(tokens)) + "\n")
    trace_lines.append(f"Baseline next token: {target_token!r}\n\n")
    trace_lines.append(
        f"ATTENTION L{args.attention_layer}H{args.head} destination={tokens[token_index]!r}\n"
    )
    trace_lines.append(
        "score(i,j) = constant_route + query_affine + key_affine + content_bilinear\n\n"
    )
    for row in attention.token_flow_rows:
        trace_lines.extend([
            f"{tokens[token_index]!r} <- {row['source_token']!r} (d={row['relative_distance']})\n",
            f"  constant_route   {row['constant_route']:+.8f}\n",
            f"  query_affine     {row['query_affine']:+.8f}\n",
            f"  key_affine       {row['key_affine']:+.8f}\n",
            f"  content_bilinear {row['content_bilinear']:+.8f}\n",
            f"  total_score      {row['program_qk_score']:+.8f}\n",
            f"  dominant_term    {row['dominant_score_term']}\n",
            f"  attention        {row['program_attention']:.8f}\n",
            f"  VO_content_norm  {row['payload_content_norm']:.8f}\n",
            f"  VO_bias_norm     {row['payload_bias_norm']:.8f}\n",
            f"  contribution     {row['contribution_norm']:.8f}\n\n",
        ])
    trace_lines.append(f"MLP L{args.mlp_layer} token={tokens[token_index]!r}\n")
    trace_lines.append("contribution_j = SiLU(Wg_j x) * (Wu_j x) * Wd[:,j]\n\n")
    for row in mlp.atom_rows:
        causal = row.get("causal_target_logit_delta")
        causal_text = "not run" if causal is None else f"{causal:+.8f}"
        trace_lines.extend([
            f"atom {row['neuron']}\n",
            f"  gate              {row['gate_after_silu']:+.8f}\n",
            f"  read              {row['read']:+.8f}\n",
            f"  coefficient       {row['coefficient_gate_x_read']:+.8f}\n",
            f"  write_norm        {row['write_norm']:.8f}\n",
            f"  atom_output_norm  {row['atom_output_norm']:.8f}\n",
            f"  causal_delta      {causal_text}\n\n",
        ])
    trace_lines.append("NATIVE-FORWARD REPLACEMENT / INTERVENTIONS\n")
    for row in interventions:
        trace_lines.append(
            f"{row['name']}: full_logit_rel={row['logit_rel_percent']:.8f}% "
            f"delta_target={row['target_logit_delta']:+.8f} "
            f"top1={row['top1_token']!r} preserved={row['top1_preserved']}\n"
        )
    (out_dir / "trace.txt").write_text("".join(trace_lines), encoding="utf-8")

    print(f"\nSaved plain-text trace: {out_dir / 'trace.txt'}")
    print(f"Saved HTML report: {out_dir / 'report.html'}")
    print(f"Saved Markdown report: {out_dir / 'report.md'}")
    print(f"Saved CSV tables: {out_dir}")


if __name__ == "__main__":
    main()
