#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_matrix_program_attention_mlp_v4_1_github_fixed.py

Purpose
-------
Full non-movement matrix-program extraction for Qwen2-style pretrained transformers:
attention affine QK/VO targets + exact gated-MLP rank-1 atom programs.

Real circuit-matrix extraction for one/more Qwen2 attention heads, using the user's
matrix-program principle:

  code path -> exact affine/bilinear circuit targets -> basis/search diagnostics -> functional checks

v4 final-article extension:
  * global replacement of all heads/all MLP/all backbone
  * native identity/recompute controls
  * v4.1: independent seeds for each held-out random atom control
  * full all-head/all-MLP causal atlas with downstream propagation
  * held-out atom selection/evaluation split
  * resumable per-experiment reports

v3 patching fix:
  * supports recent Transformers/PyTorch calls where attention hidden_states is passed only in kwargs
  * uses robust args/kwargs input capture for both attention and MLP hooks

v2 affine fixes retained:
  * q/k/v projection BIASES are included via homogeneous coordinates x_aug=[x,1]
  * QK is exported as an affine-bilinear target:
        score_ij = x_aug_i^T M_qk_aug[delta] x_aug_j
  * VO is exported as an affine read/write target:
        payload_j = x_aug_j @ C_vo_aug.T
        Y_i = sum_j A[i,j] payload_j

This is NOT a generic probe/PCA script. The exact circuit targets are the main result.
Basis diagnostics are optional/secondary and are evaluated functionally in the same
style as qwen_program_decompiler_v6: A_rel/KL/top1/Z_rel/Y_rel, not by pretty plots.

Targets
-------
Normalized input x = RMSNorm(raw):

  q_pre = x_aug @ Wq_aug.T,  Wq_aug=[Wq | bq]
  k_pre = x_aug @ Wk_aug.T,  Wk_aug=[Wk | bk]
  v     = x_aug @ Wv_aug.T,  Wv_aug=[Wv | bv]

  q_i = q_pre_i @ R_i.T
  k_j = k_pre_j @ R_j.T

  M_qk_aug[d] = Wq_aug.T @ R_i.T @ R_j @ Wk_aug / sqrt(head_dim), d=i-j
  score_ij    = x_aug_i @ M_qk_aug[d] @ x_aug_j.T

  C_vo_aug = Wo_head @ Wv_aug
  Y_i      = sum_j A[i,j] * (x_aug_j @ C_vo_aug.T)

Raw input block form:

  x_norm_j = D_j @ x_raw_j, D_j=diag(gamma/rms(x_raw_j))
  x_aug_j  = [D_j @ x_raw_j, 1]
  block(i,j) = A[i,j] * [C_vo_linear @ D_j, b_vo]

Outputs
-------
  summary.json / summary.md
  per_prompt_checks.csv
  basis_pca_functional.csv
  token_flow_examples.json
  circuit_targets.pt optional
  mlp_matrix_program/summary.json, mlp_faithfulness_table.csv, mlp_top_atom_examples.json
  patching/attention_exact_replacement.csv, attention_interventions.csv
  patching/mlp_exact_replacement.csv, mlp_interventions.csv, article_results.md

Example
-------
  python qwen_circuit_matrix_targets_v2_affine_basis.py \
    --base-script ./qwen_program_decompiler_v6_scorehybrid.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --dtype fp16 \
    --attn-implementation eager --heads 2:1 \
    --prompt-suites all --prompts-per-suite 8 --same-text-repeats 2 \
    --max-length 192 --max-delta 64 --svd-ranks 4,8,16,32,64,128 \
    --basis-ranks 4,8,16,32,64 --fit-learned-qk-basis --learned-steps 120 \
    --save-tensors --out-dir ./qwen_circuit_targets_v2_L2H1_fast

Self-test:
  python qwen_circuit_matrix_targets_v2_affine_basis.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


# ---------------- utilities ----------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.detach().float()
    b = b.detach().float()
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(eps))


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def json_sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return str(obj) if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        if obj.numel() <= 16:
            return obj.detach().cpu().tolist()
        return {"__tensor__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if np is not None:
        if isinstance(obj, np.generic):
            return json_sanitize(obj.item())
        if isinstance(obj, np.ndarray):
            if obj.size <= 16:
                return obj.tolist()
            return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]
    return str(obj)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(json_sanitize(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json_sanitize(r.get(k, "")) for k in keys})


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def parse_heads(s: str, n_layers: int, n_heads: int) -> List[Tuple[int, int]]:
    if s.strip().lower() == "all":
        return [(l, h) for l in range(n_layers) for h in range(n_heads)]
    out: List[Tuple[int, int]] = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad head spec {part!r}; use L:H")
        a, b = part.split(":", 1)
        l, h = int(a), int(b)
        if not (0 <= l < n_layers and 0 <= h < n_heads):
            raise ValueError(f"head out of range {l}:{h}; model has layers={n_layers}, heads={n_heads}")
        out.append((l, h))
    return list(dict.fromkeys(out))


def get_dtype(name: str):
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(name)


def load_base_module(path: str):
    p = Path(path)
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("qwen_decomp_base", str(p))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen_decomp_base"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ---------------- RoPE explicit matrices ----------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_matrix(D: int, device=None, dtype=torch.float32) -> torch.Tensor:
    device = device or torch.device("cpu")
    P = torch.zeros(D, D, device=device, dtype=dtype)
    h = D // 2
    for i in range(h):
        P[i, h + i] = -1.0
        P[h + i, i] = 1.0
    return P


def rope_col_matrix(cos_row: torch.Tensor, sin_row: torch.Tensor) -> torch.Tensor:
    """Column-space matrix R: q_rot_col = R @ q_col. Row form: q_rot_row = q_row @ R.T."""
    cos_row = cos_row.detach().float()
    sin_row = sin_row.detach().float()
    D = int(cos_row.numel())
    P = rotate_half_matrix(D, cos_row.device, cos_row.dtype)
    return torch.diag(cos_row) + torch.diag(sin_row) @ P


def apply_rope_rows(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return q * cos + rotate_half(q) * sin


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1)


# ---------------- collected data ----------------

@dataclass
class CircuitSeq:
    prompt_id: int
    suite: str
    text: str
    token_ids: List[int]
    tokens: List[str]
    Xraw: torch.Tensor       # [T,H] residual before RMSNorm
    Xn: torch.Tensor         # [T,H] post RMSNorm
    Xaug: torch.Tensor       # [T,H+1] [Xn,1]
    rms_scale: torch.Tensor  # [T,H], Xn ~= Xraw*rms_scale
    Q_pre: torch.Tensor      # [T,D]
    K_pre: torch.Tensor      # [T,D]
    Q: torch.Tensor          # [T,D] after RoPE
    K: torch.Tensor          # [T,D] after RoPE
    V: torch.Tensor          # [T,D]
    A: torch.Tensor          # [T,T]
    Y: torch.Tensor          # [T,H]
    cos: torch.Tensor        # [T,D]
    sin: torch.Tensor        # [T,D]
    rms_rec_err: float


def get_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find decoder layers on model")


def compute_position_embeddings(model: Any, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    rotary = getattr(model.model, "rotary_emb", None) if hasattr(model, "model") else None
    if rotary is None:
        raise RuntimeError("model.model.rotary_emb not found")
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return rotary(position_ids)


def build_prompts(base_mod: Any, suites: str, prompts_per_suite: int, same_text_repeats: int) -> List[Dict[str, str]]:
    if base_mod is not None and hasattr(base_mod, "build_prompts"):
        return base_mod.build_prompts(suites, prompts_per_suite, same_text_repeats)
    base = [
        {"suite": "text", "text": "Explain why rivers are important for cities."},
        {"suite": "code", "text": "Write a Python function that sums a list."},
        {"suite": "math", "text": "Solve 3x + 5 = 20."},
        {"suite": "symbols", "text": "JSON: {\"name\": \"Alice\", \"score\": 42}"},
    ]
    rows = []
    for _ in range(max(1, prompts_per_suite)):
        rows.extend(base)
    for _ in range(same_text_repeats):
        rows.append({"suite": "same_repeat", "text": "The same calibration sentence repeated."})
    return rows


@torch.no_grad()
def collect_circuit_data(model: Any, tokenizer: Any, prompts: List[Dict[str, str]],
                         layer_idx: int, head_idx: int, max_length: int, device: str) -> Tuple[List[CircuitSeq], Dict[str, Any]]:
    layers = get_layers(model)
    layer = layers[layer_idx]
    attn = layer.self_attn
    cfg = model.config
    H = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    D = int(getattr(cfg, "head_dim", H // n_heads))
    kv_groups = n_heads // n_kv
    kv_idx = int(head_idx) // kv_groups

    gamma = layer.input_layernorm.weight.detach().float().to(device)
    eps = float(getattr(layer.input_layernorm, "variance_epsilon", getattr(layer.input_layernorm, "eps", 1e-6)))

    o_w = attn.o_proj.weight.detach().float()[:, head_idx * D:(head_idx + 1) * D].to(device)
    out: List[CircuitSeq] = []
    for pi, pr in enumerate(prompts):
        enc = tokenizer(pr["text"], return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        T = int(input_ids.shape[1])
        if T < 2:
            continue
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
        Xraw = outputs.hidden_states[layer_idx].detach()  # [1,T,H]
        Xn = layer.input_layernorm(Xraw).detach()
        rms = torch.sqrt(Xraw.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        scale = gamma.view(1, 1, H) / rms
        Xn_recon = Xraw.float() * scale
        rms_err = rel_err(Xn_recon[0], Xn[0].float())

        q = attn.q_proj(Xn).view(1, T, n_heads, D).transpose(1, 2).contiguous()
        k = attn.k_proj(Xn).view(1, T, n_kv, D).transpose(1, 2).contiguous()
        v = attn.v_proj(Xn).view(1, T, n_kv, D).transpose(1, 2).contiguous()
        pos = torch.arange(T, device=device).unsqueeze(0)
        cos, sin = compute_position_embeddings(model, Xn, pos)
        cos2 = cos[0] if cos.dim() == 3 else cos
        sin2 = sin[0] if sin.dim() == 3 else sin
        q_rot = apply_rope_rows(q, cos2.unsqueeze(0).unsqueeze(0), sin2.unsqueeze(0).unsqueeze(0))
        k_rot = apply_rope_rows(k, cos2.unsqueeze(0).unsqueeze(0), sin2.unsqueeze(0).unsqueeze(0))
        Qh = q_rot[0, head_idx].float()
        Kh = k_rot[0, kv_idx].float()
        Vh = v[0, kv_idx].float()
        scores = (Qh @ Kh.T) / math.sqrt(D)
        A = causal_softmax(scores.float())
        Y = (A @ Vh) @ o_w.T
        tok_ids = input_ids[0].detach().cpu().tolist()
        try:
            toks = tokenizer.convert_ids_to_tokens(tok_ids)
        except Exception:
            toks = [str(x) for x in tok_ids]
        Xn_cpu = Xn[0].float().cpu()
        Xaug_cpu = torch.cat([Xn_cpu, torch.ones(T, 1)], dim=1)
        out.append(CircuitSeq(
            prompt_id=pi, suite=str(pr.get("suite", "?")), text=pr["text"], token_ids=tok_ids, tokens=toks,
            Xraw=Xraw[0].float().cpu(), Xn=Xn_cpu, Xaug=Xaug_cpu, rms_scale=scale[0].float().cpu(),
            Q_pre=q[0, head_idx].float().cpu(), K_pre=k[0, kv_idx].float().cpu(),
            Q=Qh.cpu(), K=Kh.cpu(), V=Vh.cpu(), A=A.cpu(), Y=Y.cpu(), cos=cos2.float().cpu(), sin=sin2.float().cpu(),
            rms_rec_err=float(rms_err),
        ))
        del outputs
        if torch.cuda.is_available() and (pi + 1) % 16 == 0:
            torch.cuda.empty_cache()
    meta = {"hidden_size": H, "num_heads": n_heads, "num_kv": n_kv, "head_dim": D, "kv_idx": kv_idx, "rms_eps": eps}
    return out, meta


# ---------------- target construction ----------------

def _slice_bias(module: Any, start: int, end: int) -> torch.Tensor:
    b = getattr(module, "bias", None)
    if b is None:
        return torch.zeros(end - start, dtype=torch.float32)
    return b.detach().float()[start:end].cpu()


def build_weight_slices(model: Any, layer_idx: int, head_idx: int, meta: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    H = int(meta["hidden_size"])
    D = int(meta["head_dim"])
    kv_idx = int(meta["kv_idx"])
    Wq = attn.q_proj.weight.detach().float()[head_idx * D:(head_idx + 1) * D, :].cpu()  # [D,H]
    Wk = attn.k_proj.weight.detach().float()[kv_idx * D:(kv_idx + 1) * D, :].cpu()      # [D,H]
    Wv = attn.v_proj.weight.detach().float()[kv_idx * D:(kv_idx + 1) * D, :].cpu()      # [D,H]
    Wo = attn.o_proj.weight.detach().float()[:, head_idx * D:(head_idx + 1) * D].cpu() # [H,D]
    bq = _slice_bias(attn.q_proj, head_idx * D, (head_idx + 1) * D)
    bk = _slice_bias(attn.k_proj, kv_idx * D, (kv_idx + 1) * D)
    bv = _slice_bias(attn.v_proj, kv_idx * D, (kv_idx + 1) * D)
    # o_proj bias is full attention-output bias, not a per-head contribution. Keep for metadata only.
    bo = getattr(attn.o_proj, "bias", None)
    bo = torch.zeros(H, dtype=torch.float32) if bo is None else bo.detach().float().cpu()
    Wq_aug = torch.cat([Wq, bq[:, None]], dim=1)  # [D,H+1]
    Wk_aug = torch.cat([Wk, bk[:, None]], dim=1)
    Wv_aug = torch.cat([Wv, bv[:, None]], dim=1)
    return {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "bq": bq, "bk": bk, "bv": bv, "bo_full": bo,
            "Wq_aug": Wq_aug, "Wk_aug": Wk_aug, "Wv_aug": Wv_aug}


def build_rope_by_pos(seqs: List[CircuitSeq], max_pos: int) -> Dict[int, torch.Tensor]:
    pos_cos: Dict[int, torch.Tensor] = {}
    pos_sin: Dict[int, torch.Tensor] = {}
    for s in seqs:
        T = int(s.cos.shape[0])
        for p in range(min(T, max_pos + 1)):
            if p not in pos_cos:
                pos_cos[p] = s.cos[p]
                pos_sin[p] = s.sin[p]
    return {p: rope_col_matrix(pos_cos[p], pos_sin[p]).cpu() for p in sorted(pos_cos)}


def qk_delta_matrices_affine(weights: Dict[str, torch.Tensor], Rpos: Dict[int, torch.Tensor], max_delta: int, head_dim: int) -> Dict[int, torch.Tensor]:
    Wq, Wk = weights["Wq_aug"], weights["Wk_aug"]
    out: Dict[int, torch.Tensor] = {}
    if 0 not in Rpos:
        return out
    R0 = Rpos[0]
    for d in range(max_delta + 1):
        if d not in Rpos:
            continue
        Rrel = Rpos[d].T @ R0
        out[d] = (Wq.T @ Rrel @ Wk) / math.sqrt(head_dim)  # [H+1,H+1]
    return out


def qk_delta_matrices_linear(weights: Dict[str, torch.Tensor], Rpos: Dict[int, torch.Tensor], max_delta: int, head_dim: int) -> Dict[int, torch.Tensor]:
    Wq, Wk = weights["Wq"], weights["Wk"]
    out: Dict[int, torch.Tensor] = {}
    if 0 not in Rpos:
        return out
    R0 = Rpos[0]
    for d in range(max_delta + 1):
        if d not in Rpos:
            continue
        Rrel = Rpos[d].T @ R0
        out[d] = (Wq.T @ Rrel @ Wk) / math.sqrt(head_dim)  # [H,H]
    return out


def rope_delta_invariance(seqs: List[CircuitSeq], Rpos: Dict[int, torch.Tensor], max_delta: int, sample_pairs: int = 256) -> Dict[str, Any]:
    rows = []
    rng = random.Random(123)
    for d in range(max_delta + 1):
        mats = []
        for s in seqs:
            T = int(s.cos.shape[0])
            if T <= d:
                continue
            pairs = [(i, i - d) for i in range(d, T)]
            if len(pairs) > sample_pairs:
                pairs = rng.sample(pairs, sample_pairs)
            for i, j in pairs:
                if i in Rpos and j in Rpos:
                    mats.append(Rpos[i].T @ Rpos[j])
        if len(mats) <= 1:
            continue
        base = mats[0]
        errs = [rel_err(m, base) for m in mats[1:]]
        rows.append({"delta": d, "n": len(mats), "mean_rel_to_first": safe_mean(errs), "max_rel_to_first": max(errs) if errs else 0.0})
    if not rows:
        return {"enabled": False, "reason": "no pairs"}
    return {"enabled": True, "max_delta_checked": max(r["delta"] for r in rows), "worst_max_rel": max(float(r["max_rel_to_first"]) for r in rows), "rows": rows[:16]}


# ---------------- metrics and basis ----------------

def svd_rank_table(M: torch.Tensor, ranks: List[int]) -> List[Dict[str, Any]]:
    M = M.detach().float().cpu()
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    except Exception as e:
        return [{"error": str(e)}]
    rows = []
    total_e = float((S * S).sum().item())
    for r in ranks:
        rr = min(int(r), int(S.numel()))
        Mr = (U[:, :rr] * S[:rr]) @ Vh[:rr]
        rows.append({"rank": rr, "rel_err": rel_err(Mr, M), "energy": float((S[:rr] * S[:rr]).sum().item() / max(1e-12, total_e))})
    return rows


def matrix_shape_stats(M: torch.Tensor) -> Dict[str, Any]:
    M = M.detach().float().cpu()
    norm = torch.linalg.norm(M).clamp_min(1e-12)
    is_square = M.ndim == 2 and M.shape[0] == M.shape[1]
    diag = torch.diag(torch.diag(M)) if is_square else torch.zeros_like(M)
    sym_err = rel_err((M + M.T) * 0.5, M) if is_square else None
    skew_err = rel_err((M - M.T) * 0.5, M) if is_square else None
    return {
        "shape": list(M.shape), "norm": float(norm.item()), "mean_abs": float(M.abs().mean().item()), "max_abs": float(M.abs().max().item()),
        "diag_energy_frac": float((torch.linalg.norm(diag) / norm).item()) if is_square else None,
        "sym_rel_to_M": sym_err, "skew_rel_to_M": skew_err,
    }


def _pca_basis(M: torch.Tensor, rank: int) -> torch.Tensor:
    # rows=samples, columns=features; no centering, same reason as v6 QK geometry.
    M = M.detach().float()
    if M.shape[0] == 0:
        return torch.empty(M.shape[1], 0)
    _, _, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, Vh.shape[0])
    return Vh[:r].T.contiguous()


def _project_basis(X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    return (X @ P) @ P.T


def evaluate_qkv_bases(seqs: List[CircuitSeq], weights: Dict[str, torch.Tensor], ranks: List[int], device: str = "cpu") -> List[Dict[str, Any]]:
    """Basis search in the same spirit as v6: choose Pq/Pk/Pv in real head space, then evaluate A/Z/Y."""
    Qall = torch.cat([s.Q for s in seqs], dim=0).float().to(device)
    Kall = torch.cat([s.K for s in seqs], dim=0).float().to(device)
    Vall = torch.cat([s.V for s in seqs], dim=0).float().to(device)
    Wo = weights["Wo"].float().to(device)
    rows: List[Dict[str, Any]] = []
    for r in ranks:
        Pq = _pca_basis(Qall, r)
        Pk = _pca_basis(Kall, r)
        Pv = _pca_basis(Vall, r)
        num_a = den_a = 0.0; kl_sum = 0.0; top_ok = 0; nrows = 0
        num_z = den_z = 0.0; num_y = den_y = 0.0
        num_vbasis_y = den_vbasis_y = 0.0
        for s in seqs:
            Q = s.Q.float().to(device); K = s.K.float().to(device); V = s.V.float().to(device); A = s.A.float().to(device); Y = s.Y.float().to(device)
            D = Q.shape[1]
            Qh = _project_basis(Q, Pq); Kh = _project_basis(K, Pk)
            Ah = causal_softmax((Qh @ Kh.T) / math.sqrt(D))
            da = (Ah - A).float()
            num_a += float((da * da).sum().detach().cpu()); den_a += float((A * A).sum().detach().cpu())
            kl_sum += float(F.kl_div((Ah + 1e-12).log(), A, reduction="sum").detach().cpu())
            top_ok += int((Ah.argmax(dim=-1) == A.argmax(dim=-1)).sum().detach().cpu()); nrows += int(A.shape[0])
            Zt = A @ V; Zh = Ah @ V
            dz = Zh - Zt
            num_z += float((dz * dz).sum().detach().cpu()); den_z += float((Zt * Zt).sum().detach().cpu())
            Yh = Zh @ Wo.T
            dy = Yh - Y
            num_y += float((dy * dy).sum().detach().cpu()); den_y += float((Y * Y).sum().detach().cpu())
            Vp = _project_basis(V, Pv)
            Yv = (A @ Vp) @ Wo.T
            dyv = Yv - Y
            num_vbasis_y += float((dyv * dyv).sum().detach().cpu()); den_vbasis_y += float((Y * Y).sum().detach().cpu())
        rows.append({
            "basis_kind": "pca_no_center_functional", "rank": int(min(r, Qall.shape[1])),
            "A_rel": math.sqrt(num_a / max(1e-12, den_a)), "KL": kl_sum / max(1, nrows), "top1": top_ok / max(1, nrows),
            "Z_rel": math.sqrt(num_z / max(1e-12, den_z)), "Y_rel_from_QK_basis": math.sqrt(num_y / max(1e-12, den_y)),
            "Y_rel_from_V_basis_true_A": math.sqrt(num_vbasis_y / max(1e-12, den_vbasis_y)),
        })
    return rows


def _pack_qk_batches(seqs: List[CircuitSeq], device: str, batch_size: int = 16) -> List[Dict[str, torch.Tensor]]:
    out = []
    for off in range(0, len(seqs), batch_size):
        chunk = seqs[off:off + batch_size]
        B = len(chunk); Tm = max(s.Q.shape[0] for s in chunk); D = chunk[0].Q.shape[1]
        Q = torch.zeros(B, Tm, D, device=device); K = torch.zeros_like(Q); A = torch.zeros(B, Tm, Tm, device=device)
        mask = torch.zeros(B, Tm, Tm, dtype=torch.bool, device=device); row_sel = torch.zeros(B, Tm, dtype=torch.bool, device=device)
        for b, s in enumerate(chunk):
            T = s.Q.shape[0]
            Q[b, :T] = s.Q.to(device); K[b, :T] = s.K.to(device); A[b, :T, :T] = s.A.to(device)
            row_sel[b, :T] = True
            ar = torch.arange(Tm, device=device)
            mask[b] = (ar[None, :] < T) & (ar[:, None] < T) & (ar[:, None] >= ar[None, :])
        out.append({"Q": Q, "K": K, "A": A, "mask": mask, "row_sel": row_sel})
    return out


def evaluate_learned_qk_basis(seqs: List[CircuitSeq], weights: Dict[str, torch.Tensor], ranks: List[int], steps: int, lr: float, batch_size: int, device: str) -> List[Dict[str, Any]]:
    """Small learned Pq/Pk score-basis search, same idea as v6 but self-contained and no patching."""
    rows: List[Dict[str, Any]] = []
    if steps <= 0:
        return rows
    batches = _pack_qk_batches(seqs, device, batch_size=batch_size)
    Wo = weights["Wo"].to(device).float()
    Qall = torch.cat([s.Q for s in seqs], dim=0).float().to(device)
    Kall = torch.cat([s.K for s in seqs], dim=0).float().to(device)
    for r in ranks:
        Pq0 = _pca_basis(Qall, r).to(device); Pk0 = _pca_basis(Kall, r).to(device)
        Pq = torch.nn.Parameter(Pq0.clone()); Pk = torch.nn.Parameter(Pk0.clone())
        opt = torch.optim.AdamW([Pq, Pk], lr=lr, weight_decay=1e-4)
        denom = math.sqrt(int(Pq.shape[1]))
        for _ in range(int(steps)):
            loss_sum = 0.0; n = 0
            for bd in batches:
                Ql = bd["Q"] @ Pq; Kl = bd["K"] @ Pk
                scores = torch.bmm(Ql, Kl.transpose(1, 2)) / denom
                Ah = torch.softmax(scores.masked_fill(~bd["mask"], -1e9), dim=-1)
                sel = bd["row_sel"]
                loss_b = F.kl_div((Ah[sel] + 1e-12).log(), bd["A"][sel], reduction="sum")
                loss_sum = loss_sum + loss_b; n += int(sel.sum())
            loss = loss_sum / max(1, n)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([Pq, Pk], 1.0); opt.step()
        # eval
        num_a = den_a = 0.0; kl_sum = 0.0; top_ok = 0; nrows = 0; num_z = den_z = 0.0; num_y = den_y = 0.0
        with torch.no_grad():
            for s in seqs:
                Q=s.Q.to(device); K=s.K.to(device); V=s.V.to(device); A=s.A.to(device); Y=s.Y.to(device)
                Ah = causal_softmax(((Q @ Pq) @ (K @ Pk).T) / denom)
                da=Ah-A; num_a += float((da*da).sum().cpu()); den_a += float((A*A).sum().cpu())
                kl_sum += float(F.kl_div((Ah+1e-12).log(), A, reduction="sum").cpu())
                top_ok += int((Ah.argmax(-1)==A.argmax(-1)).sum().cpu()); nrows += int(A.shape[0])
                Zt=A@V; Zh=Ah@V; dz=Zh-Zt; num_z += float((dz*dz).sum().cpu()); den_z += float((Zt*Zt).sum().cpu())
                Yh=Zh@Wo.T; dy=Yh-Y; num_y += float((dy*dy).sum().cpu()); den_y += float((Y*Y).sum().cpu())
        rows.append({"basis_kind":"learned_score_qk", "rank":int(Pq.shape[1]), "steps":int(steps),
                     "A_rel":math.sqrt(num_a/max(1e-12,den_a)), "KL":kl_sum/max(1,nrows), "top1":top_ok/max(1,nrows),
                     "Z_rel":math.sqrt(num_z/max(1e-12,den_z)), "Y_rel_from_QK_basis":math.sqrt(num_y/max(1e-12,den_y))})
    return rows


# ---------------- verification ----------------

def verify_targets(seqs: List[CircuitSeq], weights: Dict[str, torch.Tensor], Mdelta_aug: Dict[int, torch.Tensor], Mdelta_lin: Dict[int, torch.Tensor], max_delta: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    Wq, Wk, Wv, Wo = weights["Wq"], weights["Wk"], weights["Wv"], weights["Wo"]
    bq, bk, bv = weights["bq"], weights["bk"], weights["bv"]
    Wq_aug, Wk_aug, Wv_aug = weights["Wq_aug"], weights["Wk_aug"], weights["Wv_aug"]
    Cvo_lin = Wo @ Wv          # [H,H]
    Cvo_aug = Wo @ Wv_aug      # [H,H+1]
    b_vo = Wo @ bv             # [H]
    rows: List[Dict[str, Any]] = []
    acc = {k: 0.0 for k in ["score_aug_num","score_aug_den","score_lin_num","score_lin_den","A_num","A_den","Qpre_num","Qpre_den","Kpre_num","Kpre_den","V_num","V_den","Y_aug_num","Y_aug_den","Y_raw_num","Y_raw_den"]}
    for s in seqs:
        T = int(s.Xn.shape[0]); D = int(s.Q.shape[-1])
        scores_true = (s.Q @ s.K.T) / math.sqrt(D)
        scores_aug = torch.zeros_like(scores_true); scores_lin = torch.zeros_like(scores_true)
        mask_eval = torch.zeros_like(scores_true, dtype=torch.bool)
        for i in range(T):
            for j in range(i + 1):
                d = i - j
                if d <= max_delta and d in Mdelta_aug:
                    scores_aug[i, j] = s.Xaug[i] @ Mdelta_aug[d] @ s.Xaug[j]
                    scores_lin[i, j] = s.Xn[i] @ Mdelta_lin[d] @ s.Xn[j] if d in Mdelta_lin else 0.0
                    mask_eval[i, j] = True
        if int(mask_eval.sum()) > 0:
            diff_aug = (scores_aug[mask_eval] - scores_true[mask_eval]).float()
            diff_lin = (scores_lin[mask_eval] - scores_true[mask_eval]).float()
            den = float(scores_true[mask_eval].float().pow(2).sum().item())
            acc["score_aug_num"] += float((diff_aug * diff_aug).sum().item()); acc["score_aug_den"] += den
            acc["score_lin_num"] += float((diff_lin * diff_lin).sum().item()); acc["score_lin_den"] += den
        Ahat = causal_softmax(scores_aug.masked_fill(~mask_eval, torch.finfo(scores_aug.dtype).min))
        row_covered = torch.zeros((T,), dtype=torch.bool)
        for i in range(T):
            row_covered[i] = all(((i - j) <= max_delta and (i - j) in Mdelta_aug) for j in range(i + 1))
        if int(row_covered.sum()) > 0:
            da = (Ahat[row_covered] - s.A[row_covered]).float()
            acc["A_num"] += float((da * da).sum().item()); acc["A_den"] += float((s.A[row_covered].float().pow(2)).sum().item())
        Qpre_hat = s.Xaug @ Wq_aug.T
        Kpre_hat = s.Xaug @ Wk_aug.T
        V_hat = s.Xaug @ Wv_aug.T
        Y_aug = s.A @ (s.Xaug @ Cvo_aug.T)
        # raw dynamic block: linear normalized part plus bias per row. Since sum_j Aij=1, bias is added once.
        payload_raw = (s.Xraw * s.rms_scale) @ Cvo_lin.T + b_vo.view(1, -1)
        Y_raw = s.A @ payload_raw
        for name, pred, true in [("Qpre", Qpre_hat, s.Q_pre), ("Kpre", Kpre_hat, s.K_pre), ("V", V_hat, s.V), ("Y_aug", Y_aug, s.Y), ("Y_raw", Y_raw, s.Y)]:
            diff = (pred - true).float(); acc[f"{name}_num"] += float((diff*diff).sum().item()); acc[f"{name}_den"] += float((true.float()*true.float()).sum().item())
        rows.append({
            "prompt_id": s.prompt_id, "suite": s.suite, "T": T, "text": s.text[:120],
            "rms_rec_err": s.rms_rec_err,
            "qk_score_aug_rel_covered": math.sqrt(float((scores_aug[mask_eval]-scores_true[mask_eval]).pow(2).sum().item()) / max(1e-12, float(scores_true[mask_eval].pow(2).sum().item()))) if int(mask_eval.sum()) else None,
            "qk_score_linear_no_bias_rel_covered": math.sqrt(float((scores_lin[mask_eval]-scores_true[mask_eval]).pow(2).sum().item()) / max(1e-12, float(scores_true[mask_eval].pow(2).sum().item()))) if int(mask_eval.sum()) else None,
            "qk_rows_fully_covered": int(row_covered.sum().item()),
            "A_rel_fully_covered": rel_err(Ahat[row_covered], s.A[row_covered]) if int(row_covered.sum()) else None,
            "Qpre_from_affine_rel": rel_err(Qpre_hat, s.Q_pre),
            "Kpre_from_affine_rel": rel_err(Kpre_hat, s.K_pre),
            "V_from_affine_Wv_rel": rel_err(V_hat, s.V),
            "Y_from_Cvo_aug_rel": rel_err(Y_aug, s.Y),
            "Y_block_raw_affine_rel": rel_err(Y_raw, s.Y),
            "A_entropy_mean": float((-(s.A * (s.A + 1e-12).log()).sum(dim=-1)).mean().item()),
            "Y_norm_mean": float(torch.linalg.norm(s.Y, dim=-1).mean().item()),
        })
    summary = {
        "qk_score_aug_rel_covered_normX": math.sqrt(acc["score_aug_num"] / max(1e-12, acc["score_aug_den"])),
        "qk_score_linear_no_bias_rel_covered_normX": math.sqrt(acc["score_lin_num"] / max(1e-12, acc["score_lin_den"])),
        "A_rel_rows_fully_covered_from_aug_scores": math.sqrt(acc["A_num"] / max(1e-12, acc["A_den"])) if acc["A_den"] > 0 else None,
        "Qpre_from_affine_rel": math.sqrt(acc["Qpre_num"] / max(1e-12, acc["Qpre_den"])),
        "Kpre_from_affine_rel": math.sqrt(acc["Kpre_num"] / max(1e-12, acc["Kpre_den"])),
        "V_from_affine_Wv_rel": math.sqrt(acc["V_num"] / max(1e-12, acc["V_den"])),
        "Y_from_Cvo_aug_rel": math.sqrt(acc["Y_aug_num"] / max(1e-12, acc["Y_aug_den"])),
        "Y_block_raw_affine_rel": math.sqrt(acc["Y_raw_num"] / max(1e-12, acc["Y_raw_den"])),
    }
    return summary, rows


# ---------------- optional generic channel baseline ----------------

def make_light_ops(D: int, dtype=torch.float32) -> Tuple[List[str], torch.Tensor]:
    ops: List[torch.Tensor] = []; names: List[str] = []
    I = torch.eye(D, dtype=dtype); ops.append(I); names.append("Identity")
    ops.append(torch.diag(torch.linspace(-1, 1, D, dtype=dtype))); names.append("RampDiag")
    ops.append(torch.ones(D, D, dtype=dtype) / D); names.append("MeanProject")
    for sh in [1, 2, 4, 8]:
        M = torch.zeros(D, D, dtype=dtype)
        for i in range(D):
            j = i - sh
            if 0 <= j < D: M[i, j] = 1.0
        ops.append(M); names.append(f"ShiftRight{sh}")
        ops.append(M.T); names.append(f"ShiftLeft{sh}")
    for block in [2, 4, 8, 16, 32, 64]:
        if block >= D: continue
        M = torch.zeros(D, D, dtype=dtype)
        for start in range(0, D, block):
            end = min(D, start + block); M[start:end, start:end] = 1.0 / max(1, end - start)
        ops.append(M); names.append(f"BlockAvg{block}")
    A = torch.stack([m.reshape(-1) / torch.linalg.norm(m.reshape(-1)).clamp_min(1e-12) for m in ops], dim=1)
    return names, A


def fit_light_ops(M: torch.Tensor, ridge: float = 1e-4, max_terms: int = 8) -> Dict[str, Any]:
    M = M.detach().float().cpu()
    if M.shape[0] != M.shape[1]:
        return {"enabled": False, "reason": "non-square matrix"}
    D = int(M.shape[0]); names, A = make_light_ops(D)
    b = M.reshape(-1); bnorm = torch.linalg.norm(b).clamp_min(1e-12)
    c = torch.linalg.solve(A.T @ A + ridge * torch.eye(A.shape[1]), A.T @ b)
    if 0 < max_terms < c.numel():
        keep = torch.topk(c.abs(), k=max_terms).indices; A2 = A[:, keep]
        c2 = torch.linalg.solve(A2.T @ A2 + ridge * torch.eye(A2.shape[1]), A2.T @ b)
        cc = torch.zeros_like(c); cc[keep] = c2; c = cc
    rec = A @ c
    terms = []
    for i in torch.argsort(c.abs(), descending=True).tolist()[:max_terms]:
        if abs(float(c[i])) > 1e-8:
            terms.append({"name": names[i], "coef_normalized_atom": float(c[i])})
    return {"enabled": True, "rel_err": float(torch.linalg.norm(rec - b) / bnorm), "terms": terms, "bank_size": len(names)}



# ---------------- MLP exact matrix-program analysis (NO movement operators) ----------------

def parse_layers(spec: str, n_layers: int) -> List[int]:
    """Parse layer specs like 'all', '0,6,12', '14-20'."""
    s = str(spec).strip().lower()
    if s in ("", "none"):
        return []
    if s == "all":
        return list(range(n_layers))
    out: List[int] = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            out.extend(range(max(0, lo), min(n_layers - 1, hi) + 1))
        else:
            x = int(part)
            if 0 <= x < n_layers:
                out.append(x)
    return sorted(dict.fromkeys(out))


def clean_tok(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        try:
            return tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        except Exception:
            return str(token_id)


def tok_ok(txt: str) -> bool:
    t = str(txt).strip()
    if not t:
        return False
    return sum(c.isalnum() for c in t) >= max(1, len(t) // 2)


def rmsnorm_rows(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    v = x.float()
    return v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()


class GateLens:
    """Gate/read/write lens for SwiGLU MLP neurons.

    Reads are intentionally computed through gate_proj Wg on RMSNorm(embed), not through up_proj.
    Writes are computed by projecting Wd[:,j] through the final LM head.
    """
    def __init__(self, model: Any, tokenizer: Any, device: str, pool: int = 300):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pool = int(pool)
        self.W_E = model.model.embed_tokens.weight.detach().float().to(device)
        self.W_U = model.lm_head.weight.detach().float().to(device)
        self.final_norm_w = model.model.norm.weight.detach().float().to(device)
        good = [tok_ok(clean_tok(tokenizer, i)) for i in range(self.W_E.shape[0])]
        self.good = torch.tensor(good, device=device, dtype=torch.bool)

    def _top(self, scores: torch.Tensor, k: int) -> List[str]:
        sc = scores.detach().float().clone()
        sc[~self.good] = -1e30
        out: List[str] = []
        seen = set()
        for tid in sc.topk(min(self.pool, sc.numel())).indices.detach().cpu().tolist():
            t = clean_tok(self.tokenizer, int(tid)).strip()
            key = t.lower()
            if key in seen or not tok_ok(t):
                continue
            out.append(t)
            seen.add(key)
            if len(out) >= k:
                break
        return out

    @torch.no_grad()
    def neuron_reads(self, layer_idx: int, neuron_idx: int, k: int = 5) -> List[str]:
        layer = self.model.model.layers[layer_idx]
        Wg = layer.mlp.gate_proj.weight.detach().float().to(self.device)
        ln = layer.post_attention_layernorm.weight.detach().float().to(self.device)
        En = rmsnorm_rows(self.W_E, ln)
        return self._top(En @ Wg[int(neuron_idx)], k)

    @torch.no_grad()
    def neuron_writes(self, layer_idx: int, neuron_idx: int, a_sign: float = 1.0, k: int = 5) -> List[str]:
        layer = self.model.model.layers[layer_idx]
        Wd = layer.mlp.down_proj.weight.detach().float().to(self.device)
        v = Wd[:, int(neuron_idx)] * (1.0 if float(a_sign) >= 0 else -1.0)
        pn = rmsnorm_rows(v.unsqueeze(0), self.final_norm_w)[0]
        return self._top(self.W_U @ pn, k)


@dataclass
class MLPSeq:
    prompt_id: int
    suite: str
    text: str
    token_ids: List[int]
    tokens: List[str]
    X: torch.Tensor       # [T,H], exact MLP input captured by forward-pre hook
    Y: torch.Tensor       # [T,H], exact MLP output captured by forward hook
    target_id: int        # argmax next-token id at final position for examples
    target_token: str


def get_mlp_weights(model: Any, layer_idx: int, device: str) -> Dict[str, torch.Tensor]:
    layer = model.model.layers[layer_idx]
    return {
        "Wg": layer.mlp.gate_proj.weight.detach().float().to(device),
        "Wu": layer.mlp.up_proj.weight.detach().float().to(device),
        "Wd": layer.mlp.down_proj.weight.detach().float().to(device),
    }


@torch.no_grad()
def collect_mlp_data(model: Any, tokenizer: Any, prompts: List[Dict[str, str]],
                     layer_idx: int, max_length: int, device: str) -> List[MLPSeq]:
    """Collect exact inputs/outputs of one MLP layer for all prompt examples."""
    layer = model.model.layers[layer_idx]
    seqs: List[MLPSeq] = []
    for pi, pr in enumerate(prompts):
        cap: Dict[str, torch.Tensor] = {}
        def hook(mod, inp, out):
            cap["x"] = inp[0].detach().float()
            cap["y"] = out.detach().float()
        h = layer.mlp.register_forward_hook(hook)
        try:
            enc = tokenizer(pr["text"], return_tensors="pt", truncation=True, max_length=max_length).to(device)
            out = model(**enc, use_cache=False)
        finally:
            h.remove()
        if "x" not in cap or "y" not in cap:
            continue
        ids = enc["input_ids"][0].detach().cpu().tolist()
        toks = [clean_tok(tokenizer, t) for t in ids]
        target_id = int(out.logits[0, -1].detach().float().argmax().item())
        seqs.append(MLPSeq(
            prompt_id=pi,
            suite=str(pr.get("suite", "?")),
            text=pr["text"],
            token_ids=ids,
            tokens=toks,
            X=cap["x"][0].detach().float().cpu(),
            Y=cap["y"][0].detach().float().cpu(),
            target_id=target_id,
            target_token=clean_tok(tokenizer, target_id),
        ))
        del out
        if torch.cuda.is_available() and (pi + 1) % 16 == 0:
            torch.cuda.empty_cache()
    return seqs


@torch.no_grad()
def mlp_reconstruct_from_weights(X: torch.Tensor, weights: Dict[str, torch.Tensor],
                                 device: str, batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact SwiGLU MLP matrix program on captured MLP input.

    y = sum_j a_j(x) Wd[:,j], where a_j(x)=silu(Wg_j x) * (Wu_j x).
    Returns (Y_rec, A_coeff, Gate, Read).
    """
    Wg, Wu, Wd = weights["Wg"], weights["Wu"], weights["Wd"]
    Ys: List[torch.Tensor] = []
    As: List[torch.Tensor] = []
    Gs: List[torch.Tensor] = []
    Rs: List[torch.Tensor] = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device).float()
        gate = F.silu(xb @ Wg.T)
        read = xb @ Wu.T
        a = gate * read
        y = a @ Wd.T
        Ys.append(y.detach().cpu())
        As.append(a.detach().cpu())
        Gs.append(gate.detach().cpu())
        Rs.append(read.detach().cpu())
    return torch.cat(Ys, 0), torch.cat(As, 0), torch.cat(Gs, 0), torch.cat(Rs, 0)


def mlp_atom_matrix(model: Any, layer_idx: int, neuron_idx: int, device: str) -> torch.Tensor:
    """Explicit rank-1 MLP atom: Wd[:,j] outer Wu[j,:]."""
    w = get_mlp_weights(model, layer_idx, device)
    return torch.outer(w["Wd"][:, int(neuron_idx)], w["Wu"][int(neuron_idx)])


def mlp_atom_shape_stats(weights: Dict[str, torch.Tensor], neuron_idx: int) -> Dict[str, Any]:
    Wd, Wu = weights["Wd"], weights["Wu"]
    j = int(neuron_idx)
    wd_norm = float(torch.linalg.norm(Wd[:, j]).item())
    wu_norm = float(torch.linalg.norm(Wu[j]).item())
    # Frobenius norm of outer product equals product of vector norms.
    return {"rank": 1, "write_norm": wd_norm, "read_norm": wu_norm, "atom_fro_norm": wd_norm * wu_norm}


def _pca_columns(X: torch.Tensor, rank: int) -> torch.Tensor:
    X = X.detach().float()
    Xc = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    r = min(int(rank), int(Vh.shape[0]))
    return Vh[:r].T.contiguous()


@torch.no_grad()
def mlp_projected_atom_report(seqs: List[MLPSeq], weights: Dict[str, torch.Tensor],
                              rank: int, top_k: int, device: str) -> Dict[str, Any]:
    """Projected atom report from v34 idea, but without movement clustering.

    atom_j^U = (U^T Wd[:,j]) outer (Wu[j] U), reported only as stats/top norms.
    """
    if int(rank) <= 0 or not seqs:
        return {"enabled": False}
    X = torch.cat([s.X for s in seqs], 0)
    U = _pca_columns(X, int(rank)).to(device)
    Wd, Wu = weights["Wd"], weights["Wu"]
    A = Wd.T @ U      # [m,p], projected write
    C = Wu @ U        # [m,p], projected read
    norms = A.norm(dim=1) * C.norm(dim=1)
    vals, idx = torch.topk(norms, k=min(int(top_k), norms.numel()))
    return {
        "enabled": True,
        "rank": int(U.shape[1]),
        "n_atoms": int(norms.numel()),
        "mean_projected_atom_norm": float(norms.mean().item()),
        "max_projected_atom_norm": float(norms.max().item()),
        "top_atoms": [{"neuron": int(j), "projected_atom_norm": float(v)} for v, j in zip(vals.detach().cpu().tolist(), idx.detach().cpu().tolist())],
        "formula": "atom_j^U = (U^T Wd[:,j]) outer (Wu[j] U)",
    }


@torch.no_grad()
def mlp_neuron_ablation_verify(model: Any, tokenizer: Any, prompt: str, layer_idx: int,
                               neuron_idx: int, target_id: int, device: str, seed: int = 0) -> Dict[str, Any]:
    """Causal check: subtract one neuron's exact last-token MLP write and compare to random/unrelated controls."""
    layer = model.model.layers[layer_idx]
    weights = get_mlp_weights(model, layer_idx, device)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    cap: Dict[str, torch.Tensor] = {}
    def pre(mod, inp):
        cap["x"] = inp[0].detach().float()
    hp = layer.mlp.register_forward_pre_hook(pre)
    try:
        base_out = model(**enc, use_cache=False)
    finally:
        hp.remove()
    base = float(base_out.logits[0, -1, int(target_id)].detach().float().item())
    x = cap["x"][0, -1].to(device).float()
    Wg, Wu, Wd = weights["Wg"], weights["Wu"], weights["Wd"]
    gate = F.silu(x @ Wg.T)
    read = x @ Wu.T
    a = gate * read
    j = int(neuron_idx)
    write_j = float(a[j].item()) * Wd[:, j]

    def logit_after_subtract(vec: torch.Tensor) -> float:
        def hook(mod, inp, out):
            oo = out.clone()
            oo[:, -1, :] = oo[:, -1, :] - vec.to(out.dtype)
            return oo
        hh = layer.mlp.register_forward_hook(hook)
        try:
            o = model(**enc, use_cache=False)
        finally:
            hh.remove()
        return float(o.logits[0, -1, int(target_id)].detach().float().item())

    real = base - logit_after_subtract(write_j)
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 100 * int(layer_idx) + j)
    rnd = torch.randn(write_j.shape, generator=g).to(device)
    rnd = rnd / rnd.norm().clamp_min(1e-12) * write_j.norm().clamp_min(1e-12)
    rand = base - logit_after_subtract(rnd)
    nrm = a.abs() * Wd.norm(dim=0)
    diff = (nrm - nrm[j]).abs()
    diff[j] = 1e30
    ju = int(diff.argmin().item())
    unrel_vec = float(a[ju].item()) * Wd[:, ju]
    unrel_vec = unrel_vec / unrel_vec.norm().clamp_min(1e-12) * write_j.norm().clamp_min(1e-12)
    unrel = base - logit_after_subtract(unrel_vec)
    return {
        "neuron": j,
        "target_id": int(target_id),
        "base_logit": base,
        "delta_real": real,
        "delta_random_control": rand,
        "delta_unrelated_control": unrel,
        "real_over_random_abs": abs(real) / (abs(rand) + 1e-9),
        "real_over_unrelated_abs": abs(real) / (abs(unrel) + 1e-9),
        "coefficient_a": float(a[j].item()),
    }


@torch.no_grad()
def run_mlp_analysis(model: Any, tokenizer: Any, prompts: List[Dict[str, str]],
                     args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    """Full non-movement MLP matrix-program analysis across selected layers."""
    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    mlp_layers = parse_layers(args.mlp_layers, n_layers)
    mdir = out_dir / "mlp_matrix_program"
    ensure_dir(mdir)
    if not mlp_layers:
        return {"enabled": False, "reason": "no mlp layers selected"}
    lens = None
    if args.enable_token_lens:
        lens_device = args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        lens = GateLens(model, tokenizer, lens_device, pool=args.lens_pool)
    compute_device = args.mlp_compute_device
    if compute_device == "same":
        compute_device = args.device
    if str(compute_device).startswith("cuda") and not torch.cuda.is_available():
        compute_device = "cpu"
    print(f"\n=== MLP exact matrix-program analysis layers={mlp_layers} ===", flush=True)
    layer_summaries: List[Dict[str, Any]] = []
    faith_rows: List[Dict[str, Any]] = []
    example_rows_all: List[Dict[str, Any]] = []
    exact_program = {
        "MLP_input": "x is the exact input captured at layer.mlp forward pre-hook (already post_attention_layernorm for Qwen2)",
        "gate": "g_j(x) = silu(Wg_j x)",
        "read": "r_j(x) = Wu_j x",
        "coefficient": "a_j(x) = g_j(x) * r_j(x)",
        "rank1_atom": "atom_j = Wd[:,j] outer Wu[j,:]",
        "output_identity": "y(x) = sum_j a_j(x) Wd[:,j] = (silu(x Wg^T) * (x Wu^T)) Wd^T",
        "gate_lens": "read semantics are estimated with RMSNorm(embed) @ Wg[j], writes with lm_head @ RMSNorm(Wd[:,j])",
    }
    for layer_idx in mlp_layers:
        ldir = mdir / f"L{layer_idx}"
        ensure_dir(ldir)
        print(f"  MLP L{layer_idx}: collecting...", flush=True)
        seqs = collect_mlp_data(model, tokenizer, prompts, layer_idx, args.max_length, args.device)
        if not seqs:
            layer_summaries.append({"layer": layer_idx, "enabled": False, "reason": "no sequences"})
            continue
        weights = get_mlp_weights(model, layer_idx, compute_device)
        per_prompt: List[Dict[str, Any]] = []
        rels: List[float] = []
        max_abs_all: List[float] = []
        ntok = 0
        for s in seqs:
            y_rec, Acoef, Gate, Read = mlp_reconstruct_from_weights(s.X, weights, compute_device, batch_size=args.mlp_batch_size)
            e = rel_err(y_rec, s.Y)
            max_abs = float((y_rec - s.Y).abs().max().item())
            rels.append(e)
            max_abs_all.append(max_abs)
            ntok += int(s.X.shape[0])
            per_prompt.append({
                "layer": layer_idx,
                "prompt_id": s.prompt_id,
                "suite": s.suite,
                "tokens": int(s.X.shape[0]),
                "mlp_output_rel": e,
                "mlp_output_max_abs": max_abs,
                "target_id": s.target_id,
                "target_token": s.target_token,
            })
        write_csv(ldir / "mlp_per_prompt_checks.csv", per_prompt)
        faith = {
            "layer": layer_idx,
            "nseq": len(seqs),
            "ntokens": ntok,
            "mlp_output_rel_mean": safe_mean(rels),
            "mlp_output_rel_max": max(rels) if rels else None,
            "mlp_output_max_abs_max": max(max_abs_all) if max_abs_all else None,
        }
        faith_rows.append(faith)

        # Examples: top atoms at final token, by contribution to the model's predicted next-token logit.
        example_rows: List[Dict[str, Any]] = []
        Wd = weights["Wd"]
        for s in seqs[: int(args.mlp_example_prompts)]:
            x_last = s.X[-1:].to(compute_device).float()
            y_rec, Acoef, Gate, Read = mlp_reconstruct_from_weights(x_last.cpu(), weights, compute_device, batch_size=1)
            a = Acoef[0].to(compute_device)
            gate = Gate[0].to(compute_device)
            read = Read[0].to(compute_device)
            target_vec = model.lm_head.weight[int(s.target_id)].detach().float().to(compute_device)
            write_to_target = Wd.T @ target_vec
            contrib = a * write_to_target
            vals, idx = torch.topk(contrib.abs(), k=min(int(args.mlp_top_neurons), contrib.numel()))
            atoms = []
            for _, jj in zip(vals.detach().cpu().tolist(), idx.detach().cpu().tolist()):
                j = int(jj)
                atom_stats = mlp_atom_shape_stats(weights, j)
                row = {
                    "neuron": j,
                    "a": float(a[j].item()),
                    "gate": float(gate[j].item()),
                    "read": float(read[j].item()),
                    "target_contribution": float(contrib[j].item()),
                    "atom": atom_stats,
                    "gate_reads": lens.neuron_reads(layer_idx, j, k=args.lens_topk) if lens is not None else [],
                    "writes": lens.neuron_writes(layer_idx, j, a_sign=float(a[j].item()), k=args.lens_topk) if lens is not None else [],
                    "token_lens_enabled": bool(lens is not None),
                }
                if args.mlp_verify_ablation:
                    row["ablation_verify"] = mlp_neuron_ablation_verify(model, tokenizer, s.text, layer_idx, j, s.target_id, args.device, seed=args.seed)
                atoms.append(row)
                if args.save_mlp_top_atoms:
                    atom_tensor = mlp_atom_matrix(model, layer_idx, j, compute_device).detach().cpu()
                    torch.save(atom_tensor, ldir / f"atom_prompt{s.prompt_id}_n{j}.pt")
            example_rows.append({
                "layer": layer_idx,
                "prompt_id": s.prompt_id,
                "suite": s.suite,
                "text": s.text,
                "last_token": s.tokens[-1] if s.tokens else "",
                "target_id": s.target_id,
                "target_token": s.target_token,
                "top_atoms_by_target_logit": atoms,
            })
        write_json(ldir / "mlp_top_atom_examples.json", example_rows)
        example_rows_all.extend(example_rows)

        proj_report = mlp_projected_atom_report(seqs, weights, args.mlp_proj_rank, args.mlp_top_neurons, compute_device)
        summary = {
            "layer": layer_idx,
            "enabled": True,
            "exact_program": exact_program,
            "faithfulness": faith,
            "projected_atoms": proj_report,
            "n_intermediate_atoms": int(weights["Wg"].shape[0]),
            "hidden_size": int(weights["Wd"].shape[0]),
            "files": {
                "per_prompt_checks": str(ldir / "mlp_per_prompt_checks.csv"),
                "top_atom_examples": str(ldir / "mlp_top_atom_examples.json"),
            },
        }
        write_json(ldir / "summary.json", summary)
        md = []
        md.append(f"# MLP matrix program L{layer_idx}\n\n")
        md.append("## Exact program\n```text\n")
        for k, v in exact_program.items():
            md.append(f"{k}: {v}\n")
        md.append("```\n\n")
        md.append("## Faithfulness\n```json\n" + json.dumps(json_sanitize(faith), ensure_ascii=False, indent=2) + "\n```\n")
        md.append("\n## Notes\n")
        md.append("- This is not a movement-operator analysis. No sym/skew/rotation clustering is used.\n")
        md.append("- The rank-1 atom is explicit: `Wd[:,j] outer Wu[j,:]`; the live nonlinear coefficient is `silu(Wg_j x)*(Wu_j x)`.\n")
        (ldir / "summary.md").write_text("".join(md), encoding="utf-8")
        layer_summaries.append(summary)
        del weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(mdir / "mlp_faithfulness_table.csv", faith_rows)
    top = {
        "enabled": True,
        "layers": mlp_layers,
        "exact_program": exact_program,
        "faithfulness_table": faith_rows,
        "layer_summaries": layer_summaries,
    }
    write_json(mdir / "summary.json", top)
    md = ["# MLP matrix-program summary\n\n", "## Faithfulness table\n\n",
          "| layer | nseq | ntokens | mean rel | max rel | max abs |\n",
          "|---:|---:|---:|---:|---:|---:|\n"]
    for r in faith_rows:
        md.append(f"| {r['layer']} | {r['nseq']} | {r['ntokens']} | {r['mlp_output_rel_mean']:.3e} | {r['mlp_output_rel_max']:.3e} | {r['mlp_output_max_abs_max']:.3e} |\n")
    (mdir / "summary.md").write_text("".join(md), encoding="utf-8")
    print(f"  MLP DONE wrote {mdir}", flush=True)
    return top



# ==============================================================================
# EXECUTABLE MATRIX-PROGRAM PATCHING BENCHMARK
# ==============================================================================


def _first_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, list):
        return output[0]
    return output


def _replace_first_output(output: Any, new_first: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_first,) + tuple(output[1:])
    if isinstance(output, list):
        return [new_first] + list(output[1:])
    return new_first


def _register_pre_hook_kwargs(module: Any, fn: Any):
    """Register a pre-hook that receives kwargs when supported by the local PyTorch."""
    try:
        return module.register_forward_pre_hook(fn, with_kwargs=True)
    except TypeError:
        def old_fn(mod, args):
            return fn(mod, args, {})
        return module.register_forward_pre_hook(old_fn)


def _hook_input_tensor(args: Tuple[Any, ...], kwargs: Dict[str, Any],
                       keys: Tuple[str, ...] = ("hidden_states", "x")) -> torch.Tensor:
    """Return the main tensor input from either positional args or keyword args.

    Recent Transformers versions call Qwen2Attention as
    ``self_attn(hidden_states=..., position_embeddings=..., ...)``. In that case a
    forward pre-hook receives an empty ``args`` tuple and the tensor only in ``kwargs``.
    Older versions often pass it positionally. Supporting both forms keeps patching
    compatible across Transformers/PyTorch releases.
    """
    if args:
        x = args[0]
        if torch.is_tensor(x):
            return x
    for key in keys:
        x = kwargs.get(key)
        if torch.is_tensor(x):
            return x
    tensor_keys = [str(k) for k, v in kwargs.items() if torch.is_tensor(v)]
    raise RuntimeError(
        "Could not capture the module input tensor from hook args/kwargs. "
        f"positional_args={len(args)} tensor_kwargs={tensor_keys} all_kwargs={list(kwargs.keys())}"
    )


def _position_cos_sin(model: Any, hidden_states: torch.Tensor,
                      kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    pe = kwargs.get("position_embeddings")
    if isinstance(pe, (tuple, list)) and len(pe) >= 2:
        cos, sin = pe[0], pe[1]
    else:
        B, T = hidden_states.shape[:2]
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(T, device=hidden_states.device).view(1, T).expand(B, T)
        cos, sin = compute_position_embeddings(model, hidden_states, position_ids)
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.shape[0] == 1 and hidden_states.shape[0] > 1:
        cos = cos.expand(hidden_states.shape[0], -1, -1)
        sin = sin.expand(hidden_states.shape[0], -1, -1)
    return cos, sin


def _masked_attention_probs(scores: torch.Tensor, attention_mask: Optional[torch.Tensor],
                            batch_idx: int, head_idx: int) -> torch.Tensor:
    """Causal softmax plus an optional HF additive/binary mask for one sequence/head."""
    Tq, Tk = scores.shape[-2], scores.shape[-1]
    s = scores.float()
    causal = torch.triu(torch.ones(Tq, Tk, device=s.device, dtype=torch.bool), diagonal=1 + max(0, Tk - Tq))
    s = s.masked_fill(causal, torch.finfo(s.dtype).min)
    if torch.is_tensor(attention_mask):
        am = attention_mask
        try:
            if am.dim() == 4:
                h = head_idx if am.shape[1] > 1 else 0
                add = am[batch_idx, h, :Tq, :Tk].to(s.device).float()
                s = s + add
            elif am.dim() == 3:
                s = s + am[batch_idx, :Tq, :Tk].to(s.device).float()
            elif am.dim() == 2:
                valid = am[batch_idx, :Tk].to(s.device).bool()
                s = s.masked_fill(~valid.view(1, Tk), torch.finfo(s.dtype).min)
        except Exception:
            # A malformed/implementation-specific mask must not silently corrupt shape.
            pass
    return torch.softmax(s, dim=-1, dtype=torch.float32)


def _attn_bias_slice(module: Any, start: int, end: int, device: torch.device) -> torch.Tensor:
    b = getattr(module, "bias", None)
    if b is None:
        return torch.zeros(end - start, device=device, dtype=torch.float32)
    return b.detach().float()[start:end].to(device)


@torch.no_grad()
def attention_heads_native_and_program(model: Any, layer_idx: int, heads: List[int],
                                       hidden_states: torch.Tensor, hook_kwargs: Dict[str, Any],
                                       qk_scale: float = 1.0, vo_scale: float = 1.0
                                       ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Return summed native and executable matrix-program contributions for selected heads.

    Program path does not call q_proj/k_proj/v_proj/o_proj as a chain. It executes the
    extracted affine targets in factored form:
      q_pre = Xaug Wq_aug^T, k_pre = Xaug Wk_aug^T,
      score = q_i^T k_j / sqrt(D) == Xaug_i^T M_qk[d] Xaug_j,
      payload = Xaug C_vo_aug^T, Y = A payload.
    """
    layer = model.model.layers[int(layer_idx)]
    attn = layer.self_attn
    cfg = model.config
    B, T, H = hidden_states.shape
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    D = int(getattr(cfg, "head_dim", H // n_heads))
    groups = n_heads // n_kv
    dev = hidden_states.device
    x_native = hidden_states
    x_float = hidden_states.detach().float()
    x_aug = torch.cat([x_float, torch.ones(B, T, 1, device=dev)], dim=-1)
    cos, sin = _position_cos_sin(model, hidden_states, hook_kwargs)
    cos = cos.to(dev)
    sin = sin.to(dev)
    attention_mask = hook_kwargs.get("attention_mask")

    # Native projection path, evaluated once and sliced by head.
    q_all = attn.q_proj(x_native).view(B, T, n_heads, D).transpose(1, 2).contiguous()
    k_all = attn.k_proj(x_native).view(B, T, n_kv, D).transpose(1, 2).contiguous()
    v_all = attn.v_proj(x_native).view(B, T, n_kv, D).transpose(1, 2).contiguous()
    q_rot_all = apply_rope_rows(q_all, cos.unsqueeze(1), sin.unsqueeze(1))
    k_rot_all = apply_rope_rows(k_all, cos.unsqueeze(1), sin.unsqueeze(1))

    native_sum = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    program_sum = torch.zeros_like(native_sum)
    details: List[Dict[str, Any]] = []

    for head in heads:
        head = int(head)
        kv = head // groups
        qh = q_rot_all[:, head].float()
        kh = k_rot_all[:, kv].float()
        vh = v_all[:, kv].float()
        Wo = attn.o_proj.weight.detach().float()[:, head * D:(head + 1) * D].to(dev)

        # Extract affine targets directly from weights.
        Wq = attn.q_proj.weight.detach().float()[head * D:(head + 1) * D].to(dev)
        Wk = attn.k_proj.weight.detach().float()[kv * D:(kv + 1) * D].to(dev)
        Wv = attn.v_proj.weight.detach().float()[kv * D:(kv + 1) * D].to(dev)
        bq = _attn_bias_slice(attn.q_proj, head * D, (head + 1) * D, dev)
        bk = _attn_bias_slice(attn.k_proj, kv * D, (kv + 1) * D, dev)
        bv = _attn_bias_slice(attn.v_proj, kv * D, (kv + 1) * D, dev)
        Wq_aug = torch.cat([Wq, bq[:, None]], dim=1)
        Wk_aug = torch.cat([Wk, bk[:, None]], dim=1)
        Wv_aug = torch.cat([Wv, bv[:, None]], dim=1)
        Cvo_aug = Wo @ Wv_aug

        qpre_prog = x_aug @ Wq_aug.T
        kpre_prog = x_aug @ Wk_aug.T
        qprog = apply_rope_rows(qpre_prog, cos, sin)
        kprog = apply_rope_rows(kpre_prog, cos, sin)
        payload_prog = x_aug @ Cvo_aug.T

        native_b: List[torch.Tensor] = []
        program_b: List[torch.Tensor] = []
        a_rels: List[float] = []
        score_rels: List[float] = []
        for b in range(B):
            score_native = (qh[b] @ kh[b].T) / math.sqrt(D)
            A_native = _masked_attention_probs(score_native, attention_mask, b, head)
            y_native = (A_native @ vh[b]) @ Wo.T

            score_program = (qprog[b] @ kprog[b].T) / math.sqrt(D)
            A_program = _masked_attention_probs(score_program * float(qk_scale), attention_mask, b, head)
            y_program = (A_program @ payload_prog[b]) * float(vo_scale)

            native_b.append(y_native)
            program_b.append(y_program)
            score_rels.append(rel_err(score_program, score_native))
            a_rels.append(rel_err(A_program, A_native))

        yn = torch.stack(native_b, dim=0)
        yp = torch.stack(program_b, dim=0)
        native_sum += yn
        program_sum += yp
        details.append({
            "layer": int(layer_idx),
            "head": head,
            "qk_scale": float(qk_scale),
            "vo_scale": float(vo_scale),
            "score_program_vs_native_rel": safe_mean(score_rels),
            "attention_program_vs_native_rel": safe_mean(a_rels),
            "head_output_program_vs_native_rel": rel_err(yp, yn),
            "head_output_program_vs_native_rel_percent": 100.0 * rel_err(yp, yn),
            "native_output_norm": float(yn.norm().item()),
            "program_output_norm": float(yp.norm().item()),
        })
    return native_sum, program_sum, details


def _install_attention_program_patch(model: Any, layer_to_heads: Dict[int, List[int]],
                                     qk_scale: float, vo_scale: float,
                                     local_records: List[Dict[str, Any]]) -> List[Any]:
    handles: List[Any] = []
    for layer_idx, heads in layer_to_heads.items():
        attn = model.model.layers[int(layer_idx)].self_attn
        state: Dict[str, Any] = {}

        def pre(mod, args, kwargs, state=state):
            state["hidden_states"] = _hook_input_tensor(args, kwargs, ("hidden_states",))
            state["kwargs"] = dict(kwargs)

        def post(mod, args, output, layer_idx=layer_idx, heads=list(heads), state=state):
            if "hidden_states" not in state:
                return output
            native, program, details = attention_heads_native_and_program(
                model, int(layer_idx), heads, state["hidden_states"], state.get("kwargs", {}),
                qk_scale=float(qk_scale), vo_scale=float(vo_scale))
            first = _first_output_tensor(output)
            patched = first.float() - native + program
            local_records.extend(details)
            local_records.append({
                "layer": int(layer_idx),
                "heads": [int(h) for h in heads],
                "group_output_program_vs_native_rel": rel_err(program, native),
                "group_output_program_vs_native_rel_percent": 100.0 * rel_err(program, native),
                "qk_scale": float(qk_scale),
                "vo_scale": float(vo_scale),
            })
            return _replace_first_output(output, patched.to(first.dtype))

        handles.append(_register_pre_hook_kwargs(attn, pre))
        handles.append(attn.register_forward_hook(post))
    return handles


def _install_mlp_program_patch(model: Any, layer_indices: List[int], layer_scale: float,
                               atom_scale: Optional[float], target_id: int,
                               local_records: List[Dict[str, Any]],
                               atom_control: str = "real", seed: int = 0) -> List[Any]:
    handles: List[Any] = []
    for layer_idx in layer_indices:
        mlp = model.model.layers[int(layer_idx)].mlp
        state: Dict[str, Any] = {}

        def pre(mod, args, kwargs, state=state):
            state["x"] = _hook_input_tensor(args, kwargs, ("hidden_states", "x"))

        def post(mod, args, output, layer_idx=layer_idx, state=state):
            if "x" not in state:
                return output
            x = state["x"].detach().float()
            layer = model.model.layers[int(layer_idx)]
            Wg = layer.mlp.gate_proj.weight.detach().float().to(x.device)
            Wu = layer.mlp.up_proj.weight.detach().float().to(x.device)
            Wd = layer.mlp.down_proj.weight.detach().float().to(x.device)
            gate = F.silu(x @ Wg.T)
            read = x @ Wu.T
            a = gate * read
            program = a @ Wd.T
            native = output.detach().float()
            patched = program * float(layer_scale)
            selected_atom = None
            selected_contribution = None
            control_neuron = None
            intervention_vector_norm = None
            if atom_scale is not None:
                target_vec = model.lm_head.weight[int(target_id)].detach().float().to(x.device)
                write_to_target = Wd.T @ target_vec
                final_contrib = a[0, -1] * write_to_target
                j = int(final_contrib.abs().argmax().item())
                real_atom_write = a[:, -1, j].unsqueeze(-1) * Wd[:, j].view(1, -1)
                patch_vec = real_atom_write
                control = str(atom_control).lower()
                if control == "random":
                    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1009 * int(layer_idx) + j)
                    rv = torch.randn(real_atom_write.shape[-1], generator=gen).to(x.device)
                    rv = rv / rv.norm().clamp_min(1e-12)
                    rv = rv * real_atom_write[0].norm().clamp_min(1e-12)
                    patch_vec = rv.view(1, -1).expand_as(real_atom_write)
                elif control == "unrelated":
                    norms = a[0, -1].abs() * Wd.norm(dim=0)
                    diff = (norms - norms[j]).abs()
                    diff[j] = torch.finfo(diff.dtype).max
                    ju = int(diff.argmin().item())
                    uv = a[:, -1, ju].unsqueeze(-1) * Wd[:, ju].view(1, -1)
                    uv = uv / uv.norm().clamp_min(1e-12) * real_atom_write.norm().clamp_min(1e-12)
                    patch_vec = uv
                    control_neuron = ju
                elif control != "real":
                    raise ValueError(f"unknown atom_control={atom_control!r}")
                # Atom experiment keeps every other atom exact (layer_scale should be 1).
                patched[:, -1, :] += (float(atom_scale) - float(layer_scale)) * patch_vec
                selected_atom = j
                selected_contribution = float(final_contrib[j].item())
                intervention_vector_norm = float(patch_vec.norm().item())
            local_records.append({
                "layer": int(layer_idx),
                "layer_scale": float(layer_scale),
                "atom_scale": None if atom_scale is None else float(atom_scale),
                "atom_control": str(atom_control),
                "selected_atom": selected_atom,
                "control_neuron": control_neuron,
                "selected_atom_target_logit_contribution_local": selected_contribution,
                "intervention_vector_norm": intervention_vector_norm,
                "mlp_program_vs_native_rel": rel_err(program, native),
                "mlp_program_vs_native_rel_percent": 100.0 * rel_err(program, native),
                "native_output_norm": float(native.norm().item()),
                "program_output_norm": float(program.norm().item()),
            })
            return patched.to(output.dtype)

        handles.append(_register_pre_hook_kwargs(mlp, pre))
        handles.append(mlp.register_forward_hook(post))
    return handles


def _remove_handles(handles: List[Any]) -> None:
    for h in reversed(handles):
        try:
            h.remove()
        except Exception:
            pass


@torch.no_grad()
def _last_logits(model: Any, tokenizer: Any, text: str, device: str, max_length: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out = model(**enc, use_cache=False)
    logits = out.logits[0, -1].detach().float()
    ids = enc["input_ids"][0].detach().cpu().tolist()
    return logits, {"input_ids": ids, "n_tokens": len(ids)}


def _top_tokens(tokenizer: Any, logits: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
    probs = torch.softmax(logits.float(), dim=-1)
    vals, ids = torch.topk(probs, k=min(int(k), probs.numel()))
    return [{"id": int(i), "token": clean_tok(tokenizer, int(i)), "prob": float(v)}
            for v, i in zip(vals.detach().cpu().tolist(), ids.detach().cpu().tolist())]


def _compare_last_logits(tokenizer: Any, base: torch.Tensor, patched: torch.Tensor,
                         target_id: int, topk: int) -> Dict[str, Any]:
    base = base.float()
    patched = patched.float()
    logp0 = F.log_softmax(base, dim=-1)
    logp1 = F.log_softmax(patched, dim=-1)
    p0 = logp0.exp()
    p1 = logp1.exp()
    m = 0.5 * (p0 + p1)
    kl = float((p0 * (logp0 - logp1)).sum().item())
    js = float(0.5 * (p0 * (logp0 - (m + 1e-30).log())).sum().item() +
               0.5 * (p1 * (logp1 - (m + 1e-30).log())).sum().item())
    r = rel_err(patched, base)
    top0 = int(base.argmax().item())
    top1 = int(patched.argmax().item())
    top0_set = set(torch.topk(base, k=min(topk, base.numel())).indices.detach().cpu().tolist())
    top1_set = set(torch.topk(patched, k=min(topk, patched.numel())).indices.detach().cpu().tolist())
    return {
        "logit_rel": r,
        "logit_rel_percent": 100.0 * r,
        "logit_max_abs": float((patched - base).abs().max().item()),
        "logit_cosine": float(F.cosine_similarity(base.view(1, -1), patched.view(1, -1)).item()),
        "kl_base_to_patch": kl,
        "js_divergence": js,
        "top1_same": bool(top0 == top1),
        "base_top1_id": top0,
        "base_top1_token": clean_tok(tokenizer, top0),
        "patched_top1_id": top1,
        "patched_top1_token": clean_tok(tokenizer, top1),
        "topk_overlap": len(top0_set & top1_set) / max(1, len(top0_set)),
        "target_id": int(target_id),
        "target_token": clean_tok(tokenizer, int(target_id)),
        "base_target_logit": float(base[int(target_id)].item()),
        "patched_target_logit": float(patched[int(target_id)].item()),
        "target_logit_delta": float((patched[int(target_id)] - base[int(target_id)]).item()),
        "base_target_prob": float(p0[int(target_id)].item()),
        "patched_target_prob": float(p1[int(target_id)].item()),
        "target_prob_delta": float((p1[int(target_id)] - p0[int(target_id)]).item()),
    }


def _float_list(spec: str) -> List[float]:
    return [float(x.strip()) for x in str(spec).replace(";", ",").split(",") if x.strip()]


def _group_heads(head_specs: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for l, h in head_specs:
        out.setdefault(int(l), []).append(int(h))
    return out


def _metric_stats(values: List[float]) -> Dict[str, Any]:
    """Released per-experiment aggregation; torch.median is lower-middle for even N."""
    if not values:
        return {"n": 0}
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "n": int(t.numel()),
        "mean": float(t.mean().item()),
        "median": float(t.median().item()),
        "p95": float(torch.quantile(t, 0.95).item()),
        "max": float(t.max().item()),
    }




def _conventional_median_v4(values: List[float]) -> float:
    """Conventional sample median; averages two middle values for even N."""
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(t, 0.5).item())

def _summarize_patch_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get("experiment", "?")), []).append(r)
    out: Dict[str, Any] = {}
    for name, rr in groups.items():
        out[name] = {
            "n": len(rr),
            "logit_rel_percent": _metric_stats([float(x["logit_rel_percent"]) for x in rr]),
            "kl_base_to_patch": _metric_stats([float(x["kl_base_to_patch"]) for x in rr]),
            "abs_target_logit_delta": _metric_stats([abs(float(x["target_logit_delta"])) for x in rr]),
            "top1_preservation_rate": safe_mean([1.0 if bool(x["top1_same"]) else 0.0 for x in rr]),
            "top1_change_rate": safe_mean([0.0 if bool(x["top1_same"]) else 1.0 for x in rr]),
        }
    return out


def _write_patching_article_md(path: Path, summary: Dict[str, Any], args: argparse.Namespace) -> None:
    lines: List[str] = []
    lines.append("# Matrix-program replacement and intervention results\n\n")
    lines.append("The benchmark separates exact executable replacement from causal intervention.\n\n")
    lines.append("## Exact replacement\n\n")
    lines.append("| Experiment | N | logit rel %, median | p95 | max | KL median | top-1 preserved |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for name, s in summary.items():
        if "exact" not in name:
            continue
        lr = s["logit_rel_percent"]; kl = s["kl_base_to_patch"]
        lines.append(f"| {name} | {s['n']} | {lr.get('median', 0):.6g} | {lr.get('p95', 0):.6g} | {lr.get('max', 0):.6g} | {kl.get('median', 0):.6g} | {100*s['top1_preservation_rate']:.2f}% |\n")
    lines.append("\n## Controlled interventions\n\n")
    lines.append("| Experiment | N | |Δ target logit| median | p95 | KL median | top-1 changed |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for name, s in summary.items():
        if "exact" in name:
            continue
        dl = s["abs_target_logit_delta"]; kl = s["kl_base_to_patch"]
        lines.append(f"| {name} | {s['n']} | {dl.get('median', 0):.6g} | {dl.get('p95', 0):.6g} | {kl.get('median', 0):.6g} | {100*s['top1_change_rate']:.2f}% |\n")
    lines.append("\n## Configuration\n\n```json\n")
    lines.append(json.dumps(json_sanitize({
        "patch_heads": args.patch_heads,
        "patch_mlp_layers": args.patch_mlp_layers,
        "patch_prompts": args.patch_prompts,
        "qk_scales": args.patch_qk_scales,
        "vo_scales": args.patch_vo_scales,
        "mlp_scales": args.patch_mlp_scales,
        "atom_scales": args.patch_atom_scales,
    }), ensure_ascii=False, indent=2))
    lines.append("\n```\n")
    path.write_text("".join(lines), encoding="utf-8")


def paper_patching_prompts() -> List[Dict[str, str]]:
    """Fixed, heterogeneous prompt bank for the paper patching benchmark."""
    return [
        {"suite": "capital", "text": "The capital of France is"},
        {"suite": "capital", "text": "The capital of Germany is"},
        {"suite": "capital", "text": "The capital of Japan is"},
        {"suite": "capital", "text": "The capital of Italy is"},
        {"suite": "factual", "text": "Water freezes at a temperature of"},
        {"suite": "math", "text": "If x = 7, then 3 * x + 2 ="},
        {"suite": "code", "text": "def add(a, b):\n    return"},
        {"suite": "russian", "text": "Столица Франции — это"},
        {"suite": "ukrainian", "text": "Столиця Франції — це"},
        {"suite": "reasoning", "text": "A triangle has three sides, therefore a square has"},
        {"suite": "text", "text": "The opposite of hot is"},
        {"suite": "json", "text": "JSON: {\"country\": \"France\", \"capital\":"},
    ]


def select_patching_prompts(prompts: List[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    source = str(args.patch_prompt_source).strip().lower()
    bank = paper_patching_prompts() if source == "paper" else list(prompts)
    n = max(1, int(args.patch_prompts))
    return bank[:n]


@torch.no_grad()
def run_patching_experiments(model: Any, tokenizer: Any, prompts: List[Dict[str, str]],
                             args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    pdir = out_dir / "patching"
    ensure_dir(pdir)
    cfg = model.config
    patch_heads_spec = args.patch_heads
    if str(patch_heads_spec).strip().lower() in ("same", "from-heads"):
        patch_heads_spec = args.heads
    head_specs = parse_heads(str(patch_heads_spec), int(cfg.num_hidden_layers), int(cfg.num_attention_heads))
    mlp_layers = parse_layers(args.patch_mlp_layers, int(cfg.num_hidden_layers))
    selected_prompts = select_patching_prompts(prompts, args)
    write_json(pdir / "patch_prompts.json", selected_prompts)
    qk_scales = _float_list(args.patch_qk_scales)
    vo_scales = _float_list(args.patch_vo_scales)
    mlp_scales = _float_list(args.patch_mlp_scales)
    atom_scales = _float_list(args.patch_atom_scales)
    rows: List[Dict[str, Any]] = []
    details_json: List[Dict[str, Any]] = []

    print(f"\n=== EXECUTABLE PATCHING heads={head_specs} mlp_layers={mlp_layers} prompts={len(selected_prompts)} ===", flush=True)

    for prompt_id, pr in enumerate(selected_prompts):
        text = pr["text"]
        base_logits, base_meta = _last_logits(model, tokenizer, text, args.device, args.max_length)
        target_id = int(base_logits.argmax().item())
        base_top = _top_tokens(tokenizer, base_logits, args.patch_topk)

        def execute(experiment: str, component: str,
                    attn_map: Optional[Dict[int, List[int]]] = None,
                    qk_scale: float = 1.0, vo_scale: float = 1.0,
                    mlp_ls: Optional[List[int]] = None, mlp_scale: float = 1.0,
                    atom_scale: Optional[float] = None, atom_control: str = "real",
                    component_label: str = "") -> None:
            local: List[Dict[str, Any]] = []
            handles: List[Any] = []
            try:
                if attn_map:
                    handles += _install_attention_program_patch(model, attn_map, qk_scale, vo_scale, local)
                if mlp_ls:
                    handles += _install_mlp_program_patch(model, mlp_ls, mlp_scale, atom_scale, target_id, local,
                                                          atom_control=atom_control, seed=args.seed + prompt_id)
                patched_logits, _ = _last_logits(model, tokenizer, text, args.device, args.max_length)
            finally:
                _remove_handles(handles)
            metrics = _compare_last_logits(tokenizer, base_logits, patched_logits, target_id, args.patch_topk)
            local_rels = []
            for z in local:
                for key in ("head_output_program_vs_native_rel", "group_output_program_vs_native_rel", "mlp_program_vs_native_rel"):
                    if z.get(key) is not None:
                        local_rels.append(float(z[key]))
            row = {
                "prompt_id": prompt_id,
                "suite": str(pr.get("suite", "?")),
                "text": text,
                "n_tokens": base_meta["n_tokens"],
                "experiment": experiment,
                "component": component,
                "component_label": component_label,
                "qk_scale": float(qk_scale),
                "vo_scale": float(vo_scale),
                "mlp_scale": float(mlp_scale),
                "atom_scale": atom_scale,
                "atom_control": atom_control,
                "local_component_rel_mean": safe_mean(local_rels) if local_rels else None,
                "local_component_rel_mean_percent": 100.0 * safe_mean(local_rels) if local_rels else None,
                **metrics,
            }
            rows.append(row)
            details_json.append({
                "row": row,
                "base_top_tokens": base_top,
                "patched_top_tokens": _top_tokens(tokenizer, patched_logits, args.patch_topk),
                "local_component_records": local,
            })
            print(f"  p{prompt_id:02d} {experiment:<30} {component_label:<12} "
                  f"logit_rel={row['logit_rel_percent']:.5f}% KL={row['kl_base_to_patch']:.3e} "
                  f"dlogit={row['target_logit_delta']:+.4f} top1_same={row['top1_same']}", flush=True)

        # Each head separately: exact replacement and independent QK/VO controls.
        for l, h in head_specs:
            amap = {int(l): [int(h)]}
            label = f"L{l}H{h}"
            execute("attention_exact", "attention", amap, 1.0, 1.0, component_label=label)
            for scale in qk_scales:
                if abs(scale - 1.0) < 1e-12:
                    continue
                execute(f"attention_qk_scale_{scale:g}", "attention", amap, scale, 1.0, component_label=label)
            for scale in vo_scales:
                if abs(scale - 1.0) < 1e-12:
                    continue
                execute(f"attention_vo_scale_{scale:g}", "attention", amap, 1.0, scale, component_label=label)

        # Group exact replacement proves composability across selected heads.
        if args.patch_group_tests and head_specs:
            execute("attention_group_exact", "attention", _group_heads(head_specs), 1.0, 1.0,
                    component_label="selected_heads")

        # MLP layer replacement/intervention.
        for l in mlp_layers:
            label = f"L{l}"
            execute("mlp_exact", "mlp", mlp_ls=[int(l)], mlp_scale=1.0, component_label=label)
            for scale in mlp_scales:
                if abs(scale - 1.0) < 1e-12:
                    continue
                execute(f"mlp_scale_{scale:g}", "mlp", mlp_ls=[int(l)], mlp_scale=scale, component_label=label)
            for scale in atom_scales:
                execute(f"mlp_top_atom_scale_{scale:g}", "mlp", mlp_ls=[int(l)], mlp_scale=1.0,
                        atom_scale=scale, atom_control="real", component_label=label)
                if not args.no_patch_atom_controls:
                    execute(f"mlp_random_control_scale_{scale:g}", "mlp", mlp_ls=[int(l)], mlp_scale=1.0,
                            atom_scale=scale, atom_control="random", component_label=label)
                    execute(f"mlp_unrelated_control_scale_{scale:g}", "mlp", mlp_ls=[int(l)], mlp_scale=1.0,
                            atom_scale=scale, atom_control="unrelated", component_label=label)

        if args.patch_group_tests and mlp_layers:
            execute("mlp_group_exact", "mlp", mlp_ls=mlp_layers, mlp_scale=1.0,
                    component_label="selected_layers")
        if args.patch_group_tests and head_specs and mlp_layers:
            execute("attention_mlp_combined_exact", "combined", attn_map=_group_heads(head_specs),
                    qk_scale=1.0, vo_scale=1.0, mlp_ls=mlp_layers, mlp_scale=1.0,
                    atom_scale=None, atom_control="real", component_label="selected_components")

    exact_rows = [r for r in rows if "exact" in str(r["experiment"])]
    intervention_rows = [r for r in rows if "exact" not in str(r["experiment"])]
    write_csv(pdir / "all_patching_results.csv", rows)
    write_csv(pdir / "attention_exact_replacement.csv", [r for r in exact_rows if r["component"] in ("attention", "combined")])
    write_csv(pdir / "attention_interventions.csv", [r for r in intervention_rows if r["component"] == "attention"])
    write_csv(pdir / "mlp_exact_replacement.csv", [r for r in exact_rows if r["component"] in ("mlp", "combined")])
    write_csv(pdir / "mlp_interventions.csv", [r for r in intervention_rows if r["component"] == "mlp"])
    write_json(pdir / "per_run_details.json", details_json)
    summary = _summarize_patch_rows(rows)
    top = {
        "enabled": True,
        "setup": {
            "heads": head_specs,
            "mlp_layers": mlp_layers,
            "n_prompts": len(selected_prompts),
            "qk_scales": qk_scales,
            "vo_scales": vo_scales,
            "mlp_scales": mlp_scales,
            "atom_scales": atom_scales,
            "exact_definition": "native component output is subtracted and replaced by executable weight-derived matrix-program output",
            "intervention_definition": "QK routing, VO payload, MLP layer, or selected MLP atom is scaled inside the executable program",
        },
        "summary_by_experiment": summary,
        "files": {
            "all": str(pdir / "all_patching_results.csv"),
            "details": str(pdir / "per_run_details.json"),
            "article_table": str(pdir / "article_results.md"),
        },
    }
    write_json(pdir / "summary.json", top)
    _write_patching_article_md(pdir / "article_results.md", summary, args)
    print(f"PATCHING DONE wrote {pdir}", flush=True)
    return top


# ---------------- v4 final-article global replacement / causal atlas ----------------


def final_article_prompt_bank() -> List[Dict[str, str]]:
    """Fixed 64-prompt heterogeneous bank. Targets are always the model's own baseline top-1."""
    prompts: List[Dict[str, str]] = []
    capitals = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("Italy", "Rome"), ("Spain", "Madrid"), ("Canada", "Ottawa"),
        ("Australia", "Canberra"), ("Brazil", "Brasilia"),
        ("Norway", "Oslo"), ("Poland", "Warsaw"),
        ("Ukraine", "Kyiv"), ("Portugal", "Lisbon"),
    ]
    for country, _ in capitals:
        prompts.append({"suite": "capital", "text": f"The capital of {country} is"})
    factual = [
        "Water freezes at a temperature of",
        "Water boils at sea level at",
        "The chemical symbol for oxygen is",
        "The largest planet in the Solar System is",
        "The nearest star to Earth is",
        "A year normally contains",
        "The opposite of hot is",
        "The color produced by mixing blue and yellow is",
        "A triangle has three sides, while a square has",
        "The process by which plants use light is called",
        "The primary language spoken in Brazil is",
        "The instrument used to measure temperature is a",
    ]
    prompts += [{"suite": "factual", "text": x} for x in factual]
    maths = [
        "If x = 7, then 3 * x + 2 =",
        "12 + 19 =",
        "81 divided by 9 equals",
        "The square root of 144 is",
        "If a rectangle has sides 4 and 6, its area is",
        "2 to the power of 8 is",
        "The next number in the sequence 2, 4, 8, 16 is",
        "If y - 5 = 9, then y =",
        "One half of 50 is",
        "The sum of the angles of a triangle is",
    ]
    prompts += [{"suite": "math", "text": x} for x in maths]
    code = [
        "def add(a, b):\n    return",
        "def square(x):\n    return",
        "for i in range(5):\n    print(",
        "items = [1, 2, 3]\nlength = len(",
        "if value is None:\n    value =",
        "const add = (a, b) =>",
        "function square(x) { return",
        "SELECT name FROM users WHERE id =",
        "JSON: {\"country\": \"France\", \"capital\":",
        "try:\n    result = 10 / 2\nexcept Exception as e:\n    print(",
    ]
    prompts += [{"suite": "code", "text": x} for x in code]
    multilingual = [
        ("russian", "Столица Франции — это"),
        ("russian", "Вода замерзает при температуре"),
        ("russian", "Противоположность слову горячий —"),
        ("russian", "Два плюс два равно"),
        ("ukrainian", "Столиця Франції — це"),
        ("ukrainian", "Вода замерзає за температури"),
        ("ukrainian", "Протилежне до слова гарячий —"),
        ("ukrainian", "Два плюс два дорівнює"),
        ("german", "Die Hauptstadt von Frankreich ist"),
        ("spanish", "La capital de Francia es"),
    ]
    prompts += [{"suite": s, "text": x} for s, x in multilingual]
    text_cont = [
        "Once upon a time, in a small village,",
        "The scientist examined the data and concluded that",
        "To solve the problem step by step, first",
        "The main advantage of renewable energy is",
        "Machine learning models improve when",
        "A careful explanation should include",
        "The customer opened the package and found",
        "After the rain stopped, the sky became",
        "The book begins with a description of",
        "In a democratic system, citizens usually",
    ]
    prompts += [{"suite": "continuation", "text": x} for x in text_cont]
    assert len(prompts) == 64, len(prompts)
    return prompts


def _chunks(xs: List[Any], n: int) -> List[List[Any]]:
    n = max(1, int(n))
    return [xs[i:i+n] for i in range(0, len(xs), n)]


def _safe_slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:180]


def _all_head_map(model: Any) -> Dict[int, List[int]]:
    nl = int(model.config.num_hidden_layers)
    nh = int(model.config.num_attention_heads)
    return {l: list(range(nh)) for l in range(nl)}


def _layer_head_map(model: Any, layer: int) -> Dict[int, List[int]]:
    return {int(layer): list(range(int(model.config.num_attention_heads)))}


def _install_attention_patch_v4(model: Any, layer_to_heads: Dict[int, List[int]],
                                replacement_mode: str, qk_scale: float, vo_scale: float,
                                local_records: List[Dict[str, Any]]) -> List[Any]:
    """Install program replacement or recomputed-native subtract/add control."""
    mode = str(replacement_mode).lower()
    if mode not in ("program", "native_control"):
        raise ValueError(f"unknown attention replacement_mode={replacement_mode!r}")
    handles: List[Any] = []
    for layer_idx, heads in layer_to_heads.items():
        attn = model.model.layers[int(layer_idx)].self_attn
        state: Dict[str, Any] = {}

        def pre(mod, args, kwargs, state=state):
            state["hidden_states"] = _hook_input_tensor(args, kwargs, ("hidden_states",))
            state["kwargs"] = dict(kwargs)

        def post(mod, args, output, layer_idx=layer_idx, heads=list(heads), state=state):
            if "hidden_states" not in state:
                return output
            native, program, details = attention_heads_native_and_program(
                model, int(layer_idx), heads, state["hidden_states"], state.get("kwargs", {}),
                qk_scale=float(qk_scale), vo_scale=float(vo_scale))
            first = _first_output_tensor(output)
            inserted = program if mode == "program" else native
            patched = first.float() - native + inserted
            local_records.extend(details)
            local_records.append({
                "layer": int(layer_idx), "heads": [int(h) for h in heads],
                "replacement_mode": mode,
                "group_output_program_vs_native_rel": rel_err(program, native),
                "group_output_program_vs_native_rel_percent": 100.0 * rel_err(program, native),
                "qk_scale": float(qk_scale), "vo_scale": float(vo_scale),
            })
            return _replace_first_output(output, patched.to(first.dtype))

        handles.append(_register_pre_hook_kwargs(attn, pre))
        handles.append(attn.register_forward_hook(post))
    return handles


def _install_mlp_patch_v4(model: Any, layer_indices: List[int], replacement_mode: str,
                           layer_scale: float, local_records: List[Dict[str, Any]]) -> List[Any]:
    """Install float32 matrix program or native-dtype recomputation control."""
    mode = str(replacement_mode).lower()
    if mode not in ("program", "native_control"):
        raise ValueError(f"unknown MLP replacement_mode={replacement_mode!r}")
    handles: List[Any] = []
    for layer_idx in layer_indices:
        mlp = model.model.layers[int(layer_idx)].mlp
        state: Dict[str, Any] = {}

        def pre(mod, args, kwargs, state=state):
            state["x"] = _hook_input_tensor(args, kwargs, ("hidden_states", "x"))

        def post(mod, args, output, layer_idx=layer_idx, state=state):
            if "x" not in state:
                return output
            x_native = state["x"]
            x = x_native.detach().float()
            layer = model.model.layers[int(layer_idx)]
            Wg = layer.mlp.gate_proj.weight.detach().float().to(x.device)
            Wu = layer.mlp.up_proj.weight.detach().float().to(x.device)
            Wd = layer.mlp.down_proj.weight.detach().float().to(x.device)
            program = (F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
            native = output.detach().float()
            if mode == "native_control":
                # Re-execute the module's own native-dtype chain inside the hook.
                native_recomputed = layer.mlp.down_proj(
                    F.silu(layer.mlp.gate_proj(x_native)) * layer.mlp.up_proj(x_native))
                patched = native_recomputed.float() * float(layer_scale)
                control_rel = rel_err(native_recomputed, output)
            else:
                patched = program * float(layer_scale)
                control_rel = None
            local_records.append({
                "layer": int(layer_idx), "replacement_mode": mode,
                "layer_scale": float(layer_scale),
                "mlp_program_vs_native_rel": rel_err(program, native),
                "mlp_program_vs_native_rel_percent": 100.0 * rel_err(program, native),
                "native_recompute_vs_native_rel": control_rel,
                "native_output_norm": float(native.norm().item()),
                "program_output_norm": float(program.norm().item()),
            })
            return patched.to(output.dtype)

        handles.append(_register_pre_hook_kwargs(mlp, pre))
        handles.append(mlp.register_forward_hook(post))
    return handles


def _install_component_collectors(model: Any, store: Dict[str, Dict[int, torch.Tensor]]) -> List[Any]:
    handles: List[Any] = []
    for l, layer in enumerate(model.model.layers):
        def attn_post(mod, args, output, l=l):
            store.setdefault("attn", {})[int(l)] = _first_output_tensor(output).detach().float().cpu()
        def mlp_post(mod, args, output, l=l):
            store.setdefault("mlp", {})[int(l)] = output.detach().float().cpu()
        handles.append(layer.self_attn.register_forward_hook(attn_post))
        handles.append(layer.mlp.register_forward_hook(mlp_post))
    return handles


@torch.no_grad()
def _forward_batch_capture_v4(model: Any, tokenizer: Any, prompt_batch: List[Dict[str, str]],
                              device: str, max_length: int, capture_components: bool = True
                              ) -> Dict[str, Any]:
    texts = [str(p["text"]) for p in prompt_batch]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=int(max_length)).to(device)
    tokenizer.padding_side = old_side
    collector: Dict[str, Dict[int, torch.Tensor]] = {}
    handles = _install_component_collectors(model, collector) if capture_components else []
    try:
        out = model(**enc, use_cache=False, output_hidden_states=True)
    finally:
        _remove_handles(handles)
    mask = enc["attention_mask"].detach().cpu().bool()
    pos = torch.arange(mask.shape[1]).view(1, -1)
    last_idx = (mask.long() * pos).max(dim=1).values.long()
    logits_all = out.logits.detach().float().cpu()
    batch_idx = torch.arange(logits_all.shape[0])
    last_logits = logits_all[batch_idx, last_idx]
    hidden = [h.detach().float().cpu() for h in out.hidden_states]
    return {
        "prompts": prompt_batch,
        "texts": texts,
        "input_ids": enc["input_ids"].detach().cpu(),
        "attention_mask": mask,
        "last_idx": last_idx,
        "last_logits": last_logits,
        "hidden_states": hidden,
        "attn_outputs": collector.get("attn", {}),
        "mlp_outputs": collector.get("mlp", {}),
    }


def _per_example_logit_rows_v4(tokenizer: Any, base: torch.Tensor, patched: torch.Tensor,
                                prompts: List[Dict[str, str]], prompt_offset: int,
                                topk: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for b in range(base.shape[0]):
        target = int(base[b].argmax().item())
        m = _compare_last_logits(tokenizer, base[b], patched[b], target, topk)
        rows.append({
            "prompt_id": int(prompt_offset + b),
            "suite": str(prompts[b].get("suite", "?")),
            "text": str(prompts[b]["text"]),
            **m,
        })
    return rows


def _masked_delta_sums(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor,
                       last_idx: torch.Tensor) -> Dict[str, Any]:
    a = a.float(); b = b.float(); mask = mask.bool()
    diff = a - b
    valid = mask.unsqueeze(-1).expand_as(diff)
    num2 = float(diff[valid].pow(2).sum().item())
    den2 = float(b[valid].pow(2).sum().item())
    last_rels: List[float] = []
    for i in range(a.shape[0]):
        p = int(last_idx[i].item())
        last_rels.append(rel_err(a[i, p], b[i, p]))
    return {"num2": num2, "den2": den2, "last_rels": last_rels}


def _propagation_batch_v4(base: Dict[str, Any], patched: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    mask = base["attention_mask"]
    last = base["last_idx"]
    for idx, (bp, pp) in enumerate(zip(base["hidden_states"], patched["hidden_states"])):
        st = _masked_delta_sums(pp, bp, mask, last)
        out.append({"target_type": "residual", "target_index": int(idx), **st})
    for typ, key in (("attention_output", "attn_outputs"), ("mlp_output", "mlp_outputs")):
        common = sorted(set(base[key]) & set(patched[key]))
        for l in common:
            st = _masked_delta_sums(patched[key][l], base[key][l], mask, last)
            out.append({"target_type": typ, "target_index": int(l), **st})
    return out


def _aggregate_propagation_v4(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in rows:
        k = (str(r["target_type"]), int(r["target_index"]))
        g = groups.setdefault(k, {"num2": 0.0, "den2": 0.0, "last_rels": []})
        g["num2"] += float(r["num2"]); g["den2"] += float(r["den2"])
        g["last_rels"].extend(float(x) for x in r["last_rels"])
    out: List[Dict[str, Any]] = []
    for (typ, idx), g in sorted(groups.items()):
        rel = math.sqrt(g["num2"] / max(g["den2"], 1e-30))
        st = _metric_stats(g["last_rels"])
        out.append({
            "target_type": typ, "target_index": idx,
            "valid_tensor_rel": rel, "valid_tensor_rel_percent": 100.0 * rel,
            "last_token_rel_median": st.get("median"),
            "last_token_rel_p95": st.get("p95"),
            "last_token_rel_max": st.get("max"),
        })
    return out


def _summarize_prompt_rows_v4(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n": len(rows),
        "logit_rel_percent": _metric_stats([float(r["logit_rel_percent"]) for r in rows]),
        "kl_base_to_patch": _metric_stats([max(0.0, float(r["kl_base_to_patch"])) for r in rows]),
        "js_divergence": _metric_stats([max(0.0, float(r["js_divergence"])) for r in rows]),
        "abs_target_logit_delta": _metric_stats([abs(float(r["target_logit_delta"])) for r in rows]),
        "top1_preservation_rate": safe_mean([1.0 if r["top1_same"] else 0.0 for r in rows]),
        "top1_change_rate": safe_mean([0.0 if r["top1_same"] else 1.0 for r in rows]),
        "topk_overlap": _metric_stats([float(r["topk_overlap"]) for r in rows]),
        "logit_cosine": _metric_stats([float(r["logit_cosine"]) for r in rows]),
    }


def _run_v4_experiment(model: Any, tokenizer: Any, baseline_batches: List[Dict[str, Any]],
                       experiment: Dict[str, Any], args: argparse.Namespace,
                       exp_dir: Path) -> Dict[str, Any]:
    slug = _safe_slug(str(experiment["name"]))
    path = exp_dir / f"{slug}.json"
    if args.final_resume and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    local: List[Dict[str, Any]] = []
    handles: List[Any] = []
    attn_map = experiment.get("attn_map")
    mlp_layers = experiment.get("mlp_layers")
    try:
        if attn_map:
            handles += _install_attention_patch_v4(
                model, attn_map, experiment.get("attn_mode", "program"),
                float(experiment.get("qk_scale", 1.0)), float(experiment.get("vo_scale", 1.0)), local)
        if mlp_layers:
            handles += _install_mlp_patch_v4(
                model, list(mlp_layers), experiment.get("mlp_mode", "program"),
                float(experiment.get("mlp_scale", 1.0)), local)
        prompt_rows: List[Dict[str, Any]] = []
        prop_parts: List[Dict[str, Any]] = []
        offset = 0
        for base in baseline_batches:
            patched = _forward_batch_capture_v4(
                model, tokenizer, base["prompts"], args.device, args.max_length,
                capture_components=bool(args.final_capture_propagation))
            prompt_rows.extend(_per_example_logit_rows_v4(
                tokenizer, base["last_logits"], patched["last_logits"],
                base["prompts"], offset, args.patch_topk))
            if args.final_capture_propagation:
                prop_parts.extend(_propagation_batch_v4(base, patched))
            offset += len(base["prompts"])
    finally:
        _remove_handles(handles)
    local_rels = []
    for z in local:
        for key in ("head_output_program_vs_native_rel", "group_output_program_vs_native_rel", "mlp_program_vs_native_rel"):
            if z.get(key) is not None:
                local_rels.append(float(z[key]))
    result = {
        "experiment": {k: v for k, v in experiment.items() if k not in ("attn_map",)},
        "attn_map": attn_map,
        "summary": _summarize_prompt_rows_v4(prompt_rows),
        "local_component_rel": _metric_stats(local_rels),
        "prompt_rows": prompt_rows,
        "propagation": _aggregate_propagation_v4(prop_parts) if prop_parts else [],
    }
    write_json(path, result)
    s = result["summary"]
    print(f"[{experiment['name']}] N={s['n']} median={s['logit_rel_percent'].get('median',0):.6g}% "
          f"p95={s['logit_rel_percent'].get('p95',0):.6g}% top1_keep={100*s['top1_preservation_rate']:.2f}%", flush=True)
    return result


def _build_final_experiments_v4(model: Any, args: argparse.Namespace) -> List[Dict[str, Any]]:
    nl = int(model.config.num_hidden_layers)
    nh = int(model.config.num_attention_heads)
    all_heads = _all_head_map(model)
    all_layers = list(range(nl))
    exps: List[Dict[str, Any]] = []
    if not args.final_skip_global:
        exps += [
            {"name":"global_attention_native_control", "category":"global_exact", "attn_map":all_heads, "attn_mode":"native_control"},
            {"name":"global_attention_program", "category":"global_exact", "attn_map":all_heads, "attn_mode":"program"},
            {"name":"global_mlp_native_control", "category":"global_exact", "mlp_layers":all_layers, "mlp_mode":"native_control"},
            {"name":"global_mlp_program", "category":"global_exact", "mlp_layers":all_layers, "mlp_mode":"program"},
            {"name":"global_backbone_native_control", "category":"global_exact", "attn_map":all_heads, "attn_mode":"native_control", "mlp_layers":all_layers, "mlp_mode":"native_control"},
            {"name":"global_backbone_program", "category":"global_exact", "attn_map":all_heads, "attn_mode":"program", "mlp_layers":all_layers, "mlp_mode":"program"},
        ]
    if not args.final_skip_layer_groups:
        for l in range(nl):
            hm = {l: list(range(nh))}
            exps += [
                {"name":f"layer_{l:02d}_attention_native_control", "category":"layer_exact", "source_layer":l, "attn_map":hm, "attn_mode":"native_control"},
                {"name":f"layer_{l:02d}_attention_program", "category":"layer_exact", "source_layer":l, "attn_map":hm, "attn_mode":"program"},
                {"name":f"layer_{l:02d}_mlp_native_control", "category":"layer_exact", "source_layer":l, "mlp_layers":[l], "mlp_mode":"native_control"},
                {"name":f"layer_{l:02d}_mlp_program", "category":"layer_exact", "source_layer":l, "mlp_layers":[l], "mlp_mode":"program"},
                {"name":f"layer_{l:02d}_full_native_control", "category":"layer_exact", "source_layer":l, "attn_map":hm, "attn_mode":"native_control", "mlp_layers":[l], "mlp_mode":"native_control"},
                {"name":f"layer_{l:02d}_full_program", "category":"layer_exact", "source_layer":l, "attn_map":hm, "attn_mode":"program", "mlp_layers":[l], "mlp_mode":"program"},
                {"name":f"layer_{l:02d}_qk_off_all_heads", "category":"layer_intervention", "source_layer":l, "attn_map":hm, "attn_mode":"program", "qk_scale":0.0, "vo_scale":1.0},
                {"name":f"layer_{l:02d}_vo_off_all_heads", "category":"layer_intervention", "source_layer":l, "attn_map":hm, "attn_mode":"program", "qk_scale":1.0, "vo_scale":0.0},
                {"name":f"layer_{l:02d}_mlp_off", "category":"layer_intervention", "source_layer":l, "mlp_layers":[l], "mlp_mode":"program", "mlp_scale":0.0},
                {"name":f"layer_{l:02d}_full_off", "category":"layer_intervention", "source_layer":l, "attn_map":hm, "attn_mode":"program", "qk_scale":1.0, "vo_scale":0.0, "mlp_layers":[l], "mlp_mode":"program", "mlp_scale":0.0},
            ]
    if not args.final_skip_head_atlas:
        for l in range(nl):
            for h in range(nh):
                hm = {l:[h]}
                exps += [
                    {"name":f"head_L{l:02d}H{h:02d}_native_control", "category":"head_exact", "source_layer":l, "source_head":h, "attn_map":hm, "attn_mode":"native_control"},
                    {"name":f"head_L{l:02d}H{h:02d}_program", "category":"head_exact", "source_layer":l, "source_head":h, "attn_map":hm, "attn_mode":"program"},
                    {"name":f"head_L{l:02d}H{h:02d}_qk_off", "category":"head_intervention", "source_layer":l, "source_head":h, "attn_map":hm, "attn_mode":"program", "qk_scale":0.0, "vo_scale":1.0},
                    {"name":f"head_L{l:02d}H{h:02d}_vo_off", "category":"head_intervention", "source_layer":l, "source_head":h, "attn_map":hm, "attn_mode":"program", "qk_scale":1.0, "vo_scale":0.0},
                ]
    if not args.final_skip_mlp_atlas:
        for l in range(nl):
            exps += [
                {"name":f"mlp_L{l:02d}_native_control", "category":"mlp_exact", "source_layer":l, "mlp_layers":[l], "mlp_mode":"native_control"},
                {"name":f"mlp_L{l:02d}_program", "category":"mlp_exact", "source_layer":l, "mlp_layers":[l], "mlp_mode":"program"},
                {"name":f"mlp_L{l:02d}_off", "category":"mlp_intervention", "source_layer":l, "mlp_layers":[l], "mlp_mode":"program", "mlp_scale":0.0},
            ]
    return exps


def _prop_features_v4(result: Dict[str, Any], source_layer: Optional[int]) -> Dict[str, Any]:
    if source_layer is None:
        return {}
    rr = [r for r in result.get("propagation", []) if r["target_type"] == "residual" and int(r["target_index"]) >= int(source_layer)+1]
    if not rr:
        return {}
    vals = [(int(r["target_index"]), float(r["valid_tensor_rel_percent"])) for r in rr]
    peak_idx, peak_val = max(vals, key=lambda x: x[1])
    first = next((v for i, v in vals if i == int(source_layer)+1), vals[0][1])
    final_idx, final_val = vals[-1]
    return {
        "residual_first_percent": first,
        "residual_peak_percent": peak_val,
        "residual_peak_index": peak_idx,
        "residual_final_percent": final_val,
        "downstream_amplification": peak_val / max(first, 1e-12),
        "final_retention_from_peak": final_val / max(peak_val, 1e-12),
    }


def _load_all_experiment_json_v4(exp_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for p in sorted(exp_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"warning: cannot read {p}: {e}", file=sys.stderr)
    return out


def _index_results_v4(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["experiment"]["name"]): r for r in results}


def _median_metric_v4(r: Optional[Dict[str, Any]], metric: str="logit_rel_percent") -> Optional[float]:
    if not r:
        return None
    return r.get("summary", {}).get(metric, {}).get("median")


def _build_atlas_tables_v4(model: Any, results: List[Dict[str, Any]], final_dir: Path) -> Dict[str, Any]:
    idx = _index_results_v4(results)
    nl = int(model.config.num_hidden_layers); nh = int(model.config.num_attention_heads)
    head_rows: List[Dict[str, Any]] = []
    for l in range(nl):
        for h in range(nh):
            native = idx.get(f"head_L{l:02d}H{h:02d}_native_control")
            exact = idx.get(f"head_L{l:02d}H{h:02d}_program")
            qk = idx.get(f"head_L{l:02d}H{h:02d}_qk_off")
            vo = idx.get(f"head_L{l:02d}H{h:02d}_vo_off")
            replacement_discrepancy = max(_median_metric_v4(native) or 0.0, _median_metric_v4(exact) or 0.0, 1e-12)
            row = {
                "layer": l, "head": h,
                "native_control_logit_rel_percent": _median_metric_v4(native),
                "program_logit_rel_percent": _median_metric_v4(exact),
                "program_excess_over_control_percent": None if not exact else max(0.0, (_median_metric_v4(exact) or 0)-(_median_metric_v4(native) or 0)),
                "qk_off_logit_rel_percent": _median_metric_v4(qk),
                "vo_off_logit_rel_percent": _median_metric_v4(vo),
                "qk_effect_to_replacement_discrepancy_ratio": None if not qk else (_median_metric_v4(qk) or 0)/replacement_discrepancy,
                "vo_effect_to_replacement_discrepancy_ratio": None if not vo else (_median_metric_v4(vo) or 0)/replacement_discrepancy,
                "qk_top1_change_rate": None if not qk else qk["summary"]["top1_change_rate"],
                "vo_top1_change_rate": None if not vo else vo["summary"]["top1_change_rate"],
                **{f"qk_{k}":v for k,v in _prop_features_v4(qk,l).items()},
                **{f"vo_{k}":v for k,v in _prop_features_v4(vo,l).items()},
            }
            row["qk_internal_peak_to_final_logit_ratio"] = (float(row.get("qk_residual_peak_percent") or 0.0) / max(float(row.get("qk_off_logit_rel_percent") or 0.0), 1e-12))
            row["vo_internal_peak_to_final_logit_ratio"] = (float(row.get("vo_residual_peak_percent") or 0.0) / max(float(row.get("vo_off_logit_rel_percent") or 0.0), 1e-12))
            row["qk_delayed_peak_layers"] = None if row.get("qk_residual_peak_index") is None else int(row["qk_residual_peak_index"]) - (l + 1)
            row["vo_delayed_peak_layers"] = None if row.get("vo_residual_peak_index") is None else int(row["vo_residual_peak_index"]) - (l + 1)
            head_rows.append(row)
    mlp_rows: List[Dict[str, Any]] = []
    for l in range(nl):
        native = idx.get(f"mlp_L{l:02d}_native_control")
        exact = idx.get(f"mlp_L{l:02d}_program")
        off = idx.get(f"mlp_L{l:02d}_off")
        replacement_discrepancy = max(_median_metric_v4(native) or 0.0, _median_metric_v4(exact) or 0.0, 1e-12)
        mr = {
            "layer": l,
            "native_control_logit_rel_percent": _median_metric_v4(native),
            "program_logit_rel_percent": _median_metric_v4(exact),
            "program_excess_over_control_percent": None if not exact else max(0.0, (_median_metric_v4(exact) or 0)-(_median_metric_v4(native) or 0)),
            "off_logit_rel_percent": _median_metric_v4(off),
            "off_effect_to_replacement_discrepancy_ratio": None if not off else (_median_metric_v4(off) or 0)/replacement_discrepancy,
            "off_top1_change_rate": None if not off else off["summary"]["top1_change_rate"],
            **_prop_features_v4(off,l),
        }
        mr["internal_peak_to_final_logit_ratio"] = float(mr.get("residual_peak_percent") or 0.0) / max(float(mr.get("off_logit_rel_percent") or 0.0), 1e-12)
        mr["delayed_peak_layers"] = None if mr.get("residual_peak_index") is None else int(mr["residual_peak_index"]) - (l + 1)
        mlp_rows.append(mr)
    layer_rows: List[Dict[str, Any]] = []
    for l in range(nl):
        row: Dict[str, Any] = {"layer":l}
        for suffix in ("attention_program","mlp_program","full_program","qk_off_all_heads","vo_off_all_heads","mlp_off","full_off"):
            r = idx.get(f"layer_{l:02d}_{suffix}")
            row[f"{suffix}_logit_rel_percent"] = _median_metric_v4(r)
            if r and "off" in suffix:
                row[f"{suffix}_top1_change_rate"] = r["summary"]["top1_change_rate"]
                pf = _prop_features_v4(r,l)
                row[f"{suffix}_peak_residual_percent"] = pf.get("residual_peak_percent")
                row[f"{suffix}_peak_residual_index"] = pf.get("residual_peak_index")
                row[f"{suffix}_final_residual_percent"] = pf.get("residual_final_percent")
        layer_rows.append(row)
    write_csv(final_dir / "head_causal_atlas.csv", head_rows)
    write_csv(final_dir / "mlp_causal_atlas.csv", mlp_rows)
    write_csv(final_dir / "layer_group_atlas.csv", layer_rows)
    return {"heads": head_rows, "mlp": mlp_rows, "layers": layer_rows}


def _capture_mlp_inputs_v4(model: Any, tokenizer: Any, prompt_batch: List[Dict[str,str]],
                           device: str, max_length: int) -> Dict[str, Any]:
    store: Dict[int, torch.Tensor] = {}
    handles: List[Any] = []
    for l, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, l=l):
            store[int(l)] = _hook_input_tensor(args, kwargs, ("hidden_states","x")).detach().float().cpu()
        handles.append(_register_pre_hook_kwargs(layer.mlp, pre))
    try:
        base = _forward_batch_capture_v4(model, tokenizer, prompt_batch, device, max_length, capture_components=False)
    finally:
        _remove_handles(handles)
    base["mlp_inputs"] = store
    return base


def _select_atoms_heldout_v4(model: Any, tokenizer: Any, selection_prompts: List[Dict[str,str]],
                             args: argparse.Namespace) -> List[Dict[str, Any]]:
    batches = _chunks(selection_prompts, args.final_batch_size)
    captures = [_capture_mlp_inputs_v4(model, tokenizer, b, args.device, args.max_length) for b in batches]
    nl = int(model.config.num_hidden_layers)
    selected: List[Dict[str, Any]] = []
    for l in range(nl):
        layer = model.model.layers[l]
        Wg = layer.mlp.gate_proj.weight.detach().float().to(args.device)
        Wu = layer.mlp.up_proj.weight.detach().float().to(args.device)
        Wd = layer.mlp.down_proj.weight.detach().float().to(args.device)
        score_sum = torch.zeros(Wg.shape[0], device=args.device)
        norm_sum = torch.zeros_like(score_sum)
        n = 0
        for cap in captures:
            xall = cap["mlp_inputs"][l].to(args.device)
            li = cap["last_idx"].to(args.device)
            bi = torch.arange(xall.shape[0], device=args.device)
            x = xall[bi, li]
            logits = cap["last_logits"].to(args.device)
            targets = logits.argmax(dim=-1)
            target_vec = model.lm_head.weight.detach().float().to(args.device)[targets]
            a = F.silu(x @ Wg.T) * (x @ Wu.T)
            wt = target_vec @ Wd
            contrib = a * wt
            score_sum += contrib.abs().sum(dim=0)
            norm_sum += (a.abs() * Wd.norm(dim=0).view(1,-1)).sum(dim=0)
            n += x.shape[0]
        mean_score = score_sum / max(1,n)
        mean_norm = norm_sum / max(1,n)
        j = int(mean_score.argmax().item())
        diff = (mean_norm - mean_norm[j]).abs(); diff[j] = torch.finfo(diff.dtype).max
        ju = int(diff.argmin().item())
        selected.append({"layer":l,"atom":j,"unrelated_atom":ju,
                         "selection_mean_abs_target_contribution":float(mean_score[j].item()),
                         "selection_mean_output_norm":float(mean_norm[j].item())})
    return selected


def _install_fixed_atom_patch_v4(model: Any, layer_idx: int, atom_idx: int,
                                 control: str, unrelated_idx: Optional[int],
                                 last_positions: torch.Tensor, seed: int,
                                 local_records: List[Dict[str,Any]]) -> List[Any]:
    mlp = model.model.layers[int(layer_idx)].mlp
    state: Dict[str, Any] = {}
    control = str(control)
    def pre(mod,args,kwargs):
        state["x"] = _hook_input_tensor(args,kwargs,("hidden_states","x"))
    def post(mod,args,output):
        x = state["x"].detach().float()
        layer = model.model.layers[int(layer_idx)]
        Wg = layer.mlp.gate_proj.weight.detach().float().to(x.device)
        Wu = layer.mlp.up_proj.weight.detach().float().to(x.device)
        Wd = layer.mlp.down_proj.weight.detach().float().to(x.device)
        a = F.silu(x@Wg.T)*(x@Wu.T)
        program = a@Wd.T
        patched = program.clone()
        pos = last_positions.to(x.device)
        for b in range(x.shape[0]):
            p = int(pos[b].item())
            real = a[b,p,int(atom_idx)] * Wd[:,int(atom_idx)]
            vec = real
            if control == "random":
                gen = torch.Generator(device="cpu").manual_seed(int(seed)+10007*b+997*int(layer_idx)+int(atom_idx))
                rv = torch.randn(real.numel(),generator=gen).to(x.device)
                rv = rv/rv.norm().clamp_min(1e-12)*real.norm().clamp_min(1e-12)
                vec = rv
            elif control == "unrelated":
                ju = int(unrelated_idx)
                uv = a[b,p,ju]*Wd[:,ju]
                vec = uv/uv.norm().clamp_min(1e-12)*real.norm().clamp_min(1e-12)
            elif control != "real":
                raise ValueError(control)
            patched[b,p] -= vec
        local_records.append({"layer":int(layer_idx),"atom":int(atom_idx),"control":control,
                              "mlp_program_vs_native_rel":rel_err(program,output.detach().float())})
        return patched.to(output.dtype)
    return [_register_pre_hook_kwargs(mlp,pre), mlp.register_forward_hook(post)]


def _run_atom_split_v4(model: Any, tokenizer: Any, prompts: List[Dict[str,str]],
                       args: argparse.Namespace, final_dir: Path) -> Dict[str,Any]:
    if args.final_skip_atom_split:
        return {"enabled":False}
    nsel = min(max(1,int(args.final_atom_selection_prompts)), len(prompts)//2)
    order = list(range(len(prompts)))
    random.Random(int(args.seed) + 4242).shuffle(order)
    selection = [prompts[i] for i in order[:nsel]]
    evaluation = [prompts[i] for i in order[nsel:]]
    if args.final_atom_eval_prompts > 0:
        evaluation = evaluation[:int(args.final_atom_eval_prompts)]
    selected = _select_atoms_heldout_v4(model,tokenizer,selection,args)
    write_json(final_dir/"atom_selection.json", {"selection_prompts":selection,"selected":selected})
    eval_batches = _chunks(evaluation,args.final_batch_size)
    base_batches = [_forward_batch_capture_v4(model,tokenizer,b,args.device,args.max_length,False) for b in eval_batches]
    rows: List[Dict[str,Any]] = []
    for sel in selected:
        l=int(sel["layer"]); j=int(sel["atom"]); ju=int(sel["unrelated_atom"])
        controls = ["real","unrelated"] + [f"random_{r}" for r in range(int(args.final_atom_random_controls))]
        for ctrl_index, ctrl_name in enumerate(controls):
            ctrl = "random" if ctrl_name.startswith("random_") else ctrl_name
            # Different random_N controls must receive independent directions.
            # v4 accidentally reused the same seed for random_0/random_1/random_2.
            random_seed_offset = 1000003 * int(ctrl_index) if ctrl == "random" else 0
            offset=0
            for bi,base in enumerate(base_batches):
                local=[]
                handles=_install_fixed_atom_patch_v4(
                    model,l,j,ctrl,ju,base["last_idx"],
                    args.seed+bi+1000*l+random_seed_offset,local)
                try:
                    patched=_forward_batch_capture_v4(model,tokenizer,base["prompts"],args.device,args.max_length,False)
                finally:
                    _remove_handles(handles)
                pr=_per_example_logit_rows_v4(tokenizer,base["last_logits"],patched["last_logits"],base["prompts"],offset,args.patch_topk)
                for r in pr:
                    rows.append({"layer":l,"atom":j,"unrelated_atom":ju,"control":ctrl_name,
                                 "selection_mean_abs_target_contribution":sel["selection_mean_abs_target_contribution"],**r})
                offset += len(base["prompts"])
    write_csv(final_dir/"atom_heldout_prompt_results.csv",rows)
    summary_rows=[]
    groups: Dict[Tuple[int,str],List[Dict[str,Any]]]={}
    for r in rows: groups.setdefault((int(r["layer"]),str(r["control"])),[]).append(r)
    for (l,c),rr in sorted(groups.items()):
        summary_rows.append({"layer":l,"control":c,"n":len(rr),
                             "logit_rel_percent_median":_metric_stats([float(x["logit_rel_percent"]) for x in rr]).get("median"),
                             "abs_target_logit_delta_median":_metric_stats([abs(float(x["target_logit_delta"])) for x in rr]).get("median"),
                             "kl_median":_metric_stats([max(0,float(x["kl_base_to_patch"])) for x in rr]).get("median"),
                             "top1_change_rate":safe_mean([0 if x["top1_same"] else 1 for x in rr])})
    write_csv(final_dir/"atom_heldout_summary.csv",summary_rows)
    return {"enabled":True,"selection_n":len(selection),"evaluation_n":len(evaluation),"selected":selected,"summary":summary_rows}


def _write_final_article_report_v4(final_dir: Path, results: List[Dict[str,Any]],
                                   atlas: Dict[str,Any], atom: Dict[str,Any], args: argparse.Namespace) -> None:
    idx=_index_results_v4(results)
    lines=["# Final article experiment results\n\n"]
    lines.append("## Global executable replacement\n\n")
    lines.append("| Experiment | N | logit rel median % | p95 % | max % | KL median | top-1 preserved |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for name in ("global_attention_native_control","global_attention_program","global_mlp_native_control","global_mlp_program","global_backbone_native_control","global_backbone_program"):
        r=idx.get(name)
        if not r: continue
        s=r["summary"]; lr=s["logit_rel_percent"]; kl=s["kl_base_to_patch"]
        lines.append(f"| {name} | {s['n']} | {lr.get('median',0):.6g} | {lr.get('p95',0):.6g} | {lr.get('max',0):.6g} | {kl.get('median',0):.6g} | {100*s['top1_preservation_rate']:.2f}% |\n")
    if atlas.get("heads"):
        h=atlas["heads"]
        qkr=[x["qk_effect_to_replacement_discrepancy_ratio"] for x in h if x.get("qk_effect_to_replacement_discrepancy_ratio") is not None]
        vor=[x["vo_effect_to_replacement_discrepancy_ratio"] for x in h if x.get("vo_effect_to_replacement_discrepancy_ratio") is not None]
        lines.append("\n## Full head atlas aggregate\n\n")
        lines.append(f"- Heads evaluated: {len(h)}\n")
        lines.append(f"- QK-off / replacement-discrepancy ratio median: {_conventional_median_v4(qkr):.4g}×\n")
        lines.append(f"- VO-off / replacement-discrepancy ratio median: {_conventional_median_v4(vor):.4g}×\n")
        prep=sorted(h,key=lambda x:max(float(x.get('qk_residual_peak_percent') or 0),float(x.get('vo_residual_peak_percent') or 0)),reverse=True)[:10]
        lines.append("\nTop internal-propagation heads are in `head_causal_atlas.csv`.\n")
    if atlas.get("mlp"):
        m=atlas["mlp"]
        ratios=[x["off_effect_to_replacement_discrepancy_ratio"] for x in m if x.get("off_effect_to_replacement_discrepancy_ratio") is not None]
        lines.append("\n## MLP atlas aggregate\n\n")
        lines.append(f"- MLP layers evaluated: {len(m)}\n")
        lines.append(f"- MLP-off / replacement-discrepancy ratio median: {_conventional_median_v4(ratios):.4g}×\n")
    lines.append("\n## Held-out atom experiment\n\n")
    lines.append(f"Enabled: {atom.get('enabled',False)}; selection prompts={atom.get('selection_n',0)}; evaluation prompts={atom.get('evaluation_n',0)}.\n")
    lines.append("\n## Configuration\n\n```json\n")
    lines.append(json.dumps(json_sanitize({k:v for k,v in vars(args).items() if k.startswith('final_')}),ensure_ascii=False,indent=2))
    lines.append("\n```\n")
    (final_dir/"final_article_results.md").write_text("".join(lines),encoding="utf-8")


def run_final_article_suite(model: Any, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> Dict[str,Any]:
    final_dir=out_dir/"final_article"
    exp_dir=final_dir/"experiments"
    ensure_dir(exp_dir)
    bank=final_article_prompt_bank()[:max(1,int(args.final_prompts))]
    write_json(final_dir/"final_prompts.json",bank)
    batches=_chunks(bank,int(args.final_batch_size))
    print(f"\n=== FINAL ARTICLE SUITE prompts={len(bank)} batches={len(batches)} ===",flush=True)
    baseline=[]
    for i,b in enumerate(batches):
        print(f"baseline batch {i+1}/{len(batches)} size={len(b)}",flush=True)
        baseline.append(_forward_batch_capture_v4(model,tokenizer,b,args.device,args.max_length,
                                                   capture_components=bool(args.final_capture_propagation)))
    experiments=_build_final_experiments_v4(model,args)
    write_json(final_dir/"experiment_manifest.json",[{k:v for k,v in e.items() if k!='attn_map'}|({"attn_map":e.get("attn_map")} if e.get("attn_map") else {}) for e in experiments])
    print(f"experiments={len(experiments)} resume={args.final_resume}",flush=True)
    for i,e in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {e['name']}",flush=True)
        _run_v4_experiment(model,tokenizer,baseline,e,args,exp_dir)
    results=_load_all_experiment_json_v4(exp_dir)
    # Flatten core reports.
    exp_rows=[]; prompt_rows=[]; prop_rows=[]
    for r in results:
        e=r["experiment"]; s=r["summary"]
        exp_rows.append({"name":e.get("name"),"category":e.get("category"),"source_layer":e.get("source_layer"),"source_head":e.get("source_head"),
                         "n":s.get("n"),"logit_rel_percent_median":s["logit_rel_percent"].get("median"),
                         "logit_rel_percent_p95":s["logit_rel_percent"].get("p95"),"logit_rel_percent_max":s["logit_rel_percent"].get("max"),
                         "kl_median":s["kl_base_to_patch"].get("median"),"abs_target_logit_delta_median":s["abs_target_logit_delta"].get("median"),
                         "top1_preservation_rate":s.get("top1_preservation_rate"),"top1_change_rate":s.get("top1_change_rate")})
        for x in r.get("prompt_rows",[]): prompt_rows.append({"experiment":e.get("name"),"category":e.get("category"),**x})
        for x in r.get("propagation",[]): prop_rows.append({"experiment":e.get("name"),"category":e.get("category"),"source_layer":e.get("source_layer"),"source_head":e.get("source_head"),**x})
    write_csv(final_dir/"all_experiment_summaries.csv",exp_rows)
    write_csv(final_dir/"all_prompt_metrics.csv",prompt_rows)
    write_csv(final_dir/"propagation_profiles.csv",prop_rows)
    atlas=_build_atlas_tables_v4(model,results,final_dir)
    atom=_run_atom_split_v4(model,tokenizer,bank,args,final_dir)
    summary={"enabled":True,"n_prompts":len(bank),"n_experiments":len(results),"atlas_files":{
        "heads":str(final_dir/"head_causal_atlas.csv"),"mlp":str(final_dir/"mlp_causal_atlas.csv"),"layers":str(final_dir/"layer_group_atlas.csv")},
        "atom_split":atom,"args":{k:v for k,v in vars(args).items() if k.startswith("final_")}}
    write_json(final_dir/"summary.json",summary)
    _write_final_article_report_v4(final_dir,results,atlas,atom,args)
    print(f"FINAL ARTICLE SUITE DONE: {final_dir}",flush=True)
    return summary


# ---------------- main analysis ----------------

def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    base_mod = load_base_module(args.base_script) if args.base_script else None
    set_seed(args.seed)
    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    print("loading model...", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required; use --self-test for synthetic check") from e

    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {"dtype": dtype, "trust_remote_code": True}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(args.device)
    except TypeError:
        model_kwargs.pop("dtype", None)
        model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(args.device)
    model.eval()

    cfg = model.config
    head_specs = parse_heads(args.heads, int(cfg.num_hidden_layers), int(cfg.num_attention_heads))
    prompts = build_prompts(base_mod, args.prompt_suites, args.prompts_per_suite, args.same_text_repeats)
    write_json(out_dir / "prompts.json", prompts)
    print(f"heads={head_specs} prompts={len(prompts)}", flush=True)

    all_summaries = []
    mlp_summary = None
    patching_summary = None
    extraction_head_specs = [] if (args.patch_only or args.final_article_suite) else head_specs
    for layer_idx, head_idx in extraction_head_specs:
        hdir = out_dir / f"L{layer_idx}H{head_idx}"; ensure_dir(hdir)
        print(f"\n=== affine circuit targets L{layer_idx}H{head_idx} ===", flush=True)
        t0 = time.perf_counter()
        seqs, meta = collect_circuit_data(model, tok, prompts, layer_idx, head_idx, args.max_length, args.device)
        if not seqs:
            raise RuntimeError("no sequences collected")
        print(f"collected {len(seqs)} seqs in {time.perf_counter()-t0:.1f}s", flush=True)
        weights = build_weight_slices(model, layer_idx, head_idx, meta)
        max_observed_T = max(int(s.Xn.shape[0]) for s in seqs)
        max_delta = min(int(args.max_delta), max_observed_T - 1)
        Rpos = build_rope_by_pos(seqs, max_delta)
        Mdelta_aug = qk_delta_matrices_affine(weights, Rpos, max_delta, int(meta["head_dim"]))
        Mdelta_lin = qk_delta_matrices_linear(weights, Rpos, max_delta, int(meta["head_dim"]))
        verify, per_prompt = verify_targets(seqs, weights, Mdelta_aug, Mdelta_lin, max_delta)
        rope_inv = rope_delta_invariance(seqs, Rpos, max_delta=min(max_delta, 16))

        Cvo_lin = weights["Wo"] @ weights["Wv"]
        Cvo_aug = weights["Wo"] @ weights["Wv_aug"]
        ranks = parse_int_list(args.svd_ranks)
        mat_reports: Dict[str, Any] = {
            "C_vo_linear_HxH": {"stats": matrix_shape_stats(Cvo_lin), "svd": svd_rank_table(Cvo_lin, ranks)},
            "C_vo_aug_HxHplus1": {"stats": matrix_shape_stats(Cvo_aug), "svd": svd_rank_table(Cvo_aug, ranks)},
        }
        for d in sorted(Mdelta_aug.keys())[: min(len(Mdelta_aug), int(args.report_deltas))]:
            mat_reports[f"M_qk_aug_delta_{d}"] = {"stats": matrix_shape_stats(Mdelta_aug[d]), "svd": svd_rank_table(Mdelta_aug[d], ranks)}
            mat_reports[f"M_qk_linear_no_bias_delta_{d}"] = {"stats": matrix_shape_stats(Mdelta_lin[d]), "svd": svd_rank_table(Mdelta_lin[d], ranks)}
        if args.decode_generic_ops:
            mat_reports["C_vo_linear_HxH"]["generic_ops_fit"] = fit_light_ops(Cvo_lin, ridge=args.generic_ridge, max_terms=args.generic_max_terms)
            for d in sorted(Mdelta_lin.keys())[: min(len(Mdelta_lin), int(args.report_deltas))]:
                mat_reports[f"M_qk_linear_no_bias_delta_{d}"]["generic_ops_fit"] = fit_light_ops(Mdelta_lin[d], ridge=args.generic_ridge, max_terms=args.generic_max_terms)

        basis_ranks = parse_int_list(args.basis_ranks or args.svd_ranks)
        basis_rows: List[Dict[str, Any]] = []
        if not args.skip_basis_diagnostics:
            basis_rows = evaluate_qkv_bases(seqs, weights, basis_ranks, device=args.basis_device)
            if args.fit_learned_qk_basis:
                basis_rows += evaluate_learned_qk_basis(seqs, weights, basis_ranks, args.learned_steps, args.learned_lr, args.basis_batch_size, args.device)
        write_csv(hdir / "basis_functional.csv", basis_rows)

        exact_program = {
            "RMS": "Xn_t = D_t x_raw_t, D_t = diag(gamma / sqrt(mean(x_raw_t^2)+eps))",
            "QK_affine_normalized": "score_ij = Xaug_i^T M_qk_aug[i-j] Xaug_j, Xaug=[Xn,1]",
            "M_qk_aug_delta": "M_qk_aug[d] = Wq_aug^T @ (R_i^T @ R_j) @ Wk_aug / sqrt(head_dim), d=i-j",
            "VO_affine_normalized": "Y_i = sum_j A[i,j] * Xaug_j @ C_vo_aug.T",
            "C_vo_aug": "C_vo_aug = Wo_head @ [Wv_kv | bv_kv]",
            "FullHead_raw_block_sparse": "payload_j = C_vo_linear @ D_j @ x_raw_j + b_vo; y_i=sum_j Aij*payload_j",
            "Basis_search": "Pq/Pk/Pv are searched in real Q/K/V head-space and judged by A/Z/Y errors, same spirit as v6 QK basis search.",
        }
        summary = {
            "layer": layer_idx, "head": head_idx, "meta": meta, "nseq": len(seqs), "max_observed_T": max_observed_T,
            "max_delta": max_delta, "bias_norms": {"bq": float(weights["bq"].norm()), "bk": float(weights["bk"].norm()), "bv": float(weights["bv"].norm())},
            "verify": verify, "rope_delta_invariance": rope_inv, "exact_program": exact_program,
            "basis_best": sorted(basis_rows, key=lambda r: (r.get("Y_rel_from_QK_basis", 999), r.get("A_rel", 999)))[:5],
            "matrix_reports": mat_reports,
        }
        all_summaries.append(summary)
        write_json(hdir / "summary.json", summary)
        write_csv(hdir / "per_prompt_checks.csv", per_prompt)
        examples = []
        for s in seqs[: int(args.example_prompts)]:
            top_rows = []
            T = int(s.A.shape[0])
            for i in range(min(T, int(args.example_tokens))):
                vals, idx = torch.topk(s.A[i, : i + 1], k=min(5, i + 1))
                top_rows.append({
                    "pos": i, "token": s.tokens[i] if i < len(s.tokens) else "?",
                    "top_read": [{"pos": int(j), "token": s.tokens[int(j)] if int(j) < len(s.tokens) else "?", "p": float(v)} for v, j in zip(vals.tolist(), idx.tolist())],
                    "Y_norm": float(torch.linalg.norm(s.Y[i]).item()), "rms_scale_mean": float(s.rms_scale[i].mean().item()),
                })
            examples.append({"prompt_id": s.prompt_id, "suite": s.suite, "text": s.text, "rows": top_rows})
        write_json(hdir / "token_flow_examples.json", examples)
        if args.save_tensors:
            torch.save({
                "layer": layer_idx, "head": head_idx, "meta": meta,
                "weights": weights, "C_vo_linear": Cvo_lin, "C_vo_aug": Cvo_aug,
                "M_qk_aug_delta": Mdelta_aug, "M_qk_linear_no_bias_delta": Mdelta_lin,
                "note": "Main exact targets are affine: score=x_aug M_aug x_aug; payload=x_aug C_vo_aug.T.",
            }, hdir / "circuit_targets_v2.pt")
        md = []
        md.append(f"# Affine circuit matrix targets L{layer_idx}H{head_idx}\n\n")
        md.append("## Exact matrix program\n```python\n")
        for k, v in exact_program.items(): md.append(f"{k}: {v}\n")
        md.append("```\n\n## Verification\n```json\n" + json.dumps(json_sanitize(verify), indent=2, ensure_ascii=False) + "\n```\n")
        md.append("\n## Read this\n")
        md.append("- `qk_score_linear_no_bias_rel` is the old v1-style target; it should be worse if q/k bias matters.\n")
        md.append("- `qk_score_aug_rel`, `V_from_affine_Wv_rel`, `Y_from_Cvo_aug_rel`, `Y_block_raw_affine_rel` should be near zero when the circuit formula is correct.\n")
        md.append("- `basis_functional.csv` is not generic PCA interpretation; it is functional score/read/write basis validation like your v6 QK basis search.\n")
        (hdir / "summary.md").write_text("".join(md), encoding="utf-8")

    if not args.patch_only and not args.final_article_suite and not args.skip_mlp:
        mlp_summary = run_mlp_analysis(model, tok, prompts, args, out_dir)

    if args.run_patching or (args.patch_only and not args.final_article_suite):
        patching_summary = run_patching_experiments(model, tok, prompts, args, out_dir)

    final_article_summary = None
    if args.final_article_suite:
        final_article_summary = run_final_article_suite(model, tok, args, out_dir)

    top_summary = {"args": vars(args), "heads": all_summaries, "mlp": mlp_summary, "patching": patching_summary, "final_article": final_article_summary}
    write_json(out_dir / "summary.json", top_summary)
    print(f"\nDONE wrote {out_dir}", flush=True)
    return top_summary


# ---------------- self-test ----------------

def self_test() -> None:
    torch.manual_seed(0)
    H, D, T = 16, 8, 7
    Wq = torch.randn(D, H) / math.sqrt(H); Wk = torch.randn(D, H) / math.sqrt(H); Wv = torch.randn(D, H) / math.sqrt(H); Wo = torch.randn(H, D) / math.sqrt(D)
    bq = torch.randn(D) * 0.2; bk = torch.randn(D) * 0.2; bv = torch.randn(D) * 0.2
    Wq_aug = torch.cat([Wq, bq[:, None]], dim=1); Wk_aug = torch.cat([Wk, bk[:, None]], dim=1); Wv_aug = torch.cat([Wv, bv[:, None]], dim=1)
    gamma = torch.rand(H) + 0.5; Xraw = torch.randn(T, H); eps = 1e-6
    scale = gamma.view(1, H) / torch.sqrt(Xraw.pow(2).mean(dim=-1, keepdim=True) + eps)
    Xn = Xraw * scale; Xaug = torch.cat([Xn, torch.ones(T, 1)], dim=1)
    freq = torch.linspace(0.01, 0.2, D // 2)
    angles_h = torch.arange(T).float().view(T, 1) * freq.view(1, D // 2)
    cos = torch.cat([torch.cos(angles_h), torch.cos(angles_h)], dim=1)
    sin = torch.cat([torch.sin(angles_h), torch.sin(angles_h)], dim=1)
    Rpos = {i: rope_col_matrix(cos[i], sin[i]) for i in range(T)}
    Qpre = Xaug @ Wq_aug.T; Kpre = Xaug @ Wk_aug.T; V = Xaug @ Wv_aug.T
    Q = torch.stack([Qpre[i] @ Rpos[i].T for i in range(T)], dim=0)
    K = torch.stack([Kpre[i] @ Rpos[i].T for i in range(T)], dim=0)
    A = causal_softmax((Q @ K.T) / math.sqrt(D)); Y = (A @ V) @ Wo.T
    weights = {"Wq":Wq,"Wk":Wk,"Wv":Wv,"Wo":Wo,"bq":bq,"bk":bk,"bv":bv,"Wq_aug":Wq_aug,"Wk_aug":Wk_aug,"Wv_aug":Wv_aug}
    Mdelta = qk_delta_matrices_affine(weights, Rpos, T-1, D)
    Mlin = qk_delta_matrices_linear(weights, Rpos, T-1, D)
    scores_hat = torch.zeros(T,T); scores_lin=torch.zeros(T,T)
    for i in range(T):
        for j in range(i+1):
            scores_hat[i,j] = Xaug[i] @ Mdelta[i-j] @ Xaug[j]
            scores_lin[i,j] = Xn[i] @ Mlin[i-j] @ Xn[j]
    scores_true=(Q@K.T)/math.sqrt(D); mask=torch.tril(torch.ones(T,T,dtype=torch.bool))
    Cvo_aug=Wo@Wv_aug; Cvo_lin=Wo@Wv; bvo=Wo@bv
    Y2=A@(Xaug@Cvo_aug.T); Y3=A@((Xraw*scale)@Cvo_lin.T + bvo.view(1,-1))
    print("SELF TEST v4 final article affine + patch hooks")
    print(f"  qk_score_aug_err={rel_err(scores_hat[mask], scores_true[mask]):.3e}")
    print(f"  qk_score_linear_no_bias_err={rel_err(scores_lin[mask], scores_true[mask]):.3e}  # expected nonzero")
    print(f"  vo_Y_aug_err={rel_err(Y2,Y):.3e}")
    print(f"  raw_block_affine_Y_err={rel_err(Y3,Y):.3e}")
    # MLP identity self-test: y=(silu(X Wg^T)*(X Wu^T)) Wd^T
    m = 32
    Wg_m = torch.randn(m, H) / math.sqrt(H)
    Wu_m = torch.randn(m, H) / math.sqrt(H)
    Wd_m = torch.randn(H, m) / math.sqrt(m)
    Xm = torch.randn(T, H)
    Ym = (F.silu(Xm @ Wg_m.T) * (Xm @ Wu_m.T)) @ Wd_m.T
    Ym2 = torch.zeros_like(Ym)
    for j in range(m):
        Ym2 += (F.silu(Xm @ Wg_m[j]) * (Xm @ Wu_m[j])).unsqueeze(1) * Wd_m[:, j].view(1, H)
    print(f"  mlp_rank1_atom_identity_err={rel_err(Ym2,Ym):.3e}")
    assert rel_err(scores_hat[mask], scores_true[mask]) < 1e-5
    assert rel_err(Y2, Y) < 1e-5
    assert rel_err(Y3, Y) < 1e-5
    assert rel_err(Ym2, Ym) < 1e-5
    print("  OK")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base-script", default="")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--heads", default="2:1")
    ap.add_argument("--prompt-suites", default="all")
    ap.add_argument("--prompts-per-suite", type=int, default=8)
    ap.add_argument("--same-text-repeats", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--max-delta", type=int, default=64)
    ap.add_argument("--svd-ranks", default="4,8,16,32,64,128")
    ap.add_argument("--basis-ranks", default="4,8,16,32,64")
    ap.add_argument("--basis-device", default="cpu", help="cpu is safer; cuda is faster for basis eval")
    ap.add_argument("--fit-learned-qk-basis", action="store_true")
    ap.add_argument("--learned-steps", type=int, default=120)
    ap.add_argument("--learned-lr", type=float, default=0.002)
    ap.add_argument("--basis-batch-size", type=int, default=16)
    ap.add_argument("--report-deltas", type=int, default=8)
    ap.add_argument("--example-prompts", type=int, default=2)
    ap.add_argument("--example-tokens", type=int, default=16)
    ap.add_argument("--decode-generic-ops", action="store_true")
    ap.add_argument("--generic-ridge", type=float, default=1e-4)
    ap.add_argument("--generic-max-terms", type=int, default=8)
    ap.add_argument("--save-tensors", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./qwen_matrix_program_attention_mlp_full")

    # Non-movement MLP exact matrix-program options.
    ap.add_argument("--skip-mlp", action="store_true", help="only run attention affine QK/VO extractor")
    ap.add_argument("--mlp-layers", default="all", help="MLP layers to analyze: all, 0,6,12 or 14-20")
    ap.add_argument("--mlp-compute-device", default="same", help="same/cpu/cuda; device for MLP reconstruction math")
    ap.add_argument("--mlp-batch-size", type=int, default=128)
    ap.add_argument("--mlp-top-neurons", type=int, default=8)
    ap.add_argument("--mlp-example-prompts", type=int, default=2)
    ap.add_argument("--mlp-proj-rank", type=int, default=32, help="0 disables projected atom report")
    ap.add_argument("--mlp-verify-ablation", action="store_true", help="expensive: causal ablation checks for shown top MLP atoms")
    ap.add_argument("--save-mlp-top-atoms", action="store_true", help="save explicit HxH rank-1 tensors only for shown top atoms")
    ap.add_argument("--enable-token-lens", action="store_true", help="optional qualitative vocabulary lens; not required for faithfulness/patching")
    ap.add_argument("--lens-pool", type=int, default=300)
    ap.add_argument("--lens-topk", type=int, default=5)

    # Executable replacement / intervention benchmark for the paper.
    ap.add_argument("--run-patching", action="store_true", help="run executable attention/MLP replacement and intervention benchmark")
    ap.add_argument("--patch-only", action="store_true", help="skip extraction/basis reports and run only patching benchmark")
    ap.add_argument("--skip-basis-diagnostics", action="store_true", help="skip expensive PCA/learned basis diagnostics")
    ap.add_argument("--patch-heads", default="20:5,21:1,21:6", help="heads for patching, L:H list; use same to reuse --heads")
    ap.add_argument("--patch-mlp-layers", default="20,21,22,23", help="MLP layers for executable replacement/intervention")
    ap.add_argument("--patch-prompt-source", default="paper", choices=["paper", "analysis"], help="fixed heterogeneous paper bank or prompts from the extraction suite")
    ap.add_argument("--patch-prompts", type=int, default=8, help="number of prompts used for patch benchmark")
    ap.add_argument("--patch-qk-scales", default="0,0.5,1.5", help="QK score scales; 1 is exact and added automatically")
    ap.add_argument("--patch-vo-scales", default="0,0.5,1.5", help="VO payload/output scales")
    ap.add_argument("--patch-mlp-scales", default="0,0.5,1.5", help="full MLP program scales")
    ap.add_argument("--patch-atom-scales", default="0", help="scales for top target-contributing MLP atom at final token")
    ap.add_argument("--no-patch-atom-controls", action="store_true", help="disable same-norm random and unrelated-neuron controls")
    ap.add_argument("--patch-group-tests", action="store_true", help="also replace all selected heads/layers jointly and combined")
    ap.add_argument("--patch-topk", type=int, default=5, help="top-k token report and overlap metric")

    # v4 final article suite: global equivalence + full causal atlas + held-out atoms.
    ap.add_argument("--final-article-suite", action="store_true", help="run resumable final article benchmark and skip normal extraction")
    ap.add_argument("--final-prompts", type=int, default=64, help="number of fixed heterogeneous prompts, max 64")
    ap.add_argument("--final-batch-size", type=int, default=32, help="batched prompts per forward; lower if VRAM is insufficient")
    ap.add_argument("--final-resume", action=argparse.BooleanOptionalAction, default=True, help="reuse completed per-experiment JSON files")
    ap.add_argument("--final-capture-propagation", action=argparse.BooleanOptionalAction, default=True, help="capture residual/attention/MLP downstream propagation")
    ap.add_argument("--final-skip-global", action="store_true", help="skip all-attention/all-MLP/full-backbone replacements")
    ap.add_argument("--final-skip-layer-groups", action="store_true", help="skip whole-layer exact and intervention tests")
    ap.add_argument("--final-skip-head-atlas", action="store_true", help="skip all 336 head native/program/QK-off/VO-off tests")
    ap.add_argument("--final-skip-mlp-atlas", action="store_true", help="skip all 24 MLP native/program/off tests")
    ap.add_argument("--final-skip-atom-split", action="store_true", help="skip held-out atom selection/evaluation")
    ap.add_argument("--final-atom-selection-prompts", type=int, default=24, help="prompt count used only to select fixed atoms")
    ap.add_argument("--final-atom-eval-prompts", type=int, default=32, help="held-out prompts used only for atom evaluation; 0 means all remaining")
    ap.add_argument("--final-atom-random-controls", type=int, default=3, help="same-norm random controls per layer in held-out atom test")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(); return
    run_analysis(args)


if __name__ == "__main__":
    main()
