#!/usr/bin/env python3
"""
Real-KV Audit — executed by the ORIGINAL research group (OG)
=============================================================

Purpose: independently settle the three questions the external team's work left open,
using genuine Qwen2.5-0.5B-Instruct KV on genuine (non-repetitive) text.

  Q1. What is the true high-frequency retention of direct averaging vs de-RoPE merging?
      -> Their two reports disagree: 0.1612 (synthetic, plan-02 §4.1)
         vs 0.4430 (real, track_a §5.3). We measure it on real text, per layer.

  Q2. THE MISSING M0 CRITERION — recall@64.
      Retention is a norm statistic; it says nothing about whether retrieval ORDER
      survives. v2 §5.7 required recall@64 vs exhaustive Mean-K ground truth.
      It has no implementation anywhere in the repo. We implement it here.

  Q3. Is sigma ~= 8.03 semantic spread or Qwen massive-activation artefacts?
      -> Per-layer profile + per-channel variance decomposition.

Everything here is CPU-only, dependency-light, and reproducible.

Usage:  python benchmarks/real_kv_audit_og.py [--outdir docs]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK = 32

# Real corpus: distinct Wikipedia articles -> non-repetitious, heterogeneous prose.
# The external audit used one paragraph * 10, which makes every block a near-duplicate.
WIKI_TITLES = [
    "Kv cache",
    "Attention mechanism",
    "Transformer (deep learning architecture)",
    "Vector quantization",
    "Content-addressable storage",
    "Computer storage",
]


# ----------------------------------------------------------------------------- corpus
def fetch_wiki_corpus(max_chars: int = 60000) -> Tuple[str, str]:
    """Fetch real prose from Wikipedia. Returns (text, provenance)."""
    chunks: List[str] = []
    for title in WIKI_TITLES:
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            "&explaintext=1&format=json&redirects=1&titles=" + urllib.parse.quote(title)
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kv-audit/1.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.loads(r.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                ext = page.get("extract")
                if ext:
                    chunks.append(f"### {title}\n{ext}")
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  [warn] failed to fetch '{title}': {exc}")
    text = "\n\n".join(chunks)
    if len(text) < 4000:
        print("  [warn] corpus fetch too small; using built-in fallback prose")
        text = FALLBACK_CORPUS
        return text, "built-in fallback"
    return text[:max_chars], "Wikipedia extracts (6 distinct articles)"


FALLBACK_CORPUS = (
    "Kv cache is a memory buffer storing intermediate results of attention computations. "
    "Attention mechanisms weight tokens according to learned relevance scores. "
    "Vector quantization maps continuous vectors to discrete symbols from a codebook. "
    "Content-addressable storage locates data by content hash rather than by pointer. "
    "Computer storage retains information across time using persistent media. "
    "Rotary position embeddings encode absolute position through rotation of query and key vectors. "
    "Grouped query attention shares key and value heads across multiple query heads to reduce memory. "
    "Speculative decoding proposes candidate tokens that a verifier model scores in parallel. "
) * 40


# ----------------------------------------------------------------------------- helpers
def get_kv(past_key_values, n_layers: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return [(K, V)] per layer, each [B, H, T, D].

    Handles the three Cache APIs across transformers versions:
      * transformers >= 5.x : DynamicCache.layers[i].keys / .values
      * transformers 4.x    : DynamicCache.key_cache[i] / .value_cache[i]
      * legacy tuple         : past_key_values[i] == (K, V)
    """
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if hasattr(past_key_values, "layers"):                 # transformers >= 5
        for layer in past_key_values.layers:
            out.append((layer.keys, layer.values))
        return out
    if hasattr(past_key_values, "key_cache"):               # transformers 4.x
        for l in range(n_layers):
            out.append((past_key_values.key_cache[l], past_key_values.value_cache[l]))
        return out
    for l in range(n_layers):                              # legacy tuple
        k, v = past_key_values[l]
        out.append((k, v))
    return out


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen2 / Llama half-split rotation: cat(-x2, x1)."""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def rope_forward(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def rope_inverse(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Exact inverse of rope_forward (uses rotate_half(rotate_half(v)) == -v)."""
    return x * cos - rotate_half(x) * sin


def band_idx(head_dim: int, k: int = 8) -> Tuple[List[int], List[int]]:
    """Half-split layout: dims 0..D/2-1 are the HIGHEST frequency pairs."""
    h = head_dim // 2
    hi = list(range(0, k)) + list(range(h, h + k))
    lo = list(range(h - k, h)) + list(range(head_dim - k, head_dim))
    return hi, lo


def retention(mean_vec: torch.Tensor, tokens: torch.Tensor, idx: List[int]) -> float:
    """|mean(tokens[band])| / mean(|tokens[band]|). 1.0 = no phase cancellation."""
    ref = tokens[:, idx].norm(dim=-1).mean().item()
    if ref <= 0:
        return float("nan")
    return mean_vec[idx].norm().item() / ref


def mad(x: torch.Tensor) -> float:
    """Mean per-token L2 distance from the block mean (NOT a standard deviation)."""
    m = x.mean(dim=0, keepdim=True)
    return (x - m).norm(dim=-1).mean().item()


def rope_at(vec: torch.Tensor, pos: int, cos_b: torch.Tensor, sin_b: torch.Tensor,
            head_dim: int) -> torch.Tensor:
    """Apply a rotation at a single position to a [D] vector -> [D].

    NOTE: HF returns cos/sin with last dim == head_dim (freqs catfreqs), not D//2.
    Broadcasting therefore promotes [D] -> [1,1,D]; we flatten back to [D].
    """
    c = cos_b[:, :, pos:pos + 1, :]
    s = sin_b[:, :, pos:pos + 1, :]
    return rope_inverse(vec.unsqueeze(0), c, s).reshape(head_dim)


# ----------------------------------------------------------------------------- audit
@dataclass
class Result:
    meta: Dict = field(default_factory=dict)
    selfcheck: Dict = field(default_factory=dict)
    retention: Dict = field(default_factory=dict)
    recall: Dict = field(default_factory=dict)
    sigma: Dict = field(default_factory=dict)


def run(max_len: int = 4096, topk: int = 64, n_query: int = 64, outdir: str = "docs") -> Result:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    res = Result()

    # ---------------- corpus
    corpus, provenance = fetch_wiki_corpus()
    print(f"[corpus] {provenance}, {len(corpus):,} chars")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    ids = tok(corpus, return_tensors="pt").input_ids[:, :max_len]
    T = ids.shape[1]
    n_blocks = T // BLOCK
    T = n_blocks * BLOCK
    ids = ids[:, :T]
    print(f"[model] {MODEL_ID}  tokens={T}  blocks={n_blocks}")

    with torch.no_grad():
        out = model(ids, use_cache=True)
    n_layers = len(out.past_key_values)
    kv = get_kv(out.past_key_values, n_layers)
    K0 = kv[0][0]                     # [1, H, T, D]
    H, D = K0.shape[1], K0.shape[3]
    res.meta = dict(model=MODEL_ID, corpus=provenance, layers=n_layers,
                    kv_heads=H, head_dim=D, tokens=T, blocks=n_blocks)

    # ---------------- cos / sin
    rot = model.model.rotary_emb
    pos = torch.arange(0, T, dtype=torch.long).unsqueeze(0)
    cos, sin = rot(K0[:, :, :T, :], pos)            # [1, T, D//2]
    cos_b = cos.unsqueeze(1).double()                # [1,1,T,D//2]
    sin_b = sin.unsqueeze(1).double()

    # ---------------- Q1a. de-RoPE self-check (report BOTH abs and rel)
    k0 = K0[:, :, :T, :].double()
    rt = rope_inverse(rope_forward(k0, cos_b, sin_b), cos_b, sin_b)
    abs_err = (k0 - rt).abs().max().item()
    rel_err = abs_err / (k0.abs().max().item() + 1e-12)
    res.selfcheck = dict(abs_err=abs_err, rel_err=rel_err,
                         v2_gate_atol=1e-5, v2_gate_atol_pass=bool(abs_err < 1e-5),
                         rel_gate=1e-5, rel_gate_pass=bool(rel_err < 1e-5))
    print(f"[self-check] abs={abs_err:.4e}  rel={rel_err:.4e}  "
          f"(v2 atol<=1e-5 -> {'PASS' if abs_err < 1e-5 else 'FAIL'})")

    hi, lo = band_idx(D)

    # ---------------- Q1b. per-layer three-arm retention
    per_layer = []
    for l in range(n_layers):
        Kl = kv[l][0][0, :, :T, :].double()                 # [H, T, D]
        Kd = rope_inverse(Kl.unsqueeze(0), cos_b, sin_b).squeeze(0)
        ra, rb, rlo_a, rlo_b = [], [], [], []
        for h in range(H):
            for b in range(n_blocks):
                s, e = b * BLOCK, (b + 1) * BLOCK
                rot_blk = Kl[h, s:e, :]
                der_blk = Kd[h, s:e, :]
                mean_a = rot_blk.mean(dim=0)                            # Arm A
                mid = s + BLOCK // 2
                mean_b = rope_at(der_blk.mean(dim=0), mid, cos_b, sin_b, D)   # Arm B
                ra.append(retention(mean_a, rot_blk, hi))
                rb.append(retention(mean_b, rot_blk, hi))
                rlo_a.append(retention(mean_a, rot_blk, lo))
                rlo_b.append(retention(mean_b, rot_blk, lo))
        per_layer.append(dict(layer=l,
                              hi_a=float(np.mean(ra)), hi_b=float(np.mean(rb)),
                              lo_a=float(np.mean(rlo_a)), lo_b=float(np.mean(rlo_b)),
                              n=len(ra)))
        if l % 6 == 0 or l == n_layers - 1:
            p = per_layer[-1]
            print(f"  L{l:>2}  hiA={p['hi_a']:.4f}  hiB={p['hi_b']:.4f}  "
                  f"B/A={p['hi_b']/max(p['hi_a'],1e-9):.2f}x   (n={p['n']})")

    mid_layer = n_layers // 2
    agg_hi_a = float(np.mean([p["hi_a"] for p in per_layer]))
    agg_hi_b = float(np.mean([p["hi_b"] for p in per_layer]))
    agg_lo_a = float(np.mean([p["lo_a"] for p in per_layer]))
    agg_lo_b = float(np.mean([p["lo_b"] for p in per_layer]))
    # std across layers -> dispersion of the estimate
    sd_hi_a = float(np.std([p["hi_a"] for p in per_layer]))
    sd_hi_b = float(np.std([p["hi_b"] for p in per_layer]))
    res.retention = dict(per_layer=per_layer,
                         hi_a=agg_hi_a, hi_b=agg_hi_b, hi_b_sd=sd_hi_b, hi_a_sd=sd_hi_a,
                         lo_a=agg_lo_a, lo_b=agg_lo_b,
                         ratio=agg_hi_b / max(agg_hi_a, 1e-9),
                         mid_layer=mid_layer,
                         mid_layer_agg=dict(hi_a=per_layer[mid_layer]["hi_a"],
                                            hi_b=per_layer[mid_layer]["hi_b"]))
    print(f"[retention] HI  A={agg_hi_a:.4f}(sd {sd_hi_a:.4f})  "
          f"B={agg_hi_b:.4f}(sd {sd_hi_b:.4f})  B/A={agg_hi_a and agg_hi_b/agg_hi_a:.2f}x")
    print(f"[retention] LO  A={agg_lo_a:.4f}  B={agg_lo_b:.4f}")

    # ---------------- Q2. recall@topk  (THE MISSING M0 CRITERION)
    # Definition (index-space self-retrieval consistency):
    #   query      = TRUE de-RoPE mean of a block (position-free, uncompressed)
    #   groundTruth= rank blocks by cosine(query, TRUE de-RoPE mean)          [exhaustive Mean-K]
    #   armX       = rank blocks by cosine(query, armX merged vector)
    #   recall@k   = |top-k(armX) INTERSECT top-k(truth)| / k
    # Arm A = mean of rotated keys (position contaminated) -> what merging without de-RoPE does.
    # Arm B = de-RoPE, mean, re-RoPE                            -> LADDER L2'.
    layer = mid_layer
    Kl = kv[layer][0][0, :, :T, :].double()
    Kd = rope_inverse(Kl.unsqueeze(0), cos_b, sin_b).squeeze(0)

    armA = torch.zeros(n_blocks, D, dtype=torch.float64)
    armB = torch.zeros(n_blocks, D, dtype=torch.float64)
    truth = torch.zeros(n_blocks, D, dtype=torch.float64)
    for h in range(H):
        for b in range(n_blocks):
            s, e = b * BLOCK, (b + 1) * BLOCK
            rot_blk, der_blk = Kl[h, s:e, :], Kd[h, s:e, :]
            mid = s + BLOCK // 2
            armA[b] += rot_blk.mean(dim=0) / H
            armB[b] += rope_at(der_blk.mean(dim=0), mid, cos_b, sin_b, D) / H
            truth[b] += der_blk.mean(dim=0) / H

    def cos_rank(q: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        Mn = M / (M.norm(dim=-1, keepdim=True) + 1e-12)
        return Mn @ q

    k = min(topk, n_blocks)
    rng = np.random.RandomState(0)
    q_blocks = rng.choice(n_blocks, size=min(n_query, n_blocks), replace=False)
    rec = {"A": [], "B": []}
    for qb in q_blocks:
        q = truth[qb] / (truth[qb].norm() + 1e-12)
        gt = set(torch.topk(cos_rank(q, truth), k).indices.tolist())
        for name, M in (("A", armA), ("B", armB)):
            top = set(torch.topk(cos_rank(q, M), k).indices.tolist())
            rec[name].append(len(gt & top) / k)
    recA, recB = float(np.mean(rec["A"])), float(np.mean(rec["B"]))
    n_q = len(q_blocks)
    # normal-approx CI on the mean of a bounded [0,1] sample
    seA = float(np.std(rec["A"], ddof=1) / math.sqrt(n_q)) if n_q > 1 else 0.0
    seB = float(np.std(rec["B"], ddof=1) / math.sqrt(n_q)) if n_q > 1 else 0.0
    res.recall = dict(topk=k, n_queries=n_q,
                      recall_at_k_A=recA, ci_A=[recA - 1.96 * seA, recA + 1.96 * seA],
                      recall_at_k_B=recB, ci_B=[recB - 1.96 * seB, recB + 1.96 * seB],
                      b_over_a=recB / max(recA, 1e-12))
    print(f"[recall@{k}] A={recA:.4f}  B={recB:.4f}  B/A={recB/max(recA,1e-12):.2f}x  "
          f"(n={n_q} queries)")
    print(f"[M0 gate]  B drop vs A must be <= A/3 -> "
          f"{'PASS' if (recA - recB) <= recA / 3 else 'FAIL'}")

    # ---------------- Q3. sigma: semantic spread or massive-activation artefact?
    prof, chan_share, normed_sigma, excl_sigma = [], [], [], []
    for l in range(n_layers):
        Kl = kv[l][0][0, :, :T, :].double()
        Kd = rope_inverse(Kl.unsqueeze(0), cos_b, sin_b).squeeze(0)
        vals = []
        for h in range(H):
            for b in range(n_blocks):
                vals.append(mad(Kd[h, b * BLOCK:(b + 1) * BLOCK, :]))
        prof.append(float(np.mean(vals)))

        if l == mid_layer:
            flat = Kd.reshape(-1, D)                       # [H*T, D]
            v = flat.var(dim=0)
            order = torch.argsort(v, descending=True)
            tot = v.sum()
            chan_share = [float(v[order[i]] / tot) for i in range(8)]
            cum = 0.0
            for i, c in enumerate(chan_share):
                cum += c
                if cum >= 0.8:
                    break
            n_ch_80 = i + 1
            keep = order[n_ch_80:]
            sub = flat[:, keep]
            excl_sigma.append(float((sub - sub.mean(0, keepdim=True)).norm(dim=-1).mean()))
            un = flat / (flat.norm(dim=-1, keepdim=True) + 1e-12)
            normed_sigma.append(float((un - un.mean(0, keepdim=True)).norm(dim=-1).mean()))

    res.sigma = dict(per_layer=prof, overall=float(np.mean(prof)),
                     per_channel_top8_share=chan_share,
                     n_channels_for_80pct=n_ch_80,
                     sigma_excluding_top_channels=excl_sigma[0] if excl_sigma else None,
                     sigma_on_l2_normalised_keys=normed_sigma[0] if normed_sigma else None)
    print(f"[sigma] per-layer mean MAD  min={min(prof):.3f} max={max(prof):.3f} "
          f"overall={np.mean(prof):.3f}")
    print(f"[sigma] top-8 channels carry {sum(chan_share)*100:.1f}% of variance; "
          f"{n_ch_80} channels carry 80%")
    print(f"[sigma] after dropping those: {res.sigma['sigma_excluding_top_channels']:.4f}   "
          f"on L2-normalised keys: {res.sigma['sigma_on_l2_normalised_keys']:.4f}")

    # ---------------- persist
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    jpath = os.path.join(outdir, f"real_kv_audit_og_{ts}.json")
    with open(jpath, "w") as f:
        json.dump(res.__dict__, f, indent=2)
    print(f"\n[saved] {jpath}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxlen", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--nquery", type=int, default=64)
    ap.add_argument("--outdir", default="docs")
    a = ap.parse_args()
    run(max_len=a.maxlen, topk=a.topk, n_query=a.nquery, outdir=a.outdir)