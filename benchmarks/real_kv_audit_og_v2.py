#!/usr/bin/env python3
"""
Real-KV Audit v2 — executed by the ORIGINAL research group (OG)
===============================================================
Second pass. Fixes the one methodological gap in v1 and turns the negative
result into a quantitative design rule.

WHAT v1 LEFT OPEN
-----------------
v1 compared, at recall@64:
    Arm A = mean of ROTATED keys          (merging without de-RoPE)
    Arm B = de-RoPE -> mean -> re-RoPE at block midpoint
and reported B/A = 0.98 (no gain).
But Arm B was re-rotated to the block midpoint while the query was left
de-RoPE'd at position 0. cos(R·x, y) != cos(x, y), so Arm B carried a
residual position mismatch that Arm A did not. v2 removes it.

WHAT v2 ADDS
------------
S.  Self-checks
    S1  de-RoPE round-trip (abs + rel), reproduced in double precision.
    S2  Manual Q/K/RoPE reconstruction validated against transformers'
        own eager attention on a short sequence. If this fails, every
        "attention alignment" number below is void.

P1. Index fidelity with POSITION-MATCHED arms.
    Query  q   = de-RoPE'd key of a random token (and, second mode, the
                 de-RoPE'd mean of a random 16-token span).
    Truth  T[b]= mean over tokens t in block b, mean over kv-heads h,
                 of cos(q_de_h, k_de_h(t))            <- "mean-of-cos"
    Arms:
      A      cos(q_rot(p), mean_t k_rot(t))           no de-RoPE anywhere
      B      cos(q_de,     mean_t k_de(t))            position-free, NO re-RoPE
      Bmid   cos(q_de,     rope(mean_t k_de(t), mid)) v1's Arm B (the artefact)
      C_m    mean_j cos(q_de, c_{b,j})  with m sub-centroids per block
      Cmax_m max_j  cos(q_de, c_{b,j})
      Rand   random ranking                            <- required baseline
    Note C_1 == B and C_32 == mean-of-cos == Truth exactly. So the ladder
    m = 1,2,4,8,16,32 interpolates continuously from "cos-of-mean" to
    "exhaustive". THAT is the compression error, isolated.

P2. Attention alignment. Does the Mean-K index rank blocks the way real
    attention does? Spearman(index block score, real attention mass per
    block) at the last position, using the model's OWN Q vector.

P3. Block-size design curve. BLOCK in {8,16,32,64,128} at m=1: recall vs
    index-entry count. Gives the memory/fidelity trade-off directly.

P4. Robustness: 5 layers x 5 seeds x 64 queries, two models (Qwen2.5-0.5B
    and Qwen3-0.6B, the latter has 8 kv heads, head_dim 128 and q_norm).

Usage:  python benchmarks/real_kv_audit_og_v2.py [--outdir docs] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

BLOCK = 32
QUERY_MODES = ("token", "span")          # token: single key; span: mean of 16 keys
SEEDS = (0, 1, 2, 3, 4)
N_QUERY = 64
KS = (16, 64)
M_LIST = (1, 2, 4, 8, 16, 32)
BLOCK_SWEEP = (8, 16, 32, 64, 128)

WIKI_TITLES = [
    "Kv cache",
    "Attention mechanism",
    "Transformer (deep learning architecture)",
    "Vector quantization",
    "Content-addressable storage",
    "Computer storage",
    "Paged attention",
    "Locality-sensitive hashing",
]

FALLBACK_CORPUS = (
    "Kv cache is a memory buffer storing intermediate results of attention computations. "
    "Attention mechanisms weight tokens according to learned relevance scores. "
    "Vector quantization maps continuous vectors to discrete symbols from a codebook. "
    "Content-addressable storage locates data by content hash rather than by pointer. "
    "Computer storage retains information across time using persistent media. "
    "Rotary position embeddings encode absolute position through rotation of query and key. "
    "Grouped query attention shares key and value heads across several query heads. "
    "Speculative decoding proposes candidate tokens that a verifier model scores at once. "
) * 60


# ------------------------------------------------------------------ corpus
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "wiki_corpus.json")


def fetch_wiki_corpus(max_chars: int = 90000) -> Tuple[str, str]:
    """Fetch real prose from Wikipedia, with a disk cache (the API rate-limits hard)."""
    chunks: List[str] = []
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cached: Dict[str, str] = {}
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                cached = json.load(f)
        except Exception:
            cached = {}
    fresh = 0
    for title in WIKI_TITLES:
        if title in cached and len(cached[title]) > 500:
            chunks.append(cached[title])
            continue
        url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
               "&explaintext=1&format=json&redirects=1&titles=" + urllib.parse.quote(title))
        got = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "kv-audit/2.0"})
                with urllib.request.urlopen(req, timeout=25) as r:
                    data = json.loads(r.read().decode("utf-8"))
                for _, page in data.get("query", {}).get("pages", {}).items():
                    if page.get("extract"):
                        got = f"### {title}\n{page['extract']}"
                if got:
                    break
            except Exception as exc:  # pragma: no cover
                time.sleep(1.5 * (attempt + 1))
                got = None
        if got:
            cached[title] = got
            chunks.append(got)
            fresh += 1
            time.sleep(0.4)
        elif title in cached:
            chunks.append(cached[title])
        else:
            print(f"  [warn] '{title}' unavailable and not cached")
    try:
        with open(CACHE, "w") as f:
            json.dump(cached, f)
    except Exception:
        pass
    if len(chunks) < 2:
        return FALLBACK_CORPUS, "built-in fallback"
    # IMPORTANT: do NOT simply concatenate and truncate. A 4096-token window is
    # only ~17k characters, which at article length would sit entirely inside
    # ONE article -- we would be measuring intra-document homogeneity while
    # claiming cross-document heterogeneity. Interleave instead: cut every
    # article into segments and round-robin them, so any prefix of the corpus
    # spans every article.
    text = interleave(chunks, seg_len=2200)
    prov = (f"Wikipedia extracts ({len(chunks)} articles, {fresh} fresh), "
            f"round-robin interleaved at 2200-char segments")
    return text[:max_chars], prov


def interleave(chunks: List[str], seg_len: int = 2200) -> str:
    """Round-robin fixed-length segments from each document."""
    segs: List[List[str]] = []
    for ch in chunks:
        segs.append([ch[i:i + seg_len] for i in range(0, len(ch), seg_len)])
    depth = max(len(s) for s in segs)
    outp: List[str] = []
    for d in range(depth):
        for s in segs:
            if d < len(s):
                outp.append(s[d])
    return "\n\n".join(outp)


# ------------------------------------------------------------------ helpers
def get_kv(past, n_layers: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past, "layers"):                       # transformers >= 5
        return [(la.keys, la.values) for la in past.layers]
    if hasattr(past, "key_cache"):                    # transformers 4.x
        return [(past.key_cache[l], past.value_cache[l]) for l in range(n_layers)]
    return [(past[l][0], past[l][1]) for l in range(n_layers)]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def rope_forward(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def rope_inverse(x, cos, sin):
    return x * cos - rotate_half(x) * sin


def rope_at_batch(x, pos, cos_b, sin_b):
    """Re-apply RoPE at position `pos` to a batch of de-RoPE'd vectors.
    x [n,D] float32, pos [n] long, cos_b/sin_b [1,1,T,D]."""
    c = cos_b[0, 0, pos, :][None, None]
    s = sin_b[0, 0, pos, :][None, None]
    return rope_forward(x[None, None], c, s).reshape(x.shape)


def norm_rows(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def recall_at_k(pred_order: np.ndarray, truth_order: np.ndarray, k: int) -> float:
    """Share of the truth top-k recovered by the arm's top-k.

    k is clamped to the number of candidates: asking for top-64 out of 32
    blocks would silently return 32/64 = 0.5 for EVERY arm, including Random.
    That is an artefact, not a measurement.
    """
    n = min(len(pred_order), len(truth_order))
    k = min(k, n)
    if k <= 0:
        return float("nan")
    gt, tp = set(truth_order[:k].tolist()), set(pred_order[:k].tolist())
    return len(gt & tp) / k


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman(x, y) with the effect of z removed (rank-space residualisation).

    Needed because real attention mass is strongly driven by RECENCY, i.e. by
    block position. A position-free index cannot reproduce that by construction,
    so a raw Spearman of index-vs-attention conflates "the index is bad" with
    "the index is deliberately position-free".
    """
    rx, ry, rz = [np.argsort(np.argsort(v)).astype(float) for v in (x, y, z)]
    rz = rz - rz.mean()
    rz = np.column_stack([rz, np.ones_like(rz)])
    bx = np.linalg.lstsq(rz, rx, rcond=None)[0]
    by = np.linalg.lstsq(rz, ry, rcond=None)[0]
    ex, ey = rx - rz @ bx, ry - rz @ by
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def ci95(vals: List[float]) -> List[float]:
    if len(vals) < 2:
        return [float("nan"), float("nan")]
    se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
    m = float(np.mean(vals))
    return [m - 1.96 * se, m + 1.96 * se]


def agg(vals: List[float]) -> Dict:
    m = float(np.mean(vals))
    return dict(mean=m, ci95=ci95(vals), n=len(vals),
                sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


# ------------------------------------------------------------------ per-layer prep
def layer_pack(Kl_rot, cos_b, sin_b):
    """Return rotated and de-RoPE'd keys for one layer, float32.
    Kl_rot [H,T,D] -> (Krot [H,T,D], Kde [H,T,D])."""
    Krot = Kl_rot.float()
    Kde = rope_inverse(Krot.unsqueeze(0), cos_b.float(), sin_b.float()).squeeze(0)
    return Krot, Kde


def centroids(K: torch.Tensor, block: int, m: int):
    """K [H,T,D] -> C [m, n_blocks, H, D] (mean over de-RoPE'd sub-spans).

    Sub-spans are contiguous: block b covers [b*block,(b+1)*block);
    sub-centroid j covers the j-th (block//m)-token slice of it.
    """
    H, T, D = K.shape
    nb = T // block
    sub = block // m
    assert sub * m == block, "block must be divisible by m"
    K = K[:, :nb * block, :]
    C = torch.zeros(m, nb, H, D, dtype=K.dtype)
    for j in range(m):
        for b in range(nb):
            s = b * block + j * sub
            C[j, b] = K[:, s:s + sub, :].mean(dim=1)
    return C


def block_means(K: torch.Tensor, block: int):
    """K [H,T,D] -> M [n_blocks, H, D]."""
    H, T, D = K.shape
    nb = T // block
    return K[:, :nb * block, :].reshape(H, nb, block, D).mean(dim=2).permute(1, 0, 2).contiguous()


# ------------------------------------------------------------------ main
def audit_model(model_id: str, max_len: int, layer_ids: List[int], outdir: str,
                quick: bool) -> Dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)          # inference only; keeps tensors numpy()-able

    print(f"\n{'='*78}\nMODEL {model_id}\n{'='*78}")
    out: Dict = dict(model=model_id)

    corpus, provenance = fetch_wiki_corpus()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    n_layers = len(model.model.layers)
    layer_ids = [l for l in layer_ids if l < n_layers]

    ids_full = tok(corpus, return_tensors="pt").input_ids[:, :max_len]
    T_orig = ids_full.shape[1]
    T = (T_orig // BLOCK) * BLOCK
    ids_full = ids_full[:, :T]
    print(f"[corpus] {provenance}  tokens={T}")

    # =============================================== S1 de-RoPE round trip
    with torch.no_grad():
        probe = model(ids_full[:, :512], use_cache=True)
    kv_probe = get_kv(probe.past_key_values, n_layers)
    Kp = kv_probe[0][0][:, :, :512, :]
    H, D = Kp.shape[1], Kp.shape[3]
    H_q = model.config.num_attention_heads
    rot_emb = model.model.rotary_emb
    pos_ids = torch.arange(0, 512).unsqueeze(0)
    with torch.no_grad():
        cos_p, sin_p = rot_emb(Kp, pos_ids)
    cb = cos_p.unsqueeze(1).double()
    sb = sin_p.unsqueeze(1).double()
    kd = Kp.double()
    rt = rope_inverse(rope_forward(kd, cb, sb), cb, sb)
    abs_err = (kd - rt).abs().max().item()
    rel_err = abs_err / (kd.abs().max().item() + 1e-12)
    out["S1_derope_roundtrip"] = dict(abs_err=abs_err, rel_err=rel_err,
                                      v2_atol_gate_1e5_pass=bool(abs_err < 1e-5),
                                      rel_gate_1e5_pass=bool(rel_err < 1e-5))
    print(f"[S1] de-RoPE round-trip  abs={abs_err:.4e} rel={rel_err:.4e}  "
          f"atol<1e-5:{'PASS' if abs_err<1e-5 else 'FAIL'}  rel<1e-5:{'PASS' if rel_err<1e-5 else 'FAIL'}")

    # =============================================== S2 attention reconstruction
    L2 = 64
    # A SECOND model instance in eager mode: sdpa/sage refuse to return attention
    # weights, and forcing eager on the 4096-token forward would be ruinously slow.
    model_eager = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager")
    model_eager.eval()
    with torch.no_grad():
        short = model_eager(ids_full[:, :L2], output_attentions=True, use_cache=True)
        short_h = model_eager(ids_full[:, :L2], output_hidden_states=True, use_cache=True)
    del model_eager
    kv_short = get_kv(short.past_key_values, n_layers)
    hs_short = short_h.hidden_states                       # tuple, len n_layers+1
    cos_s, sin_s = rot_emb(kv_short[0][0], torch.arange(0, L2).unsqueeze(0))
    cos_bs, sin_bs = cos_s.unsqueeze(1).float(), sin_s.unsqueeze(1).float()
    worst = []
    if not short.attentions:
        out["S2_attention_reconstruction_max_abs_err"] = None
        out["S2_pass"] = False
        print("[S2] transformers returned no attention weights -> NOT VERIFIED")
    for l in layer_ids if short.attentions else []:
        attn = model.model.layers[l].self_attn
        # hs[l] is the RESIDUAL STREAM entering layer l; Qwen applies RMSNorm
        # (input_layernorm) INSIDE the layer before q_proj. Without this the
        # reconstruction is wrong by a large margin.
        h_in = model.model.layers[l].input_layernorm(hs_short[l][0])
        q = attn.q_proj(h_in).reshape(L2, H_q, D).permute(1, 0, 2)   # [H_q, L2, D]
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        c = cos_bs[0, 0, :, :][None]                       # [1, L2, D]
        s = sin_bs[0, 0, :, :][None]
        q_rot = rope_forward(q, c, s)
        K = kv_short[l][0][0].float()                      # [H_kv, L2, D]
        grp = H_q // K.shape[0]
        # GQA: softmax is per QUERY head. Averaging q inside a group first is
        # NOT the same as averaging the softmaxes, so expand K per q head.
        Kexp = K[torch.arange(H_q) // grp]                 # [H_q, L2, D]
        scale = 1.0 / math.sqrt(D)
        logits = torch.einsum("htd,hsd->hts", q_rot, Kexp) * scale     # [H_q, L2, L2]
        mask = torch.triu(torch.ones(L2, L2, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))
        mine = torch.softmax(logits.float(), dim=-1)
        real = short.attentions[l][0]                                  # [H_q, L2, L2]
        worst.append(float((mine - real).abs().max()))
    if worst:
        out["S2_attention_reconstruction_max_abs_err"] = float(max(worst))
        ok_s2 = max(worst) < 2e-3
        out["S2_pass"] = bool(ok_s2)
        print(f"[S2] manual attention vs transformers eager: max|diff|={max(worst):.3e}  "
              f"{'PASS' if ok_s2 else 'FAIL'}")

    # =============================================== full forward
    with torch.no_grad():
        full = model(ids_full, output_hidden_states=True, use_cache=True)
    kv = get_kv(full.past_key_values, n_layers)
    hs = full.hidden_states
    cos_f, sin_f = rot_emb(kv[0][0][:, :, :T, :], torch.arange(0, T).unsqueeze(0))
    cos_b, sin_b = cos_f.unsqueeze(1).float(), sin_f.unsqueeze(1).float()
    n_blocks = T // BLOCK
    print(f"[model] layers={n_layers} q_heads={H_q} kv_heads={H} head_dim={D} "
          f"tokens={T} blocks={n_blocks}")

    # =============================================== P1 index fidelity
    p1: Dict = {}
    for l in layer_ids:
        Krot, Kde = layer_pack(kv[l][0][0, :, :T, :], cos_b, sin_b)
        Kde_n = norm_rows(Kde.reshape(H * T, D)).reshape(H, T, D)     # for truth
        Mrot = block_means(Krot, BLOCK)                               # [nb,H,D]
        Mde = block_means(Kde, BLOCK)
        mid_pos = torch.arange(n_blocks) * BLOCK + BLOCK // 2
        Mde_re = torch.stack([rope_at_batch(Mde[:, h, :], mid_pos, cos_b, sin_b)
                              for h in range(H)], dim=1)              # [nb,H,D]
        Mr_n = norm_rows(Mrot.reshape(-1, D)).reshape(n_blocks, H, D)
        Md_n = norm_rows(Mde.reshape(-1, D)).reshape(n_blocks, H, D)
        Mre_n = norm_rows(Mde_re.reshape(-1, D)).reshape(n_blocks, H, D)
        Cs = {m: centroids(Kde, BLOCK, m) for m in M_LIST if m != 1}
        Cs[1] = Mde.unsqueeze(0)                                      # [1,nb,H,D]
        Cs_n = {m: norm_rows(Cs[m].reshape(-1, D)).reshape(Cs[m].shape[0], n_blocks, H, D)
                for m in Cs}

        for mode in QUERY_MODES:
            rec: Dict[str, Dict[int, List[float]]] = {}
            for seed in SEEDS:
                rng = np.random.RandomState(1000 * l + seed)
                if mode == "token":
                    qpos = rng.choice(T, size=N_QUERY, replace=False)
                else:
                    qpos = rng.choice(T - 16, size=N_QUERY, replace=False)
                for p in qpos:
                    if mode == "token":
                        qde = Kde[:, p, :]                            # [H,D]
                    else:
                        qde = Kde[:, p:p + 16, :].mean(dim=1)
                    qde_n = norm_rows(qde)                            # [H,D]
                    qrot = rope_at_batch(qde, torch.tensor([int(p)] * H), cos_b, sin_b)
                    qrot_n = norm_rows(qrot)
                    # truth = mean-of-cos over tokens, averaged across kv heads
                    per_tok = torch.einsum("hd,htd->ht", qde_n, Kde_n)      # [H,T]
                    truth = per_tok[:, :n_blocks * BLOCK].reshape(H, n_blocks, BLOCK).mean(dim=2)
                    truth = truth.mean(dim=0)                                # [nb]
                    t_order = torch.argsort(truth, descending=True).numpy()
                    arms: Dict[str, np.ndarray] = {}
                    # NOTE: einsum "hd,nhd->n" already contracts BOTH h and d,
                    # so the result is already [n]. Do NOT add .mean(dim=0).
                    arms["A"] = torch.einsum("hd,nhd->n", qrot_n, Mr_n)
                    arms["B"] = torch.einsum("hd,nhd->n", qde_n, Md_n)
                    arms["Bmid"] = torch.einsum("hd,nhd->n", qde_n, Mre_n)
                    for m in M_LIST:
                        sc = torch.einsum("hd,mnhd->mn", qde_n, Cs_n[m])     # [m,nb]
                        arms[f"C_mean_m{m}"] = sc.mean(dim=0)
                        arms[f"C_max_m{m}"] = sc.max(dim=0).values
                    arms["C_mean_m1"] = arms["B"]                              # identical by construction
                    if BLOCK in M_LIST:
                        # Arm scores SUM over kv heads; `truth` AVERAGES over them.
                        # So C_mean_m{BLOCK} == H * truth exactly -- a constant
                        # factor that cannot change any ranking. Compare after
                        # removing that known factor.
                        s32 = arms[f"C_mean_m{BLOCK}"] / H
                        dev = (s32 - truth).abs().max().item()
                        den = truth.abs().max().item() + 1e-12
                        out["_s3a"] = max(out.get("_s3a", 0.0), dev / den)
                    arms["Rand"] = torch.from_numpy(rng.permutation(n_blocks).astype(np.float32))
                    for name, sc in arms.items():
                        order = torch.argsort(sc, descending=True).numpy()
                        for k in KS:
                            rec.setdefault(f"{name}|{k}", []).append(
                                recall_at_k(order, t_order, k))
            key = f"layer{l}|{mode}"
            p1[key] = {nk: agg(v) for nk, v in rec.items()}
            b = p1[key][f"B|64"]["mean"]
            a = p1[key][f"A|64"]["mean"]
            bm = p1[key][f"Bmid|64"]["mean"]
            c32 = p1[key]["C_mean_m32|64"]["mean"]
            rnd = p1[key]["Rand|64"]["mean"]
            print(f"  L{l:>2} {mode:<5} k=64  A={a:.4f}  B={b:.4f}  Bmid={bm:.4f}  "
                  f"C_m2={p1[key]['C_mean_m2|64']['mean']:.4f}  "
                  f"C_m4={p1[key]['C_mean_m4|64']['mean']:.4f}  "
                  f"C_m8={p1[key]['C_mean_m8|64']['mean']:.4f}  "
                  f"C_m32(exhaustive)={c32:.4f}  Rand={rnd:.4f}")
    out["P1_index_fidelity"] = p1

    # --------- S3 built-in sanity: with m == BLOCK the centroid IS the token,
    # so C_mean_m32 IS the ground truth, computed by a different code path.
    # Two independent checks:
    #   S3a numerical: max |score(C_m32) - score(truth)| / |score(truth)|
    #   S3b behavioural: recall(C_m32) vs recall(truth) -- can still deviate by
    #       a hair when two blocks are near-tied and the two code paths order
    #       them differently. That is a tie artefact, not a wiring bug.
    worst_s3a = out.get("_s3a", 0.0)
    worst_s3b = 0.0
    for kk, v in p1.items():
        r = v.get("C_mean_m32|64", {}).get("mean")
        if r is not None:
            worst_s3b = max(worst_s3b, abs(1.0 - r))
    out["S3_ladder_endpoint"] = dict(
        max_relative_score_deviation=worst_s3a,
        worst_recall_deviation_from_1=worst_s3b,
        numerical_pass=bool(worst_s3a < 1e-5),
        behavioural_pass=bool(worst_s3b < 5e-3),
        note="behavioural deviation is bounded by tie-breaking, not wiring")
    print(f"[S3] ladder endpoint (m=32 == exhaustive): "
          f"max rel score dev={worst_s3a:.2e} ({'PASS' if worst_s3a<1e-5 else 'FAIL'}), "
          f"worst |1-recall|={worst_s3b:.2e} ({'PASS(tie-limited)' if worst_s3b<5e-3 else 'FAIL'})")

    # =============================================== P1b  NATIVE space + relocation
    # P1 above scores in the DE-ROPE'd (position-free) space, which is where a
    # Mean-K index lives. But a block that has been selected is then RELOCATED
    # and re-RoPE'd, and from that moment on it is compared against queries in
    # the NATIVE space. KVMem scores in one space and consumes in the other.
    # P1b measures the price of that crossing:
    #   truth[b] = mean_t mean_h cos(q_rot(p), k_rot(t))         <- exhaustive, native
    #   Arm A     = cos(q_rot(p), mean_t k_rot(t))               <- naive average
    #   Arm D(Δ)  = cos(q_rot(p), rope(mean_t k_de(t), s_b + Δ)) <- de-RoPE, mean,
    #                                                               re-RoPE Δ later
    # Δ = 0 means "put the block back where it started"; Δ > 0 is a real
    # compaction-induced move, which is exactly what KVMem does.
    DELTAS = (0, 16, 32, 128, 512, 2048)
    p1b: Dict = {}
    for l in layer_ids:
        Krot, Kde = layer_pack(kv[l][0][0, :, :T, :], cos_b, sin_b)
        Krot_n = norm_rows(Krot.reshape(H * T, D)).reshape(H, T, D)
        Mrot = block_means(Krot, BLOCK)
        Mde = block_means(Kde, BLOCK)
        Mr_n = norm_rows(Mrot.reshape(-1, D)).reshape(n_blocks, H, D)
        base_pos = torch.arange(n_blocks) * BLOCK
        Dcent: Dict[int, torch.Tensor] = {}
        for d in DELTAS:
            tgt = (base_pos + d).clamp(max=T - 1)
            Md = torch.stack([rope_at_batch(Mde[:, h, :], tgt, cos_b, sin_b)
                              for h in range(H)], dim=1)              # [nb,H,D]
            Dcent[d] = norm_rows(Md.reshape(-1, D)).reshape(n_blocks, H, D)
        rec: Dict[str, Dict[int, List[float]]] = {}
        for seed in SEEDS:
            rng = np.random.RandomState(5000 * l + seed)
            for p in rng.choice(T, size=N_QUERY, replace=False):
                qrot_n = norm_rows(Krot[:, p, :])
                per_tok = torch.einsum("hd,htd->ht", qrot_n, Krot_n)
                truth = per_tok[:, :n_blocks * BLOCK].reshape(H, n_blocks, BLOCK).mean(dim=2)
                truth = truth.mean(dim=0)
                t_order = torch.argsort(truth, descending=True).numpy()
                arms: Dict[str, torch.Tensor] = {}
                arms["A_naive"] = torch.einsum("hd,nhd->n", qrot_n, Mr_n)
                for d in DELTAS:
                    arms[f"D_delta{d}"] = torch.einsum("hd,nhd->n", qrot_n, Dcent[d])
                arms["Rand"] = torch.from_numpy(rng.permutation(n_blocks).astype(np.float32))
                for name, sc in arms.items():
                    o = torch.argsort(sc, descending=True).numpy()
                    for k in KS:
                        rec.setdefault(f"{name}|{k}", []).append(recall_at_k(o, t_order, k))
        p1b[f"layer{l}"] = {nk: agg(v) for nk, v in rec.items()}
        row = p1b[f"layer{l}"]
        a = row["A_naive|64"]["mean"]
        print(f"  L{l:>2} NATIVE  A_naive={a:.4f} | " + "  ".join(
            f"D(+{d})={row[f'D_delta{d}|64']['mean']:.4f}" for d in DELTAS) +
            f" | Rand={row['Rand|64']['mean']:.4f}")
    out["P1b_native_relocation"] = dict(deltas=list(DELTAS), per_layer=p1b)

    # =============================================== P2 attention alignment
    p2: Dict = {}
    p_last = T - 1
    grp = H_q // H
    gid = torch.arange(H_q) // grp                 # q-head -> kv-head map
    for l in layer_ids:
        attn = model.model.layers[l].self_attn
        h_in = model.model.layers[l].input_layernorm(hs[l][0])          # [T, hidden]
        q = attn.q_proj(h_in[p_last:p_last + 1]).reshape(1, H_q, D).permute(1, 0, 2)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        c = cos_b[0, 0, p_last:p_last + 1, :][None]
        s = sin_b[0, 0, p_last:p_last + 1, :][None]
        q_rot = rope_forward(q, c, s)[:, 0, :]      # [H_q, D]
        q_de = rope_inverse(q_rot[:, None], c, s)[:, 0, :]
        Krot, Kde = layer_pack(kv[l][0][0, :, :T, :], cos_b, sin_b)
        scale = 1.0 / math.sqrt(D)
        # REAL attention: softmax per query head, K expanded per q head (GQA).
        Kex = Krot[gid]                                                  # [H_q, T, D]
        logits = (Kex @ q_rot.unsqueeze(-1)).squeeze(-1) * scale          # [H_q, T]
        w = torch.softmax((logits - logits.max(-1, keepdim=True).values).float(), dim=-1)
        mass = w[:, :n_blocks * BLOCK].reshape(H_q, n_blocks, BLOCK).sum(dim=2).mean(dim=0)
        mass_np = mass.numpy()
        qr_n = norm_rows(q_rot)
        qd_n = norm_rows(q_de)
        Mrot = block_means(Krot, BLOCK)                                   # [nb,H,D]
        Mde = block_means(Kde, BLOCK)
        Mr_n = norm_rows(Mrot.reshape(-1, D)).reshape(n_blocks, H, D)
        Md_n = norm_rows(Mde.reshape(-1, D)).reshape(n_blocks, H, D)
        row: Dict = {}
        # index score: per q-head cosine against its own kv head's centroid
        scA = torch.einsum("hd,nhd->hn", qr_n, Mr_n[:, gid, :]).mean(dim=0).numpy()
        scB = torch.einsum("hd,nhd->hn", qd_n, Md_n[:, gid, :]).mean(dim=0).numpy()
        row["A"] = float(spearmanr(scA, mass_np).statistic)
        row["B"] = float(spearmanr(scB, mass_np).statistic)
        C4 = centroids(Kde, BLOCK, 4)
        C4n = norm_rows(C4.reshape(-1, D)).reshape(4, n_blocks, H, D)
        s4 = torch.einsum("hd,mnhd->hmn", qd_n, C4n[:, :, gid, :])
        row["C_max_m4"] = float(spearmanr(s4.max(dim=0).values.mean(dim=0).numpy(), mass_np).statistic)
        row["C_mean_m4"] = float(spearmanr(s4.mean(dim=0).mean(dim=0).numpy(), mass_np).statistic)
        rng = np.random.RandomState(7)
        row["Rand"] = float(spearmanr(rng.rand(n_blocks), mass_np).statistic)
        row["top1_block_real_attention"] = int(np.argmax(mass_np))
        row["top1_block_indexB"] = int(np.argmax(scB))
        row["attention_top1_block_mass"] = float(mass_np.max())

        # ---- decompose: how much of attention mass is POSITION (recency)?
        posn = np.arange(n_blocks, dtype=float)
        row["rho_position_vs_attention"] = float(spearmanr(posn, mass_np).statistic)
        row["A_partial_position"] = partial_spearman(scA, mass_np, posn)
        row["B_partial_position"] = partial_spearman(scB, mass_np, posn)
        row["C_max_m4_partial_position"] = partial_spearman(
            s4.max(dim=0).values.mean(dim=0).numpy(), mass_np, posn)
        # Spearman restricted to non-recent blocks (drop the last 16 = 512 tokens)
        cut = max(1, n_blocks - 16)
        row["A_excl_recent16"] = float(spearmanr(scA[:cut], mass_np[:cut]).statistic)
        row["B_excl_recent16"] = float(spearmanr(scB[:cut], mass_np[:cut]).statistic)
        row["pos_excl_recent16"] = float(spearmanr(posn[:cut], mass_np[:cut]).statistic)
        p2[f"layer{l}"] = row
        print(f"  L{l:>2} rho(attention)  A={row['A']:+.3f}  B={row['B']:+.3f}  "
              f"Cmax4={row['C_max_m4']:+.3f}  Rand={row['Rand']:+.3f}  |  "
              f"rho(pos,attn)={row['rho_position_vs_attention']:+.3f}  "
              f"partial(pos): A={row['A_partial_position']:+.3f} B={row['B_partial_position']:+.3f}  "
              f"| excl-recent16: A={row['A_excl_recent16']:+.3f} B={row['B_excl_recent16']:+.3f}")
    out["P2_attention_alignment"] = p2

    # =============================================== P4  positional prior blend
    # P2 shows a pure content index (de-RoPE'd Mean-K) tracks real attention
    # poorly, largely because attention mass is strongly positional. That is
    # not a reason to abandon the index -- it is a reason to ADD a positional
    # prior. P4 asks: what blend of content and position best predicts where
    # attention actually goes?
    #   score(alpha) = alpha * rank(content) + (1-alpha) * rank(position)
    # Spearman is rank-based, so any monotone recency transform gives the same
    # answer -- only alpha matters. alpha=0 is pure recency, alpha=1 pure content.
    p4: Dict = {}
    ALPHAS = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))
    for l in layer_ids:
        row = p2[f"layer{l}"]
        # recompute the two rank vectors at this layer
        attn = model.model.layers[l].self_attn
        h_in = model.model.layers[l].input_layernorm(hs[l][0])
        q = attn.q_proj(h_in[p_last:p_last + 1]).reshape(1, H_q, D).permute(1, 0, 2)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        c = cos_b[0, 0, p_last:p_last + 1, :][None]
        s = sin_b[0, 0, p_last:p_last + 1, :][None]
        q_rot = rope_forward(q, c, s)[:, 0, :]
        q_de = rope_inverse(q_rot[:, None], c, s)[:, 0, :]
        Krot, Kde = layer_pack(kv[l][0][0, :, :T, :], cos_b, sin_b)
        scale = 1.0 / math.sqrt(D)
        Kex = Krot[gid]
        lg = (Kex @ q_rot.unsqueeze(-1)).squeeze(-1) * scale
        w = torch.softmax((lg - lg.max(-1, keepdim=True).values).float(), dim=-1)
        mass = w[:, :n_blocks * BLOCK].reshape(H_q, n_blocks, BLOCK).sum(dim=2).mean(dim=0).numpy()
        Mde = block_means(Kde, BLOCK)
        Md_n = norm_rows(Mde.reshape(-1, D)).reshape(n_blocks, H, D)[:, gid, :]
        qd_n = norm_rows(q_de)
        content = torch.einsum("hd,nhd->hn", qd_n, Md_n).mean(dim=0).numpy()
        rc = np.argsort(np.argsort(content)).astype(float)
        rp = np.arange(n_blocks, dtype=float)
        curve = {}
        for al in ALPHAS:
            curve[f"{al:.1f}"] = float(spearmanr(al * rc + (1 - al) * rp, mass).statistic)
        best_a = max(curve, key=lambda kk: curve[kk])
        p4[f"layer{l}"] = dict(curve=curve, best_alpha=best_a,
                               best_rho=curve[best_a],
                               rho_alpha0=curve["0.0"], rho_alpha1=curve["1.0"],
                               gain_over_pure_content=curve[best_a] - curve["1.0"])
        print(f"  L{l:>2} blend  best alpha={best_a} rho={curve[best_a]:+.3f} "
              f"(pure recency {curve['0.0']:+.3f}, pure content {curve['1.0']:+.3f}, "
              f"gain {curve[best_a]-curve['1.0']:+.3f})")
    out["P4_positional_prior_blend"] = dict(alphas=[f"{a:.1f}" for a in ALPHAS], per_layer=p4)

    # =============================================== P3 block-size curve
    p3: Dict = {}
    # Selection FRACTION is the variable that matters, not absolute k. A 4096-
    # token probe naturally operates at 50% selection; a 1M-token context
    # operates near 1%. Compression error behaves very differently at the two
    # ends, so sweep the fraction explicitly instead of reporting one k.
    FRACS = (0.5, 0.25, 0.125, 0.0625, 0.03125)
    MS = (1, 2, 4, 8)
    l3 = layer_ids[len(layer_ids) // 2]
    Krot, Kde = layer_pack(kv[l3][0][0, :, :T, :], cos_b, sin_b)
    Kde_n = norm_rows(Kde.reshape(H * T, D)).reshape(H, T, D)
    for bs in BLOCK_SWEEP:
        if T % bs or bs < 2:
            continue
        nb = T // bs
        Cs: Dict[int, torch.Tensor] = {}
        for m in MS:
            if bs % m:
                continue
            C = centroids(Kde, bs, m)                       # [m, nb, H, D]
            Cs[m] = norm_rows(C.reshape(-1, D)).reshape(m, nb, H, D)
        bucket: Dict[str, List[float]] = {}
        rng = np.random.RandomState(0)
        for p in rng.choice(T, size=N_QUERY, replace=False):
            qde_n = norm_rows(Kde[:, p, :])
            per_tok = torch.einsum("hd,htd->ht", qde_n, Kde_n)
            truth = per_tok[:, :nb * bs].reshape(H, nb, bs).mean(dim=2).mean(dim=0)
            t_order = torch.argsort(truth, descending=True).numpy()
            for m in MS:
                if m not in Cs:
                    continue
                sc = torch.einsum("hd,mnhd->mn", qde_n, Cs[m]).mean(dim=0)
                o = torch.argsort(sc, descending=True).numpy()
                for fr in FRACS:
                    k = max(1, int(round(fr * nb)))
                    bucket.setdefault(f"m{m}|{fr}", []).append(recall_at_k(o, t_order, k))
        entry = dict(n_index_entries=int(nb), index_bytes=int(nb * H * D * 4))
        for kk, v in bucket.items():
            entry[kk] = agg(v)
        p3[f"block{bs}"] = entry
        line = "  ".join(
            f"m{m}@12.5%={np.mean(bucket[f'm{m}|0.125']):.3f}"
            for m in MS if f"m{m}|0.125" in bucket)
        half = np.mean(bucket["m1|0.5"]) if "m1|0.5" in bucket else float("nan")
        print(f"  block={bs:>3} entries={nb:>4} index={nb*H*D*4/1024:6.1f}KiB | "
              f"{line} | m1@50%={half:.3f}")
    out["P3_blocksize_curve"] = dict(layer=l3, fracs=list(FRACS), ms=list(MS),
                                     per_block=p3, bytes_per_entry=H * D * 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs")
    ap.add_argument("--maxlen", type=int, default=4096)
    ap.add_argument("--models", nargs="*",
                    default=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    layer_ids = [0, 1, 6, 12, 18, 23] if not a.quick else [0, 12]
    results = {}
    for m in a.models:
        try:
            results[m] = audit_model(m, a.maxlen, layer_ids, a.outdir, a.quick)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[m] = dict(error=str(exc))

    os.makedirs(a.outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(a.outdir, f"real_kv_audit_og_v2_{ts}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
