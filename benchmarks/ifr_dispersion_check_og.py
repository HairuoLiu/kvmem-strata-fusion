#!/usr/bin/env python3
"""
ifr_dispersion_check_og.py -- audit of Track A's c5e76c8 anti-collapse dispersion fix.

WHAT THEY ADDED
---------------
`kvmem_fusion/ifr.py::compute_normalized_dispersion()` returns

    max_i || k_i/||k_i|| - mean_j(k_j/||k_j||) ||          <- a MAX

and the docstring calibrates it as "Euclidean MAD ~8.32 but normalized MAD
~0.7965". The 0.7965 figure comes from our OG v1 audit, where it was a MEAN
(`mad()` = mean per-token distance from the block mean), computed over ALL
tokens of a layer -- not a per-block MAX.

max and mean are different statistics. This script measures all four
combinations on REAL Qwen keys so the threshold can be set on the right one,
and reports the resulting anti-collapse trigger rate.

Usage: python benchmarks/ifr_dispersion_check_og.py [--maxlen 4096]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from real_kv_audit_og_v2 import (BLOCK, fetch_wiki_corpus, get_kv,
                                 layer_pack, norm_rows)


def block_stats(Kde: torch.Tensor, block: int):
    """Kde [H,T,D] de-RoPE'd -> per-block (raw_max, raw_mean, norm_max, norm_mean)."""
    H, T, D = Kde.shape
    nb = T // block
    raw_max, raw_mean, n_max, n_mean = [], [], [], []
    for h in range(H):
        for b in range(nb):
            blk = Kde[h, b * block:(b + 1) * block, :].numpy().astype(np.float64)
            m = blk.mean(axis=0)
            d = np.linalg.norm(blk - m, axis=-1)
            raw_max.append(float(d.max()))
            raw_mean.append(float(d.mean()))
            nbk = blk / np.maximum(np.linalg.norm(blk, axis=-1, keepdims=True), 1e-12)
            nm = nbk.mean(axis=0)
            nd = np.linalg.norm(nbk - nm, axis=-1)
            n_max.append(float(nd.max()))
            n_mean.append(float(nd.mean()))
    return (np.array(raw_max), np.array(raw_mean),
            np.array(n_max), np.array(n_mean))


def triggers(metric: np.ndarray, theta: float) -> float:
    return float((metric > theta).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxlen", type=int, default=4096)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--outdir", default="docs")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)

    corpus, prov = fetch_wiki_corpus()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model.eval()
    ids = tok(corpus, return_tensors="pt").input_ids[:, :a.maxlen]
    T = (ids.shape[1] // BLOCK) * BLOCK
    ids = ids[:, :T]
    with torch.no_grad():
        out = model(ids, use_cache=True)
    n_layers = len(out.past_key_values)
    kv = get_kv(out.past_key_values, n_layers)
    K0 = kv[0][0][:, :, :T, :]
    H, D = K0.shape[1], K0.shape[3]
    rot = model.model.rotary_emb
    with torch.no_grad():
        cos, sin = rot(K0, torch.arange(0, T).unsqueeze(0))
    cos_b, sin_b = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
    print(f"[{a.model}] layers={n_layers} kv_heads={H} head_dim={D} "
          f"tokens={T} blocks={T // BLOCK}")
    print(f"[corpus] {prov}\n")

    print(f"{'layer':>5} | {'raw_max':>8} {'raw_mean':>8} | "
          f"{'norm_max':>8} {'norm_mean':>8} | {'max/mean':>8}")
    agg = {"raw_max": [], "raw_mean": [], "n_max": [], "n_mean": []}
    for l in range(n_layers):
        _, Kde = layer_pack(kv[l][0][0, :, :T, :], cos_b, sin_b)
        rm, ra, nm, nn = block_stats(Kde, BLOCK)
        agg["raw_max"].append(rm.mean())
        agg["raw_mean"].append(ra.mean())
        agg["n_max"].append(nm.mean())
        agg["n_mean"].append(nn.mean())
        if l % 6 == 0 or l == n_layers - 1:
            print(f"{l:>5} | {rm.mean():8.3f} {ra.mean():8.3f} | "
                  f"{nm.mean():8.3f} {nn.mean():8.3f} | {nm.mean()/nn.mean():8.2f}")

    R = {k: float(np.mean(v)) for k, v in agg.items()}
    print(f"\n[all layers] raw_max={R['raw_max']:.3f}  raw_mean={R['raw_mean']:.3f}  "
          f"norm_max={R['n_max']:.3f}  norm_mean={R['n_mean']:.3f}")
    print(f"[units] their metric is a MAX; the 0.7965 they calibrated against is a MEAN.")
    print(f"        max/mean ratio = {R['n_max']/R['n_mean']:.2f}  -> a threshold copied "
          f"from the mean is off by roughly this factor.\n")

    # full distributions for trigger-rate evaluation
    all_rm, all_ra, all_nm, all_nn = [], [], [], []
    for l in range(n_layers):
        _, Kde = layer_pack(kv[l][0][0, :, :T, :], cos_b, sin_b)
        rm, ra, nm, nn = block_stats(Kde, BLOCK)
        all_rm.append(rm); all_ra.append(ra); all_nm.append(nm); all_nn.append(nn)
    all_rm = np.concatenate(all_rm); all_ra = np.concatenate(all_ra)
    all_nm = np.concatenate(all_nm); all_nn = np.concatenate(all_nn)

    print("Anti-collapse trigger rate (share of blocks that would be split):")
    print(f"  {'threshold':>10} | {'raw_max':>9} | {'norm_max':>9} | {'norm_mean':>10}")
    for th in (0.85, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0):
        print(f"  {th:>10} | {triggers(all_rm, th)*100:8.1f}% | "
              f"{triggers(all_nm, th)*100:8.1f}% | {triggers(all_nn, th)*100:9.1f}%")
    print("\n  NOTE: `AntiCollapseSplitter` defaults to use_normalized=False, i.e. it")
    print("  gates on raw_max with threshold 0.85 -> read the first column.")

    # what threshold gives a sane (say 5%) trigger rate on each metric?
    out = dict(model=a.model, tokens=int(T), blocks=int(T // BLOCK),
               per_layer_mean={k: [float(x) for x in v] for k, v in agg.items()},
               overall=R,
               trigger_rate_at={str(th): {
                   "raw_max": triggers(all_rm, th),
                   "norm_max": triggers(all_nm, th),
                   "norm_mean": triggers(all_nn, th)} for th in
                   (0.85, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0)},
               pctl=dict(
                   raw_max=[float(np.percentile(all_rm, p)) for p in (50, 90, 95, 99)],
                   norm_max=[float(np.percentile(all_nm, p)) for p in (50, 90, 95, 99)],
                   norm_mean=[float(np.percentile(all_nn, p)) for p in (50, 90, 95, 99)]),
               theta_for_5pct=dict(
                   raw_max=float(np.percentile(all_rm, 95)),
                   norm_max=float(np.percentile(all_nm, 95)),
                   norm_mean=float(np.percentile(all_nn, 95))),
               corpus=prov)
    os.makedirs(a.outdir, exist_ok=True)
    p = os.path.join(a.outdir, f"ifr_dispersion_check_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {p}")
    print(f"[recommend] theta_norm_max for a 5% split rate = "
          f"{out['theta_for_5pct']['norm_max']:.3f}  (they ship 0.85)")


if __name__ == "__main__":
    main()
