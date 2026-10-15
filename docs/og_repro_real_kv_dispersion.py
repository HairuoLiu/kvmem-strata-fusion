#!/usr/bin/env python3
"""
OG reproduction: measure REAL model KV intra-block dispersion and IFR retrieval
latency on that real KV.

Used to falsify the Phase 1 benchmark (`benchmarks/benchmark_flashnext_hybrid_e2e.py`,
commit 96f7865), whose synthetic blocks are built with np.tile(k_base, (32, 1)) and
therefore have norm_dispersion == 0.0000 exactly -- a value that cannot occur in any
real model, and one that makes IFR's defining anti-collapse doublet splitting
mathematically unable to fire.

Run from the repo root with the project venv:
    .venv/bin/python docs/og_repro_real_kv_dispersion.py

Requires local Qwen3-0.6B weights in the HF cache (the only real weights present).
"""
import glob, os, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from transformers import AutoModelForCausalLM, AutoTokenizer
from kvmem_fusion.ifr import IFRRetriever, IFRBlock, AntiCollapseSplitter

MP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"))[0]

tok = AutoTokenizer.from_pretrained(MP)
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, attn_implementation="sdpa")
model.eval()

# Real natural-language text from the repo's own documents (varied prose).
text = ""
for p in sorted(glob.glob(os.path.join(_HERE, "*.md"))):
    try:
        text += open(p, encoding="utf-8", errors="ignore").read() + "\n\n"
    except Exception:
        pass
ids = tok(text, return_tensors="pt")["input_ids"][0]
print(f"corpus tokens available: {ids.shape[0]}")
CTX = 4096
ids = ids[:CTX].unsqueeze(0)

captured = {}
layer0 = model.model.layers[0].self_attn


def hook(mod, inp, out):
    captured["k"] = out.detach()


h = layer0.k_proj.register_forward_hook(hook)
with torch.no_grad():
    model(ids)
h.remove()

k = captured["k"]  # [1, S, num_kv_heads*head_dim]
S = k.shape[1]
cfg = model.config
nkv = cfg.num_key_value_heads
hd = cfg.head_dim
k = k.view(1, S, nkv, hd).transpose(1, 2)[0]  # [nkv, S, hd]
print(f"layer0 K: {tuple(k.shape)}  (num_kv_heads={nkv}, head_dim={hd}, S={S})")

# Mean over KV heads -> [S, hd], i.e. one vector per token (matches IFR mean-K space)
K = k.float().mean(dim=0).numpy()
print(f"per-token K matrix: {K.shape}")

BLOCK = 32
n_blocks = S // BLOCK
disps = []
sp = AntiCollapseSplitter(dispersion_threshold=0.868, use_normalized=True)
r = IFRRetriever(dim=hd, target_top_k=64)
fires = 0
for b in range(n_blocks):
    km = K[b * BLOCK:(b + 1) * BLOCK]
    blk = IFRBlock(block_id=f"r{b}", tokens=list(range(b * BLOCK, (b + 1) * BLOCK)),
                   keys=km, values=km, orig_pos_start=b * BLOCK)
    disps.append(blk.norm_dispersion)
    is_split, _ = sp.inspect_and_split(blk)
    fires += int(is_split)
    r.add_block(blk)

disps = np.array(disps)
print()
print(f"REAL Qwen3-0.6B layer-0 K, {n_blocks} blocks of 32 tokens:")
print(f"  norm_dispersion  mean={disps.mean():.4f}  median={np.median(disps):.4f} "
      f" p95={np.percentile(disps,95):.4f}  max={disps.max():.4f}")
print(f"  anti-collapse splits (theta=0.868): {fires}/{n_blocks} = {fires/n_blocks*100:.1f}%")

# Retrieval latency on REAL KV
rng = np.random.RandomState(0)
qs = [K[i] for i in rng.choice(S, 20, replace=False)]
r.retrieve(qs[0], top_k=56)
ts = []
for q in qs:
    t = time.perf_counter()
    bids, _ = r.retrieve(q, top_k=56)
    ts.append((time.perf_counter() - t) * 1000)
ts = np.array(ts)
print()
print(f"REAL-KV IFR retrieval (dim={hd}, top_k=56, {n_blocks} blocks = {n_blocks*BLOCK} tokens):")
print(f"  mean={ts.mean():.3f} ms   p95={np.percentile(ts,95):.3f} ms   n_ret={len(bids)}")
print(f"  -> linear extrapolation to 10M tokens ({10_000_000/(n_blocks*BLOCK):.1f}x): "
      f"{ts.mean()*10_000_000/(n_blocks*BLOCK):.1f} ms   (E-gate = 350 ms)")
