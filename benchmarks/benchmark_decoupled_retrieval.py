#!/usr/bin/env python3
"""
Benchmark: Decoupled Multi-Centroid Retrieval vs Naive Strata Block-Mean
Evaluates recall@topk across real Wikipedia articles on Qwen2.5-0.5B-Instruct.
Compares:
  - Arm A (Naive Strata Mean-K pooling)
  - Arm B (de-RoPE monolithic mean)
  - Decoupled Sub-Centroids (K=2, K=4)
"""

import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from benchmarks.real_kv_audit_og import fetch_wiki_corpus, get_kv, rope_forward, rope_inverse, rope_at

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK = 32

def run_decoupled_benchmark(max_len: int = 4096, topk: int = 64, n_query: int = 64):
    print("=" * 70)
    print("Starting Decoupled Retrieval Benchmark on Qwen2.5-0.5B-Instruct")
    print("=" * 70)
    
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    corpus, provenance = fetch_wiki_corpus(max_chars=60000)
    ids = tok(corpus, return_tensors="pt").input_ids[:, :max_len]
    T = (ids.shape[1] // BLOCK) * BLOCK
    ids = ids[:, :T]
    n_blocks = T // BLOCK
    print(f"[Corpus] {provenance} | Tokens: {T} | Blocks: {n_blocks}")

    with torch.no_grad():
        out = model(ids, use_cache=True)
    n_layers = len(out.past_key_values)
    kv = get_kv(out.past_key_values, n_layers)
    mid_layer = n_layers // 2

    K_mid = kv[mid_layer][0][0, :, :T, :].double()  # [H, T, D]
    H, D = K_mid.shape[0], K_mid.shape[2]

    rot = model.model.rotary_emb
    pos = torch.arange(0, T, dtype=torch.long).unsqueeze(0)
    cos, sin = rot(K_mid.unsqueeze(0), pos)
    cos_b, sin_b = cos.unsqueeze(1).double(), sin.unsqueeze(1).double()

    Kd = rope_inverse(K_mid.unsqueeze(0), cos_b, sin_b).squeeze(0)  # [H, T, D]

    armA = torch.zeros(n_blocks, D, dtype=torch.float64)
    armB = torch.zeros(n_blocks, D, dtype=torch.float64)
    truth = torch.zeros(n_blocks, D, dtype=torch.float64)
    sub2 = torch.zeros(n_blocks, 2, D, dtype=torch.float64)
    sub4 = torch.zeros(n_blocks, 4, D, dtype=torch.float64)

    for h in range(H):
        for b in range(n_blocks):
            s, e = b * BLOCK, (b + 1) * BLOCK
            rot_blk, der_blk = K_mid[h, s:e, :], Kd[h, s:e, :]
            mid = s + BLOCK // 2

            armA[b] += rot_blk.mean(dim=0) / H
            armB[b] += rope_at(der_blk.mean(dim=0), mid, cos_b, sin_b, D) / H
            truth[b] += der_blk.mean(dim=0) / H

            # Sub-centroids (tokens 0-15 and 16-31)
            sub2[b, 0] += der_blk[:16].mean(dim=0) / H
            sub2[b, 1] += der_blk[16:].mean(dim=0) / H

            # Sub-centroids (4 groups of 8 tokens)
            for sub_i in range(4):
                sub4[b, sub_i] += der_blk[sub_i*8:(sub_i+1)*8].mean(dim=0) / H

    def cos_rank(q, M):
        Mn = M / (M.norm(dim=-1, keepdim=True) + 1e-12)
        return Mn @ q

    def max_cos_rank_multi(q, sub_centroids):
        norms = sub_centroids.norm(dim=-1, keepdim=True) + 1e-12
        normed = sub_centroids / norms
        scores = torch.einsum("bsd,d->bs", normed, q)
        max_scores, _ = scores.max(dim=-1)
        return max_scores

    k = min(topk, n_blocks)
    rng = np.random.RandomState(42)
    q_blocks = rng.choice(n_blocks, size=min(n_query, n_blocks), replace=False)

    results = {"Strata_Naive_A": [], "Monolithic_B": [], "Decoupled_K2": [], "Decoupled_K4": []}

    for qb in q_blocks:
        q = truth[qb] / (truth[qb].norm() + 1e-12)
        gt = set(torch.topk(cos_rank(q, truth), k).indices.tolist())

        # 1. Strata Naive Mean-K (Arm A)
        top_A = set(torch.topk(cos_rank(q, armA), k).indices.tolist())
        results["Strata_Naive_A"].append(len(gt & top_A) / k)

        # 2. Monolithic de-RoPE (Arm B)
        top_B = set(torch.topk(cos_rank(q, armB), k).indices.tolist())
        results["Monolithic_B"].append(len(gt & top_B) / k)

        # 3. Decoupled Sub-Centroids (K=2)
        top_sub2 = set(torch.topk(max_cos_rank_multi(q, sub2), k).indices.tolist())
        results["Decoupled_K2"].append(len(gt & top_sub2) / k)

        # 4. Decoupled Sub-Centroids (K=4)
        top_sub4 = set(torch.topk(max_cos_rank_multi(q, sub4), k).indices.tolist())
        results["Decoupled_K4"].append(len(gt & top_sub4) / k)

    print("\n" + "=" * 70)
    print("Empirical Results on Layer 12 (n=64 queries, top-64 retrieval):")
    print("=" * 70)
    for name, vals in results.items():
        mean_val = float(np.mean(vals))
        se_val = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        ci = [mean_val - 1.96 * se_val, mean_val + 1.96 * se_val]
        gain = ((mean_val - np.mean(results["Strata_Naive_A"])) / np.mean(results["Strata_Naive_A"])) * 100
        print(f"  {name:<20} : recall@64 = {mean_val:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])  Gain: {gain:+.2f}%")

if __name__ == "__main__":
    run_decoupled_benchmark()
