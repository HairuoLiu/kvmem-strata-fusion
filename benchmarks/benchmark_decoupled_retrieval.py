#!/usr/bin/env python3
"""
Benchmark: Decoupled Multi-Centroid Retrieval vs Naive Strata Block-Mean
Evaluates recall@topk across real Wikipedia articles on Qwen2.5-0.5B-Instruct.
Audits under the rigorous Token-Level Mean-of-Cosine Attention Ground Truth:
  Truth:    mean_{t in block} cos(q_token, k_token(t))
  Arm A:    Naive Strata Mean-K pooling (rotated keys)
  Arm Bmid: de-RoPE -> mean -> re-RoPE at block midpoint (v1 position-mismatch artefact)
  Arm B:    Position-matched monolithic de-RoPE mean (m=1)
  Decoupled K=2: Dual sub-centroids per block (m=2)
  Decoupled K=4: Quad sub-centroids per block (m=4)

Produces verifiable audit JSON artifact at docs/decoupled_retrieval_benchmark.json.
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer, AutoModelForCausalLM
from benchmarks.real_kv_audit_og_v2 import (
    fetch_wiki_corpus,
    get_kv,
    rope_inverse,
    rope_at_batch,
    norm_rows,
    centroids,
    block_means,
    recall_at_k
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK = 32

def run_decoupled_benchmark(max_len: int = 4096, topk: int = 64, n_query: int = 64, out_json: str = "docs/decoupled_retrieval_benchmark.json"):
    print("=" * 75)
    print(f"Decoupled Retrieval Benchmark on {MODEL_ID}")
    print("Ground Truth: Exact Token-Level Mean-of-Cosine Attention (Mean-of-Cos)")
    print("=" * 75)
    
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

    K_rot = kv[mid_layer][0][0, :, :T, :].float()  # [H, T, D]
    H, D = K_rot.shape[0], K_rot.shape[2]

    rot = model.model.rotary_emb
    pos = torch.arange(0, T, dtype=torch.long).unsqueeze(0)
    cos, sin = rot(K_rot.unsqueeze(0), pos)
    cos_b, sin_b = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()

    K_de = rope_inverse(K_rot.unsqueeze(0), cos_b, sin_b).squeeze(0)  # [H, T, D]
    Kde_n = norm_rows(K_de.reshape(H * T, D)).reshape(H, T, D)

    # Compute block representations
    M_rot = block_means(K_rot, BLOCK)                               # [nb, H, D]
    M_de = block_means(K_de, BLOCK)                                 # [nb, H, D]
    mid_pos = torch.arange(n_blocks) * BLOCK + BLOCK // 2
    M_re = torch.stack([rope_at_batch(M_de[:, h, :], mid_pos, cos_b, sin_b) for h in range(H)], dim=1)

    Mr_n = norm_rows(M_rot.reshape(-1, D)).reshape(n_blocks, H, D)
    Md_n = norm_rows(M_de.reshape(-1, D)).reshape(n_blocks, H, D)
    Mre_n = norm_rows(M_re.reshape(-1, D)).reshape(n_blocks, H, D)

    # Sub-centroids for K=2 and K=4
    C2 = centroids(K_de, BLOCK, 2)  # [2, nb, H, D]
    C4 = centroids(K_de, BLOCK, 4)  # [4, nb, H, D]
    C2_n = norm_rows(C2.reshape(-1, D)).reshape(2, n_blocks, H, D)
    C4_n = norm_rows(C4.reshape(-1, D)).reshape(4, n_blocks, H, D)

    rng = np.random.RandomState(42)
    q_positions = rng.choice(T, size=min(n_query, T), replace=False)

    results = {
        "Strata_Naive_A": [],
        "Monolithic_Bmid": [],
        "Monolithic_B_matched": [],
        "Decoupled_K2_Mean": [],
        "Decoupled_K2_Max": [],
        "Decoupled_K4_Mean": [],
        "Decoupled_K4_Max": [],
        "Random_Baseline": []
    }

    k = min(topk, n_blocks)

    for p in q_positions:
        q_de = K_de[:, p, :]  # [H, D]
        q_de_n = norm_rows(q_de)
        q_rot = rope_at_batch(q_de, torch.tensor([int(p)] * H), cos_b, sin_b)
        q_rot_n = norm_rows(q_rot)

        # 1. Ground Truth: Mean-of-Cosine attention over individual tokens in each block
        per_tok = torch.einsum("hd,htd->ht", q_de_n, Kde_n)  # [H, T]
        truth = per_tok[:, :n_blocks * BLOCK].reshape(H, n_blocks, BLOCK).mean(dim=2).mean(dim=0)
        t_order = torch.argsort(truth, descending=True).numpy()

        # 2. Arm A: Naive Strata (Mean over rotated keys)
        score_A = torch.einsum("hd,nhd->n", q_rot_n, Mr_n)
        results["Strata_Naive_A"].append(recall_at_k(torch.argsort(score_A, descending=True).numpy(), t_order, k))

        # 3. Arm Bmid: v1 Re-RoPE at block midpoint
        score_Bmid = torch.einsum("hd,nhd->n", q_de_n, Mre_n)
        results["Monolithic_Bmid"].append(recall_at_k(torch.argsort(score_Bmid, descending=True).numpy(), t_order, k))

        # 4. Arm B: Position-matched de-RoPE monolithic mean (m=1)
        score_B = torch.einsum("hd,nhd->n", q_de_n, Md_n)
        results["Monolithic_B_matched"].append(recall_at_k(torch.argsort(score_B, descending=True).numpy(), t_order, k))

        # 5. Decoupled K=2: Mean and Max over 2 sub-centroids
        sc2 = torch.einsum("hd,mnhd->mn", q_de_n, C2_n)
        sc2_mean = sc2.mean(dim=0)
        sc2_max = sc2.max(dim=0).values
        results["Decoupled_K2_Mean"].append(recall_at_k(torch.argsort(sc2_mean, descending=True).numpy(), t_order, k))
        results["Decoupled_K2_Max"].append(recall_at_k(torch.argsort(sc2_max, descending=True).numpy(), t_order, k))

        # 6. Decoupled K=4: Mean and Max over 4 sub-centroids
        sc4 = torch.einsum("hd,mnhd->mn", q_de_n, C4_n)
        sc4_mean = sc4.mean(dim=0)
        sc4_max = sc4.max(dim=0).values
        results["Decoupled_K4_Mean"].append(recall_at_k(torch.argsort(sc4_mean, descending=True).numpy(), t_order, k))
        results["Decoupled_K4_Max"].append(recall_at_k(torch.argsort(sc4_max, descending=True).numpy(), t_order, k))

        # 7. Random Baseline
        rand_order = rng.permutation(n_blocks)
        results["Random_Baseline"].append(recall_at_k(rand_order, t_order, k))

    summary = {}
    print("\n" + "=" * 75)
    print(f"Empirical Results on Layer {mid_layer} (n={len(q_positions)} queries, top-{k} retrieval):")
    print("=" * 75)
    base_a = np.mean(results["Strata_Naive_A"])
    for name, vals in results.items():
        mean_val = float(np.mean(vals))
        se_val = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        ci = [mean_val - 1.96 * se_val, mean_val + 1.96 * se_val]
        gain = ((mean_val - base_a) / base_a) * 100.0
        summary[name] = {
            "mean_recall": mean_val,
            "se": se_val,
            "ci_95": ci,
            "gain_vs_strata_naive_pct": gain,
            "raw_scores": vals
        }
        print(f"  {name:<22} : recall@{k} = {mean_val:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])  Gain: {gain:+.2f}%")

    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_id": MODEL_ID,
        "tokens": T,
        "n_blocks": n_blocks,
        "mid_layer": mid_layer,
        "n_query": len(q_positions),
        "top_k": k,
        "ground_truth_definition": "mean_t cos(q_de, k_de(t)) [exact token-level attention]",
        "summary": {k: {kk: vv for kk, vv in v.items() if kk != "raw_scores"} for k, v in summary.items()}
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\n[SAVED] Benchmark audit artifact written to: {out_json}")


if __name__ == "__main__":
    run_decoupled_benchmark()
