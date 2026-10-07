# Track A: Real Model Evaluation & Needle-in-a-Haystack Benchmark Report

> **Lead Researcher**: Track A Architecture Team (Calibrated via External Review Audit)  
> **Status**: **Synthetic Self-Consistency Validation（合成自洽验证）**  
> **Hardware Environment**: Apple Silicon / CPU Host (PyTest 73/73 passing)  
> **Hardware Gates**: G-CAS-1 (I/O DMA) & G-CAS-2 (Kernel GEMM) marked as `UNMEASURED (Design Intent)`  
> **Target Contexts Evaluated**: 8,192 (8K) to 32,768 (32K) tokens, with 1M tokens scale extrapolation  

---

## 1. Executive Summary & Calibration Notice

Following external peer critique (`docs/external_review_critique.md`), this report explicitly distinguishes between:
1. **Mathematical Self-Consistency (Verified)**: Proves that the fused algorithm pipeline (CASA $\to$ IFR $\to$ UBBA $\to$ LADDER) is internally self-consistent, numerical errors in de-RoPE / PagedAttention are below machine epsilon ($< 10^{-12}$), and tier-bias cancels Jensen's inequality drift under synthetic Gaussian assumptions.
2. **Empirical LLM Limitations (Calibrated)**:
   - Values of $\Delta\text{PPL} = 0.0000$ and $\text{Cosine} = 1.0000$ were mathematically constructed properties of the synthetic test harness (where target logits were calibrated to dominate attention mass). Real next-token prediction perplexity on natural text requires P0 model weights (e.g. Qwen2.5 / GPT-2).
   - Problem scale at 32K token context represents a ~10% selection rate (103 / 1,024 blocks), whereas 10M tokens represents a 0.033% selection rate (300× harder).
   - Hardware bandwidths $\ge 45\text{ GB/s}$ are design targets, unmeasured on CPU/Mac dev environments.

### Core Benchmark Summary (Synthetic Calibration)

| System / Method | Needle Recall | Needle Rank | Context CosSim | Avg PPL Drift ($\Delta\text{PPL}$) | Compression Ratio | Hierarchical Footprint (32K) | U-E-F-C Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full-Context (FP8 Baseline)** | **100.0%** | **1.0** | **1.0000** | **0.0000** | **1.0x** | 1024.0 KB | REFERENCE |
| **Naive Flat Compression (INT2)** | **0.0%** | **> 999999** | **0.0639** | **+922.07** | **4.0x** | 256.0 KB | **FAIL** |
| **KVMem-Strata-Fusion (Pipeline)** | **100.0%** | **1.0** | **1.0000\*** | **0.0000\*** | **7.7x** | **130.4 KB** | **PASS (Synthetic)** |

*\*Note: 1.0000 CosSim / 0.0000 PPL Drift reflects synthetic self-consistency under target-logit alignment; see §4 for realistic SNR sweep.*

---

## 2. Experimental Methodology & Architecture

### 2.1 The Long-Context Needle-in-a-Haystack Setup
- **Atomic Unit**: 32 tokens per block, head dimension $d = 64$.
- **Context Lengths**: 8,192 (256 blocks), 16,384 (512 blocks), and 32,768 (1024 blocks).
- **Haystack Generation**: Structured background topic clusters with local semantic coherence, realistic intra-block variance ($\sigma \approx 0.02$), and RoPE rotary position embeddings.
- **Needle Embedding**: Bursty factual token injected at variable depths (10%, 25%, 50%, 75%, 90%). The needle key $k_{\text{needle}}$ in unrotated space aligns with the needle query topic $q_{\text{semantic}}$ ($\|k_{\text{needle}}\| = 2.5$).
- **Rotary Position Alignment**: Attention query $q_{\text{raw}}$ matches relative RoPE phase at logical position $L$: $(R_L q)^T (R_n k) = q^T R_{L-n} k$.

### 2.2 Why Naive Flat Compression Collapses
Naive KV compression schemes exhibit two fatal failure modes:
1. **Mean-K Dilution**: In a 32-token block containing 1 burst needle token and 31 background tokens, naive block averaging dilutes the needle signal by $1/32$. Coarse cluster probes rank the needle block at rank $> 50$ (or $> 150$ at 16K/32K), dropping it from the candidate set (0% recall).
2. **Jensen's Inequality Attention Mass Theft**: When uncalibrated quantization noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is added without tier bias, the softmax expectation inflates:
   $$\mathbb{E}[\exp(s + \epsilon)] = \exp(s) \exp\left(\frac{\sigma^2}{2}\right)$$
   For INT2 ($\rho = 0.343$, $\sigma \approx 0.635$), background tokens receive an artificial $+22\%$ logit boost, stealing attention mass and corrupting next-token predictions.

---

## 3. Detailed Benchmark Results (Context Sweep 8K - 32K)

| Context | Depth | Full-Context Recall | Naive Recall | Fusion Recall | Full Footprint | Naive Footprint | Fusion Footprint | Fusion Ratio | $\Delta\text{PPL}$ (Synth) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8,192** | 10% | 100.0% | 0.0% | **100.0%** | 256.0 KB | 64.0 KB | **34.2 KB** | **7.5x** | **0.0000** |
| 8,192 | 50% | 100.0% | 0.0% | **100.0%** | 256.0 KB | 64.0 KB | **34.4 KB** | **7.4x** | **0.0000** |
| 8,192 | 90% | 100.0% | 0.0% | **100.0%** | 256.0 KB | 64.0 KB | **34.4 KB** | **7.4x** | **0.0000** |
| **16,384** | 10% | 100.0% | 0.0% | **100.0%** | 512.0 KB | 128.0 KB | **66.2 KB** | **7.7x** | **0.0000** |
| 16,384 | 50% | 100.0% | 0.0% | **100.0%** | 512.0 KB | 128.0 KB | **66.4 KB** | **7.7x** | **0.0000** |
| 16,384 | 90% | 100.0% | 0.0% | **100.0%** | 512.0 KB | 128.0 KB | **66.4 KB** | **7.7x** | **0.0000** |
| **32,768** | 10% | 100.0% | 0.0% | **100.0%** | 1024.0 KB | 256.0 KB | **130.2 KB** | **7.9x** | **0.0000** |
| 32,768 | 50% | 100.0% | 0.0% | **100.0%** | 1024.0 KB | 256.0 KB | **130.4 KB** | **7.9x** | **0.0000** |
| 32,768 | 90% | 100.0% | 0.0% | **100.0%** | 1024.0 KB | 256.0 KB | **130.4 KB** | **7.9x** | **0.0000** |

---

## 4. Empirical Audits Addressing External Critique

### 4.1 SNR Sensitivity & Failure Mode Analysis (`benchmarks/eval_snr_sensitivity.py`)
To prevent circular logic (detecting artificially high dispersion), we swept the Signal-to-Noise Ratio $\|k_{\text{needle}}\| / \|k_{\text{bg}}\|$ from 1.0 (unseparable) to 5.0:

| SNR Level | Needle Burst | Full-Context Recall | Standard KVMem Mean-K | Naive INT2 Recall | IFR Doublet Recall | Realistic CosSim |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.00** | 1.00 | 100.0% | **80.0%** | 100.0% | **100.0%** | 0.2830 |
| **1.25** | 1.25 | 100.0% | **80.0%** | 100.0% | **100.0%** | 0.2817 |
| **1.50** | 1.50 | 100.0% | 100.0% | 100.0% | 100.0% | 0.2804 |
| **2.00** | 2.00 | 100.0% | 100.0% | 100.0% | 100.0% | 0.2776 |
| **3.00** | 3.00 | 100.0% | 100.0% | 100.0% | 100.0% | 0.2645 |
| **5.00** | 5.00 | 100.0% | 100.0% | 100.0% | 100.0% | 0.2662 |

**Key Finding**:
- Under low SNR (1.00 - 1.25), standard KVMem Mean-K suffers a **20% recall drop**, whereas IFR's doublet candidate generation maintains 100% recall.
- Without artificial logit dominance, uncompressed-to-compressed cosine similarity across the full context is **~0.28**, accurately exposing the compression trade-off.

### 4.2 False Alarm Rate on Background Blocks (Assumption A15)
Testing pure background blocks against a fixed dispersion threshold ($\theta = 0.85$):
- At $\sigma = 0.02$: False Alarm Rate = **0.00%** (p95 dispersion 0.199).
- At $\sigma = 0.05$: False Alarm Rate = **0.00%** (p95 dispersion 0.497).
- At $\sigma = 0.10$: False Alarm Rate = **99.85%** (p95 dispersion 0.994).
- At $\sigma \ge 0.15$: False Alarm Rate = **100.00%** (p95 dispersion $\ge 1.49$).

**Conclusion**: Fixed $\theta = 0.85$ is brittle under varied background noise. The production implementation must use **dynamic dispersion thresholds** ($\mu_{\text{block}} + 3\sigma_{\text{block}}$) to prevent false-alarm saturation.

### 4.3 Scale & Selection-Rate Joint Scaling (`benchmarks/eval_scale_selection_rate.py`)
Evaluating candidate pool expansion from 32K to 1M tokens with fixed $K=103$ block budget:

| Context | Candidate Blocks ($N$) | Selection Rate ($103/N$) | Standard KVMem (Low SNR) | IFR Doublet (Low SNR) | Index RAM Footprint | Re-RoPE Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32K** | 1,024 | 10.059% | 0.0% | **100.0%** | 0.4 MiB | Native ($\le 256\text{K}$) |
| **64K** | 2,048 | 5.029% | 0.0% | **100.0%** | 0.8 MiB | Native ($\le 256\text{K}$) |
| **128K** | 4,096 | 2.515% | 0.0% | **100.0%** | 1.5 MiB | Native ($\le 256\text{K}$) |
| **256K** | 8,192 | 1.257% | 0.0% | **100.0%** | 3.0 MiB | Native ($\le 256\text{K}$) |
| **512K** | 16,384 | 0.629% | 0.0% | **100.0%** | 6.0 MiB | Paged Table ($\gt 256\text{K}$) |
| **1M** | 32,768 | **0.314%** | 0.0% | **100.0%** | 12.0 MiB | Paged Table ($\gt 256\text{K}$) |

---

## 5. Architecture Governance: Subsystem Degradation

We formally adopt the external review recommendation to replace **Plan-level Elimination** with **Subsystem Degradation**:

1. **CASA PagedAttention Degradation**: If PagedAttention virtual table indexing encounters non-standard memory layouts, degrade gracefully to uncompressed local FIFO sliding window.
2. **IFR Doublet Degradation**: If dispersion detection saturates (false-alarm $> 10\%$), degrade dynamically to single-centroid Mean-K clustering with enlarged $n_{\text{probe}}$.
3. **UBBA Knapsack Degradation**: If Lagrangian knapsack solver fails SLA ($> 0.1\text{ ms}$), fallback to static 2-tier greedy allocation.
4. **LADDER Tier Degradation**: If Quad-Merge introduces non-recoverable high-frequency degradation, freeze compression at Tier L2 (FP8/INT4 4.35×) without INT2 merge.
