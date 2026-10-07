# 对 Track A/B/C 与外部专家评审产出的技术质疑与验证建议

> **致**：参与 KVMem-Strata-Fusion 评审的各位教授与工程团队
> **来自**：原始研究组（10 研究员 + 圆桌收敛，主报告见 `docs/master-research-report.md`）
> **日期**：2026-10-07
> **性质**：技术质疑（falsification request），**不是否定**。本文的唯一目的是让这些数字在真实模型上站得住。
> **配套**：本文件所有引用均给出 `文件:行号`，可逐条复核。

---

## 0. 先说我们认输的三处

在质疑之前，先明确承认贵组的三处成果**优于**我们的原方案。这不是客套，是我们建议并入主线的真实升级：

1. **K-Freeze 用 PagedAttention 页表实现，而非 Δ-QRoPE / Q-remap。**
   我们原方案把位置校正放在 Q 侧，并把它列为最大内核风险（内部评估置信度仅 50%，R4 自曝）。贵组指出：既然规范 K 已按全局位置预旋转为 `R_n k`，而 `(R_m q)ᵀ(R_n k) = qᵀR_{m−n}k`，**PagedAttention 的页表天然给出相对位置，无需任何重旋转**。这比我们简单得多，且直接消解了 Gate G-CAS-2。我们接受，并建议替换我们原来的表述。

2. **因果前缀哈希链（HiRadix Merkle DAG）。**
   我们有去重（L0 HiRadixTree），但从未形式化"因果性"——相同词面在不同上下文产生不同 KV 激活值，扁平分块哈希会造成跨会话语义污染。贵组的定理 2 是真正的正确性补强。

3. **G-CAS-1 的 7.8 GB/s 根因假设。**
   我们判定这是"三界矛盾（不被 DRAM/PCIe 解释）"并作为最大单点风险搁置。贵组提出可测假设：NVMe 小块（<64 KB）I/O 命令队列饥饿。这是一个可以被 `gdsio`/`fio` 直接证伪的假设，比我们的"未解"前进了一步。

以下质疑均建立在这三处贡献之上。**我们质疑的是证据强度，不是设计方向。**

---

## 1. 结论摘要（Verdict）

贵组的**设计方向**大多成立，但**证据不足以支撑报告中的数字与 Gate 结论**。

| 报告主张 | 我们的判定 | 原因 |
|---|---|---|
| 100% 针尖召回 / PPL 漂移 0.0000 / cosine 1.0000 | ❌ **不可采信** | 合成数据，未跑真实模型（§2.1） |
| 7.7×–7.85× 压缩 @32K | ⚠️ 仅可作合成验证 | 规模低于问题域两个数量级（§2.2） |
| G-CAS-1 PASS（≥45 GB/s） | ❌ **无效判定** | 用断言常量冒充实测（§2.3） |
| G-CAS-2 PASS（回退 ≤15%） | ❌ **无效判定** | 理论 + numpy 等价性，无 GPU（§2.3） |
| M4 bit-exact <1e-12 | ✅ 可接受 | 数值恒等性，与硬件无关 |
| U-E-F-C 全 PASS | ⚠️ **口径错误** | 与 10M 定义的门限不可比（§2.2） |
| UBBA 求解器 69.75 µs / 10K 块 | ✅ 可信（待补测量条件） | 纯 CPU，可实测 |
| 控制面 C++ vs Python 150.9× | ✅ 可信 | 同上 |

---

## 2. 致命问题（逐条）

### 2.1 【致命】benchmark 未跑任何真实模型

**主张**（`docs/track_a_benchmark_report.md` §1）："**Verified & Reproducible**"、"ΔPPL = 0.0000"、"Context Cosine Sim = 1.0000"、"100.0% Needle Recall"。

**证据**：
- `benchmarks/eval_end_to_end.py:122` — `self.rng = np.random.RandomState(seed)`
- `benchmarks/eval_end_to_end.py:132` — 注释原文：`Generates a synthetic long-context session:`
- `benchmarks/eval_end_to_end.py:148–177` — `query_semantic = rng.randn(self.head_dim)`；`raw_keys[idx] = bg_topic + diffuse_proj * query_semantic + 0.02 * rng.randn(self.head_dim)`；`v_tok = rng.randn(self.head_dim) * 0.1`
- 全仓库检索 `torch|transformers|from_pretrained|AutoModel|AutoModelForCausalLM` **仅命中** `kernels/fused_tier_bias_attention.py:39`，且该处为
  ```python
  try:
      import torch
      HAS_TORCH = True
  except ImportError:
      torch = None; HAS_TORCH = False
  ```
  即可选**张量后端**，**不是模型加载**。

**为什么这使结论失效**：

PPL 只能对真实 next-token 分布计算交叉熵。贵组报告的 ΔPPL 是对自造随机向量的度量，**不是语言模型困惑度**。同理，cosine = 1.0000 衡量的是贵组自己的合成流水线是否自洽，而非压缩是否保真。在合成数据上，一个恒等映射必然给出 1.0000——**这个数字是构造出来的，不是测出来的。**

一个通用判据，建议在组内立为规矩：

> **任何测量结果精确等于 `0.0000` 或 `1.0000`，都应首先被怀疑为解析恒等式而非测量值。** 真实系统（FP8 量化、PCIe 传输、softmax 浮点累加）不可能产出精确的 0。

**应当如何验证**：

1. **P0 最小可行（2 人日，无需 GPU）**：取 Qwen2.5-0.5B-Instruct（或 GPT-2 124M），`use_cache=True, return_dict_in_generate=True` 导出 `past_key_values`。真值 = **Full-Context 重算**（这是我们 F 门的定义）。PPL 用真实 logits 对真实 continuation 算 cross-entropy。
2. **校准门槛（不可跳过）**：在 **256K** 复现 LongMemEval-S **85.6%** / AgentLongBench **60.87%**（主报告假设 A11，实验手册 §3）。**复现不出这两个数，后面所有数字都不成立。**
3. 若暂时只能用合成数据，报告口径必须改为 **"合成自洽验证（synthetic self-consistency）"**，且**不得**出现 "Verified & Reproducible"、**不得**与 Full-Context 并列宣称 PASS。

---

### 2.2 【致命】规模低于问题域两个数量级，且难度不可比

**主张**：`docs/track_a_benchmark_report.md` §4 — U-E-F-C 五门全 PASS，对照的是我们为 **10M** 定义的门限。

**证据**：§3.1 仅测 8,192 / 16,384 / 32,768。

**为什么不可比**：

- 我们的问题域是 **1M–10M token**：10M ⇒ N = 312,500 个 32-token 块，NVMe footprint **324.2 GiB**。
- **选择率是核心难度变量**。我们算过：10M 下每 window 目标约 103 块 / 312,500 候选 = **0.033%**；而 32K 下是 103/1024 ≈ **10%**——**难度相差约 300 倍**。在 10% 选择率下召回 100%，完全不能外推到 0.033%。
- **索引常驻**：按 A2（index ≈ 1 KiB/token），10M 下索引 **9.5 GiB**；32K 下仅 32 MiB。贵组 Gate C 报 **0.005 GiB**，与我们的 ≤4 GiB@10M **不是同一个量**。
- **Gate E**：≤350 ms 是我们为 10M 定的；32K 下 18.4–150.4 ms 平凡成立。
- **最关键的一点**：主报告附录 C.3 已指出 **≤256K 不需要 re-RoPE**。贵组在 32K 宣称"终结 re-RoPE 语义摩擦"——**该区间本来就没有这个摩擦**，等于在没有病人的地方宣布治愈。

**应当如何验证**：

1. 至少做到 **256K**（免 re-RoPE 区间的上界，也正是 KVMem 两个核心 benchmark 的测量区间）。
2. 目标 **1M**，并显式报 N 与选择率。
3. 必须把**选择率（103/N）与 recall 做成联合曲线**，而非单点。这是判断检索机制是否真的有效的唯一方式——单点 100% 不携带任何关于机制的信息。

---

### 2.3 【致命】硬件 Gate 用断言冒充实测

**主张**：
- `docs/comparative_study_report.md` §1："I/O 带宽由 **7.8 GB/s → 50+ GB/s**"
- `plan-04-casa/research-report.md` §5：G-CAS-1 **PASS**（须证明 ≥45 GB/s）、G-CAS-2 **PASS**（回退 ≤15%）

**证据**：
- G-CAS-1 的"验证"是 `tests/test_plan04_casa.py:219–221`：
  ```python
  assert tile.size_bytes == 131072
  assert tile.size_bytes >= 65536
  ```
  这只断言了一个 Python 对象的**尺寸常量**。它没有发起任何 I/O，没有 NVMe，没有 cuFile，没有带宽。
- G-CAS-2 的"验证"是 `test_paged_attention_tensor_core_equivalence`：numpy 等价性 + 报告自述"**理论**无算力回退"。无 GPU、无 kernel、无 Nsight。
- `csrc/main_bench.cpp:389–390` 的 "Effective Bandwidth"：
  ```cpp
  const double total_bytes = total_tokens * head_dim * sizeof(float) * 2; // K and V read
  const double effective_bandwidth_gb = (total_bytes / (paged_stats.mean_us * 1e-6)) / (1024.0*1024.0*1024.0);
  ```
  其中数据来自**已在 host 内存中的零拷贝页表**（`casa_table.compute_paged_attention_tile`，CPU 执行）。这是**内存内有效带宽**，**不是 NVMe 或 PCIe 带宽**。白皮书把它写成 I/O 带宽 7.8→50+ GB/s，属于**口径错置**。

**为什么严重**：G-CAS-1 是我们全研究标定的**最大单点风险**，且我们明确要求它必须在 **95% CI ≥15pp** 上可区分。用一个尺寸断言标记为 PASS，会让后续所有依赖该结论的工作建立在空证据上。

**应当如何验证**：

1. **G-CAS-1（真 I/O）**：在具备 NVMe + GPUDirect 的机器上，用 `gdsio` / `fio --ioengine=libaio` 扫请求尺寸 {4K, 8K, 32K, 64K, 128K, 256K}，报每个尺寸的带宽 p50/p95、≥30 次重复、bootstrap CI。判据保持"≥45 GB/s 且 CI 可区分"。
   **若无法访问该硬件，请把 G-CAS-1 状态改为 `DESIGN-INTENT / UNMEASURED`，不得写 PASS。**
2. **G-CAS-2（真 kernel）**：Hopper 上跑 paged GEMM vs cuBLAS，报 p50/p95 延迟比 + CI；并用 Nsight Systems 给 SM 占用率与 TMA 传输占比。
3. **C++ 控制面 69.75 µs**：这个数字我们接受（纯 CPU 可测），但请补：CPU 型号、是否锁频/turbo、warmup 次数、iteration 数，以及 **p50/p95 而非仅均值**。

---

### 2.4 【严重】针尖召回是循环论证，且对照是稻草人

**证据**：
- `docs/track_a_benchmark_report.md` §2.1：针尖定义为 `‖k_needle‖ = 2.5`，且与 query 语义方向对齐；背景块方差 `σ ≈ 0.02`。
- 检测机制（IFR）：`dev_meta > 0.85` 时触发双子质心分裂。

**问题**：针尖块**就是**高离散度块，而机制**正是**"分裂高离散度块"。用高离散度检测器去检测被构造成高离散度的目标，**100% 召回接近同义反复**。

此外，对照组 "Naive Flat Compression (INT2)"（0% 召回 / +922 PPL）是贵组自建的最弱基线。我们主报告 §5 列出的基线是：Full-Context 重算（F 门真值）、KVMem 穷举 Mean-K、Compact+RAG、vLLM+LMCache、TensorRT-LLM，以及真压缩方案 **KIVI / H2O / StreamingLLM / DMC**。

**应当如何验证**：

1. **信噪比扫描**：把 `‖k_needle‖ / ‖k_bg‖` 作为唯一自变量，从 **1.0（不可分）** 扫到 5.0，报 recall 曲线与失效拐点。只有在低信噪比下仍优于穷举 Mean-K，机制才算成立。
2. **虚警率**：在**不含针尖**的背景块上统计 `dev_meta > 0.85` 的比例。若背景频繁超阈，说明阈值无分辨力——这直接对应我们标为未验证的假设 **A15**。
3. **补真基线**：至少加 KIVI（2-bit）与 H2O，以及 **KVMem 穷举 Mean-K**（这是本工作的真正起点，也是唯一能说明"我们改进了什么"的对照）。

---

### 2.5 【严重】统计契约被自己违反

**证据**：`docs/track_a_benchmark_report.md` §4，Gate U 报 "Paired Success Difference Δ ≥ 0.0 → **+1.00**"。

**问题**：贵组引用了我们的 R10 统计契约，但 R10 明文要求：

- 配对单元是 (task, seed)，需 **471–525 对**（≈118–131 tasks × 4 seeds），Wilson CI；
- 按 task 聚类，design effect ≈ 1.9（ICC ≈ 0.3，假设 A14）；
- 我们已判定**任务质量终点在 1,900 GPU-h 内不可判决**，故一律不得设为主终点。

"+1.00" 是**单次**配对差，**无 N、无 CI、无 MDE**，不具统计意义。

**应当如何验证**：

- 报告 N、配对差分布、Wilson CI、MDE；
- 延迟类 ≥30 次重复 + bootstrap；
- 预注册 Lan-DeMets OBF（3–4 looks）；
- **禁用点估计**：不要把 "100.0%" 当结论，应报 CI 下界。契约原文：禁用 "up to"，只报中位数 + IQR + 全配置散点。

---

### 2.6 【中】口径与自相矛盾

- `docs/comparative_study_report.md` §2 表格列头写"原生 KVMem (**推导**)"——**推导值不能与实测值同列比较**，应分表或显式标注。
- 白皮书 §1 写 "cuFile DMA 读取速度**实测**可飙升至 50–60 GB/s"（E6 专家意见）。**专家意见 ≠ 实测**，仓库内无任何对应数据。
- 顶层 `README.md` 曾写"本报告没有任何代码实现"（现已有 `csrc/`、`kernels/`、`kvmem_fusion/`），且 plan-03/04 状态曾标"研究报告待生成"（两份报告已在仓内）。**这两处已在本轮由我们修正。**

---

## 3. 遗漏项（贵组未覆盖，但属我们标定的必测项）

| 项 | 来源 | 为什么不能省 |
|---|---|---|
| >256K 的 re-RoPE 消除验证 | 主报告 C.3 | 32K 区间本就无此摩擦，等于未测 |
| 并发 8 / 32 / 128 | A13 / M4 | "GPU 恒定 ~34.9 GiB" 在单用户下无生产意义 |
| 跨架构（MLA / GQA） | R8（index/KV = 1/2B） | head 数不是体积杠杆，**层数 L 才是**；GQA/MLA 会改变压缩比 |
| QW3 闭源可达性（G0） | 主报告 §5.3 | 仍是 G0 杀点，贵组未说明如何处理 |
| 合并不可逆性 / 影子阶梯 | LADDER 报告 §7 | 合并后无法配对比较，统计效力下降，需离线影子副本 |
| Full-Context 真值成本 | F 门 | 真值本身存在跨版本漂移风险 |

---

## 4. 架构治理矛盾（需贵组明确选择）

贵组把四方案**融合为单一流水线**（CASA → IFR → UBBA → LADDER）。我们原设计是**竞争 + 淘汰 + 投注配比**（CASA 45% / IFR 30% / LADDER 10% / 机动 15%），并配有一整套触发即冻结的 Gate（M0 LADDER 第 2 周终止、M3 IFR 第 10 周停止并出阴性报告、G-CAS-2 终止 K-Freeze 等）。

**这两套治理逻辑不能同时成立**：融合后不存在"可独立终止的方案"，我们那张 Gate 表需要整体重写。

请明确回答选哪一个。若选融合，我们建议把 Gate 从"**方案级淘汰**"改写为"**子系统级降级**"——例如 K-Freeze 失败则回退 Δ-QRoPE，而非终止整条链；L2′ 合并失败则接受 4.35× 并出阴性结论。

---

## 5. 我们的建议（供参考）

### 5.1 验证路线图（按成本从低到高）

- **P0（2 人日，无 GPU）**：Qwen2.5-0.5B 导出真实 KV；跑 LADDER 的 de-RoPE 合并证伪实验（A 直接平均已 RoPE / B de-RoPE 后平均 / C 不合并）。判据：B 的高频维保留率显著高于 A（预期 0.6–1.0 vs ≈0.18），且 recall@64 掉点不超过 A 的 1/3。
- **P1（~200 GPU-h）**：把贵组现有 harness 接到 **`kvmem-llama.cpp`**（Apache-2.0, v0.17.0，RTX 5060 Ti 16GB 实测 52.7 KiB/token）导出的真实 KV，在 **256K 复现 85.6% / 60.87%**。**这一步不过，后面都不要做。**
- **P2**：真实模型上重跑 32K → 256K → 1M 三档，报**选择率–召回联合曲线**；补并发 8/32/128；补 KIVI/H2O 基线。
- **P3（需真硬件）**：G-CAS-1/2 转实测，或降级标注为 UNMEASURED。

### 5.2 报告口径修改清单（建议立即执行）

| # | 当前写法 | 建议改为 |
|---|---|---|
| 1 | "Verified & Reproducible" | "Synthetic self-consistency validation（合成自洽验证）" |
| 2 | "PPL 漂移 0.0000" | 标注为合成向量度量，或删除，待真实模型重测后回填 |
| 3 | G-CAS-1 / G-CAS-2 = PASS | `UNMEASURED (design intent)`，直至有 `gdsio`/Nsight 数据 |
| 4 | Gate U "+1.00" | 删除，改为报告 N 与 CI，或标注"未做判决" |
| 5 | "8K~32K" 表中的"推导"值 | 与实测值分列，不得并列比较 |
| 6 | 缺失 | 补 `Reproduction` 一节：给出得到每个数字的**确切命令与硬件** |

### 5.3 我们这边的下一步

1. 把 **PagedAttention-K-Freeze**（贵组定理 1）与**因果前缀哈希链**（定理 2）并入我们的 CASA 主线，**替换**我们原来的 Δ-QRoPE / Q-remap 表述。
2. 把本文件的质疑项转为**预注册复现清单**，任何人接手都能逐条打勾。
3. 保留竞争/淘汰治理或改为子系统降级——**待贵组回答 §4 后确定**。

---

## 6. 一句话

> 贵组把我们从"四个竞争方案"推进到了"一条可实现的流水线"，并给出了两处我们确实没想到的正确设计（PagedAttention-K-Freeze、因果前缀哈希链）。
> 但**"合成数据 + 断言式验证 + 32K 规模"这三件事，让报告里那些漂亮的数字目前还不能被称为结论**。
> 把它们换成一次 256K 真实模型复现（85.6% / 60.87%），这份工作的分量会立刻不一样。
