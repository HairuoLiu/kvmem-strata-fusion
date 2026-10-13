# OG 独立审计：`752df3d`（release 整合提交）

**审计对象**：`752df3d` — *feat(release): consolidate Qwen3.8 architecture alignment, clean legacy documents, and resolve review issues*（2026-10-13，我方上次审计 `dcae08f` 之后唯一新提交）
**审计方法**：全部结论基于本地复现与一手来源核对，不采信提交信息自述。Strata 引用值逐一回到 `Niko1221/Strata` 仓库的 `docs/MODELS.md`、`docs/MAC.md` 及其捆绑论文 *Running a 125-Billion-Parameter Model on a Normal PC*（2026-09）原文核对。
**总结论**：这是一个**修复动作真实、但修复不完整、且修复过程中引入新归因错误**的提交。我方可引用的正面变更共 6 项（全部验证通过，含一个我预先准备纠正、实际证明我方先前的怀疑不成立的条目）；但三系统对比的两个核心产物（md 与 json）现在描述的是**两个不同的 "Strata"**，Recall 归因错误原样保留，`ifr.py` 的"均值聚合统一"只统一了一半语义并带来 **59× 不必要性能开销**，另新增两处查无实据的 Strata 参数（`--ple-io direct`、`4,105 槽`）。

---

## 0. 与原审计的衔接

`752df3d` 是对我方两份文档的回应：`og_audit_tripartite_8d88c3b.md`（v1/v2）与 `og_verify_aa07e56_gt_sensitivity.md`。提交信息列出 6 条改动，下面逐条核。

---

## 1. 已确认修复（✅ 全部复现通过）

### 1.1 GT 标签修正 ✅
`benchmarks/benchmark_decoupled_retrieval.py` 与 `docs/decoupled_retrieval_benchmark.json` 中：

```diff
-"ground_truth_definition": "mean_t cos(q_de, k_de(t)) [exact token-level attention]"
+"ground_truth_definition": "mean_t cos(q_de, k_de(t)) [content-space similarity proxy, NOT attention]"
```

与我方 aa07e56 审计的要求一致。README §4(A) 同步改为"严格在内容空间 Token 级真值下评测"。

### 1.2 因果项 `+0.044 / 0.8913→0.9350` 修正 ✅
`docs/track_a_benchmark_report.md`：

```diff
-2. **Sub-Centroid Resolution (+0.044)**: ... $0.8913 \to 0.9350$.
+2. **Sub-Centroid Resolution (+0.022 ~ +0.038)**: ... ($0.9004 \to 0.9224 \to 0.9382$),
+   while max-pooling exhibits false-positive saturation ($K_2\text{-Max}=0.8660 > K_4\text{-Max}=0.8369$) ...
```

数值链 0.9004→0.9224→0.9382 与 max-pooling 饱和现象（K2-Max 0.8660 > K4-Max 0.8369）与我方 `gt_probe.py` 独立复现**逐位一致**。这是我方审计中被采纳的最实质的一条。

### 1.3 `Strata_Naive_A` → `BlockMean_A` ✅
README §4(A) 臂名改正，并补入 Random 基线（0.4910）与全臂列表（A 0.6450 / Bmid 0.6042 / B_matched 0.9004 / K2 0.9224 / K4 0.9382），删除过期值 0.9275/0.5999/0.6104。

### 1.4 测试套件 ✅（我方复现）
提交信息称 "All 83 test suite tests passing"。我方在干净 HEAD 复现：

```
$ python -m pytest tests/ -q
83 passed in 3.09s
```

**确认属实。**

### 1.5 `ifr.py` max→mean pooling ✅（语义变更真实发生，见 §6 的代价）
`kvmem_fusion/ifr.py` 检索路径从"扁平化所有子质心 → 单次向量化 dot → top-2K → 按块去重"（等价 max-pooling）改为"逐子质心打分 → 按块 mean 聚合 → 排序"。我方用同分布合成数据对照两种实现：**top-8 结果仅 3/8 重合**，证明这是行为性变更而非等价重构，与我方 aa07e56 关于 max-pooling 产生阶统计假阳性的结论方向一致。

### 1.6 他们引用的 4 个 Strata 标称值全部可溯源 ✅（**我方预先准备的质疑不成立，特此记录**）

审计前我假设 "~14 tok/s (Mac M4)" 等数值系编造（因首次检索未命中）。逐字核对一手文档后，**四个数值全部真实存在于 Strata 官方文档**：

| 他们的引用 | 一手来源 | 核对结果 |
|---|---|---|
| `53 ~ 94 tok/s (RTX 5070)` | MODELS.md「How fast is each size」：Q2_0 短问答 **94**、IQ3_S **53**（RTX 5070 12 GB + Ryzen 5 7600 + 64 GB） | ✅ 准确转写 |
| `1,620 ~ 2,650 tok/s (RTX 5070 32K)` | 同表「Reads your prompt」列（脚注：measured on a 32K-token prompt）：Q2_0 **2,650**、IQ3_S **1,620** | ✅ 准确转写 |
| `~14 tok/s (Mac M4)` | MAC.md「M4 Mac mini, 32 GB」表：Q2_0 短问答 **14.1**（arena 11.5–14.7 GB，medians of 5 interleaved runs） | ✅ 准确转写 |
| `Coder 剪枝版 (29.6GB)` | MODELS.md：Coder（IQ1_M）shard 1 **29.6 GB**，"fits a PC with 32 GB of RAM" | ✅ 准确 |

同时确认 `Niko1221/Strata` 仓库真实存在（23,650★，MIT，C++，2026-09-24 创建），宿主机的 "Apple M6 (Mac18,5)" 声明亦经 `sysctl` 证实为真。

**保留意见（引用方式而非数值本身）**：`~14 tok/s` 是同一台 M4 上**最佳 arena 配置**的数字；MAC.md 同节后续测量（experts' copy、默认 arena、内存紧张复测）为 **7.37–8.62 tok/s**，服务端混合负载 7.2–7.7（短答）/9.9–10.4（story）。只引 14.1 不引 7.4–8.6 属于挑选最有利配置。另：该数字对应 125B MoE 模型，与他们表中 Ollama 的 7B dense 26.3 tok/s 不同模型、不同后端，跨列比较无意义。

---

## 2. 新引入的查无实据参数（❌ 本轮新增）

### 2.1 `--ple-io direct` 不存在
README §6 与 tripartite md §1 的 [2] 列均写 "SSD 直接 I/O (`--ple-io direct`)"。Strata 全部一手文档（MAC.md + MODELS.md + DISJOINT_EXPERT_CACHE.md + 论文）中出现的 CLI flag 全集为：

```
--api-key --build --calibrate --check --context --control-vector-scaled --data-dir
--draft-vocab --gguf-dir --host --install --model --models-dir --no-browser --no-start
--parallel --port --setup --spec --vision --yes
```

环境变量为 `STRATA_METAL_PROMPT_CHUNK=N` / `STRATA_METAL_PROMPT=window`，另有 `START-HERE.bat --setup --family coder`、`--low-ram on|off`。**无 `--ple-io`。**

### 2.2 "4,105 槽" 不存在
README/md [2] 列："动态 VRAM 专家缓存 (4,105 槽)"。论文原文为 "**~4,500** of 24,576 experts, adaptive"（RTX 5070 12 GB VRAM 下的容量），262K 上下文时降为 "**about 1,600**"。"4,105" 在任何一手文档中均无出现。

### 2.3 "QSA 稀疏注意力 (4-token indexer cell + Top-K 收集)" — 三重错误
- **范畴错误**：QSA（Qwen Sparse Attention）是 **Qwen3.8-Flash-Next 模型架构**的一部分（论文：48 层中 36 层 Gated DeltaNet + **12 层 QSA**，2 个 KV head，最多 attend ~2,048 个选中位置），不是 Strata 引擎特性。Ollama 与 Fusion 跑同一模型同样具备 QSA，不构成 [2] 列的区分度。
- **数字张冠李戴**：论文中的 "4 tokens" 指 **MTP 投机解码验证窗口**（up to 4 tokens per window），与 indexer 无关；"indexer keys" 是 QSA 的模型侧构件（262K 时与 KV cache 合计约占 4 GB VRAM）。不存在 "4-token indexer cell" 这一引擎机制。
- **Strata 没有 KV 页检索机制**：其外存流式的对象是 **experts**（MoE 权重），不是 KV cache 页。

### 2.4 [2] 列 I/O 描述仍属另一个 Strata
"细粒度分页换页 (PCIe 随机小包易遇 7.8 GB/s 瓶颈)" 与 "22% (卡死在 7.8 GB/s I/O 悬崖)" 中的 **7.8 GB/s 与 22% 来自 OSDI '26 Strata 论文**（arXiv 2508.18572，GPU 辅助 I/O、PCIe 5.0 场景）——那是与我方项目同名的另一个系统。Niko1221/Strata 的实际外存行为见 MAC.md：专家流式，单次 1.75 MB 读 **2.97 GB/s**、三次 0.58 MB 读 2.30 GB/s。表中 [2] 列同时混用两个 Strata 的参数。

### 2.5 "8-bit KV (4K 以上) + KV 流式分页 (64K 以上)" 与论文不符
论文两处自述不一（§3 "Past 8K tokens…8-bit"；§4 方法 "8-bit KV cache from 32K up"），**均非 "4K 以上"**；且论文中不存在任何 "KV 流式分页" 机制（64K 仅作为结果表的上下文长度列出现）。

---

## 3. 同一 commit 内，两个产物描述两个不同的 "Strata"（❌ 核心问题）

`752df3d` 把 **md** 的 [2] 列换成了 Niko1221/Strata，但 **json** 原样未动：

| 条目 | `tripartite_comparison_benchmark.md`（752df3d 改后） | `tripartite_comparison_benchmark.json`（752df3d 改后） |
|---|---|---|
| [2] 名称 | 原版开源 Strata (Niko1221/Strata) | **原版 Strata (OSDI '26 原生)** |
| [2] 机制 | Qwen3.8-Flash-Next C++ 引擎、专家卸载 | **HiRadix 前缀树 + GPU-assisted 128B 细粒度 I/O + 朴素旋转均值池化 (Naive Mean-K)** |
| I/O | 细粒度分页、7.8 GB/s 瓶颈 | **128 Byte 细粒度；7.8 GB/s (PCIe 5.0 利用率仅 22%)** |
| 控制面 | C++ 异步图捕获（无数字） | **16.4 ms (Python 调度器 / 原位回填冲突)** |
| Recall | 0.6450「相对 Strata」 | **0.6450 (针尖稀释，丢损 35.5% 精度)** |
| 70B/125B | ⚠️ 仅 Coder 29.6GB 或低 RAM 模式 | ❌ 否 (128B 细粒度导致 FTL 队列瘫痪) |

再加上第三个版本——`docs/comparative_study_report.md`（自 `c5e76c8` 起未更新）中，**16.4 ms 与 7.8 GB/s 都放在"原生 KVMem (基线)"列**（即他们自己的 Python 原型），且 Fusion 带宽明写 **UNMEASURED (设计>=45)**。

于是同一个仓库里，同一对数字有三个互相矛盾的归属：
1. comparative_study_report：= 我方原生 KVMem 基线；Fusion 带宽未测；
2. tripartite json：= OSDI '26 Strata；Fusion "实测 35.97 GB/s"；
3. tripartite md：= Niko1221/Strata 的瓶颈；Fusion "跑满 35.97 GB/s"。

**派生后果**：md §1 控制面行的"**提速 180×**"（= 16.4 ÷ 0.0911）在 md 内已无来源（16.4 ms 单元格被改成散文），其唯一现存来源是 json 中属于 OSDI '26 Strata 的、且经我方 v1/v2 审计证实为其自身 Python 原型误标的那一行。

---

## 4. Recall 归因错误原样保留（❌）

README §4(A) 已诚实地将臂改名为 `BlockMean_A` 并注明"生产路径已采用全块均值聚合、真值为内容空间代理"；但同一份 README 的 §6 表格、tripartite md §1/§2、json `retrieval_recall_64` 仍全部保留旧归因：

- md §1 / README §6："Recall@64 从 0.6450 提升至 0.9224 ~ 0.9382"、**"相对 Strata 召回提升 +43.0% ~ +45.5%"**；
- md §2 表格仍写 "[2] 原版 Strata (Arm A) | 旋转 Key 朴素均值池化 | 0.6450"，列头仍为"相对原版 Strata 增益"；
- json：`retrieval_recall_64: 0.6450 (针尖稀释，丢损 35.5% 精度)`，[3] 列 "(+43.0%~45.5% 相对 Strata)"。

事实（我方 aa07e56 复现）：0.6450 是**他们自己的臂**，在 Qwen2.5-0.5B、Layer 12、128 个 32-token 块、64 个查询上，对**内容相似度代理真值**测得的值；该真值与真实因果注意力的 Spearman ρ = **+0.05**。两个 Strata 都不存在可测的 "KV 块 Recall@64" 指标。同一文件内 §4(A) 说 `BlockMean_A`、§6 说 "Strata 的 0.6450"，自相矛盾。

另外 md §2 开头 "严格在真实逐 Token 注意力真值下评测" **未随脚本/JSON 一并修正**——被修正的只有 json 与脚本里的标签字段。

---

## 5. Ollama 数据：一个 commit 三个数，且最新数无日志（❌）

同一 commit 内：

| 产物 | 7B prompt / gen | 0.5B prompt / gen |
|---|---|---|
| `tripartite_comparison_benchmark.json`（本轮重跑，timestamp 2026-10-13 20:55:52 UTC） | **707.68 / 28.60** | — |
| `tripartite_comparison_benchmark.md` §3（未改） | 434.66 / 26.30 | 8478.50 / 236.02 |
| `README.md` §6（未改） | 434.7 / 26.3 | 8478.5 / 236.02 |
| `docs/logs/tripartite_comparison_run.log`（7efc5b5 上传） | — | 6310.7 / 228.2 |
| **我方本轮复现**（同 prompt、同机、同模型） | 537.55 / 29.375（run 1）；**27,445.87** / 29.388（run 2，热缓存） | 8528.31 / 225.67（前轮） |

三点结论：
1. **单次 prompt 吞吐不可引用**：同一基准同机，434.66 → 707.68 → 537.55 → 27,445.87（热缓存时 prompt_eval_duration 塌缩）。gen 速度相对稳定（26.30–29.39，±6%）。
2. **本轮重跑的 707.68/28.60 没有对应 run log**——违反他们自己在 `7efc5b5` 立下的"每个数据产物配原始日志"规矩；md/README 也未同步，仍写 434.7。
3. 我方前轮复现的 0.5B（8528.31/225.67）与他们 md 的 8478.50/236.02 互相印证，说明 0.5B 那组数大致可复现；7B 那组不能。

---

## 6. `ifr.py` "均值聚合统一"：语义只统一了一半，代价 59× 性能（❌ 本轮最重要技术发现）

### 6.1 数学事实：mean-of-logits ≡ logit-of-mean-of-sub-centroids
新代码对每个分裂块逐子质心计算 `float(compute_logits(q, [sc]))` 再 `np.mean`。由于点积线性，`mean(q·a, q·b) = q·(a+b)/2`——我方实测两种计算路径最大差 **1.8e-15**。因此：

- 分裂块的有效向量 = `(needle + residual_mean) / 2`，即 **needle 权重 1/2**；
- 基准臂 `BlockMean_A`（以及 `A_blk8`）是 32 token 均匀权重，**needle 权重 1/32**；
- 两者相差 **16×**。生产路径与基准路径对同一 block 的有效表示仍然不同，"统一"只统一了聚合算子（mean 而非 max），没有统一权重结构。

且考虑到他们自己的 C++ 日志（`csrc_main_bench_run.log`：`Detected Splits: 2000 (Expected ~400)`，即 raw 指标下 100% 分裂，见 §7），实际运行中**几乎所有 block 都走 needle-1/2 这条与基准不一致的路径**。

另：生产打分用**未归一化 dot-logit**（`np.dot(candidates, query) * scale`），基准与真值用 **cosine**。他们自己的 dispersion check 显示真实 key 原始范数重尾（overall raw_max 11.46、raw_mean 8.12），dot 与 cos 的块排序在真实数据上会系统性分叉。

### 6.2 性能：逐子质心 Python 循环 vs 等价向量化
我方基准（12,500 候选、50% 分裂率、dim 64、每 query）：

| 实现 | 时延 | 相对 |
|---|---|---|
| 旧版（向量化单次 dot + top-2K 去重，max-pooling） | 6.04 ms | 1.0× |
| **新版（752df3d，逐子质心 Python 循环）** | **37.94 ms** | **6.3× 回退** |
| 与新版语义等价的向量化均值（一次 dot） | 0.64 ms | 新版比它慢 **59×** |

即：为换取 mean 聚合，付出了 6.3× 回退；而其中 59× 是纯实现浪费，一个 `np.einsum` 即可消除且结果逐位相同。

### 6.3 对 E 门（≤350 ms @ 10M）的外推
按线性外推（10M 块、probe 覆盖 25% = 2.5M 候选）：

- 新版候选打分 ≈ 7.6 s；外加 `retrieve()` 每 query 都对全表做 `all_logits` + LSE（为算 discarded_mass，实测 10M 规模约 150–250 ms）与 `block_ids.index(bid)` 的 O(n) 查找（8 次 × 10M ≈ 400 ms）→ **合计 ≈ 8.2 s/query，为 E 门目标的 23×**。
- 旧版 ≈ 1.9 s（5.4× 超）；即使换成等价向量化改写 ≈ 0.78 s（2.2× 超，瓶颈转为全表 LSE 与 index 查找）。
- 架构观察：该参考实现的 fidelity 指标（discarded_mass）本身要求全表 logits，**IVF probe 在该实现中不节省任何计算**，"两-tier" 名不副实。

---

## 7. 反折叠检测器：他们自己的数据打自己的 C++ 基准（❌）

`docs/ifr_dispersion_check_20261011-093826.json`（真实 Qwen2.5-0.5B、4096 tokens、128 blocks）明确给出两种指标的触发率：

| 阈值 | raw_max（原始欧氏） | norm_max（归一化方向） |
|---|---|---|
| 0.85 | **1.00（100% 分裂）** | 0.0576 |
| 1.00 | 1.00 | 0.0075 |
| 1.50 | 1.00 | 0.0013 |

而 `docs/logs/csrc_main_bench_run.log` 的 anti-collapse 段：`Threshold = 0.85` → `Detected Splits: 2000 (Expected ~400)`。2000/2000 = 100%，**证明 C++ 基准用的是 raw 指标**；生产 `ifr.py` 默认 `use_normalized=True, theta=0.868`（按上表约 4–6% 触发）。即：**C++ 基准与生产路径不是同一个机制**，C++ 基准中"反折叠"在合成数据上恒真，该段测量无信息量。

阈值口径三处不一：C++ 日志 0.85；Python/README 0.868；fidelity 门 Python `RHO_MAX=0.36` vs C++ 日志 "Fidelity Cliff (< 0.365)"。（另：该日志同段 `rho: 0.9648 → EXCEEDED`，即他们自己的引擎在全合成数据上丢弃 96.5% 注意力质量而门仅记录"超限"未阻断——此点见 v2 审计，未变。）

---

## 8. 杂项（🟡）

1. **`./build/main_bench` 复现指南不成立**：`build/` 在 `.gitignore` 中且不存在于仓库；md §4 称"任何人拉取本仓库代码后，均可直接运行以下命令复现…`./build/main_bench`"，实际需先自行编译（md 未给编译命令，README §4 上方有）。
2. **"IFR 解耦多子质心 (m=2/4)"**：生产 `AntiCollapseSplitter` 只产生 doublet（needle + residual，m≤2）；K=4 仅存在于基准臂，不在生产代码路径。
3. **`og_audit_tripartite_8d88c3b.md` 未被新的 `docs/README.md` 索引收录**（只收录了 aa07e56 与 v3 两份），建议补入，避免审计链断裂。
4. 新 `docs/README.md` 第 2 节仍写 "Naive Mean-K `0.6450`"——旧归因名，与同 commit 的 `BlockMean_A` 改名不一致。
5. 他们把我方 5 份早期审计文档移入 `docs/archive/`（归档本身合理），但 `external_review_critique_v3.md` 仍留在 `docs/` 根目录，与 archive 中的 v1/v2 分离。

---

## 9. 复现材料

| 复核项 | 命令/来源 |
|---|---|
| 83 测试 | `python -m pytest tests/ -q` → `83 passed in 3.09s` |
| Strata 标称值 | `Niko1221/Strata`：`docs/MODELS.md` L26-47、`docs/MAC.md` L112-145、论文 §4 与 Table 5 |
| `--ple-io` / 4,105 不存在 | 一手文档 flag 全集见 §2.1；论文专家缓存 "~4,500 / ~1,600" |
| Ollama 波动 | `python docs/og_repro_752df3d/repro_ollama.py`（7B，两跑：537.55 / 27,445.87 tok/s） |
| ifr 聚合语义 | `python docs/og_repro_752df3d/ifr_perf2.py`（新旧 top-8 重合 3/8；37.94 vs 6.04 vs 0.64 ms） |
| mean-of-logits 恒等 | `python docs/og_repro_752df3d/ifr_perf.py`（max&#124;Δ&#124; = 1.8e-15） |
| 反折叠触发率 | `docs/ifr_dispersion_check_20261011-093826.json` vs `docs/logs/csrc_main_bench_run.log` L88-93 |

---

## 10. 修复优先级（给对方的清单）

**P0（数据诚信，改动量极小）**
1. json 的 [2] 列与 md 统一为同一个 Strata referent；若保留 OSDI '26 对比，必须单列一版"OSDI '26 Strata"并与 Niko1221/Strata 分开，不得混用参数。
2. 删除 `--ple-io direct` 与 `4,105 槽`，改为论文原文 "~4,500 of 24,576 experts, adaptive（262K 时 ~1,600）"。
3. [2] 列的 "QSA 稀疏注意力 (4-token indexer cell + Top-K 收集)" 改述为模型架构事实（12/48 层、2 KV heads、≤~2,048 选中位置），并注明 Ollama/Fusion 同样具备；"KV 流式分页 (64K 以上)" 删除（不存在）。
4. Recall 行删除 "相对 Strata 提升 +43.0%~+45.5%"，改述为 "相对本方朴素块均值臂 BlockMean_A"；md §2 的 "[2] 原版 Strata (Arm A)" 与 "真实逐 Token 注意力真值" 同步改正。
5. md 控制面行删除无源的 "提速 180×"（其唯一来源 16.4 ms 属 OSDI '26 列且系自身原型误标）。

**P1（一致性）**
6. Ollama 数字三处统一；若采用 707.68/28.60，需补 run log 并更新 md/README；同时标注 prompt 吞吐的缓存敏感性（我方复现跨度 434→27,446）。
7. "~14 tok/s (Mac M4)" 补注配置条件（Q2_0、短问答、arena 11.5–14.7 GB）与同机其他配置区间 7.4–8.6；跨列比较加"不同模型/后端"警示。
8. 反折叠：C++ 基准切换到归一化指标（或明确标注 raw），使基准与生产 `ifr.py` 同机制；阈值统一 0.868、rho 门统一 0.36/0.365 其一。
9. md §4 的 `./build/main_bench` 前补编译命令，或改为"先按 README 编译"。

**P2（性能与结构）**
10. `ifr.py` 候选择分改写为单次 `np.einsum`（语义逐位相同，59× 加速）；`block_ids.index` 改 dict 查找。
11. 明确分裂块的有效表示：若要向 benchmark 的 `BlockMean_A` 对齐，needle 权重应为 1/32（即直接用 `mean_k`），而非 doublet 均值的 1/2；二者取其一，并在文档中说明生产与基准的对应关系。
12. 生产打分若意图与基准/真值一致，应使用余弦（或对 key 归一化），而非未归一化 dot-logit。

---

*审计人：OG 组。所有数值可依 §9 复现。本文件只审计 `752df3d` 相对 `dcae08f` 的变更；8d88c3b/7efc5b5 的既有问题见 `og_audit_tripartite_8d88c3b.md`（v2），aa07e56 的既有问题见 `og_verify_aa07e56_gt_sensitivity.md`。*
