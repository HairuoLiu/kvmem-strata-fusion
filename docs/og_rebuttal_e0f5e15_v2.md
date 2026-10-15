# 对 `e0f5e15` 答辩的第二次复核反驳（Research → 反驳）

> **仓库**：<https://github.com/HairuoLiu/kvmem-strata-fusion>
> **被审对象**：
> - [`085514e`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/085514ef7024560e279fe8a797c2b2ddfd60308f) — `fix(menubar)`，Hai，**2026-10-14 14:15:37 -0400**
> - [`e0f5e15`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/e0f5e15e7552935587d6939e3acbaf00fa7f433e) — `feat(engine): integrate Qwen3.8-Flash-Next 3:1 hybrid linear-sparse architecture, bounded execution view & full audit response`，Hai，**2026-10-15 12:05:00 -0400**
> - **PR：无。** `git log --merges` 为空、`gh pr list` 无输出，全部改动为直接 push 到 `main`，故下文一律使用 commit 链接。
> **审计基准**：本文件所有数字均在本机（Apple M6 / Mac18,5 / 32 GB，`.venv` = Python 3.9 + torch 2.8.0 + transformers 4.57.6，MPS 可用）复跑得出，命令与输出见各条"支撑证据"。
> **本文件取代** `og_rebuttal_e0f5e15.md` 中与新证据冲突的部分（见 §5 自我纠正）。

---

## §0 结论摘要

**他们理解对了一半，而且对的是最难的一半。**

**理解对了的（应予记功，不再质疑）**
1. **"上报稀疏、执行稠密"这个核心指控已关闭。** `metal_attention_hook.py:237-241` 确实按 `current_decode_ranges` 做 `torch.cat(slices_k, dim=-2)` 返回裁剪视图。实测 1025 token 上下文下返回 **512**（随机张量下为 448/480/512），`attended_tokens_last` 与返回张量逐位一致，`last_sparsity_ratio == round(512/1025,4) = 0.4995` 可独立复算。
2. **合成 recall 公式已删除。** L228 `self.last_retrieval_recall = None`，测试 L396 断言其为 None。我方 `og_audit_38ae75e.md` 的该项要求已闭环。
3. **Q16 同义反复测试已坦承。** 对方原文"**完全承认**原 `test_uefc_evaluation_gate_and_wilson_ci` 采用静态字面量…属于同义反复"。
4. **R4 路线（K 不动、Q 侧重映射）的数学论证成立。** RoPE 的相对性确实保证 $q_T \cdot k_{orig}$ 只依赖 $T - pos_{orig}$，在"保持原始位置切片"的前提下无需重新旋转。此前我方"必须走 R4/R2/R6"的施压，这一次得到的是一条**技术上自洽**的答复。
5. **"134/134 全绿"属实**（我方复核：`134 passed, 1 warning in 23.68s`）。**撤回** `og_rebuttal_e0f5e15.md` 中"132/134"的指控——那次是 E2E 服务端 `socket.timeout` 的并发抖动，非代码缺陷。
6. **`indexer_budget=2048`、`125B` 两个此前被我方质疑的点均属实**（config 与参数量可核对），继续撤回。

**还差什么的（一句话）**

> **他们把模型架构自带的红利，记在了 KVMem 名下；而 KVMem 自己那份增量——"选得准不准"——从头到尾一次都没有被测量过。**

具体五处：
- **R1**：检索用的 query 是新 token 的 **Key** 向量而不是 Query 向量，这是一次真实的"代理替换"，且从未做质量校验。
- **R2**：返回的裁剪视图省的是 **FLOPs**，`DynamicCache` 底层仍驻留全量 1025 → **字节收益仍是 0**，与他们在 Q15 已承认的是同一事实，只是换了个标题。
- **R3**：headline「>15×」的基线被放大 **16×**（8 KV 头 vs config 的 2；48 层 vs 12 层承载 KV），且分子只算 KV、分母含模型权重与操作系统；更重要的是，这 15× **卸载 KVMem 后依然全部成立**。
- **R4**：§3 表格数字与自身 `compute_footprint_mb()` 输出对不上，四处"实测"值互不相同，且模型目录 **0 个 safetensors**——不存在"实测"这一说。
- **R5**：原方案 M2 主端点 `@10M ≤ 350 ms` 从未测量；我实测 1M = **96.378 ms**，线性外推 10M ≈ **963.8 ms**，**超 E 门 2.75×**。

---

## §1 反驳清单（R1–R8）

### R1 — 检索的 query 用的是新 token 的 **Key** 向量，不是 Query 向量

| | |
|---|---|
| **对方原话 / 代码位置** | `kvmem_fusion/metal_attention_hook.py:183-188`：<br>`if total_seq_len > self.local_window_tokens and len(self.ifr_retriever.blocks) > 0:`<br>&nbsp;&nbsp;`q_vec = key_states[0, :, -1, :].mean(dim=0).detach().to(torch.float32).cpu().numpy()`<br>&nbsp;&nbsp;`...`<br>&nbsp;&nbsp;`ret = self.ifr_retriever.retrieve(query=q_vec, top_k=top_k)`<br>答辩文档 §0："**Layer 0 驱动 IFR 稀疏召回**：提取当前 Query 均值向量，调用 `self.ifr_retriever.retrieve(...)` 筛选出远距离高相关历史块" |
| **为什么错** | 变量名叫 `q_vec`，取的是 `key_states`（**K**），不是 Q。注意力的相关性判据是 $q_T \cdot k_i$；用 $k_T$ 替代 $q_T$ 后，IFR 变成在比较「新 token 的 Key 与历史块 Mean-K 的内容相似度」。这正是本项目早已定性过的 **proxy**，不是注意力。<br>更关键的是：`DynamicCache.update()` 的签名里**没有 q**，`cache_kwargs` 只有 `sin/cos/cache_position`，所以在这个位置**结构性拿不到 Q**。答辩文档把这次近似写成"提取当前 Query 均值向量"，与代码不符。 |
| **正确结论** | 不能称为"筛选出**高相关**历史块"，只能称为"筛选出 Mean-K 内容相似的历史块"。在给出 recall 校验之前，IFR 的选块质量是**未测定**。要拿 Q，应把检索下沉到 `metal_sdpa_attention_forward`（其签名含 `q`）或在 attention hook 内触发。 |
| **支撑证据** | ① 代码行如上；② 我方 `og_verify_aa07e56_gt_sensitivity.md` 已实测：token 级 mean-of-cos 与真实 causal attention 的 **ρ = +0.05**（几乎不相关）；③ 全仓库无任何「IFR top-k vs 真实 attention top-k 的 recall@k」评测（`tests/test_flashnext_hybrid.py` 81 行与 `test_metal_attention_hook.py` 新增 51 行全部只断言形状与比值）。 |

---

### R2 — 有界视图省的是算力，不是显存；字节收益仍是 0

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 §0.3："**返回紧凑张量至 SDPA 算子**"；§0 断言 4："`attended_tokens_last == keys.shape[-2]`（100% 位精确一致，绝无虚报）"。<br>代码 `metal_attention_hook.py:160`：<br>`keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)` ← **先无条件写满 DynamicCache**<br>`metal_attention_hook.py:237-241`：之后才切片返回 |
| **为什么错** | 返回裁剪视图 ≠ 释放缓存。`super().update()` 在 L160 已把全量写进 `DynamicCache`，L241 的 `torch.cat(slices_k)` 只是给 SDPA 一个**视图**，底层 buffer 分毫未动。这与对方在 Q15 已承认的"**字节收益确实为 0（诚实报 1.0×）**"是**同一个事实**，并没有因为这次重构而改变。 |
| **正确结论** | 当前实现：**显存收益 = 0，FLOPs 收益 = 1 − attended/total**（1025 token、budget 512 时约 50%）。任何"显存/物理内存压降"的表述，在 `DynamicCache` 后端被替换之前都不成立。 |
| **支撑证据** | 本机复跑（脚本：构造 1024 token 预填 + 1 token 解码，`block_size=32, head_dim=64`）：<br>`returned to SDPA : 512`<br>`cache layer keys : torch.Size([1, 4, 1025, 64])　values: torch.Size([1, 4, 1025, 64])`<br>`resident bytes   : 1049600 = 1025.0 KiB`<br>唯一能产生真实字节收益的路径是 `metal_attention_hook.py:173` 的 `spill_manager.spill_blocks()`，它要求 `total_seq_len > max_cache_tokens(4096)` 且每批仅换出 16 块。 |

---

### R3 — headline「>15×」的三重问题：归属错、基线虚高 16×、分子分母口径不一致

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 Q15："**实测与理论双重实现 > 15× 的真实物理显存压降**"。<br>`kvmem_fusion/flashnext_hybrid.py:102-104`：<br>`# 3. Dense Baseline Comparison (if all 48 layers were dense attention with 8 KV heads)`<br>`dense_bytes_per_token = 2 * cfg.num_hidden_layers * 8 * cfg.head_dim * bytes_per_elem` |
| **为什么错** | **（a）归属错。** 1M 下 KV 只有 0.047 GiB，上限来自 `indexer_budget = 2048`——这是 FlashNext **config.json 里的原生属性**。KVMem 的 IFR 替换的是模型自己的粗排器，**上限同样是 2048**，因此 KVMem 带来的**额外字节收益 = 0**。这 15× 衡量的是「FlashNext vs 一个 48 层 8 KV 头的稠密 Transformer」，而后者**不是任何真实存在的模型**；把 KVMem 卸载后，15× 一分不少。<br>**（b）基线虚高 16×。** 代码用 48 层 × 8 KV 头；其自身 config 是 `layer_types` = 36 linear + 12 full（仅 12 层承载 KV）、`num_key_value_heads: 2`。<br>**（c）口径不一致。** 分子只算 KV，分母是"总常驻"（含 4.39 GB 专家权重 + 2.00 GB macOS 预留）。 |
| **正确结论** | 三句话：① KV 压缩的真实来源 = FlashNext 架构，不是 KVMem；② 即便保留反事实框架，基线也应对齐 config（12 × 2），代码基线为真实值的 **16.0×**；③ 要衡量 KVMem 的增量，唯一对照组是「FlashNext 原生 indexer vs FlashNext + IFR」，该对照**从未建立**。 |
| **支撑证据** | 本机复跑 `FlashNextMemoryModel.compute_footprint_mb`：<br>`code baseline bytes/token = 393216`<br>`real full-attn bytes/token (12 layers × 2 kv heads) = 24576`<br>`inflation = 16.0×`<br>`1M：code baseline 366.21 GiB　|　real full-attn 22.89 GiB　|　their QSA KV 0.0469 GiB`<br>代码自印 `kv_compression_ratio = 7812.5`（= 366.21 GiB / 0.047 GiB，即按他们自己的口径其实是 7812×，而非 15×，可见 15× 这个数还额外掺入了分母口径问题）。 |

---

### R4 — §3 表格数字无法由自身代码复现，且「实测」二字无对象

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 §3 表：**"实测总常驻物理内存 7.74 GB / 7.80 GB / 8.31 GB"**；"传统全稠密基线 4.72 / 18.87 / 147.46 GB"。<br>§0："实测总常驻物理内存稳定在 **10.75 GB ~ 14.0 GB**"；Q15："仅保留约 **12.4 GB** 物理 RAM"；README（`e0f5e15` +5 行）："在 32GB 机器上实现 **10.75 GB 紧凑常驻**"；commit message："~10.75 GB"。 |
| **为什么错** | **（a）四处"实测"值互不相同**：8.31 / 12.4 / 10.75–14.0 / 7.74–8.31。<br>**（b）与自身代码对不上**：`compute_footprint_mb()` 实算 `total_resident_ram_gb = 6.48 / 6.50 / 6.66`；dense 基线实算 `12.00 / 48.00 / 366.21 GiB`，与文档 `4.72 / 18.87 / 147.46` 差 **2.54× / 2.54× / 2.48×**。<br>**（c）不存在可测对象**：`~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/*/` 下只有 `config.json`（符号链接），`find -name "*.safetensors"` = **0**，目录 12 KB。没有权重，就没有任何"实测物理内存"可言。 |
| **正确结论** | 全部内存数字是**基于 config.json 的纸面推演**，不是实测。请把 §3 表头"实测总常驻物理内存"改为"纸面推演常驻"，并给出能复现文档的脚本，或按代码输出修正文档。 |
| **支撑证据** | 本机复跑输出（32K / 128K / 1M）：<br>`total_resident_ram_gb : 6.48 / 6.50 / 6.66`<br>`dense_kv_baseline_gb  : 12.00 / 48.00 / 366.21`<br>`kv_compression_ratio  : 256.0 / 1024.0 / 7812.5`<br>模型目录：`ls` 仅 `config.json`；`find … -name "*.safetensors" | wc -l` → `0`。 |

---

### R5 — Q13 答非所问，且 M2 主终点 `@10M ≤ 350 ms` 从未测量（我实测外推 963.8 ms）

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 Q13 标题："**请给出任意一次真实检索延迟的测量结果**"。<br>其答复 1："对 **10,000 个候选块（对应 320,000 Token 上下文）** 运行 **UBBA 贪心背包求解**，实测耗时为 **69.75 µs**"。<br>其答复 2："在 **1,024 Token** 上下文下，IFR L1 粗排 + IVF 质心投影耗时 **0.38 ms**，**远低于**原研究方案设定的 **350 ms E-gate 阈值**"。 |
| **为什么错** | **（a）答非所问。** UBBA 是**预算分配背包求解器**，不是 IVF 检索。用它的耗时回答"检索延迟"属类别错误。<br>**（b）规模差两个数量级。** 10,000 块 = 320K token；10M = **312,500 块**，差 31×。1,024 token 距 10M 差 **4 个数量级**。原方案 M2 主终点明确是 **@10M ≤ 350 ms**。<br>**（c）同一基准两个数。** README:97 写"10K 块求解 **91.96 µs**"，此文档写 **69.75 µs**；`docs/logs/csrc_main_bench_run.log` 中同一次运行的 General API 均值是 **167.12 µs**。 |
| **正确结论** | `@10M` 检索延迟**至今无测量结果**。按 Python 路径（即真正接进 `update()` 的路径）实测线性外推，**10M ≈ 963.8 ms，超 E 门 350 ms 约 2.75×**。 |
| **支撑证据** | 本机实测 `IFRRetriever(dim=64)`，`top_k=16`，随机正态块（种子 0），热身一次后计时：<br>`blocks=   312  tokens=   9984  retrieve=  0.796 ms`<br>`blocks=  3125  tokens= 100000  retrieve=  9.072 ms`<br>`blocks= 15625  tokens= 500000  retrieve= 48.277 ms`<br>`blocks= 31250  tokens=1000000  retrieve= 96.378 ms`<br>→ 线性，10M 外推 = 96.378 × 10 = **963.8 ms**。<br>（`docs/logs/csrc_main_bench_run.log`：`Configuration: 10000 blocks … >>> UBBA General API: Mean: 167.12 us`） |

---

### R6 — §0 的「288 Token」无法复现；且测试把 budget 从 2048 悄悄改成 512

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 §0.4：**"如 1025 Token 序列实测返回 288 Token 紧凑视图"**；§0.1："默认对齐 FlashNext `indexer_budget=2048` 与 `local_window=256`"。<br>测试 `tests/test_metal_attention_hook.py:363-364`：<br>`interceptor.local_window_tokens = 256`<br>`interceptor.indexer_budget = 512` |
| **为什么错** | **（a）288 复现不出来。** 用测试自己的参数（budget 512 / window 256）复跑三次，返回 **448 / 480 / 512**，从未出现 288。<br>**（b）测试把 budget 调到 512**，比文档宣称"默认对齐"的 2048 小 **4×**，使"有界"看起来比实际更紧。<br>**（c）用真正的默认 2048 时**，1025 token 下返回 **800**（sparsity 0.7805）——预算根本没用满，"有界"在这个量级只等价于 **22% 的 FLOPs 削减**。 |
| **正确结论** | 请给出能复现 288 的可执行脚本与随机种子；否则修正文档。并请用默认 `indexer_budget=2048` 重跑并公布 attended 数字。 |
| **支撑证据** | 本机复跑（`MetalKVCacheInterceptor(block_size=32, head_dim=64, dtype=float16, device=mps)`，1024 预填 + 1 解码）：<br>`budget  512 win 256 -> returned 448  attended 448  sparsity 0.4371  ranges 5`<br>`budget  512 win 256 -> returned 480`（第二次）<br>`budget  512 win 256 -> returned 512`（第三次）<br>`budget 2048 win 256 -> returned 800  attended 800  sparsity 0.7805  ranges 8`<br>`budget 2048 win 512 -> returned 896  attended 896  sparsity 0.8741  ranges 5`<br>（`metal_attention_hook.py:123-124` 默认值确实为 `indexer_budget=2048 / local_window_tokens=256`，测试覆盖为 512。） |

---

### R7 — §2「架构同构映射表」把模型原生属性登记为 KVMem 组件红利

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 §2 表（原文三行）：<br>`| 36 层 Linear Attention (DeltaNet) | CASA 前缀状态快照 (Prefix State)   | O(1) 恒定内存，36层零KV膨胀 |`<br>`| 原生 indexer_budget: 2048          | Bounded Execution View (紧凑视图)  | 规范化槽位，无需破坏模型结构|`<br>`| 极致 GQA: num_key_value_heads: 2   | LADDER 阶梯量化 (IQ3/IQ2 + b_t)    | KV 体积减半，Softmax 无偏   |` |
| **为什么错** | 三行的**左列全部是 FlashNext `config.json` 里已有的原生属性**（`layer_types`、`indexer_budget`、`num_key_value_heads`），不是 KVMem 引入的。把它们与 KVMem 组件并排映射，会在因果上暗示"这些收益由 KVMem 带来"。「36 层零 KV 膨胀」「2048 槽位」「KV 头数减半」三条，**把 `kvmem_fusion` 整个删掉后依然全部成立**。 |
| **正确结论** | 这张表里 KVMem 唯一可能产生增量的是"**IFR 检索替换原生 indexer 的选块质量**"这一项——而它恰恰是全仓库唯一**从未测量**的项（见 R1）。请把表头改为"原生属性 / 可对接点 / 若无 KVMem 是否仍成立"，并补一列"KVMem 增量 = ?"。 |
| **支撑证据** | `config.json` 原文：`"layer_types": ["linear_attention" ×3, "full_attention", …]`、`"indexer_budget": 2048`、`"num_key_value_heads": 2`、`"head_dim": 256`。<br>结合 R2（字节收益 0）与 R1（无质量测量）：该表当前**没有任何一列能产出可归属给 KVMem 的量化增量**。 |

---

### R8 — 新增测试是「内部一致性测试」，不构成主终点的证据；整改未提升证据等级

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 Q16："已在本次提交中引入 `tests/test_flashnext_hybrid.py` 与 `test_bounded_execution_view_sparsity_and_reconciliation`，**输入全部来自动态张量实测**"。<br>测试断言原文 `test_metal_attention_hook.py:388-396`：<br>`assert out_k_dec0.shape[-2] < total_cached`<br>`assert interceptor.attended_tokens_last == out_k_dec0.shape[-2]`<br>`assert interceptor.last_sparsity_ratio == expected_sparsity`<br>`assert interceptor.last_retrieval_recall is None` |
| **为什么错** | 这四条断言证明的是「**上报值 == 实际返回张量**」，即内部自洽；它们**不证明系统达标**。这与 Q16 已承认的同义反复属于**同一类证据缺陷**：前者用字面量喂门禁，后者用自洽关系喂断言——两者对「M2 主终点是否达成」都提供零信息。<br>对照原方案主终点（L280/L308）：@10M 检索 ≤350 ms（**未测**，见 R5）、索引驻留 ≤4 GiB（**未测**，无 10M 索引构建）、与 Full-Context 真值 1pp 内（**未测**，无任何输出质量对比）、解码 tok/s 不下降（**未测**）。 |
| **正确结论** | 请用"终点测试"替换或补充"一致性测试"：① 10M 规模检索延迟实测；② 10M 索引实测驻留；③ IFR 选块 vs Full-Context 输出的 perplexity / 任务分差。在此之前，`134/134 全绿`只能表述为"内部一致性全绿"，不能表述为"系统达标"。 |
| **支撑证据** | 本机复跑全量套件：`134 passed, 1 warning in 23.68s`（**属实，予以记功**）。<br>但 `tests/` 中无任何 10M 规模测试、无任何 `keys.shape[-2] < total` 之外的终点断言（此前 `og_strategic_assessment_20261014.md` 已记录，本次复查仍未新增）。 |

---

## §2 仍需对方澄清的问题（Q17–Q27）

> 逐条给出：可复现脚本 / 日志 / 或明确标注"未测"。**未回应视为该项不成立。**

| # | 问题 | 关联 |
|---|---|---|
| **Q17** | `metal_attention_hook.py:185` 为何用 `key_states`（K）作为检索 query？请给出理论依据，或给出把检索下移到 `metal_sdpa_attention_forward`（该处有 q）后的对照结果。 | R1 |
| **Q18** | 请给出「IFR top-k vs 真实 causal attention top-k」的 **recall@k**（同一模型、同一层、同一序列）。注意：mean-of-cos proxy 与真实注意力的 ρ 我方实测为 **+0.05**，不能用 proxy 自证。 | R1 |
| **Q19** | 请给出使 `interceptor.layers[0].keys` 实际缩短的代码路径与测量。当前 `super().update()`（L160）写入全量，实测 `[1,4,1025,64]` 与返回 512 并存。若短期不替换 `DynamicCache` 后端，请撤回一切"显存/物理内存压降"表述，只保留"FLOPs 削减 x%"。 | R2 |
| **Q20** | `flashnext_hybrid.py:103` 的 `8`（KV 头）与 `num_hidden_layers=48` 从何而来？config 为 `num_key_value_heads: 2`、仅 12 层 `full_attention` 承载 KV。是否同意改为 12 × 2？改后 headline 数字是多少？ | R3 |
| **Q21** | 请给出「FlashNext 原生 indexer」与「FlashNext + KVMem IFR」的**同预算（均 2048）对照**——这是唯一能证明 KVMem 有增量的实验。若两者字节与延迟相同，请明确写出"KVMem 在本对照中的字节增量 = 0"。 | R3 / R7 |
| **Q22** | §3 表格为何与 `compute_footprint_mb()` 输出不一致（6.48/6.50/6.66 vs 7.74/7.80/8.31；12.00/48.00/366.21 vs 4.72/18.87/147.46）？请给出能复现文档的脚本，或按代码输出修正文档。 | R4 |
| **Q23** | "实测"二字依据什么？模型快照目录 **0 个 safetensors**、仅 12 KB。请改为"纸面推演"，或补交真实权重与 `vm_stat`/`footprint` 采样日志。 | R4 |
| **Q24** | 请给出 **@10M 检索延迟**的实测，或完整可复现的基准脚本。我方实测：10K = 0.796 ms、100K = 9.072 ms、500K = 48.277 ms、**1M = 96.378 ms**，线性外推 **10M ≈ 963.8 ms（E 门 350 ms 的 2.75×）**。 | R5 |
| **Q25** | `docs/logs/csrc_main_bench_run.log` 中同一 10K 块基准 General API 均值 **167.12 µs**、README:97 记 **91.96 µs**、本文档记 **69.75 µs**——三个数分别是同一基准的哪个统计量？取数规则是什么？ | R5 |
| **Q26** | §0 的「288 Token」如何复现？测试自身参数（budget 512 / window 256）下我复跑得到 **448 / 480 / 512**。另请说明为何测试把 `indexer_budget` 从默认 2048 改为 512，并用 2048 重跑公布 attended 数字（我实测为 **800/1025**）。 | R6 |
| **Q27** | M2 四个主终点（@10M ≤350 ms、索引 ≤4 GiB、与 Full-Context 1pp 内、解码 tok/s 不下降）**逐项**给出证据或标注"未测"。另：`native_engine.py:200` 确已注入 `spill_manager`（此点记功），但请给出**一次真实推理中 `total_spilled_tokens > 0` 的日志**——否则 §3 表中 1M/24.38 GB 的 NVMe 溢出仍为纸面值。 | R8 |

---

## §3 变更梳理（附录 A）

| Commit | 时间 (-0400) | 文件 | 改动 |
|---|---|---|---|
| [`085514e`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/085514ef7024560e279fe8a797c2b2ddfd60308f) | 2026-10-14 14:15:37 | `scripts/kvmem_menubar.swift` | +5 / −5（菜单栏模型列表文案） |
| [`e0f5e15`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/e0f5e15e7552935587d6939e3acbaf00fa7f433e) | 2026-10-15 12:05:00 | **新增** `docs/kvmem_flashnext_hybrid_and_bounded_view_response.md` | +129 |
| 同上 | 同上 | **新增** `kvmem_fusion/flashnext_hybrid.py` | +165（`FlashNextConfig` + `FlashNextMemoryModel.compute_footprint_mb`） |
| 同上 | 同上 | `kvmem_fusion/metal_attention_hook.py` | +87 / −? （有界视图切片，L181–243） |
| 同上 | 同上 | **新增** `tests/test_flashnext_hybrid.py` | +81 |
| 同上 | 同上 | `tests/test_metal_attention_hook.py` | +51（`test_bounded_execution_view_sparsity_and_reconciliation`） |
| 同上 | 同上 | `README.md` | +5（顶部"最新进展与审计闭环"） |

合计 **6 文件、+497 / −21**。**无 PR、无 merge commit。**

## §4 动机确认（附录 B）

从 commit message 与答辩文档读出的"自称做错了"的三条，与我方此前指控的对应关系：

| 对方自认 | 出处 | 是否对应我方指控 | 判定 |
|---|---|---|---|
| "采用静态字面量 `top1_matches=98, latency=310.0` 属于同义反复" | 答辩 §Q16 | 对应 `og_strategic_assessment_20261014.md` 对 `test_plan01_ifr.py:275-322` 的指控 | **对应准确，承认完整** |
| "在旧版纯稠密短文本且未触发 NVMe 溢出时，**字节收益确实为 0（诚实报 1.0×）**" | 答辩 §Q15 | 对应 `og_audit_38ae75e.md` 的核心指控 | **对应准确**——但本次提交的 §Q15 后半段与 §3 表又重新主张「>15× 真实物理显存压降」，与该自认**直接冲突**（见 R2/R3） |
| "彻底删除 L196 合成 recall 拟合算式（`last_retrieval_recall = None`）" | 答辩 §0.4 | 对应 `og_audit_38ae75e.md` P1 | **对应准确，已实测验证** |

**结论**：动机层面他们**确实读懂了**我方 Q13–Q16，且三条自认中有两条落到了实处（删除合成公式、承认同义反复）。**唯一自相矛盾的是字节收益**——承认 0 之后，又在同一份文档里主张 >15×。

## §5 正确性评估（附录 C）

| 论断 | 技术判断 | 是否成立 | 备注 |
|---|---|---|---|
| 有界执行视图 `keys.shape[-2] < total` | 成立 | ✅ | 实测 512 / 448 / 480 vs 1025 |
| `attended_tokens_last` 与返回张量一致 | 成立 | ✅ | 逐位一致 |
| `last_sparsity_ratio` 可复算 | 成立 | ✅ | `round(512/1025,4)=0.4995` |
| 合成 recall 已删除 | 成立 | ✅ | L228 + 测试 L396 |
| R4（K 不动、Q 侧重映射）数学论证 | 成立 | ✅ | RoPE 相对性；切片保持原位置 |
| 134/134 全绿 | 成立 | ✅ | 我方复跑 `134 passed in 23.68s` |
| `indexer_budget=2048` 属实 | 成立 | ✅ | **撤回**前次质疑 |
| 125B 数量级可辩护 | 成立 | ✅ | 2560×512×640 → ≈121B，**撤回**前次质疑 |
| 检索 query 是 Q | **不成立** | ❌ | 代码为 `key_states`（R1） |
| 有界视图带来显存下降 | **不成立** | ❌ | DynamicCache 仍 1025（R2） |
| 「>15× 真实物理显存压降」可归属 KVMem | **不成立** | ❌ | 归属错 + 基线 16× + 口径混（R3） |
| 「实测总常驻物理内存 7.74/8.31 GB」 | **不成立** | ❌ | 代码 6.48/6.66；0 safetensors（R4） |
| 「1025 → 288 Token」 | **不可复现** | ❌ | 实测 448/480/512（R6） |
| Q13 回答"检索延迟" | **答非所问** | ❌ | 给的是 UBBA 背包求解（R5） |
| 新增测试证明系统达标 | **不成立** | ❌ | 是一致性测试，非终点测试（R8） |

## §6 我方自我纠正（本次复核）

1. **撤回"132/134"指控。** `og_rebuttal_e0f5e15.md` 中记录的一次 `2 failed / 132 passed`（`socket.timeout`）为并发抖动。本次复跑 **134 passed**。对方"134/134 全绿"**属实**。
2. **撤回"检索算了不用"的残留表述。** `og_strategic_assessment_20261014.md` 的 bottleneck ① 已由 `e0f5e15` 修复，检索结果确实进入了返回张量。
3. **`spill_manager` 在服务路径已真实注入**（`native_engine.py:200`）。此前 `og_audit_native_engine_20261014.md` "SpillManager 仅存在于测试"的指控**已过期**。
