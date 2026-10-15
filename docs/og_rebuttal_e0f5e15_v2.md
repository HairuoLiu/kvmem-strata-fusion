# 对 `e0f5e15` + `96f7865` 的复核反驳（Research → 反驳）

> **仓库**：<https://github.com/HairuoLiu/kvmem-strata-fusion>
> **被审对象**：
> - [`085514e`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/085514ef7024560e279fe8a797c2b2ddfd60308f) — `fix(menubar)`，Hai，2026-10-14 14:15:37 -0400
> - [`e0f5e15`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/e0f5e15e7552935587d6939e3acbaf00fa7f433e) — `feat(engine): …bounded execution view & full audit response`，Hai，2026-10-15 12:05:00 -0400
> - [`96f7865`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/96f78659c7b628faf1121e3389a3272b4bd495c9) — `feat(bench): add Phase 1 FlashNext hybrid architecture & bounded retrieval benchmark harness with empirical 4K-128K results`，Hai，2026-10-15 12:47:23 -0400
> - **PR：无。** `git log --merges` 为空、`gh pr list` 无输出，全部为直接 push 到 `main`。
> **审计基准**：本机复跑（Apple M6 / Mac18,5 / 32 GB；`.venv` = Python 3.9 + torch 2.8.0 + transformers 4.57.6，MPS 可用；真实权重 `Qwen3-0.6B` 本地缓存可用）。所有数字附命令与输出。
> **取代关系**：本文件取代 `og_rebuttal_e0f5e15.md`，并**纠正其中一条我方错误**（见 §6）。

---

## §0 结论摘要

**他们理解对了一半，而且对的是最难的一半；但新提交的这份"实测基准"，测的是一条在真实模型上不存在的代码路径。**

### 记功：以下指控关闭，不再质疑

1. **"上报稀疏、执行稠密"已关闭。** `metal_attention_hook.py:237-241` 确实按 `current_decode_ranges` 做 `torch.cat(slices_k, dim=-2)` 返回裁剪视图。实测 1025 token 上下文返回 **512**（随机张量下 448/480/512），`attended_tokens_last` 与返回张量逐位一致，`last_sparsity_ratio == round(512/1025,4) = 0.4995` 可独立复算。
2. **合成 recall 公式已删除。** L228 `self.last_retrieval_recall = None`，测试 L396 断言其为 None。
3. **R4（K 不动、Q 侧重映射）的数学论证成立。** RoPE 相对性保证 $q_T \cdot k_{orig}$ 只依赖 $T - pos_{orig}$，切片保持原位置，无需重新旋转。
4. **Q16 同义反复已完整承认**（原文："完全承认…属于同义反复"）。
5. **134/134 全绿属实**（我方复跑 `134 passed, 1 warning in 23.68s`）。
6. **`indexer_budget=2048`、`125B`、`spill_manager` 服务路径注入**（`native_engine.py:200`）三点此前质疑，全部撤回。
7. **新基准可复现。** 复跑 `benchmarks/benchmark_flashnext_hybrid_e2e.py`，128K 得 **1.392 ms**（对方发布 1.352 ms），在噪声内。**这是一份诚实可跑的 harness，予以记功。**
8. **新基准把指标命名为 `compute_reduction_pct`（算力削减）而非"显存压降"** —— 这个改名是对的，方向上回应了我方 R2。

### 还差什么（一句话）

> **他们的基准数据块内离散度是 `0.0000`，真实 Qwen3-0.6B 的 KV 是 `0.7388`；而 IFR 赖以区别于普通 IVF 的核心机制（anti-collapse 双子分裂）在真实 KV 上只触发 `0.8%`（1/128）。这份基准对 IFR 的定义性机制是盲的。**

九条反驳见 §1。**R5 是本轮最严重的一条**——它是新增证据推翻我方旧结论后重新立起的：

| | 我方旧结论（`og_rebuttal_e0f5e15.md`） | 本次（真实模型 KV 复核后） |
|---|---|---|
| @10M 检索延迟 | 外推 963.8 ms，超 E 门 2.75× | **撤回。** 该数基于随机高斯块（`norm_dispersion ≈ 1.02`），离散度**过高**。改用真实 Qwen3-0.6B KV（`norm_dispersion` 0.7388）实测 4K = **0.075 ms**，线性外推 10M ≈ **184 ms**，**在 350 ms 内**。 |
| 指控换成 | — | ① 该外推跨 **2,441×**，10M 从未测量；② **基准数据离散度 0.0000 vs 真实 0.7388**，anti-collapse 在基准中 0/4096 触发、真实 0.8%、随机数据 100%；③ 对方 4K 数 0.048 ms 比真实 KV 的 0.075 ms 乐观 **1.56×**。 |

---

## §1 反驳清单（R1–R9）

### R1 — 检索的 query 用的是新 token 的 **Key** 向量，不是 Query 向量

| | |
|---|---|
| **对方原话 / 代码位置** | `kvmem_fusion/metal_attention_hook.py:183-188`：<br>`q_vec = key_states[0, :, -1, :].mean(dim=0).detach().to(torch.float32).cpu().numpy()`<br>`ret = self.ifr_retriever.retrieve(query=q_vec, top_k=top_k)`<br>答辩文档 §0："**提取当前 Query 均值向量**，调用 `self.ifr_retriever.retrieve(...)`" |
| **为什么错** | 变量名为 `q_vec`，取的是 `key_states`（**K**）。注意力相关性判据是 $q_T \cdot k_i$；用 $k_T$ 替代 $q_T$ 后，IFR 变成在比较「新 token 的 Key 与历史块 Mean-K 的内容相似度」——这正是本项目早已定性的 **proxy**，不是注意力。<br>`DynamicCache.update()` 签名中**没有 q**（`cache_kwargs` 只有 `sin/cos/cache_position`），所以在此位置**结构性拿不到 Q**。文档"提取当前 Query 均值向量"与代码不符。 |
| **正确结论** | 不能称为"筛选出**高相关**历史块"。在给出 recall 校验前，IFR 选块质量**未测定**。要拿 Q 应把检索下沉到 `metal_sdpa_attention_forward`（签名含 `q`）或在 attention hook 内触发。 |
| **支撑证据** | ① 代码行如上；② `og_verify_aa07e56_gt_sensitivity.md`：token 级 mean-of-cos 与真实 causal attention **ρ = +0.05**；③ `96f7865` 新增的 81 + 51 行测试中，无任何「IFR top-k vs 真实 attention top-k」recall 评测。 |

---

### R2 — 有界视图省的是算力，不是显存；字节收益仍是 0

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 §0.3："返回紧凑张量至 SDPA 算子"。<br>`metal_attention_hook.py:160`：`keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)` ← **先无条件写满 DynamicCache**<br>`metal_attention_hook.py:237-241`：之后才切片返回 |
| **为什么错** | 返回裁剪视图 ≠ 释放缓存。L160 已把全量写入 `DynamicCache`，L241 的 `torch.cat(slices_k)` 只是**视图**，底层 buffer 未动。这与对方在 Q15 自认的"**字节收益确实为 0（诚实报 1.0×）**"是同一事实，本次重构未改变。<br>新基准用 `compute_reduction_pct`（算力）命名是对的，**但** `qsa_kv_resident_mb = 48.0`、`total_resident_ram_gb` 这两个"驻留"字段来自 `compute_footprint_mb()` 的**纸面模型**，不是测量。 |
| **正确结论** | 当前实现：**显存收益 = 0**；**FLOPs 收益 = 1 − attended/total**。JSON 中 `qsa_kv_resident_mb` / `total_resident_ram_gb` 两列应改名 `…_modeled_mb` / `…_modeled_gb`，或删除。 |
| **支撑证据** | 本机复跑（1024 预填 + 1 解码，`block_size=32, head_dim=64`）：<br>`returned to SDPA : 512`<br>`cache layer keys : torch.Size([1, 4, 1025, 64])　values: torch.Size([1, 4, 1025, 64])`<br>`resident bytes   : 1049600 = 1025.0 KiB`<br>`benchmark_flashnext_hybrid_e2e.py:134,144,146`：三个内存字段直接取自 `mem_model.compute_footprint_mb(ctx, use_kvmem_bounded_view=True)`。 |

---

### R3 — headline「>15×」：归属错、基线虚高 16×、分子分母口径不一致

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 Q15："**实测与理论双重实现 > 15× 的真实物理显存压降**"。<br>`kvmem_fusion/flashnext_hybrid.py:102-104`：<br>`# 3. Dense Baseline Comparison (if all 48 layers were dense attention with 8 KV heads)`<br>`dense_bytes_per_token = 2 * cfg.num_hidden_layers * 8 * cfg.head_dim * bytes_per_elem`<br>新基准 JSON 沿用同一基线：`4096 ctx → dense_kv_baseline_mb = 1536.0` |
| **为什么错** | **（a）归属错。** 1M 下 KV 仅 0.047 GiB，上限来自 `indexer_budget = 2048`——FlashNext **config.json 的原生属性**。KVMem 的 IFR 替换的是模型自己的粗排器，**上限同为 2048**，故 KVMem 的**额外字节收益 = 0**。这 15× 衡量的是「FlashNext vs 一个 48 层 8 KV 头稠密 Transformer」，后者**不存在**；卸掉 KVMem 后 15× 一分不少。<br>**（b）基线虚高 16×。** 代码用 48 层 × 8 KV 头；自身 config 是 36 linear + 12 full（**仅 12 层承载 KV**）、`num_key_value_heads: 2`。<br>**（c）口径混。** 分子只算 KV，分母是"总常驻"（含 4.39 GB 专家权重 + 2.00 GB macOS）。 |
| **正确结论** | ① 真实来源 = FlashNext 架构，非 KVMem；② 即便保留反事实框架，基线应对齐 config（12 × 2），代码基线为真实值的 **16.0×**；③ 唯一能证明 KVMem 增量的对照是「FlashNext 原生 indexer vs FlashNext + IFR」，**从未建立**。 |
| **支撑证据** | `FlashNextMemoryModel.compute_footprint_mb` 实算：<br>`code baseline bytes/token = 393216`<br>`real full-attn bytes/token (12 layers × 2 kv heads) = 24576`<br>`inflation = 16.0×`<br>`1M：code baseline 366.21 GiB　|　real full-attn 22.89 GiB　|　their QSA KV 0.0469 GiB`<br>`kv_compression_ratio = 7812.5`（代码自印；按他们自己的口径其实是 7812×，可见 15× 这个数还额外掺入了分母口径问题）。<br>新基准 JSON：`1536.0 MB / 4096 = 0.375 MB/token = 393,216 B/token`，**仍是虚高基线**。 |

---

### R4 — 三套互相矛盾的"实测"内存数字，且无权重可供实测

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 §3 表："**实测总常驻物理内存 7.74 / 7.80 / 8.31 GB**"；<br>§0："稳定在 **10.75 GB ~ 14.0 GB**"；Q15："约 **12.4 GB**"；README（`e0f5e15` +5）："**10.75 GB 紧凑常驻**"；<br>新基准 JSON（`96f7865`）：**6.48 / 6.48 / 6.48 / 6.48 / 6.49 / 6.50 GB**。 |
| **为什么错** | 现在**同一项目里有三套数**：文档 §3（7.74–8.31）、文档 §0/Q15/README（10.75–14.0）、新基准与代码（6.48–6.50）。新基准与代码一致，**等于默认了文档 §3 的 7.74/7.80/8.31 不可复现**，但文档未作任何修正。<br>此外：`~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/*/` 下只有 `config.json`（符号链接），`find -name "*.safetensors"` = **0**，目录 12 KB。**没有权重 → 不存在"实测"。** |
| **正确结论** | 全部内存数字是**基于 config.json 的纸面推演**。请把 §3 表头"实测总常驻物理内存"改为"纸面推演常驻"，用 6.48–6.50 统一全文，或补交真实权重与 `footprint`/`vm_stat` 采样。 |
| **支撑证据** | `compute_footprint_mb` 实算（32K/128K/1M）：`total_resident_ram_gb = 6.48 / 6.50 / 6.66`；`dense_kv_baseline_gb = 12.00 / 48.00 / 366.21`。<br>复跑 `96f7865` harness：`RAM: 6.48 / 6.48 / 6.48 / 6.48 / 6.49 / 6.50 GB`。<br>模型目录：`ls` 仅 `config.json`；`find … -name "*.safetensors" | wc -l` → `0`。 |

---

### R5 —【本轮最严重】基准数据块内离散度为 `0.0000`，真实 Qwen3-0.6B KV 是 `0.7388`；IFR 的定义性机制在基准中零触发

| | |
|---|---|
| **对方原话 / 代码位置** | `96f7865` commit message："**with empirical 4K-128K results**"。<br>`benchmarks/benchmark_flashnext_hybrid_e2e.py:85-90`：<br>`k_base = centroids[c_idx] + rng.randn(cfg.indexer_head_dim).astype(np.float32) * 0.1`<br>`k_base /= np.linalg.norm(k_base)`<br>`k_mat = np.tile(k_base, (32, 1))` ← **块内 32 个 token 的 Key 向量完全相同** |
| **为什么错** | `np.tile` 使块内 32 个 token 的 K 完全一致 → **`norm_dispersion = 0.0000`（精确为 0）**。<br>而 IFR 之所以叫 IFR——区别于普通 IVF-mean-pooling 的那个机制——是 `AntiCollapseSplitter` 的抗坍缩双子分裂（`ifr.py:149-196`：块内 1 个 burst needle + 31 个弥散 token 时，mean-pooling 把 needle 稀释 1/32）。**离散度为 0 时该机制在数学上不可能触发。**<br>我方用**真实 Qwen3-0.6B layer-0 K 投影**（未经 RoPE，8 KV 头 × 128 维，按头取均值得每 token 128 维；4,096 token 自然语料）实测：<br>`norm_dispersion  mean=0.7388  median=0.7379  p95=0.8332  max=0.8690`<br>`anti-collapse splits (theta=0.868): 1/128 = 0.8%`<br>真实 KV 的离散度是基准数据的**整个值域**（0.7388 vs 0.0000）。<br>**同时请注意**：这反过来说明该机制在真实模型上**几乎不激活**（0.8%），其宣称的 needle 保护价值未在真实数据上被证明有效——这是 IFR 方法本身的问题，须在主报告中如实写明。 |
| **正确结论** | ① 该基准**对 IFR 的定义性机制是盲的**：基准中 0/4096 触发、真实 KV 0.8%、随机数据 100%，差异达两个数量级，而基准取的是那个**永远不触发**的端点；② 因此这份基准既测不出 IFR 的代价，也测不出 IFR 的收益；③ 对方 4K 数 `0.048 ms` 与真实 KV 实测 `0.075 ms` 差 **1.56×**。 |
| **支撑证据** | **离散度—延迟扫描**（`IFRRetriever(dim=128, target_top_k=64)`，4096 块 = 128K 上下文，`top_k=56`，θ=0.868）：<br>`noise=0.0 norm_disp=0.000 split=   0/4096 ->  1.178 ms` ← **基准数据落在这一行**<br>`noise=0.2 norm_disp=0.970 split=4096/4096 -> 10.879 ms`<br>`noise=0.4 norm_disp=1.010 split=4096/4096 -> 11.190 ms`<br>`noise=1.4 norm_disp=1.016 split=4096/4096 -> 11.603 ms`<br>→ 是**阶跃**不是渐变：任何非零离散度都使延迟约 **9.2×**。<br>**真实模型复核**（脚本见 `docs/og_repro_real_kv_dispersion.py`）：<br>`layer0 K: (8, 4096, 128)  (num_kv_heads=8, head_dim=128, S=4096)`<br>`REAL Qwen3-0.6B norm_dispersion  mean=0.7388  p95=0.8332  splits=1/128 (0.8%)`<br>`REAL-KV IFR retrieval (dim=128, top_k=56, 4096 tokens): mean=0.075 ms  p95=0.083 ms`<br>→ 对方同规模发布值 **0.048 ms**（复跑 0.045 ms），乐观 **1.56×**。 |

---

### R6 — @10M 主终点仍是 2,441× 的外推，且 harness 从未运行任何模型

| | |
|---|---|
| **对方原话 / 代码位置** | `96f7865`：harness 覆盖 `4096 … 131072`；JSON 最大规模 `context_tokens: 131072`。<br>`benchmark_flashnext_hybrid_e2e.py:53-67`：`LinearDeltaNetStateCell` 输入为 `torch.randn(1, 16, 1, 128)`（**单 token 哑张量**），`total_36_linear_us = linear_step_us * 36`。<br>`L126-131`：`bounded_tokens = min(ctx, local_window_tokens + len(ret_bids)*32)`，`sparsity_ratio = bounded_tokens / ctx`。 |
| **为什么错** | ① **10M 未测**，最大只到 128K，外推 **2,441×**（4,096 → 10,000,000）。<br>② **"36 层线性层"从未运行**——只跑了一个手写单 token cell 再乘 36，没有模型、没有 36 层、没有真实序列。<br>③ **`bounded_attended_tokens` 是算术，不是测量**：由 `len(ret_bids)` 算出，从未经过任何一次真实 attention 的 `keys.shape[-2]`。<br>④ 因此 JSON 中真正被测量的只有 `mean_retrieval_ms` / `p95_retrieval_ms` 两项（且是在 R5 所述的退化数据上），其余 7 个字段全是 `compute_footprint_mb()` 纸面值。 |
| **正确结论** | commit message 的 "empirical" 只适用于检索延迟一列。请把 JSON 拆分为 `measured`（2 列）与 `modeled`（7 列），并补 @1M / @10M 实测，或明确标注"10M 未测"。 |
| **支撑证据** | 复跑输出与 JSON 字段逐一对应：`Attended: 2048/131072 (98.4% cut)` 与 `bounded_tokens = min(131072, 256+56*32) = 2048` 完全吻合——**该数字由 `len(ret_bids)` 算术得出，不是张量形状**。 |

---

### R7 — §0 的「288 Token」无法复现；测试把 budget 从 2048 改成 512

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 §0.4："**如 1025 Token 序列实测返回 288 Token 紧凑视图**"；§0.1："默认对齐 FlashNext `indexer_budget=2048`"。<br>`tests/test_metal_attention_hook.py:363-364`：`interceptor.local_window_tokens = 256` / `interceptor.indexer_budget = 512` |
| **为什么错** | 用测试自身参数（budget 512 / window 256）复跑三次，返回 **448 / 480 / 512**，从未出现 288。且测试把 budget 调到 512，比文档宣称"默认对齐"的 2048 小 **4×**。用真正默认 2048 时，1025 token 返回 **800**（sparsity 0.7805）——预算根本没用满，"有界"在这个量级只等价于 **22% 的 FLOPs 削减**。 |
| **正确结论** | 请给出能复现 288 的脚本与随机种子；否则修正文档。并用默认 `indexer_budget=2048` 重跑并公布 attended 数字。 |
| **支撑证据** | 本机复跑（`MetalKVCacheInterceptor(block_size=32, head_dim=64, dtype=float16, device=mps)`，1024 预填 + 1 解码）：<br>`budget  512 win 256 -> returned 448 / 480 / 512（三次）  sparsity 0.4371`<br>`budget 2048 win 256 -> returned 800  sparsity 0.7805  ranges 8`<br>`budget 2048 win 512 -> returned 896  sparsity 0.8741  ranges 5`<br>（`metal_attention_hook.py:123-124` 默认值确为 2048 / 256。） |

---

### R8 — §2「架构同构映射表」把模型原生属性登记为 KVMem 组件红利

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 §2 表（原文三行）：<br>`| 36 层 Linear Attention (DeltaNet) | CASA 前缀状态快照 | O(1) 恒定内存，36层零KV膨胀 |`<br>`| 原生 indexer_budget: 2048          | Bounded Execution View | 规范化槽位，无需破坏模型结构|`<br>`| 极致 GQA: num_key_value_heads: 2   | LADDER 阶梯量化        | KV 体积减半，Softmax 无偏  |` |
| **为什么错** | 三行**左列全是 FlashNext `config.json` 的原生属性**（`layer_types`、`indexer_budget`、`num_key_value_heads`），不是 KVMem 引入的。并排映射制造了"这些收益由 KVMem 带来"的因果暗示。这三条**把 `kvmem_fusion` 整个删掉后依然全部成立**。 |
| **正确结论** | 表里唯一可能产生 KVMem 增量的是「IFR 检索替换原生 indexer 的**选块质量**」——而它恰是全仓库唯一**从未测量**的项（见 R1）。请增列"若无 KVMem 是否仍成立"与"KVMem 增量 = ?"。 |
| **支撑证据** | `config.json`：`"layer_types": ["linear_attention"×3, "full_attention", …]`、`"indexer_budget": 2048`、`"num_key_value_heads": 2`、`"head_dim": 256`。结合 R2（字节增量 0）与 R1（质量未测）：**该表当前没有任何一列能产出可归属给 KVMem 的量化增量**。 |

---

### R9 — 新增测试仍是「内部一致性测试」，不构成主终点证据

| | |
|---|---|
| **对方原话 / 代码位置** | 答辩文档 Q16："输入全部来自动态张量实测"。<br>`tests/test_metal_attention_hook.py:388-396`：<br>`assert out_k_dec0.shape[-2] < total_cached`<br>`assert interceptor.attended_tokens_last == out_k_dec0.shape[-2]`<br>`assert interceptor.last_sparsity_ratio == expected_sparsity`<br>`assert interceptor.last_retrieval_recall is None` |
| **为什么错** | 四条断言证明的是「**上报值 == 实际返回张量**」——内部自洽，不证明系统达标。与 Q16 已承认的同义反复属**同一类证据缺陷**。对照原方案 M2 主终点（L280/L308）：@10M ≤350 ms（**未测**，见 R6）、索引驻留 ≤4 GiB（**未测**）、与 Full-Context 真值 1pp 内（**未测**，无任何输出质量对比）、解码 tok/s 不下降（**未测**）。 |
| **正确结论** | 请用"终点测试"补充"一致性测试"：① 10M 检索实测；② 10M 索引实测驻留；③ IFR 选块 vs Full-Context 的 perplexity / 任务分差。在此之前，`134/134 全绿`只能表述为"内部一致性全绿"。 |
| **支撑证据** | 复跑全量套件 `134 passed, 1 warning in 23.68s`（**属实**）；`96f7865` 新增的 81 + 51 行测试中，无任何终点断言。 |

---

## §2 仍需对方澄清的问题（Q17–Q28）

> 逐条给出：可复现脚本 / 日志 / 或明确标注"未测"。**未回应视为该项不成立。**

| # | 问题 | 关联 |
|---|---|---|
| **Q17** | `metal_attention_hook.py:185` 为何用 `key_states`（K）作检索 query？请给理论依据，或给把检索下移到 `metal_sdpa_attention_forward`（该处有 q）后的对照。 | R1 |
| **Q18** | 请给出「IFR top-k vs 真实 causal attention top-k」的 **recall@k**（同一模型、同一层、同一序列）。mean-of-cos proxy 与真实注意力 ρ 我方实测 **+0.05**，不能用 proxy 自证。 | R1 |
| **Q19** | 请给出使 `interceptor.layers[0].keys` 实际缩短的代码路径与测量。当前 `super().update()`（L160）写满全量，实测 `[1,4,1025,64]` 与返回 512 并存。若不替换 `DynamicCache` 后端，请撤回一切"显存/物理内存压降"表述。 | R2 |
| **Q20** | 请把 JSON 的 `qsa_kv_resident_mb` / `total_resident_ram_gb` 改名为 `…_modeled_*`（二者来自 `compute_footprint_mb`，非测量），或补测。 | R2 |
| **Q21** | `flashnext_hybrid.py:103` 的 `8`（KV 头）与 `num_hidden_layers=48` 从何而来？config 为 `num_key_value_heads: 2`、仅 12 层 `full_attention` 承载 KV。是否同意改为 12 × 2？改后 headline 是多少？ | R3 |
| **Q22** | 请给出「FlashNext 原生 indexer」与「FlashNext + KVMem IFR」的**同预算（均 2048）对照**——这是唯一能证明 KVMem 有增量的实验。若两者字节与延迟相同，请明写"KVMem 在本对照中的字节增量 = 0"。 | R3 / R8 |
| **Q23** | 现有三套内存数：文档 §3（7.74–8.31）、文档 §0/Q15/README（10.75–14.0）、新基准与代码（6.48–6.50）。请以代码为准统一全文，或给出能复现文档值的脚本。 | R4 |
| **Q24** | "实测"二字依据什么？Flash-Next 快照 **0 个 safetensors**、仅 12 KB。请改为"纸面推演"，或补交真实权重与 `footprint` 采样。 | R4 |
| **Q25** | **`np.tile(k_base, (32,1))`（L89）使块内离散度精确为 0。** 请改用真实模型 KV，或至少使 `norm_dispersion` 落在真实区间（我方实测 Qwen3-0.6B：mean 0.7388、p95 0.8332）重跑，并公布新的 4K–128K 延迟。 | R5 |
| **Q26** | `AntiCollapseSplitter` 在真实 Qwen3-0.6B KV 上只触发 **1/128 = 0.8%**。请说明：IFR 相对普通 IVF-mean-pooling 的增量价值，在一个 0.8% 触发率的机制上如何成立？是否有真实数据支持 θ=0.868 这个"p95 of real Qwen norm_max"的标定？ | R5 |
| **Q27** | 请给出 **@1M 与 @10M** 的检索实测或完整基准脚本。当前最大 128K，10M 是 **2,441×** 外推。我方用真实 KV 实测 4K = 0.075 ms，线性外推 10M ≈ 184 ms（在 350 ms 内）——请用真实分布的数据确认或否证这个外推。 | R6 |
| **Q28** | `linear_36_layers_total_us` 来自单 token 哑张量 × 36，`bounded_attended_tokens` 由 `len(ret_bids)` 算术得出。请把 JSON 拆分为 `measured` / `modeled` 两组字段。 | R6 |

---

## §3 变更梳理（附录 A）

| Commit | 时间 (-0400) | 文件 | 改动 |
|---|---|---|---|
| [`085514e`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/085514ef7024560e279fe8a797c2b2ddfd60308f) | 2026-10-14 14:15:37 | `scripts/kvmem_menubar.swift` | +5 / −5 |
| [`e0f5e15`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/e0f5e15e7552935587d6939e3acbaf00fa7f433e) | 2026-10-15 12:05:00 | 新增 `docs/kvmem_flashnext_hybrid_and_bounded_view_response.md` | +129 |
| 同上 | 同上 | 新增 `kvmem_fusion/flashnext_hybrid.py` | +165 |
| 同上 | 同上 | `kvmem_fusion/metal_attention_hook.py` | +87 / −21（有界视图 L181–243） |
| 同上 | 同上 | 新增 `tests/test_flashnext_hybrid.py` | +81 |
| 同上 | 同上 | `tests/test_metal_attention_hook.py` | +51 |
| 同上 | 同上 | `README.md` | +5 |
| [`96f7865`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/96f78659c7b628faf1121e3389a3272b4bd495c9) | 2026-10-15 12:47:23 | 新增 `benchmarks/benchmark_flashnext_hybrid_e2e.py` | +175 |
| 同上 | 同上 | 新增 `benchmarks/flashnext_phase1_benchmark_results.json` | +85 |

**无 PR、无 merge commit**，全部直接 push 到 `main`。

## §4 动机确认（附录 B）

| 对方自认 | 出处 | 对应我方指控 | 判定 |
|---|---|---|---|
| "采用静态字面量…属于同义反复" | 答辩 §Q16 | `og_strategic_assessment_20261014.md`（`test_plan01_ifr.py:275-322`） | **对应准确，承认完整** |
| "字节收益确实为 0（诚实报 1.0×）" | 答辩 §Q15 | `og_audit_38ae75e.md` 核心指控 | **对应准确**——但同一份文档后段与 §3 表又主张「>15× 真实物理显存压降」，与该自认**直接冲突**（R2/R3） |
| "彻底删除 L196 合成 recall 拟合算式" | 答辩 §0.4 | `og_audit_38ae75e.md` P1 | **对应准确，已实测验证** |
| 新基准改用 `compute_reduction_pct`（算力）命名 | `96f7865` JSON | 我方 R2 | **部分响应**：命名对了，但同 JSON 仍保留两个纸面"驻留"字段 |
| 新基准采用代码实算值 6.48–6.50 GB | `96f7865` JSON | 我方 R4 | **默认了文档 §3 的 7.74–8.31 不可复现**，但文档未修正 |

**结论**：动机层面他们确实读懂了 Q13–Q16，三条自认有两条落到实处。**唯一自相矛盾的是字节收益**——承认 0 之后又主张 >15×。

## §5 正确性评估（附录 C）

| 论断 | 判定 | 备注 |
|---|---|---|
| 有界执行视图 `keys.shape[-2] < total` | ✅ | 实测 512 / 448 / 480 vs 1025 |
| `attended_tokens_last` 与返回张量一致 | ✅ | 逐位一致 |
| `last_sparsity_ratio` 可复算 | ✅ | `round(512/1025,4)=0.4995` |
| 合成 recall 已删除 | ✅ | L228 + 测试 L396 |
| R4（K 不动、Q 侧重映射）数学论证 | ✅ | RoPE 相对性 |
| 134/134 全绿 | ✅ | 复跑 `134 passed in 23.68s` |
| `indexer_budget=2048` / 125B / spill 注入 | ✅ | **撤回**前次质疑 |
| Phase 1 harness 可复现 | ✅ | 128K 复跑 1.392 ms vs 发布 1.352 ms |
| `compute_reduction_pct` 命名 | ✅ | 方向正确 |
| 检索 query 是 Q | ❌ | 代码为 `key_states`（R1） |
| 有界视图带来显存下降 | ❌ | DynamicCache 仍 1025（R2） |
| 「>15×」可归属 KVMem | ❌ | 归属错 + 基线 16× + 口径混（R3） |
| 「实测总常驻 7.74/8.31 GB」 | ❌ | 代码与基准均为 6.48–6.50；0 safetensors（R4） |
| 基准数据代表真实 KV 分布 | ❌ | `norm_dispersion = 0.0000` vs 真实 **0.7388**（R5） |
| 「empirical 4K-128K」 | 部分 | 仅 2 列实测，7 列为纸面（R6） |
| 「1025 → 288」 | ❌ | 实测 448/480/512（R7） |
| §2 映射表的红利归属 | ❌ | 三条均为模型原生属性（R8） |
| 新增测试证明系统达标 | ❌ | 一致性测试，非终点测试（R9） |

## §6 我方自我纠正（本次复核）

1. **撤回「@10M 外推 963.8 ms、超 E 门 2.75×」。** 该数基于**随机高斯块**（`norm_dispersion ≈ 1.02`，anti-collapse 100% 触发），离散度**显著高于真实模型**。改用真实 Qwen3-0.6B KV（`norm_dispersion` mean 0.7388 / p95 0.8332）实测：4K = **0.075 ms**，线性外推 10M ≈ **184 ms**，**在 350 ms E 门内**。此前结论作废；对应指控改为 R6（10M 从未测量，外推跨 2,441×）与 R5（基准数据分布失真）。
2. **撤回「132/134」。** 那次 `2 failed` 为 `socket.timeout` 并发抖动；复跑 **134 passed**。
3. **撤回「SpillManager 仅存在于测试」。** `native_engine.py:200` 已真实注入。
4. **撤回「检索算了不用」。** 已由 `e0f5e15` 修复。
5. **我方测试脚本曾出过一次 bug**（各块 `tokens` 相同触发 L0 去重，只返回 1 个块，误得 0.010 ms）。已修正为每块唯一 token id 后重测，§1 R5 的数字为修正后结果。

> **方法论备忘**：合成数据的**分布参数**与算法参数同等重要。本项目已连续三轮出现"用退化合成数据得出 headline"的情况（C++ bench 的 `2000/2000` 全分裂、本轮 Python bench 的 `norm_dispersion = 0.0000`）。审计时**必须先用真实权重标定分布区间**，再判断基准是否落在区间内——否则容易像我方这次一样，用一个方向错误的合成分布去反驳另一个方向错误的合成分布。
