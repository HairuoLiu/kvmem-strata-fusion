# 对 `76acb4a` + `c8f507a` + `1d508d4` 的逐条比对（我方上轮反馈 R1–R9 / Q17–Q28）

> **仓库**：<https://github.com/HairuoLiu/kvmem-strata-fusion>
> **比对对象（对方在 2026-10-15 13:31–14:00 连推的三个提交）**：
> - [`1d508d4`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/1d508d4608d544b4b8021086b79080e4447c2c6c) `build: compile native C++ libkvmem_engine.dylib with ARM Neon SIMD`，13:31:22 -0400
> - [`c8f507a`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/c8f507a9bba644ba7e2373dd9e52abbb74c4e859) `feat(bench): address OG Critique R5 with realistic Qwen3 empirical dispersion benchmark & test suite`，13:46:15 -0400
> - [`76acb4a`](https://github.com/HairuoLiu/kvmem-strata-fusion/commit/76acb4a37ee0ac5d781b024a9798309088eb3288) `feat(core): implement C++ ARM Neon SIMD engine, True-Q attention sinking, active physical eviction, and empirical Qwen3 dispersion`，14:00:17 -0400
> - **PR：无**，全部直接 push 到 `main`。
> **我方上轮反馈**：`docs/og_rebuttal_e0f5e15_v2.md`（commit `d20f77a`）。
> **审计基准**：Apple M6 / 32 GB；`.venv`（Python 3.9 + torch 2.8.0 + transformers 4.57.6）；真实权重 `Qwen3-0.6B`；C++ 已 `make` 构建。

---

## §0 结论摘要

**这是一个高质量、高速度、且大部分属实的回应——但同时引入了一个会让整个 R2 成果作废的致命缺陷。**

三个提交共 +2,760 行，明确点名回应 R1 / R2 / R5，并派了四个"Lead"分工。我方逐条实测后：

- **测试数属实**：全量 153 个测试，我复跑 **152 passed / 1 failed**（唯一失败是 `test_agent_protocol.py::test_e2e_server_chat_completions_streaming`，属网络抖动）。对方声称"153 passing in 32s"，基本属实。
- **R5 已正面且方法正确地回应**：采用我方实测的 0.7388 作为标定目标，消融对照公平（θ=0.868 vs 999.0，其余参数全同），needle 为"32 个 token 中替换 1 个"（正是反坍缩场景）。我复跑 **recall 字段逐位一致**。
- **R2 的物理压实确实实现，且数字属实**：8193 逻辑 token → 物理 buffer **2176**（上限 2304），内存 **3.77×** 下降，`get_seq_length()` 正确返回逻辑长度 8193，39–55 块真实换出 NVMe。
- **C++ 引擎真实**：`make` 可编译；Neon dot **2.30 µs**（宣称 2.2）、UBBA 10K **69.44 µs**（宣称 75）、子质心排序 **90.70 µs**。全部为真，无虚报。
- **还采纳了我方最早提的一个优化建议**（`IFRRetriever._cached_block_ids` / `_cached_all_mean_keys` 缓存块均值，避免每次重算）。

**但有一个 P0 级缺陷，必须立即指出：**

> **`_evict_and_compact` 在 prefill 阶段就被触发，而此时没有任何 query 可用（它拿最后一个 token 的 Key 冒充 query），于是永久删除了约 68% 的 token。实测后果：与全注意力的输出 cos 相似度仅 0.1336、top-10 预测重叠 0/10。**
>
> 对照实验非常干净：关闭压实时（`max_cache_tokens` 设极大），同一条 hooked 路径与原生 HF 全注意力**逐位相同**（max|Δlogit| = 0.0000，cos = 1.000000，top-10 重叠 10/10）。也就是说，**分歧 100% 由压实引入，且是灾难性的**。

---

## §1 已被采纳或落实的（6 项）

| # | 我方上轮观点 | 落实情况 | 实测证据 | 判定 |
|---|---|---|---|---|
| 1 | **R2** 有界视图省 FLOPs 不省显存；`DynamicCache` 仍驻留全量 | 新增 `MetalKVCacheInterceptor._evict_and_compact()`（`metal_attention_hook.py:535-636`），真正执行 `layer.keys = compact_k`（L629-630）；`spill_manager.spill_blocks()` 把冷块写 NVMe | 8193 逻辑 → 物理 **2176**；驻留 4,352 KiB vs 未压实 16,386 KiB = **3.77×**；换出 **39–55** 块；`get_seq_length(0)=8193`（RoPE 位置未被破坏） | ✅ **真实修复，本轮最大突破** |
| 2 | **R1** 检索 query 取的是 K 不是 Q | 在 `metal_sdpa_attention_forward` 中下沉检索并改用真 Q：`q_slice = query[:, :, -1:, :]` → `interceptor.retrieve_bounded_view(query=q_slice, ...)`（L857-864） | 见 §3 第 1 条：此路径**仅部分修复** | ⚠️ 部分落实 |
| 3 | **R5** 基准数据块内离散度 0.0000，真实 KV 是 0.7388；反坍缩机制在基准中零触发 | 新增 `benchmarks/eval_qwen3_real_kv.py` + `tests/test_real_kv_haystack.py`，docstring 明确引用我方实测值，按统计量生成非均匀 haystack | 复跑：BG dispersion 0.7384–0.7430、needle dispersion 1.18–1.22、needle split **100%**、BG split 0.8–3.4%、**IFR recall 100% vs naive 80%/15%/16%/12%**——recall 字段与我方复跑**逐位一致** | ✅ **方法学正确，已正面回应** |
| 4 | C++ 引擎（我方此前长期要求原生实现） | `csrc/c_api.h/.cpp`（+678 行）、`kvmem_fusion/c_engine_bindings.py`（+460 行），ctypes 零拷贝绑定 + 纯 Python 回退；`ifr.py` 热路径接入 | `make` 干净编译；`Neon dot 2.301 µs`、`UBBA 10K 69.44 µs`、`sub-centroid ranking 90.70 µs`（对方宣称 2.2 / 75） | ✅ **属实** |
| 5 | 我方最早一轮建议"把块均值在索引时算一次并缓存" | `IFRRetriever._cached_block_ids` / `_cached_all_mean_keys`（`ifr.py:359-360`） | 代码在位 | ✅ 已采纳 |
| 6 | 我方的免责性疑问（dylib 未入库导致干净克隆失败） | 新增 `Makefile` 目标可一键构建 | `make` 成功产出 `build/libkvmem_engine.dylib` | ⚠️ 见 §3 第 6 条 |

---

## §2 被遗漏或未处理的（6 项）

| # | 我方上轮观点 | 现状 | 位置 |
|---|---|---|---|
| 1 | **R3** `flashnext_hybrid.py` dense 基线虚高 **16.0×**（48 层 × 8 KV 头 vs config 的 12 层 × 2 头） | **未修，原样保留** | `kvmem_fusion/flashnext_hybrid.py:103` 仍是 `dense_bytes_per_token = 2 * cfg.num_hidden_layers * 8 * cfg.head_dim * bytes_per_elem`。三个提交均未触及该文件 |
| 2 | **R4** 三套互相矛盾的"实测"内存数 | **未修** | 响应文档 §0 L8（10.75–14.0 GB）、Q15 L58（12.4 GB）、§3 表 L114（7.74/7.80/8.31 GB）全部原样；`README.md:8` 亦仍写"10.75 GB 紧凑常驻"。而他们自己的代码与新基准都输出 **6.48–6.50 GB** |
| 3 | **R6** Phase 1 harness 的 `np.tile` 使离散度 = 0.0000；@10M 从未测量 | **未修** | `benchmarks/benchmark_flashnext_hybrid_e2e.py:89` 仍是 `k_mat = np.tile(k_base, (32, 1))`。harness 最大规模仍 128K，@10M 仍无实测。**注意：他们是"在旁边加了一个新基准"，没有修旧的退化基准**——旧基准仍留在仓库里继续产出 dispersion=0 的数字 |
| 4 | **R7** 文档 §0 "1025 Token 序列实测返回 288 Token"不可复现（我实测 448/480/512） | **未修** | `docs/kvmem_flashnext_hybrid_and_bounded_view_response.md:29` 原文仍在 |
| 5 | **R8** §2 同构映射表把三条模型原生属性登记为 KVMem 红利 | **未修** | 响应文档 §2 表（L79-83）原样 |
| 6 | **R9/Q28** 测试是"内部一致性测试"非终点测试；JSON 应拆 `measured`/`modeled` | **部分未修** | 新增 6 个测试文件、153 个测试，但仍无任何主终点测试（@10M、索引 ≤4 GiB、与 Full-Context 1pp 内）；`benchmark_flashnext_hybrid_e2e.py` 产出的 JSON 仍未拆分字段 |

---

## §3 我方与对方不一致、或认为其实现/做法仍有问题的（7 项，按严重度排序）

### P0-1 压实发生在 prefill 阶段，且没有 query 可用——输出被彻底破坏

| | |
|---|---|
| **对方原话 / 代码位置** | `76acb4a` commit message："Strata Tiering Lead: address OG Critique R2 by implementing active physical memory compaction & NVMe SuperTile offloading in MetalKVCacheInterceptor and DiskSpillManager; physical buffer bounded to <= 2304 tokens at 8192 context"<br>`metal_attention_hook.py:240-243`：<br>`eviction_threshold = self.max_cache_tokens`<br>`if curr_logical_len > eviction_threshold and hasattr(self, "_evict_and_compact"):`<br>&nbsp;&nbsp;`self._evict_and_compact(layer_idx, key_states, value_states, curr_logical_len)`<br>`metal_attention_hook.py:562`（在 `_evict_and_compact` 内）：<br>`q_vec = key_states[0, :, -1, :].mean(dim=0)...` ← **拿最后一个 token 的 Key 冒充 query** |
| **问题所在** | ① `_evict_and_compact` 的触发条件**只看 `curr_logical_len > max_cache_tokens`，完全不看 `q_len`**。所以它在 **prefill**（`q_len = 3000`）期间就逐层触发。<br>② prefill 阶段没有任何 query 语义可言——L562 用 `key_states[..., -1, :]`（最后一个 token 的 **Key**）当检索 query。<br>③ 于是 prefill 尚未完成，就已有约 68% 的 token 被永久删除并写入 NVMe，后续 token 再也看不到它们。<br>④ `tests/test_physical_eviction.py` 的 5 个测试**全部只断言形状/大小**，没有任何一条比对输出正确性。 |
| **我方立场** | 这不是"有界视图的预期稀疏"，这是**正确性缺陷**。压实必须至少满足：(a) prefill 阶段不做 compaction（或改为正规的 sliding-window/分块因果模式）；(b) 检索必须有真实 query；(c) 必须有端到端 logits 比对测试。 |
| **支撑证据（我方可复现）** | 单次 forward（仅 prefill，无 decode），`Qwen3-0.6B`，3000 token 上下文，CPU/fp32：<br>```<br>[B no-evict] max_cache=1000000000  physical_buffer= 3000  attended=0  offloaded=0<br>[C evict   ] max_cache=2048        physical_buffer=  960  attended=960  offloaded=39<br><br>A vs B (full attention vs bounded, NO compaction):<br>    max|d|=0.0000  mean|d|=0.000000  cos=1.000000  top10_overlap=10/10<br>B vs C (bounded no-compaction vs bounded WITH compaction):<br>    max|d|=22.9797  mean|d|=4.660890  cos=0.133631  top10_overlap=0/10<br>```<br>**结论极其干净**：不压实时 hooked 路径与原生 HF 全注意力**逐位相同**；一旦开启压实，输出完全不相关。分歧 100% 由压实引入。 |

### P0-2 R1 只修了一半，而且修的是不重要的那一半

| | |
|---|---|
| **对方原话 / 代码位置** | commit message："DeepMind Kernel Lead: address OG Critique R1 by sinking retrieval from DynamicCache.update() into metal_sdpa_attention_forward using genuine Query tensor Q"<br>已修：`metal_attention_hook.py:857-864`（真 Q）<br>**未修**：`metal_attention_hook.py:562`（仍在 `_evict_and_compact` 内用 `key_states` 当 query） |
| **问题所在** | 现在有两条检索路径：注意力时的 `retrieve_bounded_view(query=q_slice)` 用**真 Q** ✓；但决定"哪些块能留在内存里"的 `_evict_and_compact` 仍用 **K-K proxy** ✗。<br>**换出是不可逆决策**——被换出的块进了 NVMe，注意力阶段的真 Q 检索只能在幸存者里挑。所以真正决定"模型能看到什么"的，仍然是那个 K-K proxy。 |
| **我方立场** | R1 的核心不是"注意力时用不用真 Q"，而是"**不可逆的丢弃决策用什么判据**"。请把 L562 改为接收真实 query（由 `metal_sdpa_attention_forward` 把 `q_slice` 传给 interceptor 缓存，`_evict_and_compact` 复用），并给出 recall 校验。 |
| **支撑证据** | 代码行如上；另我方上轮已实测 mean-of-cos proxy 与真实 causal attention 的 ρ = +0.05。 |

### P1-3 "real_kv" 命名误导：新基准并未加载任何真实模型

| | |
|---|---|
| **对方原话 / 代码位置** | `benchmarks/eval_qwen3_real_kv.py`、`benchmarks/real_kv_haystack_results.json`、JSON 键 `"calibrated_model": "Qwen3-0.6B"`；commit message："on non-uniform **empirical Qwen3 KV** dispersion" |
| **问题所在** | `generate_qwen3_empirical_haystack()` 是**按我方实测的统计量合成的**，全文件**没有 `from_pretrained`、没有加载任何权重**。docstring 本身是诚实的（"generates realistic non-uniform KV activations **matching** Qwen3 empirical dispersion"），但文件名、JSON 键和 commit 措辞都读起来像"跑了真实模型"。 |
| **我方立场** | 这与我方上轮批评 R4"把纸面值标成实测"是**同一类问题**，只是换了方向。请把文件改名为 `eval_qwen3_calibrated_dispersion.py`，JSON 键改为 `"calibration_source": "Qwen3-0.6B layer-0 K, measured by OG"` + `"data": "synthetic, calibrated"`。 |
| **支撑证据** | 全文件 grep 无 `from_pretrained` / `AutoModel`；L39 注释 `# Empirical distribution constants calibrated against Qwen3-0.6B layer-0 de-RoPEd K representations` 已自认是标定值。 |

### P1-4 needle 与 query 是与 haystack 流形无关的随机高斯向量，100% recall 偏乐观

| | |
|---|---|
| **对方原话 / 代码位置** | `benchmarks/eval_qwen3_real_kv.py:115-127`：<br>`needle_concept = rng.randn(dim)` / `token_variation = needle_concept + rng.randn(dim)*0.02` / `query_variation = needle_concept + rng.randn(dim)*0.02` |
| **问题所在** | needle 与 query 都由同一个**随机高斯向量**生成，与 haystack 的分布完全无关。结果是 needle 成为 haystack 中的**极端离群点**，检索难度远低于真实场景（真实 needle 是 in-distribution 的语义片段）。因此 `recall_with_ifr = 100%` 是上界，不是期望值。 |
| **我方立场** | 保留该基准（它至少证明了机制在非零离散度下有效），但结论必须写成"在合成离群 needle 上 100%"，并补一个 in-distribution 版本：从真实 Qwen3 KV 中取一个 token 作为 needle、用相邻 token 作为 query。 |
| **支撑证据** | 代码行如上；needle dispersion 1.18–1.22 远高于背景 0.74，证实 needle 是构造出来的离群点。 |

### P1-5 我方 README 的免责声明现已部分过期，需更新以保持公信力

| | |
|---|---|
| **位置** | `README.md:14`（研究方维护）：<br>"⚠️ **关于"实测"数字的重要提醒**：外部团队报告的 `100% 针尖召回 / PPL 漂移 0.0000 / G-CAS-1 PASS / U-E-F-C 全 PASS` 等结论，经我们逐条复核，存在**① 未跑真实模型（合成数据）、② 规模仅 8K–32K（低于问题域两个数量级）、③ 硬件 Gate 用断言冒充实测**三类问题，**目前不可作为结论采信**。" |
| **问题所在** | ① 已被对方部分改善：现在不是零散度合成，而是按我方实测统计量标定的非均匀合成（虽仍未跑真模型，但已不是"退化数据"）。② 规模已从 8K–32K 扩展到 128K。③ 部分仍成立（`linear_36_layers_total_us` 仍是哑张量×36）。 |
| **我方立场** | 这段声明若不更新，会让我方在"哪些还站得住"上失去精确性。建议改为分条标注：哪一条已被对方整改回应、哪一条仍然成立。 |
| **支撑证据** | 本文件 §1 第 3、4 项与 §3 P1-3/P1-4。 |

### P2-6 4 个 C 引擎测试在干净克隆上硬失败，应改为 skip

| | |
|---|---|
| **对方原话 / 代码位置** | `tests/test_c_engine_bindings.py:42-49`：`assert is_c_engine_available(), "libkvmem_engine.dylib could not be loaded!"` |
| **问题所在** | dylib **未入库**（`.gitignore`/未提交），干净克隆未 `make` 时这 4 个测试**硬失败**而非 skip。我方首次复跑即为 `4 failed, 149 passed`，`make` 后才 `152 passed`。 |
| **我方立场** | 应改为 `@pytest.mark.skipif(not is_c_engine_available(), reason="run make first")`，否则 CI/外部复核会误判为代码缺陷。另建议把 dylib 或构建步骤写进 README 的快速开始。 |
| **支撑证据** | 我方两次复跑：`make` 前 `4 failed, 149 passed`；`make` 后 `152 passed, 1 failed`（唯一失败为 E2E 网络抖动）。 |

### P2-7 复现性细节：R5 基准的延迟/构建字段跨机器差异较大

| | |
|---|---|
| **对方原话 / 代码位置** | `benchmarks/real_kv_haystack_results.json` 的 `*_ms_*` 字段 |
| **问题所在** | 我方同机复跑：recall 与 dispersion 字段**逐位一致**，但延迟类字段差 2–8×（如 128K `index_build_ms_with`：对方 5.32 vs 我方 37.46）。 recall 是实质结论，不受影响；但说明这些延迟值不可跨机器引用。 |
| **我方立场** | 建议 JSON 中标注 `machine` / `run_id`，并把延迟与 recall 分节，避免读者把运行间抖动当成结论差异。 |
| **支撑证据** | 我方 `/tmp/my_r5.json` 与对方 `real_kv_haystack_results.json` 的 diff：仅 `index_build_ms_*` 与 `*_retrieval_ms_*` 不同，`recall_*`、`empirical_dispersion_*`、`anti_collapse_*` 全部相同。 |

---

## §4 仍需对方立即澄清/整改的问题

| # | 问题 | 优先级 |
|---|---|---|
| **N1** | `_evict_and_compact` 为何在 prefill 阶段触发？请给出"开启压实前后端到端 logits 一致性"测试。当前实测：开启压实后与全注意力 cos=0.1336、top-10 重叠 0/10；关闭压实则逐位相同。 | **P0，立即** |
| **N2** | 请把 `metal_attention_hook.py:562` 的 K 伪 query 改为真实 query，并说明不可逆的换出决策用什么判据。 | **P0，立即** |
| **N3** | 请把 `eval_qwen3_real_kv.py` / `real_kv_haystack_results.json` 改名并改键，明确"合成标定数据"而非"real KV"。 | P1 |
| **N4** | 请补一个 in-distribution needle 版本（needle 取自真实 KV，而非随机高斯），报告该情形下的 recall。 | P1 |
| **N5** | 请修 `benchmark_flashnext_hybrid_e2e.py:89` 的 `np.tile`，或直接删除该基准——不要用"在旁边加一个新的"来替代"修旧的"。 | P1 |
| **N6** | 请修 `flashnext_hybrid.py:103` 的 16× 虚高基线（48 层×8 KV 头 → 12 层×2 KV 头），并公布改后的 headline。 | P1 |
| **N7** | 请以代码实算值 6.48–6.50 GB 统一响应文档 §0/Q15/§3 与 README L8 的三套矛盾数字。 | P1 |
| **N8** | 请修正响应文档 L29 的"1025 Token 实测返回 288"（实测 448/480/512），或用默认 `indexer_budget=2048` 重跑公布（实测 800/1025）。 | P2 |
| **N9** | 请把 4 个 C 引擎测试改为 `skipif`，并在 README 快速开始中写明需先 `make`。 | P2 |
| **N10** | 请补 @1M / @10M 检索实测（当前最大 128K，10M 是 2,441× 外推），以及 M2 其余三个主终点的证据。 | P2 |

---

## §5 附：我方本轮使用的复现脚本

- `docs/og_repro_real_kv_dispersion.py`（上轮已入库）——真实 Qwen3-0.6B layer-0 K 的离散度与 IFR 延迟标定。
- 本轮压实正确性对照实验（A/B/C 三臂，3000 token，CPU/fp32）与 C 引擎延迟实测，命令与输出已全部记录在本文 §3 各条"支撑证据"中，可直接复跑。
