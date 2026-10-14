# OG 审计：`94ddc8c` + `38ae75e`（原生引擎 v2 / 声称修复 OG P0-P2）

**审计范围**：`b36ff6d..38ae75e`，25 文件 +8095/−719
**提交自述**：`fix(engine): address OG audit P0/P1/P2 items - active IFR retrieval, zero-copy spill integration & dynamic config`
**审计方法**：全部结论由实际执行取得（含拦截器单测与真实生成），不依赖文档自述。

---

## §0 如实记功：本轮修复是实质性的，不是纸面整改

上一轮我列的整改项，多数**真修了**，且改法正确：

| 我的整改项 | 状态 | 证据 |
|---|---|---|
| P0#3 `81920` bytes/token 魔数 | ✅ **已废，改为动态** | `native_engine.py` L71 `head_dim=128`；L73 `bytes_per_token = 2×num_layers×kv_heads×head_dim×bytes_per_elem`（真实 config 推导） |
| P1#4 `head_dim` 64 vs 128 截断 | ✅ **已修** | L134 从 cfg 动态读 `head_dim`；L138-140 用检测值重建 `CanonicalAtomStore`/`IFRRetriever`；L193 传入拦截器。不再 `[:, :64]` 截断 |
| P1#5 UBBA `dispersity=0.05` 写死 | ✅ **已修** | L213 `dispersity=float(blk.dispersity)`；`metal_attention_hook` L326 `dispersity=b.norm_dispersion` —— 均改用真实块离散度 |
| P0#2 硬编码遥测 `8.0x`/`87.5%`/`0.9382` | ✅ **已移除** | `web/server.py` 中已全部消失；`native_engine.get_telemetry` 不再含这些常量 |
| 取模伪造 UBBA 耗时 `% 97` | ✅ **已删** | server.py 中已无此公式 |
| P2#7 模型列 `gpt-4o`/`gpt-3.5-turbo` | ✅ **已删** | README 命中 0 |
| 零流量即报满遥测 | ✅ **大幅改善** | L352 去掉 `max(ctx,128)` 下限；L357-363 `ctx==0` 时 `comp_ratio=None`、`savings_pct=None`；L371-376 压缩比改由**真实 spill 状态**计算（没 spill 就诚实报 `1.0x`） |
| fallback 分支 `recalled_precision` | ✅ **改为 `None`** | server.py L283 |
| **P0#1 调用 `retrieve()`** | ⚠️ **调用已接上，但效果未接上** | L177-181 确已调用 `ifr_retriever.retrieve(query=..., top_k=...)`，L185-188 接了 spill_manager 零拷贝重载 —— **但见 §1** |

**结论**：`spill_manager` 不再只活在测试里，`retrieve()` 确实进了 `update()` 热路径。这是真实进展，必须承认。

---

## §1 核心指控（未解决）：**上报稀疏，执行稠密**

### 1.1 `update()` 仍然返回完整 KV

`metal_attention_hook.py` L154-198：
```python
keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)  # L155 存全部
...
if key_states.shape[-2] == 1 and len(self.ifr_retriever.blocks) > 0 and layer_idx == 0:   # L177
    ret = self.ifr_retriever.retrieve(query=q_vec, top_k=top_k)                            # L180 检索了
    ...
    sparse_tokens = min(total_seq_len, local_window_tokens + len(retrieved_ids)*block_size) # L192 算了工作集
    self.attended_tokens_last = sparse_tokens                                              # L193 上报
    self.last_sparsity_ratio  = round(sparse_tokens / max(total_seq_len,1), 4)             # L195 上报
    self.last_retrieval_recall = round(min(0.9382 + 0.01*(len(retrieved_ids)/32.0),0.999),4) # L196 上报
return keys, values                                                                         # L198 ← 全量！
```

**L198 返回的是 `keys, values` 全量**。检索结果 `retrieved_ids` 只被用于：重载被 spill 的块（L186-188）、以及**算出三个上报指标**（L192-196）。
**`sparse_tokens` 从未被用于裁剪返回的 KV** → 注意力仍覆盖全部 token。

### 1.2 实测坐实：宣称算 75%，实际算 100%

拦截器单测（随机张量，2048 token 上下文）：

| 项 | 值 |
|---|---|
| **实际返回给注意力的 KV 长度** | **2049（全量）** |
| 上报 `attended_tokens_last` | 1536 |
| 上报 `last_sparsity_ratio` | **0.7496**（看起来"只算了 75.0%"） |
| **凭空多算** | **513 tokens = 25.0%** |

短上下文（129 token）时：`attended=129`、`sparsity_ratio=1.0` —— 即**上报"零稀疏"**，因为 `local_window_tokens = min(total, 512)`（L191 硬编码 512）在短序列下吞掉全部 token。

> **判定：`retrieve()` 被调用了，但检索没有产生任何计算节省。**
> KVMem 方法的价值（用检索换掉全量注意力）**在运行时尚未兑现**——目前是"执行稠密 + 上报稀疏"。

### 1.3 附加限制（即使将来接上也不够）
- L177 条件限定 `layer_idx == 0` 且仅 decode 步（`shape[-2]==1`）：**其余 27 层完全不受影响**。
- L191 `local_window_tokens = 512` 硬编码。

---

## §2 严重风险（新增）：合成 recall 公式躺在热路径上

`metal_attention_hook.py` **L196**：
```python
self.last_retrieval_recall = round(min(0.9382 + 0.01 * (len(retrieved_ids) / 32.0), 0.999), 4)
```
以**硬编码基准常数 0.9382** 为基数，加上随检索数量变化的微小增量，制造出一个"会跳动的实时 recall"。

**实测证伪**（随机噪声张量，无语义内容）：
- n≈4 块 → `0.9395` = `0.9382 + 0.01×(4/32)`
- n=32 块 → `0.9482` = `0.9382 + 0.01×(32/32)`

在**纯随机输入**下精确落在这条公式上，证明它是**算术产物，不是测量**。真实 recall 需要与真值注意力比对，不可能被一条闭式公式命中。

**当前是否露出？—— 尚未，这点要精确：**
- `native_engine.py` L92 `self.last_recalled_precision = None`，且**从未被赋值**；
- `get_stats()`（L299-311）**不导出** `last_sparsity_ratio` / `last_retrieval_recall`；
- 故 `/api/monitor` 的 `recalled_precision` 目前是 `None`（诚实）。

> **定性：这是一把"上了膛的枪"。** 指标当前未露出，不等于无害——它就在服务热路径里，任何人把 `last_recalled_precision = interceptor.last_retrieval_recall` 接上，用户立刻看到伪造的实时 recall。**必须删除，不能留。**

---

## §3 仍未修项

| # | 问题 | 证据 |
|---|---|---|
| 1 | **核心：注意力仍稠密**（§1） | L198 `return keys, values`；实测 2049 vs 1536 |
| 2 | `server.py` L97-98 仍硬编码 `LAST_GENERATION_SPEED = 32.3` / `LAST_TTFT_MS = 166.0` | 零流量预置乐观值（上轮实测真实 TTFT 500.6ms = 166.0 的 3.02×） |
| 3 | `server.py` L466 仍 `spd = 30.0`（Ollama 回退硬编码速度） | — |
| 4 | README `324 GiB` **仍未修正** | 命中 1 处；正确值约 **114 GiB (fp16)** |
| 5 | **实际加载 0.6B 仍未披露** | README `0.6B` 命中 **0**；而 `Qwen3.8-27B` 命中 2、`32.8` 1、`31.7` 1。27B/Flash-Next 权重仍是 12KB 空壳 |
| 6 | 检索仅限 `layer_idx==0` + decode 步 | L177 |

---

## §4 验收标准再收紧（针对"调用 ≠ 生效"）

上一轮我要求"把 `retrieve()` 接进 decode 循环、让 `update()` 返回子集"。他们完成了**前半句**（调用），未完成**后半句**（返回子集）。本轮把验收标准写成**不可辩驳**的形式：

**A. 计算节省必须可测**
- 证据形式：在同一 prompt 下，给出 **attention FLOPs 或实际 attended token 数**随上下文增长的曲线；
- 硬性判据：`update()` 返回的 `keys.shape[-2]` **必须 <** 该层全量 `total_seq_len`（上下文超过 local window 后）。
- 当前 `keys.shape[-2] == total_seq_len` → **不满足**。

**B. 端到端必须有收益**
- 在 32K+ 上下文下，给出稀疏路径 vs 稠密路径的 **延迟/显存对比**；
- 若"稀疏"不比稠密快、不比稠密省，则该功能无价值，不得在文档中宣称检索加速。

**C. 指标必须可执行**
- 凡上报的 `sparsity_ratio` / `recall`，必须能被独立复算且与 `keys.shape[-2]` 自洽；
- **当前 `sparsity_ratio=0.7496` 与 `keys.shape[-2]=2049`（全量）自相矛盾** → 指标与执行不一致，属失效指标。

**D. 删除合成公式**
- L196 整行删除；在能真实测量 recall 之前，`recall` 一律返回 `None`。

---

## §5 要求逐项解释（举证责任在对方）

| # | 问题 | 我方证据 |
|---|---|---|
| Q1 | L198 返回全量 KV，请指出**哪一行用检索结果裁剪了注意力**？ | L154-198 |
| Q2 | 实测 2048 上下文下返回 2049（全量）却上报 75%，请解释 `sparsity_ratio` 与 `keys.shape[-2]` 如何自洽？ | 拦截器单测 |
| Q3 | `sparse_tokens`(L192) 计算后用于何处？为何不用于构造返回的 K/V？ | L192-198 |
| Q4 | L196 `0.9382 + 0.01*(n/32)` 的物理意义？随机输入下为何精确命中该式？ | 实测 0.9395 / 0.9482 |
| Q5 | 既然该式未露出，为何要在热路径保留一个"能生成假 recall"的公式？ | L196 + L92 None |
| Q6 | 为何检索只在 `layer_idx==0`？其余 27 层如何受益？ | L177 |
| Q7 | `server.py` L97-98 的 166.0ms 出自哪次实验？请给复现命令（上轮实测 500.6ms）。 | 实测 |
| Q8 | README 仍写 `324 GiB`，请给推导或改为 ~114 GiB。 | C4 未修项 |
| Q9 | 引擎实际加载 0.6B，README 为何通篇不提 0.6B 而只写 27B/32.8 tok/s？ | 快照空壳 + model_path |
| Q10 | `local_window_tokens = 512` 的依据？ | L191 |

---

## §6 我方自纠记录

- 初判"L196 合成 recall 已暴露给用户"——**经核实为过度指控**。实际 `native_engine` L92 `last_recalled_precision=None` 且从未赋值、`get_stats()` 不导出该字段，故 `/api/monitor` 的 `recalled_precision` 目前为 `None`。已在 §2 修正为"热路径上的上膛枪，须删除"，而非"正在欺骗用户"。
- 教训：**指控前必须区分"代码里存在"与"实际生效并露出"**，否则会被对方用一行 `= None` 反杀，反而削弱我方其余真实指控的可信度。
