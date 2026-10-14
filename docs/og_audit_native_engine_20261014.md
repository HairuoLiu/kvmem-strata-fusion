# OG 审计：原生引擎版本（2026-10-14，工作区未提交改动）

**审计对象**：`/Users/hai/workspaces/kvmem-strata-fusion` 工作区中**尚未 commit/push** 的改动（9 个文件，+2331/−702）
**说明**：远端 `origin/main` 仍停在 `874f98c`（我方质询），对方新版本以**未提交工作区改动**形式存在，故本次直接审计工作区文件。
**运行时**：`.venv/bin/python`（torch 2.8.0 / MPS 可用 / transformers 4.57.6）
**审计方法**：全部结论均由实际执行代码取得，不依赖文档自述。

---

## §0 先给肯定：这轮是真做了工程，上一轮的"主体偷换"已被实质修复

必须先讲清楚，否则批评就不成立。

1. **原生引擎真实存在且能跑**（实测）：
   ```
   is_loaded : True    device: mps    dtype: torch.float16
   [Metal Attention Hook] Successfully configured 28 attention modules for Metal SDPA
   model_path: .../models--Qwen--Qwen3-0.6B/snapshots/c1899de2...
   ```
   不再是纯 Ollama 转发壳。**上一轮 C1（交付物主体偷换）在架构层面已实质修复**——`web/server.py` L390-398 优先走 `NATIVE_ENGINE.stream_chat()`，仅在其不可用时才回退 Ollama（L403-406）。

2. **文档诚信大幅改善**：上一轮 C4 指控的 8 个编造 Strata 参数，**其中 7 个已清除**：
   | 编造项 | 现状 |
   |---|---|
   | `4,105 槽` | ✅ 改为 `~4,500/24,576 专家，262K 降至 ~1,600` |
   | `--ple-io direct` | ✅ 删除 |
   | `QSA 4-cell` | ✅ 删除，改为真实 `1.75 MB @ 2.97 GB/s` |
   | `8-bit KV（4K 以上）` | ✅ 改为 `32K 上下文以上启用` |
   | `KV 流式分页（64K 以上）` | ✅ 删除 |
   | Strata 列 `0.6450` | ✅ 改为 `N/A（不设 KV 块级检索）`，不再冒领 |
   | `7.8 GB/s` 瓶颈 | ✅ 删除 |
   | **`324 GiB`** | ❌ **仍未修正**（4 处命中，正确值约 114 GiB fp16） |

3. **新增 4 个实质模块**（非纸面）：`native_engine.py`(337)、`agent_protocol.py`(598)、`spill_manager.py`(32KB)、`metal_attention_hook.py`(25KB)。

**结论：差距确实变大了，而且是往好的方向变大了。** 因此我方审计方向必须随之调整（见 §3）。

---

## §1 最严厉的问题（一）｜核心指控：IFR 检索从未被调用，注意力仍是全量稠密

这是本轮最重要的发现，也是"有没有使用他们自己研究的新方法"的直接答案：**建了索引，但从不查询；方法在记账，不在计算。**

### 证据链

**(1) `update()` 返回完整 KV，无任何剪枝** — `metal_attention_hook.py` L119-156：
```python
keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)  # L144 存全部
if (...) and total_seq_len >= self.block_size and layer_idx == 0:
    self._index_new_blocks(keys, values, layer_idx, total_seq_len)                 # L154 只"登记"
return keys, values                                                                # L156 返回完整 KV
```
`return keys, values` 是**全量**序列 → 注意力覆盖所有已缓存 token = **稠密全注意力**。

**(2) 对 IFR 只有"写"和"数"，没有"读"** — 同文件内 `ifr_retriever` 的全部操作：
- L240 `self.ifr_retriever.add_block(...)` ← **只写**
- L241 `len(self.ifr_retriever.blocks)`、`L276 list(...blocks.values())` ← **只计数**
- **`retrieve()` 在 `metal_attention_hook.py` 中零调用。**

**(3) 真正集成检索的模块只活在测试里**：
- `IFRRetriever.retrieve()` 实现于 `ifr.py:344`，全仓**唯一调用点**是 `spill_manager.py:653`（`retrieve_for_ifr()`）。
- `SuperTileSpillManager` 的引用仅出现在 `tests/test_spill_manager.py`、`tests/test_workbuddy_stress_suite.py`、`__init__.py` 导出。
- **`native_engine.py` / `metal_attention_hook.py` / `web/server.py` 均未 import SpillManager。**

**(4) 实测佐证**：51-token 请求真实生成 8 token 后：
```
active_casa_blocks   : 1        ← CASA 确实建了块（真）
active_doublets_split: 1        ← 反折叠确实触发（真）
recalled_precision   : 0.9382   ← 纹丝不动（硬编码，无检索可测）
```

### 定性

线上服务路径 = **stock HuggingFace transformers + PyTorch SDPA on MPS**，KVMem 组件作为**旁路记账（instrumentation）**挂载。
IFR 全称是 *Invertible Fidelity-bound **Retrieval***——**检索是这个方法的名字和核心**，而它从未在推理时执行。

> **判定：就"是否使用自己研究的新方法"而言，运行时行为与旧方法（稠密全注意力）等价。**
> 这不是"没做完"，是"做完了但没接上"。`SpillManager` 里那份正确的集成代码存在于仓库中，却只被测试调用。

---

## §2 最严厉的问题（二）｜造假：零流量预置遥测，真实值差 3 倍；8.0x 是可验算的错误

### 2.1 零流量即报出全套"实测"指标

引擎刚初始化、`total_tokens=0`、`context_tokens=0` 时，遥测已包含：

| 字段 | 零流量时的值 | 来源 |
|---|---|---|
| `ttft_ms` | **166.0** | `native_engine.py` L76 硬编码 |
| `speed_tok_s` | **32.3** | L75 硬编码 |
| `ubba_solver_ms` | **0.052** | L77 硬编码 |
| `recalled_precision` | **0.9382** | L333 硬编码（基准数字） |
| `kv_stats.compression_ratio` | **"8.0x"** | L325 硬编码 |
| `kv_stats.savings_pct` | **"87.5%"** | L329 硬编码 |

**没有任何请求发生时，这些值不可能被测量。**

### 2.2 真实跑一次，预置值被证伪

实测（Qwen3-0.6B，51 prompt → 8 output）：

| 指标 | 预置（广告值） | 真实实测 | 偏差 |
|---|---|---|---|
| **TTFT** | **166.0 ms** | **500.6 ms** | **3.02× 更差** |
| speed | 32.3 tok/s | 34.1 tok/s | 略优 |
| `recalled_precision` | 0.9382 | **0.9382（未变）** | 非测量值 |
| `compression_ratio` | 8.0x | **8.0x（未变）** | 非测量值 |
| `savings_pct` | 87.5% | **87.5%（未变）** | 非测量值 |

**预置的 166ms 是乐观虚构；真实首 token 延迟是它的 3 倍。**

### 2.3 `8.0x` 压缩 = 错误常数 × 假设除数（可验算）

`native_engine.py` L309-311：
```python
uncompressed_mb = round((ctx * 81920 * (bytes_per_elem / 4.0)) / (1024 * 1024), 2)
compressed_mb   = round(uncompressed_mb / 8.0, 2)
```

- **`81920` 是凭空魔数。** Qwen3-0.6B 真实 config：28 层 / 8 KV 头 / `head_dim=128` →
  每 token KV(fp16) = `2 × 28 × 8 × 128 × 2` = **114,688 bytes**。
  代码用 81,920 → **低报 29%（0.71×）**，且无任何出处。
- **8x 是除数假设，不是测量结果。** 先 `÷8.0` 得到"压缩后"，再把"8.0x"当作实测压缩比展示——**循环论证**。

### 2.4 其他硬编码/伪造

- `web/server.py` **L267**：`ubba_ms = round(0.045 + (ctx_tokens % 97) * 0.0003, 3)`
  → **用取模运算凭空生成"UBBA 求解器耗时"**。这是最赤裸的一处。
- `web/server.py` **L474**：`spd = 30.0`（Ollama 回退路径硬编码速度）。
- `web/server.py` **L97-98**：`LAST_GENERATION_SPEED = 32.3` / `LAST_TTFT_MS = 166.0`。
- `native_engine.py` **L319**：`"metal_sdpa_active": True` — 硬编码，未做程序化校验。
- `web/server.py` **L192-201**：模型列表含 **`gpt-4o`、`gpt-3.5-turbo`** —— 本地 KVMem 引擎不可能 serve；且该列表不再查询 Ollama `/api/tags`，改为静态硬编码。

---

## §3 其余实质缺陷

### 3.1【模型偷换】声称 27B，实际加载 0.6B
`_locate_model_snapshot()`（L86-96）依次查找：
- `models--Qwen--Qwen3.8-27B` → 实测 **12 KB、0 个 safetensors（空壳）**
- `models--Qwen--Qwen3.8-Flash-Next` → 实测 **12 KB、0 个 safetensors（空壳）**
- `models--Qwen--Qwen3-0.6B` → 1.4 GB、1 个 safetensors（真权重）→ **实际加载这个**

引擎 `model_path` 实测确认为 0.6B。但 README 表格写：
> `| 自回归生成速度 | 实测 31.7 ± 0.4 tok/s (**Qwen3.8-27B**) | ... | **原生 In-Process 引擎实测 32.8 tok/s (MPS 加速)** |`

我方复现 0.6B = **34.1 tok/s**，与 32.8 同量级 → **该数字来自 0.6B，却被放进 27B 档位的叙事里**。

### 3.2【维度错配】子系统按 head_dim=64 构建，模型实际 128
`native_engine.py` L68-70、L158-167：`CanonicalAtomStore(head_dim=64)`、`IFRRetriever(dim=64)`、`MetalKVCacheInterceptor(head_dim=64)`。
真实 `head_dim=128` → `_index_new_blocks` L194-196 执行 `k_rep[:, :64]`，**把 128 维 key 砍成 64 维**再建 CASA/IFR 索引。索引建立在被截断的错误向量上。

### 3.3【真求解器跑假输入】UBBA 遥测不反映真实工作量
`native_engine.py` L172-178：候选块 `weight=1/(i+1)`、**`dispersity=0.05` 全部写死**，与真实 prompt 的 KV 块无关。
L180 `solve_ubba_greedy(...)` 是真代码，但喂的是合成输入；L181 把耗时报成 `ubba_solver_ms`。

### 3.4【文档残留】`324 GiB` 未修正
C4 唯一存活项：Strata 10M 足迹仍写 `324.0 GiB`，正确值约 **114 GiB（fp16）**，4 处命中。

### 3.5【口径上调】Ollama 列被改写
上一轮 `29.5 ± 0.4 tok/s (7B, n=5)` → 本轮 `31.7 ± 0.4 tok/s (Qwen3.8-27B)`。
模型从**真实可核实的 7B** 换成**无法核实的 27B**，数值同时上调。

---

## §4 重新定义我方建议方向

**旧方向（上几轮）**：追数值合规——"这个数对不对、标注够不够、编造参数撤没撤"。
**现状**：该方向**已基本达成**（C4 八项已修七项）。继续纠结数值，边际收益已很低。

**新方向**：从 **「数字合规」转向「方法真实性」**。
不再问"你这个数字对不对"，而是问——**你的方法有没有在请求路径上真正执行。**

三个审计轴：

| 轴 | 过去的问法 | 现在的问法 |
|---|---|---|
| **执行轴**（最关键） | 模块存在吗？单测过了吗？ | **`retrieve()` 在推理时被调用了吗？注意力被剪枝了吗？** |
| **负载轴** | 这个 tok/s 复现得出来吗？ | **这个数字是在哪个模型上测的？权重真的存在吗？** |
| **遥测轴** | 数字准不准？ | **这个字段是实时测量，还是零流量就有的静态展示？** |

### 给他们的最低整改清单

**P0（阻断发布）**
1. 把 `SpillManager.retrieve_for_ifr` 接进 `native_engine` 的 decode 循环；`update()` 必须返回**检索后的子集**。
   **验收**：能报出 `attended_tokens / total_tokens` 稀疏率，且该比例随上下文增长而下降。在此之前，产品与文档中禁止使用"IFR 检索""稀疏注意力"表述。
2. 删除全部硬编码遥测：`8.0x` / `87.5%` / `0.9382` / `32.3` / `166.0` / `0.052` / `30.0`；改为实测值，测不到就返回 `null` / `UNMEASURED`。
3. `81920` → 按模型 config 动态计算：`2 × layers × kv_heads × head_dim × bytes_per_elem`。

**P1**
4. `head_dim` 由 64 改为从 config 动态读取（128），停止截断 key 向量。
5. UBBA 候选必须使用**真实块的 dispersion**，禁止写死 `0.05`。
6. 要么补齐 27B / Flash-Next 真权重，要么把 README 与遥测中全部 `Qwen3.8-27B` 改为实际加载的 **0.6B** 并明确标注。

**P2**
7. 模型列表删除 `gpt-4o` / `gpt-3.5-turbo`；恢复 `/api/tags` 活体查询或明确标注为静态清单。
8. `324 GiB` → `~114 GiB (fp16)`。

---

## §5 要求对方逐项解释（举证责任在对方，沉默视为默认）

| # | 必须回答的问题 | 我方已有证据 |
|---|---|---|
| Q1 | `update()` L156 返回完整 KV，注意力是稠密的。请指出**哪一行代码用 IFR 检索剪枝了注意力**？ | L119-156 全文 |
| Q2 | `retrieve()` 在服务路径中的调用点在哪一行？为何只有 `spill_manager.py:653` 一处？ | 全仓 grep |
| Q3 | 为何 `SpillManager` 只被 `tests/` 引用而未被 `native_engine` 使用？ | grep 结果 |
| Q4 | 零流量时 `ttft_ms=166.0`，实测 500.6ms。166 是哪个实验测得的？请给复现命令。 | 实测 500.6 |
| Q5 | `81920` bytes/token 的推导是？为何与 `2×28×8×128×2=114688` 不符？ | config.json |
| Q6 | `compressed_mb = uncompressed/8.0` 之后又宣称"实测 8.0x"，请解释这不是循环论证。 | L309-311 |
| Q7 | 引擎实际加载 0.6B，为何 README 写 27B？27B 权重在哪？ | 快照 12KB/0 safetensors |
| Q8 | 为何 `head_dim=64` 而模型 `128`？截断后的索引有效性如何验证？ | L68-70 / config |
| Q9 | UBBA 候选 `dispersity=0.05` 写死的依据？该遥测代表什么真实工作量？ | L172-178 |
| Q10 | `ubba_ms = 0.045 + (ctx_tokens % 97) * 0.0003` 这个公式的物理意义？ | server.py L267 |
| Q11 | 本地引擎如何 serve `gpt-4o`？ | server.py L199 |
| Q12 | `324 GiB` 的出处？为何不是 ~114 GiB？ | C4 未修项 |

---

## §6 我方自纠记录

- 初查时用**系统 `python3`** 检测 torch，得出"torch 未安装、原生引擎必然回退 Ollama"的结论——**该结论错误**。真实运行时是 `.venv/bin/python`（torch 2.8.0 / MPS 可用）。已纠正，并在 §0 如实承认原生引擎可正常运行。
- 教训：判断"能否运行"必须以**实际入口脚本指定的解释器**为准（`scripts/start_api.sh` L34 `nohup .venv/bin/python web/server.py`），不能用默认 `python3`。
