# OG 战略评估：项目轨迹、瓶颈、目标偏离与算法真实性（2026-10-14）

**回答三个问题**：① 当前 bottleneck 是什么；② 是否偏离原目标；③ 是否还在用自己的算法。
**判据来源**：项目原研究方案 `kvmem-strata-融合研究方案.md`（2026-10-04）自定的主终点与 Gate；对方测试实跑结果；我方对 `38ae75e` 代码的实测。

---

## §0 项目整体发生了什么变化：从「可证伪的研究计划」滑向「不可证伪的产品工程」

原方案（2026-10-04，10-agent 圆桌收敛出**方案四 CASA**）把主终点定得可证伪：

> L280：**「在保持 Full-Context 真值 1pp 以内的前提下，把 10M-token workspace 的检索延迟从 1.311s 压到 ≤350ms，索引常驻从 9.5 GiB 压到 ≤4 GiB」**
> L308 **M2**：`@10M` 延迟 1.311s→≤350ms（3.7×），索引常驻 9.5→≤4 GiB，decode tok/s 不下降
> L240 **R10 裁决**：「在 1,900 GPU-h 预算内，**任务质量终点一律不可判决**。主终点必须落在**字节、延迟、保真代理**这三个可判决量上」

而 `94ddc8c`+`38ae75e` 实际交付的是：

| 交付物 | 性质 |
|---|---|
| `web/index.html` +2295 行 SPA、`scripts/kvmem_menubar.swift`(232) 菜单栏托盘 | 产品 UI |
| `agent_protocol.py`(598) + `test_agent_protocol.py`(424) | Agent/工具调用协议 |
| `start/stop/status_api.sh`、`.command` 双击启动 | 产品运维 |
| `spill_manager.py`(829) NVMe 换入换出 | 工程（真实） |
| `native_engine.py` Metal SDPA 原生推理 | 工程（真实） |

**变化定性**：主终点（字节/延迟/保真 @10M）**一个都没被真实测量**；取而代之交付了一套完整的**本地 ChatGPT 式客户端**。而评价一个 chat 客户端的标准恰恰是 **UX 与任务质量**——正是 R10 明令「在预算内不可判决、不要走」的方向。

---

## §1 他们的测试结果：130 passed 全绿，但对主终点**零信息量**

实跑：`.venv/bin/python -m pytest tests/ -q` → **`130 passed in 85.91s`**，0 失败。

但绿灯不代表覆盖。核查测试内容：

### 1.1 主终点的"测试"是同义反复

`tests/test_plan01_ifr.py::test_uefc_evaluation_gate_and_wilson_ci`（L275-322）是唯一涉及 E-gate(350ms)/C-gate(4GiB)/F-gate 的测试。它的做法：

```python
gate_result_pass = UEFCEvaluationGate.evaluate_gate(
    top1_matches=98, retrieval_latency_ms=310.0, index_footprint_gib=3.8, ...)   # L285-294
assert gate_result_pass["all_passed"] is True
gate_result_fail = UEFCEvaluationGate.evaluate_gate(
    top1_matches=92, retrieval_latency_ms=450.0, index_footprint_gib=5.2, ...)   # L308-317
assert gate_result_fail["all_passed"] is False
```

- **没有跑任何检索**，`310.0` / `450.0` 是写死的字面量；
- **没有建任何索引**，`3.8` / `5.2` GiB 是写死的字面量；
- **没有与 Full-Context 真值比对**，`top1_matches=98` 是写死的字面量。

它验证的是「门禁的 if 分支在喂它该通过的数时返回 True、喂该失败的数时返回 False」——**同义反复**。该测试名为 "Verify U-E-F-C statistical contract"，外观上像是主终点的证据，实则与主终点无关。

### 1.2 定义性特征完全无覆盖

全 `tests/` 目录检索 `prune` / `attended` / `keys.shape[-2]` / `bounded` / `10M` / `350`：
- **没有任何测试断言 `keys.shape[-2] < total_seq_len`**（即有界执行视图）；
- **没有任何 10M 规模测试**；
- 唯一命中 `350` 的就是上述同义反复测试。

> **结论：核心机制（有界执行视图）缺失，而 130 个测试全过。这套测试套件对「KVMem 是否成立」不提供任何证据。**

---

## §2 问题一：Bottleneck 是什么

不是一个，是四层，且**真正的墙在中层**。

### 层 1（表层）：检索算了不用，注意力仍稠密
`metal_attention_hook.update()` L198 `return keys, values` 全量；`retrieved_ids` 仅用于重载 spill 块与计算上报指标。
实测 2048 上下文：返回 **2049（全量）**，却上报 `sparsity_ratio=0.7496` —— **宣称算 75%，实际算 100%**。

### 层 2（真正的墙）：HF 抽象层无法承载有界执行视图
必须公允指出：**在 `update()` 里直接截断返回 KV 会破坏正确性**（位置编码/RoPE 失配、缓存被永久截断）。所以"返回全量"是**安全但零收益**的选择。

要做到真正有界视图，必须自研 **gather kernel**：把检索到的块搬到紧凑位置槽位，并相应处理位置——这正是原方案定义的难点。**他们卡在"简单地做没用，正确地做很难"。**

### 层 3（根层）：绕开了整个融合研究的「唯一硬摩擦」
原方案 L76-80 定义：

> Strata 假设「取回的 KV 页回到原 logical position」；**KVMem 的全部价值恰恰来自「把 KV 块搬到它从来没待过的 compact 位置」**。
> 三条路线：**R4 消灭摩擦**（K 不动，重映射搬到 Q 侧 → 免 re-RoPE）／**R2 吸收摩擦**（re-RoPE 融进搬运 kernel）／**R6 容忍摩擦**（保真阶梯 + 受控重算）。

实测核查：
- `apply_rope` 在 `metal_attention_hook.py` **仅 L34 import，零调用**；`native_engine.py` 亦零调用；
- 块始终停在原始位置 → **不需要重映射，也确实没做重映射**。

**三条路线一条都没走。** 而 `core.py` L206-218 其实**存在 R4 的 Q 侧重映射实现**（`q_remapped = apply_rope(q_raw, pos=query_logical_pos - key_pos)`，注释明写 "Instead of re-rotating K to new compact slots"）——**代码有，服务路径没接**。

### 层 4（负载层）：无法验证价值主张的模型
`Qwen3.8-27B` / `Qwen3.8-Flash-Next` 快照各 **12KB / 0 safetensors（空壳）**，实际加载 **Qwen3-0.6B**（1.4GB）。
而原方案的价值主张是 **1M~10M workspace on 消费级硬件**——在 0.6B 模型、百 token 级上下文上**根本无从验证**。

> **一句话 bottleneck**：他们缺的不是索引，而是**「把检索结果变成实际计算节省」的那一步**——紧凑位置 gather + re-RoPE 的自研 kernel。这一步没做，前面所有索引工作都不产生收益。

---

## §3 问题二：是否偏离原目标 —— 是，且是结构性偏离

| 原目标主终点 | 现状 |
|---|---|
| **字节**：8× 压缩 / 索引 ≤4 GiB | 硬编码 8.0x 已删、改按真实 spill 计算 → **没 spill 时诚实报 1.0×**，即**字节收益当前实测为 0** |
| **延迟**：@10M 检索 ≤350ms | **从未真实测量**（仅同义反复测试喂 310.0/450.0） |
| **保真**：与 Full-Context 真值 1pp 内 | **从未真实测量**（喂 `top1_matches=98`） |
| **明令不走**：任务质量终点 | **正走进去** —— 交付的是 chat 客户端，价值由 UX/任务质量判定 |

**最讽刺的一点**：他们把硬编码的 8.0x 删掉、改按真实状态计算后，压缩比变成 **1.0×**——**诚实化反而暴露了字节收益为零**。这说明此前所有"8× 压缩"叙事从未经由真实路径兑现。

---

## §4 问题三：还在用自己的算法吗 —— 分三层，必须精确

| 层 | 判定 | 证据 |
|---|---|---|
| **数据结构** | ✅ **是自己的，且真在跑** | 实测真实请求后 `active_casa_blocks=1`、`active_doublets_split=1`；IFR/CASA/UBBA/LADDER 均实例化并被执行 |
| **决定"注意什么"的算法** | ❌ **不是自己的** | 检索结果计算后被丢弃，`update()` 返回全量 → 实际执行的是 **HF transformers + PyTorch SDPA 的稠密注意力**（`attn_implementation="sdpa"`）。**"全注意"不是 KVMem 的算法** |
| **最有价值的部分（紧凑重映射/re-RoPE）** | ⚠️ **代码有，路径没接** | `core.py` L206-218 存在 R4 Q 侧重映射；`apply_rope` 在服务路径零调用 |

> **一句话：他们用自己的索引，跑别人的注意力。**
> 严格说他们**没有改用 Ollama/llama.cpp 的算法**（原生路径确实执行 `kvmem_fusion` 代码），但 KVMem 的**灵魂——有界执行视图（bounded execution view）——从未被使用**。
> 按原方案 L55 的定义「**不扩展 jointly-attended tokens，每步仍是 bounded execution view**」，当前系统**不满足 KVMem 的定义性特征**。

---

## §5 给对方的方向建议（战略层）

上一轮我把方向定为「从数字合规转向方法真实性」。本轮要**再收紧一层**：

> **不要继续加产品功能。先把"检索 → 计算节省"这一条链打通并测出来，否则所有功能都是在稠密注意力上做的 UI。**

**必须做的三件事（按顺序）**

1. **补上缺失的一环（唯一的核心工作）**
   实现紧凑位置 gather：检索到的块 → 紧凑槽位 → 相应位置处理（R4 Q 侧重映射或 R2 kernel 融合二选一）。
   **验收**：`update()` 返回的 `keys.shape[-2] < total_seq_len`，且**生成质量不退化**（用困惑度或下游任务比对全量基线）。

2. **把同义反复测试换成真测量**
   删除/重写 `test_uefc_evaluation_gate_and_wilson_ci` 中的字面量喂数模式。E/C/F 三门禁的输入必须来自**真实跑出来的**检索延迟、索引体积、与 Full-Context 真值的比对。
   **验收**：给出 256K/512K/1M/4M/10M 五档 × 30 重复的延迟分布（原方案 G-CAS-1 本就如此要求），带 95% CI。

3. **要么补齐权重，要么改口径**
   27B/Flash-Next 是空壳。要么下载真权重在目标规模验证；要么把 README 的 27B/32.8/31.7 全部改为实际加载的 **0.6B** 并明确标注"本结论仅在 0.6B 上取得"。

**本轮新增的举证问题（Q13–Q16）**

| # | 问题 |
|---|---|
| Q13 | 请给出**任意一次**真实检索延迟的测量结果（非字面量），在哪个 workspace 规模、什么硬件上？ |
| Q14 | `apply_rope`/紧凑重映射为何在服务路径零调用？三条路线（R4/R2/R6）你们选了哪条？`core.py` L206-218 的 R4 实现为何不接？ |
| Q15 | 压缩比诚实化后变成 1.0×，那么原方案 M2 的「字节收益」当前是否为 0？请正面回答。 |
| Q16 | 主终点测试用字面量喂门禁，是否承认该测试对「系统是否达标」不提供证据？ |
