# KVMem × Strata 融合研究：四方案执行计划

> 把一个"更快的长上下文 KV 系统"问题，重新拆成「测量方法 + 共同地基」两份真正可推进的工作。

## 诚实声明（先读这段）

本仓库的**研究主干**（`docs/master-research-report.md` 与四个 `plan-*/research-report.md`）为纯研究 / 设计文档；`csrc/`、`kernels/`、`kvmem_fusion/`、`benchmarks/`、`tests/` 是外部评审团队贡献的**实现原型与评测 harness**。

> ⚠️ **关于"实测"数字的重要提醒**：外部团队报告的 `100% 针尖召回 / PPL 漂移 0.0000 / G-CAS-1 PASS / U-E-F-C 全 PASS` 等结论，经我们逐条复核，存在**① 未跑真实模型（合成数据）、② 规模仅 8K–32K（低于问题域两个数量级）、③ 硬件 Gate 用断言冒充实测**三类问题，**目前不可作为结论采信**。
> **第一轮质疑（含证据行号）：[`docs/external_review_critique.md`](docs/external_review_critique.md)**
>
> ✅ 外部团队已在 `273ae4b` 中**修好了报告诚信问题**（状态改为合成自洽验证、硬件 Gate 标 UNMEASURED、采纳子系统降级）。
> ✅ 又在 `f093dfd` 中**完成真实模型 P0**（真跑 Qwen2.5-0.5B-Instruct）、补上 Random 基线与误差分解，并实现 `stopping.py`（Wilson / McNemar N=471 / Deff / Lan-DeMets OBF，公式已逐条验算正确）。
> ⚠️ 第三轮复核（**[`docs/external_review_critique_v3.md`](docs/external_review_critique_v3.md)**）指出两点：M0 判据被事后替换（一条从未测量、一条换用相对误差口径），且真实工作点 σ≈8.03 未被任何阈值分析覆盖。
>
> 🔬 **我们已自己把缺失的实验做掉了**（不是又一份清单）：v1 **[`docs/real_kv_audit_og_report.md`](docs/real_kv_audit_og_report.md)**，v2 **[`docs/real_kv_audit_og_v2_report.md`](docs/real_kv_audit_og_v2_report.md)**；代码 [`benchmarks/real_kv_audit_og.py`](benchmarks/real_kv_audit_og.py) / [`real_kv_audit_og_v2.py`](benchmarks/real_kv_audit_og_v2.py) / [`centroid_law.py`](benchmarks/centroid_law.py)（均 CPU 可复现，约 1 分钟）。
>
> **v1 结果**：① 独立复现 de-RoPE 自校验（abs 1.3354e-05，与外部团队一致，**确认原 atol 判据应为 FAIL**）；② 真实异质语料下高频保留率 **0.4979**，解释了 0.1612 与 0.4430 的差异来源；③ σ=8.03 是 **massive-activation 幅度伪影**（L2 归一化后仅 0.7965）。
>
> ⛔ **v1 的第 ④ 条已由我们自己撤回**：v1 报的「recall@64 B/A=0.98，de-RoPE 无增益」是**本组的实验假象**——Arm B 被重 RoPE 到块中点而 query 留在去 RoPE 空间，存在位置失配。v2 位置匹配后实测 **B/A = 1.40×**（0.6367 → 0.8913，随机 0.5005）。**LADDER L2′ 的技术前提成立。** 我们同时把新结论写进了 [`plan-02-ladder/research-report.md`](plan-02-ladder/research-report.md) §2.1。
>
> **v2 新增结果**：
> - **方法论**：召回率必须声明「相关性空间」。同一批数据，去 RoPE 空间下 B=0.891；原生空间下 A=0.965。裸 recall 数不可解释。
> - **新发现 · 迁移惩罚**：块被搬离原位越远，可检索性单调下降。Qwen2.5 在 Δ=2048 掉到 **0.563**（随机 0.497）；Qwen3 衰减较缓（0.850→0.762）。**压缩不是免费的，而压缩正是 KVMem 的核心机制。**
> - **建设性 · 质心定律**：决定保真度的唯一变量是 **tokens/质心**（组内极差 ≤0.06，组间跨度 0.55→1.00）。KVMem 默认 32 token/质心在 3% 选择率下只有 **0.598**（可实现余量的 58.5%）。同等索引预算下**加质心优于缩小块**（B32/m4 = 0.711 > B8/m1 = 0.654）。
> - **建设性 · 逐层位置先验**：纯内容索引在浅层与真实注意力**零相关**（L0 ρ=−0.091）。需 `α(layer)·content + (1−α(layer))·recency`，α 由浅层 0.1 递增到深层 0.9，浅层增益高达 +0.97。
> - **自检**：手工重建的 Q/K/RoPE/GQA 注意力与 transformers eager 实现**逐位一致**（max\|diff\| = 0.000e+00，双模型）。
> **当前成熟度：T1-PARTIAL**（双架构；限于 4096 token、CPU/fp32、0.5B–0.6B；**无端到端任务质量端点**）。

所有"预期""目标"均为未实测的假设，文中凡标注「未验证」之处均未经实验确认。融合的两大硬前提——KVMem 闭源引擎 **QW3 的可达性**、以及 **≤256K 区间不需要 re-RoPE**——是后续一切实验的 G0 杀点。请带着这个前提阅读，不要把它当作已验证结论。

## 背景：两个系统与一个硬摩擦

- **KVMem**（arXiv 2609.04852, 2026-09）：把长上下文组织为 workspace / GPU 页池 / 每步重组的有界视图；32-token 逻辑块；Mean-K 注意力空间索引；delta re-RoPE。闭源引擎 QW3（RTX PRO 6000 实测）+ 开源兄弟实现 `kvmem/kvmem-llama.cpp`（Apache-2.0，llama.cpp 移植，RTX 5060 Ti 16GB 实测）。
- **Strata**（arXiv 2508.18572, OSDI '26）：面向 SGLang 的分层上下文缓存；GPU-assisted I/O（128B 粒度，PCIe5 仅 22% 利用率）；layout 解耦；HiRadixTree；cache-aware 调度（delay-hit deferral / balanced batch / complementary overlap）。
- **唯一硬摩擦**：Strata 假设取回的 KV 页回到原 logical position（无需 re-RoPE）；KVMem 把块搬到从未待过的 compact 位置（必须 re-RoPE）。任何融合方案必须先回答「消灭 / 吸收 / 容忍摩擦」之一。详细对比见各子目录研究报告与 `docs/`。

## 四方案横向对比

评分 1–5，★ 越多越好。加权 = 技术风险 20% / 实现成本 15% / 性能预期 20% / 可发表性 25% / 迭代空间 20%（依赖闭源与最短路径为参考列，不计入总分）。

| 维度 | 方案一 IFR | 方案二 LADDER | 方案三 UBBA | 方案四 CASA |
|---|---|---|---|---|
| 技术风险 | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ |
| 实现成本 | ★★☆☆☆（53 人日 + ~1900 GPU-h） | ★★★★★（12 人日） | ★★★★☆（32 人日） | ★★★★☆（44 人日 + ~830 GPU-h） |
| 性能预期 | ★★★★☆ | ★★★★★（NVMe 8×↓） | ★★★☆☆ | ★★★★★（字节 + 延迟双线） |
| 可发表性 | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★★ |
| 迭代空间 | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ |
| 依赖闭源 | ★★☆☆☆ 高 | ★★★☆☆ 中 | ★★★★☆ 低 | ★★★☆☆ 中 |
| 加权总分 | **3.90** | **3.70** | **4.00** | **4.30** |

一句话定位：

- **IFR（方案一二融合 R10+R3）**：可证伪的块级 KV 检索——不宣称"更好"，只宣称"保真约束下把 10M workspace 的检索延迟与索引常驻压到可复现数值"（retrieval ≤350ms、index ≤4 GiB，top-1 一致率 ≥97%）。
- **LADDER（R6）**：KV 内部的保真阶梯——B_m 用尽时沿 KV 内部逐级降级（raw FP8 → 2-bit → 跨块合并 → Mean-K → 文本兜底），NVMe footprint 324 → ≤42 GiB（≈8×）。
- **UBBA（第一轮新增）**：统一字节预算器——纯控制面，按 `min bytes s.t. ρ ≤ ρ_floor ∧ coverage ≥ target` 分配字节，无 kernel 无训练；失败模式是"不够好"而非"崩掉"。
- **CASA（圆桌收敛）**：Canonical Atom Store Architecture——把 KV 字节从"绑定位置的可变状态"变成"内容寻址的规范化不可变对象"，K-Freeze + Q-remap 消除 re-RoPE 摩擦，是前三者的共同地基。

## 推荐结论（圆桌后更新）

**CASA 主线 + IFR 判定层捆绑，UBBA 换约束集后并入，LADDER 作期权。**

1. CASA 是新的系统主线（加权 4.30 居首），但单独不产生质量增益，必须挂靠 IFR 的 U-E-F-C 四维门才能被判定。
2. 投注配比：CASA 45% / IFR(eval backbone) 30% / LADDER(P0 期权) 10% / 机动 15%。
3. 四方案共用 IFR 的 U-E-F-C 四维门；R10 已判定任务质量终点在 1,900 GPU-h 内不可判决，一律不得设为主终点（主终点须落在 {字节, 延迟, ρ, 覆盖率}）。
4. 全套退役 / Gate 判据（G0、M0–M4、G-CAS-1/2 等）见原始主报告 §5.3，写死、事先承诺，触发即冻结对应研究路线。

## 本仓库当前内容（状态）

### 研究文档

| 路径 | 内容 / 状态 |
|---|---|
| `docs/master-research-report.md` | ✅ **完整主报告**（10 研究员摘要 + 圆桌 + 四方案 + 横向矩阵 + 退役 Gate），78 KB |
| `docs/experiment-manual.md` | ✅ **实验执行手册**：设备需求、容量/预算数学、云端 GPU 价格、E0–E9 逐步实验 |
| `plan-01-ifr/` | ✅ 研究报告（IFR：可证伪的块级 KV 检索，U-E-F-C 四维门） |
| `plan-02-ladder/` | ✅ 研究报告（LADDER：KV 内部保真阶梯，NVMe 8×↓） |
| `plan-03-ubba/` | ✅ 研究报告（UBBA：统一字节预算器，约束集修正版） |
| `plan-04-casa/` | ✅ 研究报告（CASA：规范原子存储架构，K-Freeze + 前缀哈希链） |
| `docs/external_review_critique.md` | ✅ 第一轮技术质疑：证据行号 + 正确验证方法 + 建议 |
| `docs/external_review_critique_v2.md` | ✅ 第二轮复核（针对 273ae4b）：机制有效性边界、必补对照清单、真实模型 P0 完整可执行协议、T0–T3 成熟度分级 |
| `docs/external_review_critique_v3.md` | ✅ **【最新】第三轮复核（针对 f093dfd）**：M0 判据事后替换、真实 σ 未被覆盖、索引漏 L×H_kv、配对检验用非配对 SE，并含**我们自身 82% 论证的修正** |
| `prompts/` | 🚧 可执行 AI prompt（每方案一份 + 共享 eval prompt）待生成 |
| 各方案 `design-doc.md` | 🚧 设计文档（UML + milestone）待生成 |

### 实现与评测（外部评审团队贡献，**验证状态存疑**）

| 路径 | 内容 | 我们的验证判定 |
|---|---|---|
| `benchmarks/eval_end_to_end.py` | 8K–32K NIAH 评测 harness | ❌ 合成数据，无真实模型 |
| `benchmarks/eval_snr_sensitivity.py` | SNR 1.0→5.0 扫描 + 虚警率 | ⚠️ 设计正确，但结论反噬机制：σ≥0.10 虚警 99.85%（见 v2 §2.1） |
| `benchmarks/eval_scale_selection_rate.py` | 32K→1M 规模/选择率扫描 | ⚠️ docstring 声明 Random 基线但结果表无该列（见 v2 §2.3） |
| `docs/track_a_benchmark_report.md` | Track A 基准报告 | ⚠️ 数字不可采信，见质疑文档 §2.1–2.2 |
| `docs/comparative_study_report.md` | 四系统能力对比白皮书 | ⚠️ 含推导/专家意见冒充实测，见 §2.6 |
| `csrc/` | C++20 UBBA 求解器、CASA 页表、benchmark | ✅ 求解器 69.75 µs 可信；但 "Effective Bandwidth" 是内存内带宽，非 I/O |
| `kernels/fused_tier_bias_attention.py` | 混档 tier-bias 注意力 | ⚠️ torch 为可选后端，未跑 GPU |
| `kvmem_fusion/` | core / ifr / ladder / ubba Python 实现 | ⚠️ 待真实模型验证 |
| `tests/` | 8 个测试文件（宣称 63–70 passing） | ⚠️ 多数为数值等价性断言，非性能/硬件实测 |

## 如何复现 / 起步

- 实验设备与逐步手册：见 `docs/实验执行手册-设备与步骤.md`（原始文件，待并入本仓库）。
- 基座实现：`kvmem/kvmem-llama.cpp`（Apache-2.0, v0.17.0，RTX 5060 Ti 16GB 实测 52.7 KiB/token）；其缺 re-RoPE 与 Mean-K，需在 P1 前补齐。
- **≤256K 区间不需要 re-RoPE**，第一年可钉在该区间绕开 CUDA kernel 工作——恰是 KVMem 两个核心 benchmark 的测量区间。

## 参考文献

KVMem arXiv 2609.04852 · Strata arXiv 2508.18572（OSDI '26）· CacheBlend 2405.16444 · KIVI 2402.02750 · KVQuant 2401.18079 · DMC 2405.16699 · H2O 2306.14048 · StreamingLLM 2309.17453 · Scissorhands 2305.17118 · ToMe 2210.09461 · RAPTOR 2401.18059 · Lan & DeMets, Biometrika 70(3):659–663, 1983。
