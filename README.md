# KVMem × Strata 融合研究：四方案执行计划

> 把一个"更快的长上下文 KV 系统"问题，重新拆成「测量方法 + 共同地基」两份真正可推进的工作。

## 诚实声明（先读这段）

本报告是 **纯研究 / 设计文档，没有任何代码实现**。所有"预期""目标"均为未实测的假设，文中凡标注「未验证」之处均未经实验确认。融合的两大硬前提——KVMem 闭源引擎 **QW3 的可达性**、以及 **≤256K 区间不需要 re-RoPE**——是后续一切实验的 G0 杀点。请带着这个前提阅读，不要把它当作已验证结论。

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

| 路径 | 内容 / 状态 |
|---|---|
| `plan-01-ifr/` | ✅ 教授级研究报告已完成；设计文档 + 可执行 prompt 待生成 |
| `plan-02-ladder/` | ✅ 教授级研究报告已完成；设计文档 + 可执行 prompt 待生成 |
| `plan-03-ubba/` | 🚧 研究报告待生成（受 API 频率限制阻塞，预计 2026-10-06 后补）；本目录 README 含方案简介 |
| `plan-04-casa/` | 🚧 研究报告待生成（同上）；本目录 README 含方案简介 |
| `prompts/` | 🚧 可执行 AI prompt（每方案一份 + 共享 eval prompt）待生成 |
| `docs/` | 🚧 统一评测 backbone 规范 + 实验执行手册待整理 |
| 完整主报告 | `kvmem-strata-融合研究方案.md`（10 研究员摘要 + 圆桌 + 四方案 + 矩阵 + Gate），位于本仓库外的工作区根目录，后续会并入 `docs/` |

## 如何复现 / 起步

- 实验设备与逐步手册：见 `docs/实验执行手册-设备与步骤.md`（原始文件，待并入本仓库）。
- 基座实现：`kvmem/kvmem-llama.cpp`（Apache-2.0, v0.17.0，RTX 5060 Ti 16GB 实测 52.7 KiB/token）；其缺 re-RoPE 与 Mean-K，需在 P1 前补齐。
- **≤256K 区间不需要 re-RoPE**，第一年可钉在该区间绕开 CUDA kernel 工作——恰是 KVMem 两个核心 benchmark 的测量区间。

## 参考文献

KVMem arXiv 2609.04852 · Strata arXiv 2508.18572（OSDI '26）· CacheBlend 2405.16444 · KIVI 2402.02750 · KVQuant 2401.18079 · DMC 2405.16699 · H2O 2306.14048 · StreamingLLM 2309.17453 · Scissorhands 2305.17118 · ToMe 2210.09461 · RAPTOR 2401.18059 · Lan & DeMets, Biometrika 70(3):659–663, 1983。
