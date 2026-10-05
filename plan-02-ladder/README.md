# 方案二 LADDER —— KV 内部的保真阶梯

> **一句话定位**：当 B_m 用尽时，不许退回文本，而沿 KV 内部的保真阶梯逐级降级，把 10M workspace 的 NVMe footprint 从 324.2 GiB 压到 ≤42 GiB（≈8×）。

## 核心主张

"退回文本"不是降级策略而是设计缺陷（模态断裂、不可逆、与自身基准冲突——KVMem 在 ≤256K 已达 60.87 > Full 的 59.54）。降级必须留在 KV 内部：

- **L0** raw FP8（324 GiB，不可变权威，可重建）
- **L1** KIVI 式 2-bit（65 GiB）
- **L2′** 跨块合并 G=4（32.5 GiB，有损）
- **L3** Mean-K index（9.5 GiB，仅检索）
- **L4** 文本兜底（0.04 GiB）

两条铁律：永远 **RoPE-then-quantize**（禁止 quantize-then-rotate）；合并必须在去位置空间进行。

## 关键设计

- **去位置空间合并**：de-RoPE → 对齐 → 质心 4-bit → re-RoPE 到规范位置，另存位置增量（8-bit）与内容残差（rank-r SVD）。
- **位置残差**：本方案唯一的"可逆性押金"；每 step 按 CacheBlend 的 HKVD 准则重算 ≤15% token 主动偿还有损。
- **tier-bias 修正**：混档 softmax 中第 t 档减常数 `b_t = −(ρ_t·s)²/2`，修正低档块被系统性高估（2-bit 块相对 raw 窃取约 22% attention 质量）。
- **运行时策略**：以字节预算影子价格 λ 下的 `score = P(进入视图)×Δ保真损失 − λ×Δbytes` 排序，只允许相邻档移动（修正 R6 用命中频率导致的死亡螺旋内生性错误）。

## 状态

- ✅ `research-report.md`：教授级研究报告（含 mermaid 阶梯图、ρ_max 推导、8× 对跨块冗余的算术依赖、82% 高频维损失论证、P0 最小证伪实验、合并不可逆特殊效度）。
- 🚧 设计文档（Google senior-level 工程师，含 UML + milestone）：待生成。
- 🚧 可执行 AI prompt：待生成。

## 最大风险（最大赌注，零证据）

**位置残差的充分性**未经检验；合并不可逆、无退路；8× 数字可能落在非瓶颈处。故以 **P0 最小证伪实验（2 人日，无需 GPU）** 先行：若"de-RoPE 后平均"不优于"直接平均已 RoPE"，核心假设被推翻。
