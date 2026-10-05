# 方案四 CASA —— Canonical Atom Store Architecture（圆桌收敛）

> **一句话定位**：把 KV 字节从"绑定位置的可变状态"变成"内容寻址的规范化不可变对象"；K-Freeze（K 永不移动，位置校正搬到瞬态 Q 侧）+ Q-remap，消除 re-RoPE 摩擦。

## 立项依据（前三条方案共同缺的那一层）

IFR / LADDER / UBBA 都在各自解决一个问题，却共享一个未被言明的地基缺失：KV 字节仍是"绑定位置的可变状态"。没有共同地基，去重、压缩、共享、搬运四件事落在不同字节上，永远要付 re-RoPE 的重建税。CASA 补上这层。

## 关键设计

- **K-Freeze**：K 一旦写入 canonical store 永不移动；位置校正搬到瞬态的 Q 侧（Q-remap），使保真误差与搬运次数脱钩——`ρ(n)=ρ₀` 是可形式化证明的性质，消除 `n*≈250` 的重建悬崖，I/O 放大 65→324 GiB 的 5× 回退消失。
- **三条不可违反约束**：
  - **G-CAS-1**：瓶颈归属必须在 95% CI ≥15pp 上区分（7.8 GB/s 三界矛盾：不被 DRAM/PCIe 解释，是最大单点风险）；
  - **G-CAS-2**：fused K-tile 回退 >15% 或加速 CI 下界 <0.85 ⇒ K-Freeze 核心路线终止；
  - **M4**：prefix-identical 重放必须 bit-exact，否则共享支路终止（R9 信任平面论证）。
- **Phase 0–4 里程碑** + Lan-DeMets OBF 预注册 stopping rule；Phase 0（8 人日 / ~200 GPU-h）即可出独立可发表的瓶颈归属结论（全场最早止损点）。
- **与前三关系（不是替代，是地基 + 一次吸收）**：A1 成立后 IFR 的 posting 变 append-only；LADDER 的读路径旋转 2→1 次；UBBA 换约束集后并入 Phase 2。任何一条出成果都增大 CASA 价值。

## 状态

- 🚧 `research-report.md`：教授级研究报告待生成（受 API 频率限制阻塞，预计 2026-10-06 08:00 UTC+8 后补）。
- 🚧 设计文档（Google senior-level 工程师，含 UML + milestone）：待生成。
- 🚧 可执行 AI prompt：待生成。
- 本 README 的方案简介取自主报告 §3.4，供接手者先行理解。

## 最大风险

G-CAS-1 的三界矛盾是最大单点风险；Δ-QRoPE 在 Hopper WGMMA 下的真实回退未知（R4 自曝 50% 置信度），需真正懂 Hopper 流水的人；CASA 单独不产生质量增益，必须挂靠 IFR 的 eval backbone 才能被判定。
