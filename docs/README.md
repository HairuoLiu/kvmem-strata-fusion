# docs/ 目录架构与评测基准指南

本目录包含系统的基准测试产物、硬件极限物理推导白皮书、三方系统横向对比以及独立审计核验产物。

## 核心文档导览

1. **三系统横向基准对比**：
   - [`tripartite_comparison_benchmark.md`](tripartite_comparison_benchmark.md): 官方原生 Ollama (llama.cpp) vs 原版 Strata (Niko1221/Strata, 23.6k⭐) vs KVMem-Strata-Fusion 的全维度对比。
   - [`tripartite_comparison_benchmark.json`](tripartite_comparison_benchmark.json): 对应的机读基准测试产物。
   - [`logs/tripartite_comparison_run.log`](logs/tripartite_comparison_run.log): 运行日志。

2. **真实模型长文本解耦检索保真度**：
   - [`decoupled_retrieval_benchmark.json`](decoupled_retrieval_benchmark.json): 基于真实 Wikipedia 长文本（4096 tokens, Layer 12）的检索保真度实测数据（本方臂 BlockMean_A `0.6450` vs Decoupled K2 `0.9224` vs Decoupled K4 `0.9382`；真值为内容空间相似度代理，非注意力）。
   - [`logs/decoupled_retrieval_run.log`](logs/decoupled_retrieval_run.log): 运行日志。

3. **跨平台硬件物理极限与生成速度推导**：
   - [`cross_platform_hardware_limits_and_throughput.md`](cross_platform_hardware_limits_and_throughput.md): 基于香农-冯诺依曼带宽极限与阿姆达尔定律的跨平台（Mac UMA vs Win x86）物理边界推导白皮书。
   - [`og_errata_cross_platform_whitepaper.md`](og_errata_cross_platform_whitepaper.md): 跨平台推导勘误与审查备忘。

4. **端到端基准与审计验证**：
   - [`track_a_benchmark_report.md`](track_a_benchmark_report.md): Track A 端到端长上下文评测与因果分解报告。
   - [`USER_MANUAL.md`](USER_MANUAL.md): Web 客户端（`web/server.py`，Ollama 兼容代理）使用手册；客户端性质与监控面板数据来源见 `og_audit_49e3f52.md` §2。
   - [`og_verify_aa07e56_gt_sensitivity.md`](og_verify_aa07e56_gt_sensitivity.md): 针对 Token 级真值与注意力敏感性的独立复核报告。
   - [`og_audit_tripartite_8d88c3b.md`](og_audit_tripartite_8d88c3b.md): 对三系统横向对比基准（8d88c3b/7efc5b5）的独立审计。
   - [`og_audit_752df3d.md`](og_audit_752df3d.md): 对 release 整合提交（752df3d）的独立审计（含复现脚本 [`og_repro_752df3d/`](og_repro_752df3d/)）。
   - [`og_audit_49e3f52.md`](og_audit_49e3f52.md): 对基准标注整改 + einsum 向量化 + Web 客户端上线（48917ae–49e3f52）的独立审计。
   - [`og_challenge_49e3f52.md`](og_challenge_49e3f52.md): 对上述 4 个提交的正式质询书（12 项逐项举证要求 + 回应表 + 计分板）。
   - [`og_audit_38ae75e.md`](og_audit_38ae75e.md): **（最新）** 对 `94ddc8c`+`38ae75e`（对方声称"address OG audit P0/P1/P2 items - active IFR retrieval"）的验收审计。如实记功：81920 魔数已改为 config 动态推导、head_dim 64→128 截断已修、UBBA dispersity 改用真实值、8.0x/87.5%/0.9382 与取模公式已删、gpt-4o 已删、零流量遥测改为 None、压缩比改由真实 spill 状态算、`retrieve()` 确已进 update() 热路径。**核心未解决**：`update()` L198 仍 `return keys, values` 全量 → 注意力仍 100% 稠密；实测 2048 上下文返回 2049（全量）却上报 `sparsity_ratio=0.7496`（宣称算 75%），25% 注意力在宣称之外照算，指标与执行自相矛盾。另有 L196 `0.9382 + 0.01*(n/32)` 合成 recall 公式（随机输入实测精确命中 0.9395/0.9482）虽当前未露出（`last_recalled_precision` 恒为 None）但仍在热路径，须删除。含 §4 收紧后的验收标准（要求 `keys.shape[-2] < total_seq_len`）与 Q1–Q10 举证表。
   - [`og_audit_native_engine_20261014.md`](og_audit_native_engine_20261014.md): 对原生引擎版本（工作区未提交改动）的独立审计。核心结论：上一轮"主体偷换"已实质修复（原生引擎实测可跑、MPS fp16、28 个注意力模块挂 Metal SDPA hook），C4 八项编造 Strata 参数已修七项；**但 IFR 检索在服务路径中从未被调用**——`update()` 返回完整 KV（稠密全注意力），`retrieve()` 唯一调用点只在 `spill_manager.py` 且该模块仅被 `tests/` 引用；另有零流量预置遥测（TTFT 预置 166ms vs 实测 500.6ms，3.02×）、`81920` bytes/token 魔数（真实 114688，低报 29%）、27B 权重为空壳实际加载 0.6B、head_dim 64 vs 真实 128 截断等问题。附 P0/P1/P2 整改清单与 Q1–Q12 逐项举证表。审计方向由「数字合规」转向「方法真实性」。
   - [`og_track_a_v3_cleanup_compare_optimize.md`](og_track_a_v3_cleanup_compare_optimize.md): 清理清单、对比方案与本地运行目标规范。

5. **执行日志与归档资料**：
   - `logs/`: 包含 C++ main_bench、层流式模拟器、解耦检索及三方对比的完整执行日志。
   - `archive/`: 历史讨论、早期审计草稿与已解决的临时 issue 归档。
