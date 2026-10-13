# docs/ 目录架构与评测基准指南

本目录包含系统的基准测试产物、硬件极限物理推导白皮书、三方系统横向对比以及独立审计核验产物。

## 核心文档导览

1. **三系统横向基准对比**：
   - [`tripartite_comparison_benchmark.md`](tripartite_comparison_benchmark.md): 官方原生 Ollama (llama.cpp) vs 原版 Strata (Niko1221/Strata, 23.6k⭐) vs KVMem-Strata-Fusion 的全维度对比。
   - [`tripartite_comparison_benchmark.json`](tripartite_comparison_benchmark.json): 对应的机读基准测试产物。
   - [`logs/tripartite_comparison_run.log`](logs/tripartite_comparison_run.log): 运行日志。

2. **真实模型长文本解耦检索保真度**：
   - [`decoupled_retrieval_benchmark.json`](decoupled_retrieval_benchmark.json): 基于真实 Wikipedia 长文本（4096 tokens, Layer 12）的检索保真度实测数据（对比 Naive Mean-K `0.6450` vs Decoupled K2 `0.9224` vs Decoupled K4 `0.9382`）。
   - [`logs/decoupled_retrieval_run.log`](logs/decoupled_retrieval_run.log): 运行日志。

3. **跨平台硬件物理极限与生成速度推导**：
   - [`cross_platform_hardware_limits_and_throughput.md`](cross_platform_hardware_limits_and_throughput.md): 基于香农-冯诺依曼带宽极限与阿姆达尔定律的跨平台（Mac UMA vs Win x86）物理边界推导白皮书。
   - [`og_errata_cross_platform_whitepaper.md`](og_errata_cross_platform_whitepaper.md): 跨平台推导勘误与审查备忘。

4. **端到端基准与审计验证**：
   - [`track_a_benchmark_report.md`](track_a_benchmark_report.md): Track A 端到端长上下文评测与因果分解报告。
   - [`og_verify_aa07e56_gt_sensitivity.md`](og_verify_aa07e56_gt_sensitivity.md): 针对 Token 级真值与注意力敏感性的独立复核报告。
   - [`og_track_a_v3_cleanup_compare_optimize.md`](og_track_a_v3_cleanup_compare_optimize.md): 清理清单、对比方案与本地运行目标规范。

5. **执行日志与归档资料**：
   - `logs/`: 包含 C++ main_bench、层流式模拟器、解耦检索及三方对比的完整执行日志。
   - `archive/`: 历史讨论、早期审计草稿与已解决的临时 issue 归档。
