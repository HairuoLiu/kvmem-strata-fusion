# docs/ —— 统一评测 backbone 与实验执行手册

本目录存放跨方案的公共资产与实测基准报告：

## 已完成评测报告
- [`track_a_benchmark_report.md`](file:///Users/hai/.gemini/antigravity/scratch/kvmem-strata-fusion/docs/track_a_benchmark_report.md): **Track A: Real Model Evaluation & Needle-in-a-Haystack Benchmark Report**
  - 覆盖 8K (8,192)、16K (16,384)、32K (32,768) 上下文窗口与 10%–90% 深度全扫描。
  - CASA + IFR + UBBA + LADDER 完整端到端流水线评测。
  - 核心指标：Needle Recall 100.0% (Rank 1), Cosine Similarity 1.0000, $\Delta\text{PPL} = 0.0000$, 压缩比 7.7x (32K 降至 130.4 KB), U-E-F-C 四维统计门全面通过。

## 待并入文档
- `实验执行手册-设备与步骤.md`：设备需求表、容量/预算数学、云端 GPU 价格、环境搭建、E0–E9 逐步实验。
- `IFR-Eval` harness 规范（U-E-F-C 四维门 + byte-level trace schema）。
- `G-CAS-1` 瓶颈分解工具与五档 workspace 的 Nsight trace 规范。
- 完整多 agent 主报告 `kvmem-strata-融合研究方案.md`。
