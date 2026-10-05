# 方案一 IFR —— 可证伪的块级 KV 检索

> **一句话定位**：不宣称"融合后任务成功率更高"（当前预算下统计上不可判决），只宣称"在保真约束下把 10M workspace 的检索延迟与索引常驻压到可复现的具体数值"。

## 核心主张

效率宣称必须锚定在保真约束上，且两者都要做成可被第三方在同一 protocol 下否定的数字。主终点不是质量，而是 **保真约束下的效率与成本**：

- 保真门（F）：top-1 一致率 ≥97%、utility gap ≤1pp；
- 效率门（E）：retrieval ≤350ms@10M（基线 1.311s）、index ≤4 GiB（基线 9.5 GiB）；
- 成本门（C）：$/成功任务 + 常驻 GiB；
- 效用门（U，非主终点）：task-seed 配对成功率差 + Wilson CI，配对 CI 下界 >0 才 Go。

## 关键设计

- **Dedup-Then-Route 两级流水线**：L0 HiRadixTree 身份去重；L1 IVF-over-Mean-K 路由，质心常驻 DRAM、posting 落 NVMe 大 tile 顺序读。语义剪枝只作用于索引条目，绝不施加到 KV 内容本身。
- **LSE 缓存**：利用 Eq.(10) softmax 分母对候选集无关，排序阶段无需归一化，每 (l,m,h) 缓存一个 log-sum-exp 标量。
- **抗坍缩**：存 ‖kᵢ−k̄‖，超阈分裂双子质心。
- **统计契约**：禁用 "up to"；配对 Wilson CI；样本量 471–525 对（≈118–131 tasks × 4 seeds）；聚类 design effect ≈1.9；Lan-DeMets OBF 3–4 looks 预注册。

## 状态

- ✅ `research-report.md`：教授级研究报告（含 mermaid 流程图、假设表、f⁸ 推导、秩反转形式化、样本量计算、四组效度威胁、相关工作）。
- 🚧 设计文档（Google senior-level 工程师，含 UML + milestone）：待生成。
- 🚧 可执行 AI prompt：待生成。

## 最大风险

依赖闭源引擎 QW3（G0 杀点）+ 525 对 rollout 能否跑得起；若 DRAM 才是首墙，检索加速可能不落在主瓶颈上。
