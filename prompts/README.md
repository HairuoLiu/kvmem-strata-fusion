# prompts/ —— 可执行 AI Prompt 目录（待生成）

本目录计划存放**面向 AI 子 agent 的可执行 prompt**，让接手者无需重新理解全文即可派发任务。当前为空，预计 2026-10-06 08:00 UTC+8（API 频率限制重置）后补齐，规划如下：

| 文件（规划） | 用途 | 目标角色 |
|---|---|---|
| `eval-backbone.md` | 统一 U-E-F-C 四维门评测 harness 的派发 prompt | 评测工程 agent |
| `plan-01-ifr.md` | IFR 索引层 + 保真门实现 prompt | Google senior-level 工程师 agent |
| `plan-02-ladder.md` | LADDER 保真阶梯 + P0 证伪实验 prompt | 量化 / 系统 agent |
| `plan-03-ubba.md` | UBBA 字节预算求解器（约束集修正版）prompt | 控制面 agent |
| `plan-04-casa.md` | CASA canonical store + K-Freeze kernel prompt | Hopper 内核 agent |

每个 prompt 将包含：明确输入 / 输出契约、禁止项（如禁用 "up to"、必须报 CI）、milestone 拆分、以及触发即冻结的 Gate 判据引用。
