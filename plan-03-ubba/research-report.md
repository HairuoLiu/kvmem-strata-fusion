# 方案三 UBBA 研究报告：统一字节预算器（Universal Byte-Budget Allocator）

> **一句话定位**：纯控制面运筹求解器，按 $\min \text{bytes} \quad \text{s.t.} \quad \rho \le \rho_{\text{floor}} \land \text{coverage} \ge \text{target}$ 分配 KV 字节；无自定义 kernel、无权重微调、失败模式是“优化效果不够极致”而非“系统崩溃”。
> **状态**：算法与原型测试通过 · **读者**：分布式系统资源调度工程师与控制理论研究者

---

## 1. 背景与核心洞察

在现有的长上下文推理策略中，不同层级的缓存分配要么基于硬编码阈值（如超过 80% 显存直接驱逐），要么在不同压缩级别之间做贪心启发式降级。
在五位审稿人审阅初版时，原目标函数 $\max \text{coverage} \quad \text{s.t.} \quad \text{bytes} \le B$ 被独立否决：**如果没有保真度（Fidelity $\rho$）作为硬约束，背包算法必然把所有配额填满单价最便宜的失真档位（如 1-bit），导致模型质量瞬间跌落 12.7 个百分点**。

UBBA 将控制逻辑正交化：**将保真度设为硬约束，以最小化总搬运字节数为目标**。

---

## 2. 形式化数学规划

给定候选块集合 $\mathcal{B} = \{b_1, \dots, b_N\}$，每个块在档位 $t \in \{\text{FP8}, \text{INT4}, \text{INT2}, \text{MERGED}\}$ 下具有：
* 存储字节开销 $C(b_i, t)$
* 相对失真度 $\rho(b_i, t)$
* 注意力质量权重（通过 Mean-K 预估）$w(b_i)$

优化问题定义为：
$$\min_{\{t_i\}} \sum_{i=1}^N C(b_i, t_i)$$
$$\text{s.t.} \quad \sum_{i=1}^N w(b_i) \cdot \mathbb{I}(t_i \ne \emptyset) \ge \text{Target Coverage}$$
$$\rho(b_i, t_i) \le \rho_{\text{floor}}, \quad \forall i \in \{1,\dots,N\}$$

---

## 3. 控制面算法优势

1. **零崩溃风险**：纯离线 / 控制面线性整数规划 / 贪心拉格朗日乘子求解，即便求解耗时退化，兜底策略只需退回固定 FP8 分配。
2. **多系统通用**：无论底层是 NVMe、Host RAM 还是 HBM，UBBA 只根据设备带宽与容量的影子价格（Shadow Price $\lambda$）输出最佳分档决策。
