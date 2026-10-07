# KVMem × Strata 融合研究：多 Agent 研究报告与四方案执行计划

> 生成日期：2026-10-04
> 研究对象：KVMem (arXiv 2609.04852, 2026-09) × Strata (arXiv 2508.18572 / USENIX OSDI '26, arXiv v1 2025-08)
> 研究方法：**第一轮** 10 个并行研究员 agent 各自独立多源检索 + 原文精读 + 方案合成；**第二轮** 同一批 10 人参加圆桌（共享简报 + 投票 + 自曝 + 合并/反对动议），收敛出方案四

---

## 0. 方法与诚实声明

**务必先读这一节，它影响你怎么看待本报告的所有结论。**

| 项目 | 实际情况 |
|---|---|
| Agent 数量 | 10 个并行子 agent，各自独立完成「多轮 WebSearch/WebFetch 检索 → 原文精读 → 交叉验证 → 方案合成」闭环 |
| 研究深度 | **不是**真实的 10 小时墙钟。每个 agent 执行的是高强度但有时限的研究循环（约等于研究员数小时的专注工作量），产物质量为"高级研究员初稿"而非"打磨后的投稿版" |
| 分工方式 | 10 个互不重叠的切口（调度 / I-O / 索引 / 数值 / 多租户 / 压缩 / 存储引擎 / 架构协同 / 编程模型 / 评测方法），避免 10 份产出同质化 |
| **第二轮机制** | 第一轮是 10 份互不通气的独立报告；第二轮改为**圆桌**——分发同一份共享简报，要求每人投票（100 点分配到 A–G 议程项）、自曝弱点、提合并动议与反对动议。**§2.5 记录的所有结论都只在互相质证后才出现** |
| 数字可信度 | 报告中带 arXiv 编号或 URL 的数字来自原文；agent 自行推算的数字均标注了推导过程与"未验证"。**凡标「未验证」者不可直接写进论文** |
| 最大系统性风险 | **QW3（KVMem 的推理引擎）并非完全开源**——这在 10 份报告里有 7 份被列为 Top-3 风险。**2026-10-05 复核后部分解除**：存在开源 Apache-2.0 的兄弟项目 `kvmem/kvmem-llama.cpp`，实测平台为 RTX 5060 Ti 16GB + 32GB RAM（远低于报告假设的 96GB 服务器卡）。详见 **附录 C** |

**反事实提示（第一轮）**：方案一的核心建议不是"做一个更快的系统"，而是"先让整个领域的主张变得可证伪"。这个判断源自 R10 的统计结果，它应该被优先质疑——如果 R10 的显著性计算有误，则方案排序需要重做。

**反事实提示（第二轮）**：方案四 CASA 建立在 A1（Δ-QRoPE + Fused kernel）成立之上。若 G-CAS-2 实测回退 >15%，K-Freeze 的经济性崩塌，整个方案四应让位于方案二。这个 Gate 设在**第 2 周**，是本报告止损最早的设计。

---

## 1. 对比分析：相似点与差异

### 1.1 一句话定位

- **KVMem**：把单个长生命周期 agent 的**累积工作区**虚拟化成分页 KV 仓储，每步按语义召回到一个有界的执行视图里。它对抗的是"**记不住**"。
- **Strata**：把多请求共享的长上下文字首 KV 做成分层缓存，用 GPU 辅助搬运和缓存感知调度把回注延迟藏掉。它对抗的是"**搬不动**"。

两者是同一问题的两端：**KVMem 发明了"该取什么"，Strata 发明了"怎么搬得快"**。

### 1.2 相似性对照

| 相似维度 | 具体表现 | 出处 |
|---|---|---|
| 核心隐喻 | 都把虚拟内存思想搬到 KV cache：**可寻址空间** 与 **物理驻留** 解耦 | KVMem §1/§3.1；Strata §2.2 |
| 存储金字塔 | 都是 GPU HBM → host DRAM → SSD/NVMe 三层 | 同上 |
| 共同敌人 | 都把"重新 prefill 已处理过的内容"视为最大浪费 | KVMem §1 (`T_text` vs `T_KV`)；Strata §1 |
| 管理单位 | 都用页/块作为分配与移动的最小单位 | KVMem 32-token logical block；Strata 1/16/32-token page |
| 碎片化问题 | 都直面 PagedAttention 小页带来的 I/O 低效 | Strata §3.1 是这个问题最系统的刻画 |
| 粒度取舍 | 都面临「小粒度利于命中/检索精度 vs 大粒度利于传输带宽」的经典矛盾 | KVMem 用 32-token 折中；Strata 用 128B GPU 搬运打破该取舍 |
| 次要设计目标 | 都追求"短/小规模场景不回退" | Strata 明确保 short-context no regression；KVMem 的 GPU 内存 footprint 恒定 |

### 1.3 差异性对照（核心）

| 维度 | KVMem | Strata |
|---|---|---|
| 服务对象 | 单个长生命周期 agent session | 多请求并发 serving（prefill-dominated） |
| 要解决的问题 | 上下文超出模型 **native window** 怎么办 | KV 缓存超出 **GPU HBM** 后 I/O 成为瓶颈怎么办 |
| 是否扩展 jointly-attended tokens | **不扩展**，每步仍是 bounded execution view | **不改变**，完整上下文仍参与 attention |
| 历史如何被取回 | **近似检索**：query-conditioned 稀疏随机块召回 | **精确匹配**：token-id 前缀树内容寻址去重 |
| 位置假设 | 必须 remap 到**新的 compact position** → **必须 re-RoPE** | 前缀复用回到**相同 logical position** → **不需要 re-RoPE** |
| 索引结构 | Mean-K（model-native attention 空间，语义近似） | HiRadixTree（token-id 前缀树，精确身份） |
| 时间尺度 | **step 级**（秒级，有 slack） | **请求级**（毫秒级，无 slack） |
| 核心优化指标 | recovery latency + 任务成功率 | TTFT + throughput |
| 主要创新面 | **选择/检索/保真**（"取什么"、"还原到什么精度"） | **搬运/调度**（"怎么搬"、"什么时候搬"） |
| 工作集变化性 | 每步剧烈变化，`D_t` 分解 retained/incoming/outgoing | 静态前缀，可跨请求 dedup 摊销 |
| 实现载体 | 自研引擎 QW3（C++/CUDA，`github.com/kvmem/kvmem-qw3`） | 开源 **SGLang** 分支 + 头部 AI 公司生产部署 |
| 评测强度 | 学术 benchmark + 单点笔记本 demo | 多模型×多硬件 + 生产部署 + OSDI 同行评审 |
| 关键数字 | 1M workspace @24GB 消费卡 / ~50 tok/s；DeepSWE Pass@1 43.8→48.4 | TTFT 比 vLLM+LMCache 低至 5×，比 TRT-LLM 快至 3.75× |
| 显式自承限制 | KV 复用**不等价于重算**；不可审计/编辑/跨模型迁移；B_m 用尽回退文本；未评测多用户 | 只针对 prefill-dominated；单实例 P-D 共置；启发式阈值需 per-hardware profiling |

### 1.4 融合的**唯一硬摩擦**（这是全篇最重要的一句话）

> **Strata 的全部 I/O 与去重机制，都建立在「取回的 KV 页会回到它原来的 logical position」这一隐含假设上；而 KVMem 的全部价值，恰恰来自「把 KV 块搬到它从来没待过的 compact 位置」。**

这就是为什么不能简单地把 KVMem 的 store 塞到 Strata 的 cache controller 下面。任何融合方案必须先回答三件事之一：

1. **消灭摩擦**（R4 路线）：让 K 永远不移动，把重映射搬到 Q 侧 → re-RoPE 不再是必需；
2. **吸收摩擦**（R2 路线）：把 re-RoPE 融进搬运 kernel，让重编码成为搬运的一部分，成本降到近零；
3. **容忍摩擦**（R6 路线）：承认必有误差，用保真阶梯 + 受控重算把误差钉在有界范围内。

三个方案分别对应这三条路线中的两条，以及一个正交的控制面路线。

### 1.5 设计取舍与适用边界速查

| 场景 | 应该用谁 | 理由 |
|---|---|---|
| 单 agent 长期任务、 history 超出模型 window | **KVMem** | 这是唯一定义了这个抽象的系统 |
| 多用户共享文档/系统提示的长上下文 dedup | **Strata** | 前缀去重 + 缓存感知调度是它存在的理由 |
| 短上下文、低请求率 | 都不需要 | Strata 自己承认短上下文只保证"不退步" |
| 24GB 消费卡跑百万 token 本地 agent | **KVMem** | 唯一被验证过的路径（vLLM 同硬件仅 ~10K context） |
| 需要审计/编辑/跨模型版本迁移 | **都不合适** | KV state 绑定权重，这是 KV-state memory 的根本缺陷（R9） |
| 超大输出量的 decode-heavy 负载 | **都不对口** | Strata 明确 prefill-dominated；KVMem 未报并发吞吐 |

---

## 2. 十研究员核心结论摘要

> 排序依据：加权评分（技术可执行性 30% / 性能收益量级 25% / 可发表性与影响力 25% / 后续迭代空间 10% / 成本可控性 10%），1–5 分。

### 2.1 总表

| 名次 | 编号 | 切口 | 主方案（简称） | 反面方案（简称） | 自评置信度 | 加权分 |  commanding officer 依据 |
|---|---|---|---|---|---|---|---|
| **1** | **R10** | 评测方法论 | **U-E-F-C 四维联合门 + 预注册序贯 stopping rule** | 加权 composite "融合总分"排行榜 | 72 | **4.10** | 证明领域头条数字与噪声不可分；是所有其它工作能否被判定"优秀/差"的前提 |
| **2** | **R3** | 索引检索融合 | **Dedup-Then-Route**（L0 身份去重 + L1 IVF-over-Mean-K） | 直接把 HiRadixTree 当 coarse router | 70 | **3.90** | 离线优先、成本最低、隐瞒收益最大（1.311s→≤350ms） |
| **3** | **R6** | KV 原生压缩 | **DeRoPE-Merge 保真阶梯**（L0→L4） | 直接对已 RoPE 的 K 做 token merging | 72 | **3.725** | 精准补上 KVMem 自己承认没做的 open problem；8× 体积收益 |
| 4 | R2 | I/O 与布局 | **Atom-IR + 单遍 Fused Permute-RoPE 搬运 kernel** | 全栈 GPU 发起 I/O（BaM/SCADA） | 70/55 | 3.575 | 最工程化、最 concretely 可测，但创新增量偏 incremental |
| 5 | R9 | Agent 编程模型 | **KVmadvise 契约层** | 全自动学习 step boundary + 纯 KV 持久化 | 68 | 3.50 | 一旦落地生态空间最大，但 API 类论文发表门槛高 |
| 6 | R4 | 数值正确性 | **Δ-QRoPE**（K 冻结，重映射搬到 Q 侧） | KV 全提 FP16 | 62 | 3.475 | 数学最优雅（置信度 90），但依赖模型用固定频率 RoPE，Tj estimate 部分仅 40 |
| 7 | R8 | 模型架构协同 | **NoPE 化 + Block-Shuffle 后训练 + Index Head 蒸馏** | 直接拿 MLA latent 当"KV 的 JPEG" | 62 | 3.425 | 上行最大（best-paper 级）但方差极大，且已偏离"系统结合"主题 |
| 8 | R5 | 多租户经济 | **席位化分层 KV 配额 + 恢复字节公平调度** | MIG 静态分区 | 62 | 3.30 | DRAM 是隐藏首墙的发现很有价值，但 dedup 假设可能过乐观 |
| 9 | R7 | 存储引擎 | **KVault**（age-band append log + ZNS + 两级索引 + kvadvise） | RocksDB 风格 LSM + compaction | 65 | 3.275 | 分级索引单点成立，消费级 ZNS 不可得压低落地性 |
| 10 | R1 | 调度层融合 | **SBSR**（带 deadline 的 WS-Job + 陈旧预算） | 把 execution view 注册进 HiRadixTree 用 LPM | 62 | 3.10 | 反问方案论证极锋利，但主方案 N 会话 harness 工程量最大 |

### 2.2 逐员要点（保留最有信息量的部分）

**R1 · 调度层**
- 主方案 **SBSR**：把 per-step working-set 切换包装成 `RefreshJob{J=L^phys_t, deadline=下一 step 边界}`，进入与请求队列并列的第二队列按 deadline 排；用 Mean-K 质心漂移做变点预测替代 delay-hit（依据：step 内 KL 0.070 bits vs 跨步 2.59 bits，37.3×，arXiv 2609.04852 §4.2）；Load/Compute 复用 Strata 默认阈值 100。
- 反方案锋芒：execution view 是**任意 block 子集**，两步视图的差别是**集合差异而非公共前缀**，前缀树不可比；且 agent 场景每 context 每 epoch 只有 1 个请求，正是 Strata 实测收益为 **0%** 的 max cache distance 分支（min distance 才 +42%，arXiv 2508.18572 §5.3.3）。
- 自算：64K view churn 25% → ~2.1 GB/refresh；PCIe5 实达 ~48 GB/s → 44 ms；NVMe ~7 GB/s → ~300 ms；500-token 步 ≈6.3 s slack → N=16 时 PCIe 占空比约 76%。**PCIe 先于 GPU 触顶**。

**R2 · I/O 层**
- 主方案 **Atom-IR**：把 128 B = (token, head) 的 FP8 切片作为统一原子；三种布局 = 五维 `(layer, block, token, head, K/V)` 上的仿射映射，一条 descriptor 表达；**单 kernel** 完成 `ld.global.nc → 寄存器内查 sin/cos LUT 旋转 → st.global 直写目标页`，消灭 staging buffer 与第二次 launch。
- 反方案判据：BaM 需整块 GPU 才追平 SPDK；NVMe 最小 4 KB、128 B 原子不可直读；定量 2.2 GiB ÷(128 op×4 KB) = 4676 批，GDS 单批栈延迟 ~160 µs → 单队列仅 ~3.2 GB/s。若实测 ≤2 SM 下同时达 ≥45 GB/s 且 decode 退化 <5%，则推翻。

**R3 · 索引层**（**重点，入选方案一**）
- 关键事实纠正：**KVMem 的 tiled 检索是穷举不是 ANN**（"等价于在 complete candidate index 上求 Eq.(10)"，§5.2）。10M ⇒ 312,500 个 32-token 块、索引 9.5 GiB ⇒ ~32.6 KB/块。
- 主方案 **Dedup-Then-Route**：L0 HiRadixTree 挂 `(content_hash, prefix_ctx_hash, refcount, tier, mean_k_ptr, dev_meta)` 做精确去重；L1 对 Mean-K 做 SPANN 式 IVF（posting 落 NVMe）；**关键 trick**——Eq.(10) 的 softmax 分母对每 `(l,m,h)` 是**与候选集无关的常量**，故可缓存粗排轮的 log-sum-exp 标量（约 10³ 个），裁剪后仅跨 `(l,m,h)` 权重有二阶偏差；每块存离散度 `‖kᵢ−k̄‖`，超阈分裂为双子质心以治池化坍缩。
- 反方案定量：树按 token 前缀（≈时间局部性）划分，与模型是否 attend 近乎独立 ⇒ 剪枝≈随机。N=312,500 保留比例 f，8 个目标块全中概率 = f⁸，要 90% 需 f≈0.987（等于没剪）。

**R4 · 数值层**
- 核心数学洞察：RoPE 旋转群加法且正交，`R(m)ᵀR(n)=R(n−m)` ⇒ delta re-RoPE 在实数域**恒等精确**，误差全部来自 FP8 反复重量化，`‖eₙ‖ ≤ ‖e₀‖+Σ‖ηᵢ‖`，RMS ≈ `ε_q√n`。e4m3 per-tensor 估 `ε_q≈1%‖k‖`，n=16 ⇒ ~4%，若 logit 量级 10–20 ⇒ 抖动 0.4–0.8，**与观测到的 1pt gap 同量级**。
- 主方案 **Δ-QRoPE**：K 进 GPU 后以烘焙位置冻结，每块 j 在 FA kernel 内对 Q tile 乘 `R(p_orig,j − p'_j)`（FP32，仅一次舍入）。额外 flops ≈ `1/32` attention 主循环（32-token block）⇒ <3%。Strata 侧前缀复用路径 Δ≡0 自动退化，**零改动**。
- 弱点：Dynamic-NTK 破坏跨步加法性，Qwen3.6 细节未验证。

**R5 · 多租户**
- 物理常量（自 Table 4 反推）：**Qwen3.6-27B FP8 KV ≈ 32 KiB/token（32 GB/Mtok），Mean-K 索引 ≈1 KiB/token**。
- 瓶颈次序：**host DRAM ≫ NVMe 容量 > GPU HBM > 带宽**。4TB NVMe ÷ 324 GiB = 12 席；host DRAM 128 GB ÷ KVMem 64 GiB cap = **仅 2 席（隐藏首墙）**；GPU 侧增量仅 2 GiB/席 ⇒ 22 席。
- 真实价格锚：Gemini cache storage **$0.50/1M tokens·小时**（2027 起 $1.00）+ read 0.1×（`ai.google.dev/gemini-api/docs/pricing`）；Anthropic write 1.25× / read 0.1× / 5 min TTL。
- 反讽结论：**边际成本比约 150×，而厂商价差只做了 10×** ⇒ cache read 还有约 10× 降价空间。
- 反方案定量：27B Q8 权重 27 GB，H100 最大 MIG 3g.40GB 只能切 2 份 ⇒ 每份 13 GB KV ⇒ 12 席，**劣于统一池的 22 席**，且 MIG 实例间无法共享前缀 KV，dedup 归零。

**R6 · KV 原生压缩**（**入选方案二**）
- 入口是 KVMem 自己的脚注 3：*"KV-native reclamation through block selection, merging, or compaction is an interesting direction for future work"*。
- 量级：10M workspace → NVMe 324.2 GiB ⇒ **~34.8 KB/token**，而原文文本约 4 B/token ⇒ KV 比原文**放大约 8×10³ 倍**。
- 主方案：**L0 raw FP8 324 GiB / L1 KIVI 式 2-bit ≈65 GiB / L2 de-RoPE 后跨块合并 ≈20–32 GiB / L3 Mean-K 9.5 GiB / L4 文本 0.04 GiB**，runtime 按命中频率自动升降级。
- **误差控制契约（最关键一条）**：**永远 RoPE-then-quantize，禁止 quantize-then-rotate**——这是"压缩误差 × 位置重建误差"双重放大的唯一根因。
- 反方案定量：32-token 块内 RoPE 高频维 ω₀≈1 rad/token ⇒ 跨 32 token ≈ 32 rad ≈ **5 个整圈**；两相位随机单位向量平均后模长 = cos(Δθ/2)，随机相位退化为 **~1/√32 ≈ 0.18** ⇒ **高频维振幅损失 82%**；且平均后的 K **不在 RoPE 流形上**，re-RoPE 无法修复。

**R7 · 存储引擎**
- 纠正一处常见误解：Mean-K index **不该放 CXL/PMem**——它是 full-scan streaming，延迟不敏感带宽敏感；实测 CXL 延迟 115→260 ns、带宽 33→24 GB/s，反而慢约 1.4×。CXL 该放 host KV tier。
- 主方案：KVault = age-band 分区 log-structured extent store；单个 0.5–4 MB 顺序 append；**整 zone reset 做 GC，不做 block 级 compaction**；两级索引 L0 PQ 粗筛 ~32 B/block（10M → ~10 MB 常驻 HBM）+ L1 全精度 DRAM。
- 反方案定量：KV 是 WORM，无多版本 ⇒ LSM compaction **只搬不减**；leveled WA ~10× ⇒ 逻辑写 94 MB/s 变成 940 MB/s 落盘，600 TBW 约 8 天写穿。

**R8 · 架构协同**
- 反直觉算术：**index/KV 体积比恒等于 `1/(2B)` = 1.56%（B=block size），与 head 数无关** ⇒ GQA/MLA 的 head 维度**根本不是杠杆**，真正的杠杆是层数 L（CLA2 减半，YOCO 达 8×）。这条推翻了"MLA 能大幅压缩 Mean-K 索引"的直觉。
- 主方案：每 4 层改 NoPE + block-shuffle continued pretraining + `d→64` 小 MLP 做蒸馏式 index head（参数 ~4.2M）。
- 反方案定量：MLA 的 512 维 latent 要同时服务 **128 个 head ⇒ 每 head 等效仅 4 维**（GQA 是 128 维），Mean-K 在 latent 上平均会分辨率坍缩；且 latent FP8 掉 2–5% 的误差被 128 head **共享、无平均相消**，估 SNR 降 √(128/8) = **4×**。

**R9 · 编程模型**
- 入口：虚拟内存成功不只因为有分页，更因为暴露了 `malloc/mmap/madvise/mprotect` 一套**可编程契约**；KVMem 与 Strata 目前都是"系统替你决定"。
- 主方案 API：`kv_open(ws, model_id, fmt_ver)` / `kv_advise(ranges, {WILLNEED,DONTNEED,PIN,COLD,...})`（**non-binding**，可安心默认忽略 → 天然向后兼容今天的 KVMem）/ `kv_pin` / `kv_resident` / `kv_scope`（RAII）/ `kv_checkpoint` / `kv_fork` / `kv_share_readonly` / `kv_recall_log` / `kv_materialize`。
- 反方案定量：step boundary 是隐变量无 ground truth，拿 37.3× 的 KL 跳变当标签是在造噪声监督；一次误判错 pin 20% ≈ 25.6K tokens ≈ 800 个 block 顶掉真实 top-R_b，按 66.5% 集中度估注意力质量损失量级 ~20%，**足以吃掉 pass@1 +4.7pp 的大半**。

**R10 · 评测方法论**（**入选方案一，排第一**）
- **最重要发现**：DeepSWE 的 43.8% vs 48.4% = 28/64 vs 31/64。两比例 z 检验：pooled p̄=59/128=0.4609，SE=√(0.4609×0.5391×2/64)=**0.0881**，z=0.0469/0.0881=**0.53**，双尾 **p≈0.59**；Wald 95% CI = **−12.6pp ~ +21.9pp**（半宽 ±17.3pp）；power ≈ **8%**。任务级 Pass@4 15/16 vs 13/16，Wilson CI [71.7%,98.9%] vs [57.0%,93.4%]，**完全重叠**。
- 补救代价：独立设计需 **~1775 条/臂**；改 task-seed **配对** McNemar（设不一致对率 0.15）需 **~525 对**（≈131 tasks × 4 seeds），CI 半宽 **±3.3pp**。
- 第二发现：**Strata 的 "5× lower TTFT" 是 metric substitution**——正文口径是 Llama-70B+LooGLE"**同 TTFT 下 5× 吞吐**"；9 个配置增益为 3.2/2.6/1.9、3.9/2.1/1.9、5/5/3.75，**中位数 3.2×**（headline 膨胀 1.56×）。
- 审计结论：KVMem 的 claim 应改述为"**效用持平 + 数十倍延迟**"而非"更高任务收益"——这样审稿攻击面立刻收缩。
- 反方案定量：composite 总分是自由参数 ⇒ 5 权重 × 3 配置 = 15 次窥视，α=0.05 下 family-wise 假阳率 1−0.95¹⁵ ≈ **54%**；E[max of 5 N(0,1)] ≈1.16σ ⇒ 表观效应虚高约 16%。

---

---

## 2.5 第二轮圆桌：投票、自曝与否决

> 第一轮是 10 份互不通气的独立报告。第二轮改为**圆桌**：给全体 10 人分发同一份共享简报（第一轮全部结论 + 三份现有方案），要求每人投票、自曝弱点、提合并动议与反对动议。这一节记录的是**只有在互相质证之后才会出现的东西**。

### 2.5.1 投票总表（每人 100 点，共 1000 点）

| 议员 | A/A1 | B | C | D | E | F | G | **第一选择** |
|---|---|---|---|---|---|---|---|---|
| R1 调度 | **32** | 18 | 22 | 10 | 12 | 6 | 0 | **A** |
| R2 I/O | **28** | 22 | 8 | 15 | 10 | 12 | 5 | **A** |
| R3 索引 | **32** | 0 | 6 | 16 | 14 | 10 | 22 | **A** |
| R4 数值 | **30** | 5 | 10 | 25 | 8 | 20 | 2 | **A** |
| R5 多租户 | 26 | 12 | 2 | **32** | 8 | 5 | 15 | **D** |
| R6 压缩 | 25 | 2 | 4 | **30** | 7 | 12 | 20 | **D** |
| R7 存储 | 14 | 20 | 10 | 18 | 6 | **24** | 8 | **F** |
| R8 架构 | **24**(A2=0) | 10 | 18 | 14 | 7 | 5 | 22 | **A1** |
| R9 契约 | 15(仅 A1) | 25 | 12 | 10 | **30** | 8 | 0 | **E** |
| R10 评测 | 22(A2=0) | 8 | 6 | 14 | 12 | **30** | 8 | **F** |
| **合计** | **248** | 122 | **98** | 184 | 114 | 132 | 102 | **A:5 · D:2 · F:2 · E:1** |

**A（消灭摩擦）以 5/10 第一选择、248/1000 点胜出。C（UBBA+）全场最低（98 点，零第一选择）。**

> 说明：R7 提出把 A 拆为 A1（Δ-QRoPE + Fused kernel）与 A2（NoPE 后训练），R8/R9/R10 采纳并给 A2 = 0；R1–R6 未拆分，按整项计入 A1 栏。

### 2.5.2 三条一致否决（负共识比正共识更硬）

**① A2「NoPE 后训练」——零票，被两位不同切口的研究员独立击杀**

- R7（成本）：27B 续训 10B token ≈ `6×27e9×1e10 = 1.62e20 FLOPs ÷ 400 TFLOP/s ≈ 1,125 GPU-h`，吃掉 IFR 全部 1,900 GPU-h 的 **59%**；而绑定约束是 R1 的 2.1 GB/refresh 与 76% 占空比，**A2 一个字节都不减**。
- R8（正确性，更狠）：重训主体权重会让 **324.2 GiB 已缓存 KV 全部失效**——A2 攻击的正是议程 E 想解决的可迁移性本身。且边际收益是 `O(1/L)`：去一层 RoPE 只把 Δ-QRoPE flops 降 0.073pp、保真误差降 `√(63/64)` = 0.8%；要降 29% 须拆掉 32 层。

**② C（UBBA+ 独立立项）——五人独立反对同一个东西，本轮最强负共识**

反对点全部落在**目标函数**而非控制面本身：

| 反对者 | 机制 | 定量 |
|---|---|---|
| R3 | 无辨识性 | 分辨 δ=2pp 需 ≈1427 对/臂（357 tasks）× 数十组合；15 次窥视下 FWER≈54% ⇒ **不可拟合的黑箱控制器** |
| R4 | 无保真项 | `max coverage s.t. bytes` 必把视图填满最便宜的 L2（单位字节覆盖 L3:L2:L1 ≈ 34:12:5）；而 L2 的 ρ≈0.43 正踩 ρ_max≈0.36 悬崖 |
| R6 | 会压到 1-bit | Lloyd-Max 2 级 ρ=0.603 ⇒ `e^{σ²/2}=1.86×` 放大 ⇒ top-16 质量 0.77/(0.77+0.23×1.86) = **64.3%，−12.7pp**，是 +4.6pp 的 2.8 倍反向 |
| R5 | 测不起 | 一个配置点的真实质量 ≈ `525/64 × 300 ≈ 2460 GPU-h`，校验 4 点 ≈ 1×10⁴ GPU-h |
| R10 | 不可判决 | ≈5.4×10⁴ GPU-h，远超预算 |

**一致修正**：目标函数从 `max coverage s.t. bytes ≤ B` 改为 **`min bytes s.t. ρ ≤ ρ_floor ∧ coverage ≥ target`**。保真从"目标里的一项"升格为"硬约束"。

**③ B（跨 session 共享独立立项）——被提出者本人否决**

R5 自曝上轮因果链反了：`K_i = f(tokens_{≤i}, pos i)` ⇒ **只有 prefix-identical 段可合并**，而共享前缀本就 Δ≡0（R4 已证明自动退化）⇒ **跨 session 去重从来不需要 re-RoPE**。B 不是 A 的下游，而是 A 的**平凡子集**。R9 补刀：σ≈0.01–0.05 ⇒ 净回收 ≈13 GiB（4%），却换来一个跨租户信任平面。R10 评级"不可判决"。

### 2.5.3 圆桌新产生的六条结论（第一轮没有）

1. **R7 算出「区带钉死」**：zone=1 GiB、extent≈1.06 MiB（324.2 GiB ÷ 312,500 块反推）⇒ 每 zone≈966 extent。若仅 **1% 块是长命共享块**，zone 全干净概率 `(0.99)^966 ≈ e^{−9.71} ≈ 6×10⁻⁵` ⇒ **99.99% 的 zone 被钉死**。修补：A 成立 ⇒ extent 不可变 ⇒ copy≡move，用一次性 Pin-Evacuator 把共享块 copy-forward 进独立 pin pool，稳态被钉 ≈3%。
2. **R8 的 7.8 GB/s 判据**：`10.24 GB ÷ 1.311 s = 7.8 GB/s`。而 DRAM 理论 ≈0.05 s、PCIe5 有效（48 GB/s）≈0.21 s、冷 NVMe（7 GB/s）≈1.46 s——**观测值只被"NVMe 顺序读"解释**，但论文称 Mean-K index 常驻 host。这个矛盾无人解决，被 R10 列为第一号消歧项。
3. **R4 修正并限定了自己的核心数字**：Δ-QRoPE 的 flops 从 "<3%" 修正为 **4.7%**，且未算 Hopper WGMMA 流水，实测可能回退 **>15%**；同时给出关键定性：令 `K̂ = R(a)k + e_K`，则 Δ-QRoPE 下 `logit = qᵀR(a′−b′)k + qᵀR(−δ)e_K`，**信号项实数域恒等、误差项模长与 δ_j 与 n 均无关** ⇒ Δ-QRoPE 是**运输机制而非保真机制**，它消灭每次重映射的 ε_q，但不改善已有 ρ₀。
4. **R4 给出保真悬崖**：由 top-8/103.1 块占 66.5% mass 反解 logit 标准差 **s≈1.85**，`N` 个噪声块最大值 ≈σ√(2lnN)，令 `3.9×(1.85ρ) < 2.63` ⇒ **ρ_max ≈ 36%**。而 R6 的 L2 估算 ρ≈0.43 正踩线上。R6 接受并把 L2′ 收窄为 G=4 quad-merge + 质心 4-bit ⇒ ρ 0.28–0.32。
5. **R3 自曝「秩反转」**：块 A 含 1 个强相关 token（cos≈1）⇒ 均值分 1/32；块 B 含 32 个弱相关 token（cos≈0.2）⇒ 均值分 0.2 > A；但真实 attention 是 max-like，A 该排前。**均值池化系统性杀死针尖式相关**。R8 判定 LPI 只能"部分治愈不能根治"（静态摘要对 q 是双线性的），并质疑 R3 的 1.19 GiB 隐含 OPQ≈24× 而非 8×。
6. **R10 的裁决表（决定方案四的主终点）**：

| 项 | 可判决性 | 判据成本 |
|---|---|---|
| A 消灭摩擦 | **勉强**（机制可判、质量不可判） | 机制 ~100 GPU-h；质量非劣 3pp 需 1308 对/臂 ⇒ 3.6×10⁴–1×10⁵ GPU-h |
| B 跨 session | **不可判决** | 净回收 4% < 噪声；无多用户 trace |
| C UBBA+ | **不可判决** | ≈5.4×10⁴ GPU-h |
| D LADDER+ | **勉强**（字节可判、质量不可判） | 字节 ~50 GPU-h；代理 ~200 GPU-h |
| E 契约 | **勉强** | 可实现性 ~100–300 GPU-h |
| F IFR-Eval | **可判决**（交付物，非检验） | 1900 GPU-h（已列） |

> **结论：在 1,900 GPU-h 预算内，任务质量终点一律不可判决。** 方案四的主终点必须落在字节、延迟、保真代理这三个可判决量上，质量只报 interim 并标"未验证"。

### 2.5.4 全员自曝清单（第一轮没说的弱点）

| 议员 | 自曝 |
|---|---|
| R1 | 用"Strata 在 max cache distance 收益 0%"论证 delay-hit 可替代，是**负迁移**；命中率/浪费曲线未测 |
| R2 | 128 B 原子漏了 MXFP8 块缩放：每 32 值 1 B ⇒ 实为 **132 B**，非 2 的幂，破坏 128 B 扇区与 DMA 下限 |
| R3 | IVF-over-Mean-K 有秩反转（见上）；10M 时 selectivity = 103/312,500 = **0.033%**，漏召回不可补偿 |
| R4 | flops 实为 4.7%；且 Hopper WGMMA 流水下 Q tile 须每块写回 32 KB 再喂 MMA，可能串行化打断 async pipeline |
| R5 | 上轮"2 席"算错：128 GB(十进制)=119.2 GiB，`119.2/64 = 1.86 ⇒ floor 1 席`；且因果链反了（见 ③） |
| R6 | 升降级用"命中频率"是**错的控制变量**，正确应是 ρ_t·s 相对 top-R 边界的 gap；L1(2-bit) ρ=0.343 ⇒ top-8 边界翻转概率 **20.2%** |
| R7 | 区带钉死（见上），上轮完全没算 |
| R8 | **给不出任何泛化界**；且"我的老师（真实 top-R_b 集合）正是我要替换的对象本身" |
| R9 | `kv_pin` 的 binding 语义在 step boundary 无 ground truth 下**不可执行**——一边否认存在可验证的正确块集，一边定义"保证常驻"的强契约，违约判定无法定义 |
| R10 | 上轮 SE=0.0881 用独立二项，忽略 4 seeds 的 task 内聚类；若 ICC≈0.3 ⇒ design effect≈1.9 ⇒ 有效 SE≈0.121、z≈0.39，power < 8%，**原估 8% 仍偏乐观** |

---

## 3. 方案收敛：四份执行计划

### 排序逻辑说明

为什么把一篇"方法论"（R10）排到所有系统工作之前？三个理由：

1. **因果优先**：三个方案要进实测对比、要淘汰弱者，判定权在评测手里。若评测本身分辨力只有 ±17.3pp，则"方案 A 优于方案 B"永远无法成立，淘汰机制沦为表演。
2. **它本身是高价值可发表成果**：证明一个领域的头条数字与噪声不可分、并给出补救设计（1775 → 525 配对，±17.3pp → ±3.3pp），这是 EuroMLSys / MLSys methodology track 级别的工作，且同行迫切需要。
3. **它是唯一能提前止损的一环**：能在投入 25+ 人日前告诉我们"这条路可能本来就测不出来"。

因此 **方案一 = R10（评测与统计门）+ R3（索引检索）融合**。这两者的融合不是拼接而是互补：R3 的核心风险是"剪枝后 recall 掉了但不知道是否影响下游"，而 R10 的 **F 维（fidelity）** 恰恰要求在有 Full-Context ground truth 的子集上测这个 gap——**R3 最需要的就是 R10 提供的判定能力**；反过来，R3 的离线优先路径（dump Mean-K + 穷举 oracle）是让 R10 那 525 对 rollout 变得可行的最便宜实验底座。

---

## 方案一：IFR —— 可证伪的块级 KV 检索

**代号 `IFR`（Index-First Falsifiable Retrieval）**
**来源：R10（第 1 名）+ R3（第 2 名）融合**

### 3.1.1 一句话命题

> 不宣称"融合后任务成功率更高"——那当前测不出来。**宣称的是：在保持 Full-Context 真值 1pp 以内的前提下，把 10M-token workspace 的检索延迟从 1.311s 压到 ≤350ms，索引常驻从 9.5 GiB 压到 ≤4 GiB，并且这两个数字带 95% 置信区间、可被第三方在同一个 protocol 下复现。**

### 3.1.2 技术组成

| 组件 | 来自 | 内容 |
|---|---|---|
| **U-E-F-C 四维门** | R10 | Utility（配对 CI 不含 0）/ Efficiency（**同 utility 下**同时报 p50/p95 TTFT 与吞吐，MDE 1.5×）/ **Fidelity**（vs Full-Context 真值，top-1 token 一致率 ≥97%、utility gap ≤1pp）/ Cost（$/成功任务 + GiB，MDE 20%） |
| **统计契约** | R10 | 禁用 "up to"，只报中位数 + IQR + 全配置散点；task-seed 配对 + Wilson CI + 聚类稳健 SE；序贯 α 支出（3 looks → 每 look α≈0.0167） |
| **L0 身份层去重** | R3 | HiRadixTree 节点扩展 `(content_hash, prefix_ctx_hash, refcount, tier, mean_k_ptr, dev_meta)`；仅当前缀上下文也相同时共享 KV 页 |
| **L1 IVF-over-Mean-K** | R3 | SPANN 式：centroid 常驻 DRAM，posting 落 NVMe；粗排用 256-token 段均值（由 32-token 均值免费聚合） |
| **Softmax 分母缓存** | R3 | 缓存粗排轮每 `(l,m,h)` 一个 log-sum-exp 标量（~10³ 个），使裁剪只引入二阶偏差 |
| **离散度抗坍缩** | R3 | 每块存 `‖kᵢ−k̄‖`，超阈分裂为双子质心 |
| **索引分层（冷热 posting）** | R3 | 热 posting 在 DRAM，冷 posting 必须大 tile 顺序读（每块 32.6 KB，随机小块读会崩） |

### 3.1.3 执行步骤与里程碑

**Phase 0 · 底座（能否做的前提）—— 2 周，8 人日**
1. 决策点 **G0**：尝试在 **SGLang（Strata 分支）** 上重实现 KVMem 的 selector + block manager。理由：QW3 若不可得，其余全部工作作废；若可得，仍需评估与 SGLang 的迁移成本。
   - **G0 杀点**：两周内拿不到 selector 的 block-score 接口 ⇒ 立即停止，改投开源重实现路线（见 §5 附注）。
2. 建立 `IFR-Eval` harness：block 级物化探针（block_id / tier / bytes / load latency / hit-miss）埋在 KVMem 物化 API + Strata GPU-assisted I/O admission 决策点，在 SGLang RadixCache/metrics 出口汇成统一 trace。

**Phase 1 · Fidelity 先行（可以在没有 GPU 集群时做）—— 3 周，12 人日**
3. 复用已 dump 的 8 条 OpenHands rollout 的 Mean-K + 穷举排序作为 **oracle**（先在 LongMemEval-S：其 ~115K < 256K native window，**Full Context 本身就是 fidelity 真值**——这是整份计划里最优雅的一点）。
4. 实现 L0 dedup + L1 IVF + LSE 缓存。
5. **M1 里程碑**：`recall@8 ≥ 0.95`（相对穷举），同时 **F 维 gap ≤ 1pp**。

**Phase 2 · 效率验证 —— 2 周，8 人日**
6. 在 256K→10M 的 workspace 阶梯上重跑 Table 4 口径；量测 retrieval latency / TTFT / decode tok/s。
7. **M2 里程碑**：`@10M` retrieval latency **1.311s → ≤350ms**（3.7×），索引常驻 **9.5 → ≤4 GiB**，且 decode tok/s 不下降。

**Phase 3 · 配对 utility（决定是否值得继续投入的分水岭）—— 5 周，20 人日**
8. 按 R10 协议跑 **131 tasks × 4 seeds ≈ 525 配对/臂**（约 1500 GPU-h）。
9. **M3 里程碑（Go/No-Go）**：要么 paired bootstrap 95% CI 完全在 0 右侧，要么**出阴性报告并停止**。
   - **关键**：这个里程碑的结果无论正负都要发布。阴性本身是本方案最可能的发表形态。

**Phase 4 · Serving 并发 —— 2 周，5 人日**
10. 并发 8 / 32 / 128 的压力测试（约 200 GPU-h）。
11. **M4 里程碑**：`Strata-only` p99 TTFT 不劣化 > 10%。

**总成本**：约 **53 人日** + **~1900 GPU-h**（约 $4–6k @ $2–3/GPU-h）。
**总工期**：约 **12 周**（Phase 0–4 串行）。

### 3.1.4 验收指标一览

| 指标 | 基线 | 目标 | 判据来源 |
|---|---|---|---|
| retrieval latency @10M | 1.311 s | ≤ 0.350 s | R3 |
| index resident | 9.5 GiB | ≤ 4 GiB | R3 |
| recall@8 vs 穷举 | 1.00 | ≥ 0.95 | R3 |
| **F 维 top-1 一致率** | 未报 | ≥ 97% | R10 |
| **F 维 utility gap** | 未报 | ≤ 1 pp | R10 |
| E 维 p95 增益 | — | ≥ 1.5× 且吞吐不降 | R10 |
| U 维 CI 半宽 | ±17.3 pp | ≤ ±3.3 pp | R10 |

---

## 方案二：DeRoPE-Merge 保真阶梯

**代号 `LADDER`**
**来源：R6（第 3 名），原样保留**

### 3.2.1 一句话命题

> 补上 KVMem 自己承认没做的那件事：**当后备 workspace 用尽时，不许退回文本，而是在 KV 内部降级。目标是 10M workspace 的 NVMe footprint 从 324.2 GiB 压到 ≤40 GiB（≥8×），同时两个基准的任务效用掉点 ≤0.5pt。**

### 3.2.2 技术组成

| 层级 | 体积（@10M） | 内容 | 可否重建衷心 KV |
|---|---|---|---|
| L0 raw FP8 KV | 324 GiB | 不可变权威副本 | ✅ |
| L1 KIVI 式 2-bit | ≈65 GiB | K per-channel / V per-token + FP8 residual 窗口 | ✅（近似） |
| L2 **跨块合并** | ≈20–32 GiB | de-RoPE → ToMe bipartite soft matching → re-RoPE 到规范位置 + **位置残差**低秩编码 | ⚠️ 有损 |
| L3 Mean-K | 9.5 GiB | 只够检索，不可重建 | ❌ |
| L4 文本 | 0.04 GiB | 兜底 | ❌ |

**两条不可违反的规则：**

1. **永远 RoPE-then-quantize，禁止 quantize-then-rotate。** 所有 re-RoPE 必须从不可变 FP8 raw K 出发。这是"压缩误差 × 位置重建误差"双重放大的唯一根因。
2. **合并必须在去位置空间进行。** 先复用 KVMem 已有的 de-RoPE 变换，合并后再 re-RoPE，并额外存位置残差。

**误差修复预算**：每 step 对进入执行视图的合并块，按 CacheBlend 的 HKVD 准则重算 ≤15% token（`arXiv 2405.16444` 证明 10–20% 即可回到 full-prefill 质量）。
**去重前置**：Strata HiRadixTree 先消 exact duplicate（零损），learned compressor 只处理 near-duplicate。

### 3.2.3 执行步骤与里程碑

| Phase | 内容 | 人日 | 里程碑 |
|---|---|---|---|
| P0 | Qwen3-27B + 512K rollout dump raw K；**先证伪再建设**——测"直接平均已 RoPE 的 K" vs "de-RoPE 后平均"的 ‖K‖ 衰减与 recall@64 | 2 | **M0**：验证 §3.2 的位置相位论证成立（否则整个 d 方案二的合并层要重写） |
| P1 | 接 KIVI 内核实现 L1 | 4 | **M1**：2-bit 下 LongMemEval-S 掉点 ≤0.5pt |
| P2 | 实现 L2 合并 + 位置残差 + HKVD 修复 | 4 | **M2**：recall@64 掉 ≤5pp |
| P3 | 端到端 footprint 与 1M/10M 指标 | 2 | **M3**：**324.2 → ≤40 GiB** 且两基准掉点 ≤0.5pt |

**总成本**：约 **12 人日**，是所有三方案里最便宜的。
**关键 ablation**：关位置残差 / 关 HKVD 修复 / 故意 quantize-then-rotate（做反向对照）/ 开关 HiRadixTree 去重 / block 32 vs 128。

### 3.2.4 为什么它排第三而不是第二

它是**单 memory 增益最大的**（8× 体积），三个理由让它落到第三：

1. **位置残差的充分性从未被验证**——作者自评为最高风险；且合并是**不可逆**操作，一旦掉点没有退路。
2. **收益是"节省存储"，而当下真瓶颈到底是不是 NVMe 容量还要打问号**（R5 说 DRAM 才是首墙，R1 说 PCIe 先挂）→ 可能有漂亮的 8× 数字却不解决主要 si瓶颈。
3. **它依赖 KIVI per-channel 量化与 KVMem 的 raw-K 权威 + 流式 stage-out 能否共存**（chunk 边界 vs block 边界冲突）。

但它**成本最低**（12 人日）且命题极干净，因此非常适合作为"快速试错的第一顺位期权"。

---

## 方案三（新增，差异化）：统一字节预算器 UBBA

**代号 `UBBA`（Unified KV Byte-Budget Allocator）**
**来源：本研究新增**，建立在全部 10 份报告的交叉发现之上，**不属任何单一 agent**

### 3.3.1 立项依据：一份被 5 个 agent 独立撞见却没人申报的结构性事实

把散落在 R1/R2/R5/R6/R7 里的观察拼起来，会得到一个**此前没有任何单一 agent 申报过**的结构性事实：

> **在 KVMem × Strata 的组合里，检索索引扫描、NVMe 读取、主机 DRAM 常驻、GPU↔host 搬运，这四件事在争夺同一个物理预算——而它们目前各自独立优化、互不知情。**

具体五份拼图：

| 来源 | 观察到的字节消耗 | 与谁冲突 |
|---|---|---|
| R5 | 32 KiB/token KV + 1 KiB/token index；host DRAM 128 GB ÷ KVMem 64 GiB cap = **仅 2 席** | **DRAM 是隐藏首墙** |
| R6 | 324.2 GiB @10M 的 NVMe footprint 可被压缩 8× | 同一笔预算的存储侧 |
| R3 | 9.5 GiB Mean-K 全量扫描 ≈0.19 s 纯 DRAM 带宽地板 | **索引与 KV 共享同一块 DRAM** |
| R2 | PCIe 只跑到 22%，且"搬运效率"与"搬运量"被分开优化、互不通气 | 字节搬运带宽侧 |
| R1 | N=16 并发时 PCIe 占空比 ~76% → 并发把字节预算竞争放大 | 多会话下的预算竞争 |

而现有系统里，这四件事分属**四个互不通气的控制器**：KVMem 的 retrieval controller（决定扫多少 index）、KVMem 的 tier manager（决定 host/NVMe 准入）、Strata 的 cache controller（决定搬多少）、Strata 的 scheduler（决定什么时候搬）。

### 3.3.2 一句话命题

> 把"每个 agent step 应该花多少字节"变成一个**显式可优化的量**：在同一个 DRAM 容量 + DRAM 带宽 + PCIe 带宽的联合预算下，**联合决定** ① 索引扫描的候选集大小（精度←→召回）、② 各块的保真层级（L0–L4）、③ host DRAM 在 index / hot KV / warm KV 之间的切分、④ 跨并发 session 的配额。目标是：在固定的字节预算下最大化 attention-mass 覆盖率；或在既定覆盖率目标下最小化字节花费。

### 3.3.3 形式化（初版）

设每步预算 ` Budget = (B_DRAM_cap, B_DRAM_bw, B_PCIE_bw) `，决策变量：
- `f ∈ (0,1]`：粗排保留比例（R3 的剪枝比）
- `ℓ_b ∈ {L0..L4}`：每个 block 的保真层级（R6 的阶梯）
- `α ∈ [0,1]`：host DRAM 在 index 与 KV 之间的分配比例
- `x_s ∈ {0,1}`：session s 是否准入（R5 的 seat）

目标（选一）：

```
max    Σ_b AttentionMass(b | f, ℓ, α)          # 固定预算下最大化 coverage
s.t.   Σ_b Bytes(ℓ_b) + Bytes_index(f, α) ≤ B_DRAM_cap
       Bytes_scan(f) / T_step ≤ B_DRAM_bw
       Σ_b Bytes_moved(ℓ_b) / T_step ≤ B_PCIE_bw
```

这是一个**多选择背包 + 覆盖函数最大化**问题，理论上可给出近似算法，工程上可用轻量贪心/拉格朗日松弛在线求解。

### 3.3.4 为什么它与前两个方案真正差异化

| | 方案一 IFR | 方案二 LADDER | **方案三 UBBA** |
|---|---|---|---|
| 层面 | 数据面（索引结构） | 数据面（表示格式） | **控制面（资源分配）** |
| 要不要写 CUDA kernel | 部分要 | 要（接 KIVI） | **几乎不需要** |
| 要不要训练模型 | 不要 | 不要 | 不要 |
| 核心贡献形态 | 一个新结构 | 一个新表示阶梯 | **一个新问题/一个可证明的分配策略** |
| 失败模式的性质 | 性能不达标 | 精度掉点 | **只能是不够好，不会崩** |

UBBA 的最大优势：**它是一个纯控制面工作，不需要任何新的 kernel 或训练，因此技术风险最低、可随时修正**；同时它把 R6 的"8× 能不能换成任务收益"、R3 的"剪枝到什么程度值得"这几件悬而未决的事变成了一个可计算的最优化问题。

### 3.3.5 执行步骤与里程碑

| Phase | 内容 | 人日 | 里程碑 |
|---|---|---|---|
| P0 | 建立字节成本模型：把 Table 4 的容量/延迟/吞吐数字反推成 per-byte 成本参数；跑 3 组 microbench 标定 `B_DRAM_bw`、`B_PCIE_bw`、**实机 mean-k 扫描带宽** | 6 | **M0**：模型预测值与实机 latency 误差 <20% |
| P1 | 实现求解器（先贪心 + 拉格朗日松弛，再探近似界）；**离线回放**：用 dump 的 rollout 数据离线重放不同预算分配 | 8 | **M1**：离线回放显示同等 coverage 下字节花费降 ≥25% |
| P2 | 接入真实 stack：挂在 Strata scheduler 之上，作为 admission + 保真层级决策器 | 10 | **M2**：端到端在 1M workspace 上真实 latency 下降 ≥20% 且任务指标不退 |
| P3 | 并发扩展；加入 `x_s` 准入变量，验证 R5 的"2 席墙"能否被打破 | 8 | **M3**：单 H100 并发 session 数 > 12 |

**总成本**：约 **32 人日**（其中 P0–P1 共 14 人日基本是离线工作，GPU 需求极低 → **这是最省钱的可以快速启动的方案**）。

### 3.3.6 风险

1. **理论贡献可能不够深**：如果只是个贪心求解器，会被审成"就是个启发式"。缓释：P1 必须attack 近似算法界的分析。
2. **字节模型与真实服务的拟合度不足**：模型标不准就全废。M0 的 20% 误差门槛就是这里的守门员。
3. **可能被方案一的 fuse 数据消化**：如果 IFR 的 index 剪枝已经把选 leaves from 预算节省做完，UBBA 的边际价值会缩水 → 建议 P0–P1 尽早启动以便早判断。

---

## 方案四（圆桌收敛）：CASA

**代号 `CASA`（Canonical Atom Store Architecture）**
**来源：第二轮圆桌投票收敛（A/A1 以 5/10 第一选择、248/1000 点胜出），非任何单一 agent 的方案**

### 3.4.1 立项依据：它解决的是前三条方案共同缺的那一层

前三份方案分别在**索引结构**（IFR）、**表示格式**（LADDER）、**资源分配**（UBBA）上做文章，但三者共享一个从未被质疑的前提：**KV 字节是"绑定位置的可变状态"**。

圆桌上 A 胜出的真正原因不是"它能加速"，而是 R2 那句被多人独立重复的观察：

> **A 一旦成立，搬运、去重、压缩、跨 session 共享这四件事，第一次落在同一份字节上——各自退化为同一对象的不同视图。**

具体四份拼图：

| 来源 | A 成立后发生什么 |
|---|---|
| R2 | 原子从"位置可变"升级为**规范化不可变字节** ⇒ 可内容寻址、可去重、可跨 session 共享 |
| R4 | K 冻结 ⇒ 重映射误差与搬运次数 n **脱钩**（`ρ(n) = ρ₀` 而非 `√(ρ₀² + nε_q²)`），且 `n* ≈ 250` 的重建悬崖消失 |
| R7 | extent 不可变 ⇒ copy≡move ⇒ 一次性 Pin-Evacuator 可解区带钉死；posting list 变 append-only ⇒ 永不 compaction |
| R9 | CAS manifest 使跨模型版本迁移第一次有了可讨论的载体（索引常迁移 + 字节按需迁移） |

所以方案四是**语义层**的工作：改变"一个 KV 字节意味着什么"。这是与前三份正交的第四条轴。

### 3.4.2 一句话命题

> **不改模型、不追求"任务成功率更高"（那在 1,900 GPU-h 内不可判决）。把 KV 字节的语义从"绑定位置的可变状态"改为"内容寻址的规范化不可变对象"：K 以烘焙位置永久冻结，位置校正全部搬到瞬态的 Q 侧；于是搬运、去重、压缩、共享四件事收敛到同一份字节上。主终点是三个可判决的量：NVMe 足迹 324.2 → ≤42 GiB、10M 检索延迟 1.311 s → ≤0.70 s、保真 ρ ≤ 0.32；任务质量只报 interim 并标"未验证"。**

### 3.4.3 技术组成

| 组件 | 来自 | 内容 |
|---|---|---|
| **Canonical Atom** | R2 | 原子 = 128 B 的 (token, head) FP8 切片；**MXFP8 块缩放旁路为 (block, head) 侧表**（312,500×8×4 B ≈ 10 MB 常驻 host）使原子严格回到 128 B；原子 ID = `FNV(model_ver, layer, head, token_hash, p_orig)` |
| **五维仿射 descriptor** | R2 | 三种布局（GPU layer-first / host page-first / KV chronological）= `(layer, block, token, head, K/V)` 上的仿射映射，编译进 scatter descriptor，GPU kernel 只做 gid→索引 + ld/st |
| **K-Freeze + Q-Remap** | R4 + R2 修正 | K 以烘焙位置 p_orig 冻结、永不重写。重映射搬到 Q 侧，但**不按 R4 原案在 Q tile 上做**（BLOCK_M=128 时需 36 TFLOP/s ≈ H100 FP32 峰 54%，不可用），而是放在 **FA 的 K-tile prologue 内一次性寄存器旋转**，并复用缓存的 FP8 scale（逐对旋转保范 ⇒ `‖R(Δ)k‖ = ‖k‖`，scale 不变、无需重定标）⇒ **0.144 TFLOP/s ≈ FP32 峰 0.2%** |
| **FP64 角规约** | R4 | δ~10⁶、θ_max=1 ⇒ FP32 相位误差 `≈10⁶×2⁻²⁴ ≈ 0.06 rad`，与 ε_q 同量级 ⇒ 必须用 FP64 做角规约 |
| **ρ_floor 硬约束** | R4 | 保真悬崖 `ρ_max ≈ 36%`。任何档位必须满足 **`ρ ≤ 0.3ρ_max ≈ 11%`**（混档场景）或 `ρ ≤ 0.32`（单一 L2′ 档） |
| **Tier-bias 去偏** | R6 | 混档时施加解析可算的 logit 偏置 `b_t = −(ρ_t·s)²/2`（s≈1.85）。同档内 `e^{σ²/2}` 在 softmax 中自动抵消，**仅混档暴露**——而混档正是阶梯常态 |
| **L2′ 收窄** | R6 | G=4 quad-merge + 质心 4-bit ⇒ ρ 0.28–0.32 < ρ_max；bytes/token 与 L1 同档：65×(4/2)×(1/4) = **32.5 GiB** |
| **BandCAS 布局** | R7 | 布局键从"年龄"换成"死亡相关" `(tier, lease_epoch, tenant_cohort)`；`WA = 1/d`，d=0.9→1.11，d=0.4→2.5。与 LADDER 联动：tier 降级即同步死亡（一次元数据 delete 释放 259 GiB，零读放大） |
| **Pin-Evacuator** | R7 | 一次性 copy-forward 解区带钉死，稳态被钉 ≈3% |
| **索引：MVR vs LPI 裁决** | R3 / R8 | Phase 3 三臂对比后按"等字节 recall@8"裁定。MVR = k̄ + 2 条最大余弦残差行 + OPQ；LPI = p×r FP8 探针（p=8, r=16，D=128，**字节零回归**），由 d→64 小 MLP 蒸馏产出（~150 GPU-h），多探针扫描与现有 Mean-K tiled scan **同形同参数**，仅 epilogue 多 log₂p 次比较 |
| **6 个 API** | R9 | `kv_open(model_ver, budget{residency, restore}, ρ_floor)` / `kv_advise(ctx, span, cls, ε)`（非绑定，`H_max = ln(1−ε)/ln(0.75)`）/ `kv_resident(ctx, span, ℓ)`（绑定，返实测 ρ，b_t 内部施加）/ `kv_scope(ctx)` RAII / `kv_share_readonly(ctx, span, prefix_root)` / `kv_checkpoint(ctx)`→CAS manifest |
| **预注册 stopping rule** | R10 | 见 §3.4.6 |

### 3.4.4 三条不可违反的约束

1. **永远 RoPE-then-quantize，禁止 quantize-then-rotate。** Δ-QRoPE 只消灭"每次重映射的 ε_q"，**不改善**已有的 ρ₀。这是"压缩误差 × 位置重建误差"双重放大的唯一根因。
2. **K 一旦冻结不可重写。** 任何 tier 降级只影响读取路径的保真，不改 canonical 字节。写入路径只允许一次 canonicalize（`R(p_can − p_i)`）。
3. **ρ ≤ ρ_floor 是硬约束，不是目标函数里的一项。** UBBA 式的最优化只能在 `ρ ≤ ρ_floor ∧ coverage ≥ target` 之下最小化字节——这是本轮五人独立反对后的一致修正（§2.5.2 ②）。

### 3.4.5 执行步骤与里程碑

**Phase 0 · G-CAS 消歧门禁 —— 2 周 / 8 人日 / ~200 GPU-h**

1. **G-CAS-1（第一号，必须先做）检索瓶颈分解。** 五档 workspace（256K/512K/1M/4M/10M）× 30 重复，Nsight 分解 gather / H2D / scatter / re-RoPE 占比 + 测纯 H2D 带宽。
   - **算术预警**：`10.24 GB ÷ 1.311 s = 7.8 GB/s`。而 DRAM 理论 ≈0.05 s、PCIe5 有效(48 GB/s) ≈0.21 s、冷 NVMe(7 GB/s) ≈1.46 s。**观测值只被"NVMe 顺序读"解释，但论文称 index 常驻 host。** 这个矛盾不解决，后续所有延迟目标都是空的。
   - 判据：搬运占比 >70% 且纯带宽 <12 GB/s ⇒ 支持"带宽受限"；H2D >30 GB/s 且 stall ≥70% ⇒ 支持 R1 的"PCIe 占空比 76%"。两组占比差 95% CI 须 ≥15pp。
2. **G-CAS-2 Δ-QRoPE 回退实测。** 三臂（Full / 现状 delta re-RoPE / fused K-tile）× 3 档 × 20，终点 TTFT。判据：fused 相对现状加速 95% CI 下界 >1.15 且对 Full 回退上界 <5% ⇒ 成立；下界 <0.85 ⇒ R4 的"回退 >15%"成立。
3. **M0**：G-CAS-1 给出明确瓶颈归属 + G-CAS-2 回退上界 <5%。

**Phase 1 · Canonical 字节层 —— 3 周 / 12 人日 / ~150 GPU-h**

4. Atom-IR（含缩放侧表化）+ 五维仿射 descriptor + 写入路径一次性 canonicalize kernel。
5. K-Freeze 元数据结构：`(p_orig, frozen_K_ptr, last_used, tier, ρ_measured)`，per-block 4 B δ_j 落盘（312,500×4 B = 1.25 MB）。
6. **M1**：@1M workspace 检索延迟下降 ≥1.5×，且 LongMemEval-S 掉点 ≤0.5pt。

**Phase 2 · 保真阶梯并入 —— 2 周 / 8 人日 / ~200 GPU-h**

7. 接 KIVI 实现 L1（2-bit）；实现 L2′（G=4 quad-merge + 质心 4-bit）；接入 tier-bias `b_t`；1/1000 块留 FP8 golden 做 ρ 在线标定。
8. **M2**：NVMe 足迹 @10M **324.2 → ≤42 GiB**（index 降至 64-token 质心粒度 4.75 GiB 时 ≤37 GiB），且 ρ 实测 ≤0.32、top-8 覆盖率掉 ≤2pp。

**Phase 3 · 索引裁决 —— 2 周 / 8 人日 / ~300 GPU-h**

9. MVR vs LPI vs Mean-K 的等字节 recall@8 三臂对比；以穷举排序为 oracle。
10. **M3**：胜出方案在 ≤4.77 GiB 下 recall@8 不低于 9.5 GiB Mean-K 减 1pp；top-8 mass 覆盖率掉 ≤1pp。

**Phase 4 · 共享与契约 —— 2 周 / 8 人日 / ~100 GPU-h**

11. **只做 prefix-identical 共享**（R5/R9 一致结论）；实现 Pin-Evacuator；落地 6 个 API + provenance chain-of-custody（atom_id / producer_session / model_ver / prefix_root / p_orig / tier / 实测 ρ）。
12. **M4**：prefix-identical 重放 Δ 应 bit-exact（Δ-QRoPE 残差 ≤1e-5）；稳态被钉 zone ≤8%。

**总成本**：约 **44 人日 + ~830 GPU-h**（≈$1.7–2.5k @ $2–3/GPU-h）。
**总工期**：约 **11 周**。

> 对比：方案一 53 人日 / 1,900 GPU-h；方案四便宜 43%（GPU-h），且**主终点全部落在 R10 判定的"可判决"区间内**。

### 3.4.6 验收指标与预注册 stopping rule

| 指标 | 基线 | 目标 | 判据来源 | 可判决性（R10） |
|---|---|---|---|---|
| 10M 检索延迟 | 1.311 s | **≤0.70 s** | R8 判据 1（D=64 ⇒ ~0.66 s） | 可判决，~80 GPU-h |
| NVMe 足迹 @10M | 324.2 GiB | **≤42 GiB** | R6 修正 L2′ | 可判决，~50 GPU-h |
| index 常驻 @10M | 9.5 GiB | **≤4.77 GiB** | R3/R8 裁决 | 可判决，~150 GPU-h |
| ρ（保真） | 未报 | **≤0.32**（混档硬约束 ≤0.11） | R4 悬崖 + R6 tier-gate | 可判决，~200 GPU-h |
| top-8 mass 覆盖率 vs 穷举 | 1.00 | ≥0.99 | R3 | 可判决 |
| Δ-QRoPE 相对 Full 回退 | — | **≤5%（上界）** | R10 消歧 ② | 可判决，~120 GPU-h |
| 任务质量（Pass@1） | 43.8% | **不设目标，只报 interim** | R10：非劣 3pp 需 1308 对/臂 ⇒ 3.6×10⁴ GPU-h | **不可判决** |

**预注册 stopping rule（若后续追投质量终点）**：

- Primary：配对 Pass@1 风险差 Δp（task×seed 配对，π_d 预注册 0.15，look1 重估）。双侧 α=0.05，target δ=5pp，power 80% ⇒ `N_max = 1.177/0.05² = 471 对/臂`（118 tasks × 4 seeds）。
- Looks at t = 0.25 / 0.50 / 0.75 / 1.00 ⇒ n = 120 / 240 / 360 / 471。
- α 支出用 **Lan-DeMets OBF**：`α(t) = 2[1 − Φ(1.96/√t)]` ⇒ 累计 α = 0.00009 / 0.0056 / 0.0239 / 0.05，边界 z = 3.92 / 2.77 / 2.26 / 1.96。
- **胜**：z_k ≥ 边界且 Δp̂ > 0，终 CI 下界 > 0。**负**：look≥2（n≥240）时条件功效(δ=5pp) < 20%，或 Δp̂ ≤ 0 且 CI 上界 < +2pp ⇒ 停。**害**：Δp̂ < −5pp 或 CI 上界 < −2pp ⇒ 停。
- 仅 primary 享 α；TTFT / 字节 / ρ 为描述性指标（Holm 不分 α）。
- **必须在注册报告里写明**：`N_max ≈ 14.7 点 × 2460 ≈ 3.6×10⁴ GPU-h` 超预算 ⇒ 预算内只能在**代理终点**走完 looks，质量最多报 t ≤ 0.5 的 interim 并标"未验证"。

### 3.4.7 与前三份方案的关系：**不是替代，是地基 + 一次吸收**

| | 方案一 IFR | 方案二 LADDER | 方案三 UBBA | **方案四 CASA** |
|---|---|---|---|---|
| 层面 | 数据面·索引结构 | 数据面·表示格式 | 控制面·资源分配 | **语义面·字节语义** |
| 是否依赖 A1 | 不依赖 | 不依赖 | 不依赖 | **就是 A1** |
| A1 成立后的变化 | 索引 posting 变 append-only，永不 compaction（R3 自认） | 阶梯的读路径旋转 2→1 次；L2′ 收窄后 ρ 回到悬崖内侧 | **目标函数被替换**（§2.5.2 ②） | — |

**对方案三的处理建议**：UBBA 的**控制面形态**有价值，但它的**原目标函数被五人独立否决**。因此**方案三不再独立立项**，其求解器并入方案四 Phase 2，作为 `min bytes s.t. ρ ≤ ρ_floor ∧ coverage ≥ target` 的在线求解器。这不是淘汰 R 的路线，而是它的问题形态被正确的约束集接管。

### 3.4.8 风险与退役 Gate

| Gate | 时点 | 判据 | 动作 |
|---|---|---|---|
| **G0** | 第 2 周 | QW3 不可得且 SGLang 重实现不可行 | 方案四转纯离线分析（Phase 0–1 仍可做，需自研 block-score dump） |
| **G-CAS-1** | 第 2 周 | 瓶颈归属无法在 95% CI ≥15pp 上区分 | 延迟目标作废，方案四退化为纯存储/保真工作（仍保留 8× 字节收益） |
| **G-CAS-2** | 第 2 周 | fused K-tile 回退 >15% 或加速 CI 下界 <0.85 | **K-Freeze 核心路线终止**，预算转投方案二 LADDER |
| **M1** | 第 5 周 | @1M 检索下降 <1.5× 或掉点 >0.5pt | 方案四降级为离线分析工具，不追求系统论文 |
| **M2** | 第 7 周 | ρ 实测 >0.375 | 砍掉 L2′，接受 4.35×（74.5 GiB），重设目标并出"8× 不可达"的阴性结论 |
| **M4** | 第 11 周 | prefix-identical 重放非 bit-exact | 共享支路终止（R9 的信任平面论证成立） |

**Top-3 风险**

1. **G-CAS-1 的三界矛盾无解**（7.8 GB/s 不被 DRAM/PCIe 解释、只被 NVMe 解释，但与"index 常驻 host"冲突）。这是方案四最大的单点风险，也是它最值得先做 Phase 0 的理由。
2. **Δ-QRoPE 在 Hopper WGMMA 流水下的真实回退未知**（R4 自曝 50% 置信度）。若 >15%，K-Freeze 的经济性崩塌。
3. **LPI 无泛化界**（R8 自曝）：探针在训练长度之外的 10M 布局上能否保住 recall，只能事后测。且"老师（真实 top-R_b）正是被替换的对象本身"，存在自指风险。

---

## 4. 横向对比矩阵

> 评分 1–5，★ 越多越好。括号为该维度评语。

| 维度 | **方案一 IFR**<br>(R10+R3 融合) | **方案二 LADDER**<br>(R6) | **方案三 UBBA**<br>(第一轮新增) | **方案四 CASA**<br>(圆桌收敛) |
|---|---|---|---|---|
| **技术风险** | ★★★☆☆ (3)<br>思想 risk 低；主风险是 QW3 闭源（G0 杀点）+ 525 对 rollout 能否跑得起 | ★★☆☆☆ (2)<br>**位置残差充分性未验证**；合并不可逆、无退路；KIVI 与 raw-K 权威可能冲突 | ★★★★☆ (4)<br>纯控制面，无 kernel 无训练；失败模式是"不够好"而非"崩掉" | ★★★☆☆ (3)<br>**G-CAS-1 三界矛盾未解**（7.8 GB/s 不被 DRAM/PCIe 解释）；Δ-QRoPE 在 Hopper WGMMA 下的真实回退未知（R4 自曝 50% 置信度）；但**每个 Gate 都在第 2 周就触发**，止损最早 |
| **实现成本** | ★★☆☆☆ (2)<br>53 人日 + ~1900 GPU-h + 12 周；**全场最贵** | ★★★★★ (5)<br>**12 人日**，最便宜；但需 GPU 做 dump | ★★★★☆ (4)<br>32 人日，其中 14 人日几乎不需要 GPU | ★★★★☆ (4)<br>**44 人日 + ~830 GPU-h**；比方案一省 43% GPU-h，且主终点全在可判决区间 |
| **性能预期** | ★★★★☆ (4)<br>retrieval 3.7× ↓、index 2.4× ↓；但端到端增益依赖瓶颈判断——若 DRAM 才是首墙，检索加速可能不落在主瓶颈上 | ★★★★★ (5)<br>**NVMe footprint 8× ↓**（324→40 GiB），单点增益最大 | ★★★☆☆ (3)<br>预期 20–25%，靠分配而非质变；上行有限但确定性高 | ★★★★★ (5)<br>**同时拿到两条线**：字节 324→≤42 GiB（吸收 LADDER）+ 检索 1.311→≤0.70 s（吸收 Strata 的 I/O 强项）；且 `ρ(n)=ρ₀` 消除了 `n*≈250` 的重建悬崖 ⇒ I/O 放大 65→324 GiB 的 5× 回退消失 |
| **可发表性与影响力** | ★★★★★ (5)<br>**双档可能**：F 维首个基准的自画像 claim 修正 + 索引方法本身；若出阴性结果，方法论 clout 反而更高 | ★★★★☆ (4)<br>精准命中 KVMem 明列的 open problem + 8× 数字很能打；但可能被审成"压缩工程" | ★★★★☆ (4)<br>提出一个新问题形态（跨层字节预算）；但**原目标函数被五人独立否决**，形态需换约束集才能存活 | ★★★★★ (5)<br>命题是本领域的**地基级**改动（"KV 字节意味着什么"）；且自带一条独立的 negative result 线——无论 G-CAS-1 的瓶颈归属落在哪一边，结论都可发表 |
| **后续迭代空间** | ★★★★★ (5)<br>Eval protocol 一旦立住，**本领域所有后续工作都得引用**；IFR 的索引层还可接 R2/R7 | ★★★★☆ (4)<br>向下衔接 R7 存储引擎、向上衔接 R3 索引，是 scaffold 的一环 | ★★★★★ (5)<br>天然吸收其它方案的成果 | ★★★★★ (5)<br>**它是前三者的共同底座**：A1 成立后 IFR 的 posting 变 append-only、LADDER 的读路径旋转 2→1 次、UBBA 换约束集后并入 Phase 2。任何一条出成果都增大 CASA 的价值 |
| **依赖闭源程度** | ★★☆☆☆ (2) 高依赖 | ★★★☆☆ (3) 中 | ★★★★☆ (4) 低 | ★★★☆☆ (3) 中（canonicalize kernel 需接入引擎，但 Phase 0–1 可用 dump 数据离线做） |
| **最短出成果路径** | Phase 1（12 人日，几乎不要 GPU）即可出 fidelity 结果 | P0–P1（6 人日）即可出 tipping 验证结果 | P0–P1（14 人日，基本离线）即可出模型与离线回放结论 | **Phase 0（8 人日 / ~200 GPU-h）即可出瓶颈归属结论**——这是全场最早的、且独立可发表的成果 |
| **加权总分** | **3.90** | **3.70** | **4.00** | **4.30** |

> 加权方式：技术风险 20% / 实现成本 15% / 性能预期 20% / 可发表性 25% / 迭代空间 20%
> （依赖闭源与最短路径作为参考列，不计入总分）

### 4.1 逐方案优势劣势

**方案一 IFR**
- ✅ 优势：唯一能同时产出"方法主张 + 领域校准"两条线的方案；Phase 1 的低 GPU 特性让它能最早出可验证结果；一旦 eval protocol 立住就是可持续引用资产。
- ❌ 劣势：最贵、最长；U 维/M3 的 525 对 rollout 是最大的执行不确定性；高度依赖 G0（QW3 可得性）；检索加速在 DRAM-wall 既定情况下可能不是主瓶颈 → "压了 3.7× 端到端却只快了一点点"。

**方案二 LADDER**
- ✅ 优势：成本极低、收益单点极大（8×）、命题干净（"KV 内部的降级而非退回文本"）、ablation 设计天然漂亮（quantize-then-rotate 反向对照）。
- ❌ 劣势：位置残差是**未经检验的赌注**，且不可逆；8× 数字可能落在非瓶颈处；需要先做 P0 证伪才能决定要不要继续，有"先付款后看货"的观感。

**方案三 UBBA**
- ✅ 优势：技术风险最低、几乎不依赖闭源、**天然兼容并吸收其它方案的成果**、14 人日就能出离线结论。
- ❌ 劣势：理论深度可能是弱项，面临"这只是个启发式"的审稿攻击；字节模型的保真度是整个方案的支点，标不准就失败；**原目标函数被 R3/R4/R6/R5/R10 五人独立否决**（无保真项 ⇒ 背包必填满最便宜档 ⇒ −12.7pp）。

**方案四 CASA（圆桌收敛）**
- ✅ 优势：**唯一同时覆盖字节与延迟两条主线的方案**；命题是地基级且正交于前三条；Phase 0（8 人日）就能产出独立可发表的瓶颈归属结论；所有 Gate 都在第 2 周触发 ⇒ 止损最早；K-Freeze 使保真误差与搬运次数脱钩（`ρ(n)=ρ₀`），这是一条**可形式化证明**的性质。
- ❌ 劣势：G-CAS-1 的三界矛盾（7.8 GB/s）是最大单点风险，若瓶颈归属测不出来，延迟目标整条作废（但字节线仍在）；Δ-QRoPE 的 kernel 实现难度被 R2/R4 双双上修（从"融合"变成"必须放 K-tile prologue"），需要真正懂 Hopper 流水的人；CASA 单独不产生质量增益，**必须挂靠 IFR 的 eval backbone 才能被判定**。

---

## 5. 推荐结论与淘汰机制

### 5.1 推荐（已随圆桌更新）：**CASA 主线 + IFR 判定层捆绑，UBBA 换约束集后并入，LADDER 作期权**

> 本节已在第二轮圆桌后更新。方案四 CASA 以加权 4.30 居首，取代原"方案一主线"的排序——但**这不意味着方案一被降级**，见下方第 1 条。

理由四条：

1. **方案四 CASA 应成为新的主线，方案一的 eval backbone 必须同时启动。** 这不是替代关系：CASA 单独不产生质量增益，**必须挂靠 IFR 的 U-E-F-C 四维门才能被判定**。所以正确表述是——**CASA 是新的系统主线，IFR 的评测协议是它的判定层**，二者捆绑，缺一不可。
2. **CASA 之所以排在 IFR 之前**，有三个理由：① 圆桌投票 5/10 第一选择、248/1000 点；② 它是唯一同时覆盖**字节**（324→≤42 GiB）与**延迟**（1.311→≤0.70 s）两条主线的方案；③ 它的所有 Gate 都在第 2 周触发，**止损最早**——而 IFR 的 M3（525 对 rollout）要到第 10 周才知道结果。
3. **方案三 UBBA 不再独立立项。** 它的控制面形态有价值，但原目标函数 `max coverage s.t. bytes` 被 R3/R4/R6/R5/R10 五人独立否决（无保真项 ⇒ 背包必填满最便宜档 ⇒ −12.7pp）。其求解器并入 CASA Phase 2，在 `min bytes s.t. ρ ≤ ρ_floor ∧ coverage ≥ target` 的约束集上运行。**这是约束集的接管，不是路线淘汰。**
4. **方案二 LADDER 的 L2′ 已被 CASA Phase 2 吸收**（收窄为 G=4 quad-merge + 质心 4-bit，ρ 0.28–0.32）。其独立价值退化为"12 人日的最便宜试错期权"，但仍建议先跑 P0（2 人日）证伪。

**投注配比建议（更新）**：方案四 CASA 45% / 方案一 IFR（eval backbone）30% / 方案二 LADDER（P0 期权）10% / 保留 15% 机动（留给 Phase 0 消歧结果决定投向）。

### 5.2 统一评测 backbone（四方案共用）

方案一的 U-E-F-C 四维门**必须同时套在四份方案上**，否则无法横向比较。四条硬约束：

1. 每份方案必须同时报 U / E / F / C 四维，缺一维视为未通过验收。
2. 禁用 "up to"，只报**中位数 + IQR + 全配置散点**。
3. fidelity 一律在 Full-Context 可得的子集上取真值：`LongMemEval-S` 的 ~115K < 256K native window，是最干净的入口。
4. **新增（圆桌裁决）**：R10 已判定**任务质量终点在 1,900 GPU-h 内不可判决**（非劣 3pp 需 1308 对/臂 ⇒ 3.6×10⁴ GPU-h）。因此四份方案一律**不得把 Pass@1 设为主终点**；主终点必须在 {字节, 延迟, ρ, 覆盖率} 这些可判决量中选取，质量只报 interim 并显式标注"未验证"。

### 5.3 退役 / 淘汰判据（写死，事先承诺）

| Gate | 时点 | 判据 | 动作 |
|---|---|---|---|
| **G0** | 第 2 周 | 拿不到 KVMem selector 的 block-score 接口 | 方案一转开源重实现路线；若再 2 周不可行 → **方案一降级**，预算转投方案三 |
| **M0 (IFR)** | 第 5 周 | `recall@8 < 0.90` 或 F 维 gap > 1pp | **方案一降级**：保留 eval backbone（它仍有独立价值），索引层工作停止 |
| **M0 (LADDER)** | 第 2 周 | "de-RoPE 后平均" 相比 "直接平均已 RoPE" 无显著优势 | **方案二整体终止**（2 人日即止损） |
| **M1 (LADDER)** | 第 6 周 | 2-bit 下 LongMemEval-S 掉点 > 1.5pt | **方案二终止** |
| **M3 (IFR)** | 第 10 周 | Paired bootstrap 95% CI 含 0 且 E < 1.5× 且 F gap > 1pp | **立即停止并出阴性报告**（这是事先承诺的 stopping rule，不许再调参挽救） |
| **M0 (UBBA)** | 第 4 周 | 字节模型预测 latency 与实机误差 > 20% | **方案三终止**（模型是它的全部支点） |
| **M2 (UBBA)** | 第 10 周 | 端到端真实 latency 下降 < 20% | **方案三降级**为离线分析工具，不追求系统论文 |
| **G-CAS-1 (CASA)** | 第 2 周 | 瓶颈归属无法在 95% CI ≥15pp 上区分 | 延迟目标作废；**CASA 退化为纯存储/保真工作**（8× 字节收益仍保留） |
| **G-CAS-2 (CASA)** | 第 2 周 | fused K-tile 回退 >15% 或加速 CI 下界 <0.85 | **K-Freeze 核心路线终止**，预算转投方案二 LADDER |
| **M1 (CASA)** | 第 5 周 | @1M 检索下降 <1.5× 或掉点 >0.5pt | **CASA 降级**为离线分析工具，不追求系统论文 |
| **M2 (CASA)** | 第 7 周 | ρ 实测 >0.375 | 砍掉 L2′，接受 4.35×（74.5 GiB），并出"8× 不可达"的阴性结论 |
| **M4 (CASA)** | 第 11 周 | prefix-identical 重放非 bit-exact | 共享支路终止（R9 的信任平面论证成立） |

**关于"agent 淘汰"**：对应 agent 应读作"该研究路线退役"。具体做法：
- 触发上述任一 Gate 的**研究路线立即冻结**；其技术债务与未竟设想转入公共 backlog（不销毁，后续可能被别的路线解锁）。
- **若触发者是 G0（QW3 不可得）这类外部依赖失败，不应归责于对应 agent**——这是指令要明确的一条例外规则。

### 5.4 GitHub 发布闭环

1. **第 3 周**：开源 `IFR-Eval` harness 与 byte-level trace schema（无论正负都必须发布，这是长期资产）。
2. **第 2–3 周**：开源 **G-CAS-1 瓶颈分解工具**与五档 workspace 的 Nsight trace——**无论瓶颈归属落在哪一边，结论都可发表**（这是全场最早的独立可发表成果）。
3. **第 6 周**：发布 LADDER P0/P1 的证伪结果与 raw-K dump 工具。
4. **第 12 周**：按 M3 / M2(CASA) 结果发布——**阳性则投 MLSys/OSDI 周期的论文 + artifact；阴性则投 methodological 负结果报告**。两种情况都走公开 repo，禁止"阴性不出结果"。
5. 全部 artifact 需附带 reproduction script 与 raw record；harness 默认带 `--seed-sweep N` 参数，强制 n≥4 才能宣布任何 end-to-end 增益。

### 5.5 一句话总结（已随圆桌更新）

> **第一轮的答案**是：这个领域现在缺的不是一个更快的系统，而是一个能分辨"更快"与"运气"的测量方法。
> **第二轮圆桌把答案推进了一步**：在测量方法之上，还缺一个共同的地基——**让 KV 字节从"绑定位置的可变状态"变成"内容寻址的规范化不可变对象"**。K 一旦冻结、位置校正搬到瞬态的 Q 侧，搬运、去重、压缩、共享四件事就第一次落在同一份字节上，各自退化为同一对象的不同视图。这是圆桌 5/10 第一选择收敛出的方向，也是本报告对"该做什么"的最终回答。
>
> 同时必须承认：**CASA 单独不产生质量增益**，它必须挂靠 IFR 的 eval backbone 才能被判定。所以最终形态是"**CASA 做系统、IFR 做裁判、LADDER 做期权、UBBA 换约束集后并入**"。

---

## 附录 A · 关键数字速查（仅供校验，标注来源）

| 量 | 值 | 来源 |
|---|---|---|
| Qwen3.6-27B FP8 KV per-token | ≈32 KiB（32 GB/Mtok） | R5 由 Table 4 反推 |
| Mean-K index per-token | ≈1 KiB | R5 同上 |
| KV vs 原文存储放大倍数 | ≈8×10³ | R6（34.8 KB/token vs ~4 B/token） |
| index/KV 体积比（理论） | `1/(2B)` = 1.56%（B=32） | R8 推导，**与 head 数无关** |
| Mean-K index @256K / @10M | 0.25 GiB / 9.5 GiB | KVMem Table 4 |
| NVMe KV footprint @10M | 324.2 GiB | KVMem Table 4 |
| retrieval latency @256K / @10M | 0.174 s / 1.311 s | KVMem Table 4 |
| GPU memory 占用（workspace 变化时） | 恒定 ~34.9 GiB | KVMem Table 4 |
| step 内 vs 跨步 attention KL | 0.070 bits vs 2.59 bits（37.3×） | KVMem §4.2 |
| 历史 attention 稀疏性 | top-8/103.1 blocks = 66.5% mass；top-16 = 77.0% | KVMem Figure 2 |
| cudaMemcpyAsync 带宽利用率 | 22% PCIe5 / 5% NVLink | Strata §3.1 |
| Strata GPU-assisted I/O 配置 | 2 blocks × 1024 threads → ~50 GB/s，prefill 退化 <5%、decode <10% | Strata §4.2 / Figure 5 |
| Strata 页大小 vs 命中率 | 1→1024 tokens：平均 TTFT +2×、P90 +2.9× | Strata Figure 2 |
| Strata 默认阈值 | delay-hit 100 token matches；loading-bound ratio = 100 | Strata §4.3 |
| DeepSWE Pass@1 显著性 | z=0.53, p≈0.59, CI −12.6~+21.9pp | **R10 自算** |
| 达成 ±3.3pp 所需样本 | ≈525 配对 rollout（≈131 tasks × 4 seeds） | **R10 自算** |

### 第二轮圆桌新增（§2.5 / §3.4）

| 量 | 值 | 来源 |
|---|---|---|
| **检索有效带宽 @10M** | `10.24 GB ÷ 1.311 s = 7.8 GB/s` | **R8 推得**；仅被冷 NVMe（≈1.46 s）解释，不被 DRAM（≈0.05 s）或 PCIe5（≈0.21 s）解释 ⇒ **第一号消歧项** |
| **保真悬崖 ρ_max** | ≈36%（由 top-8/103 占 66.5% mass 反解 logit σ s≈1.85） | **R4 推导** |
| **L2′ 收窄后 ρ** | 0.28–0.32（G=4 quad-merge + 质心 4-bit），32.5 GiB | **R6 修正** |
| **Δ-QRoPE flops（修正）** | 4.7%（原报 <3%）；K-tile prologue 方案下 0.144 TFLOP/s ≈ FP32 峰 0.2% | **R4 修正 / R2 改法** |
| **Δ-QRoPE 误差性质** | `logit = qᵀR(a′−b′)k + qᵀR(−δ)e_K` ⇒ 信号项恒等、**误差项与 δ_j 与 n 无关** ⇒ 运输机制非保真机制 | **R4 证明** |
| **区带钉死** | zone 1 GiB ÷ extent 1.06 MiB = 966 extent；1% 长命共享块 ⇒ `(0.99)^966 ≈ 6×10⁻⁵` ⇒ **99.99% zone 被钉死** | **R7 推导** |
| **host DRAM 席位（修正）** | 128 GB(十进制) = 119.2 GiB；`119.2/64 = 1.86 ⇒ floor 1 席`（原报 2 席有误） | **R5 自曝修正** |
| **A2 NoPE 后训练成本** | 27B × 10B token ≈ `1.62e20 FLOPs ÷ 400 TFLOP/s ≈ 1,125 GPU-h`（占 IFR 全部预算 59%） | **R7 推导** |
| **L1(2-bit) 边界翻转率** | ρ=0.343 ⇒ σ_e=0.635 nats ⇒ top-8 边界翻转 P = **20.2%** | **R6 推导** |
| **质量终点判据成本** | 非劣 3pp 需 1308 对/臂 ⇒ 3.6×10⁴–1×10⁵ GPU-h ⇒ **预算内不可判决** | **R10 裁决** |
| **预注册 stopping rule** | N_max = `1.177/0.05²` = 471 对/臂；Lan-DeMets OBF 边界 z = 3.92/2.77/2.26/1.96 | **R10 设计** |

## 附录 B · 明确的未知项（任何人接手前请先接受这些不确定）

1. **QW3 是否可得**——影响 7/10 份报告的执行路径。
2. **跨 session 的 KV 级重复率没有公开数据**——R3 的 dedup 与 R5 的共享假设都建立在此之上；R5 圆桌修正后取值范围为 canonical 挂载 σ∈[0.15,0.45]、自然轨迹 σ∈[0.01,0.05]（均未验证）。
3. **Qwen3.6/3.8 是否使用 Dynamic-NTK**——直接决定 R4 的 Δ-QRoPE 是否成立（跨步加法性）；R4 估用 YaRN 概率 ~75%（未验证）。
4. **KVMem 未披露 raw K 的存储精度**，也**没有做过 delta re-RoPE 上限的 ablation**——所有 fidelity 分析目前都缺真值。
5. **检索延迟的三界矛盾**（7.8 GB/s 只被 NVMe 解释，但论文称 index 常驻 host）——这是方案四 G-CAS-1 要解决的问题，**目前无人有答案**。
6. **R10 的 SE=0.0881 未校正 task 内聚类**——若 ICC≈0.3 ⇒ design effect≈1.9 ⇒ 有效 SE≈0.121、z≈0.39，power < 8%（原估仍偏乐观）。π_d=0.15 亦是 64 runs 的单点估计。
7. **本报告中标注「未验证」的推算一律不可写入论文**，需要实机标定。

---

---

## 附录 C · 代码可得性与实验可行性复核（2026-10-05 补充）

> 第一轮报告把"QW3 不可得"写成 G0 杀点。复核后发现情况有变，且**这个变化同时降低了门槛、也砍掉了部分实验范围**。

### C.1 开源实现确实存在：`kvmem/kvmem-llama.cpp`

| 项 | 内容 |
|---|---|
| 许可 | Apache-2.0（与 `kvmem-qw3` 同） |
| 版本 | 源码 v0.17.0（master pin llama.cpp v0.5.0 `7fe450e19`）；预编译 v0.16.0-rc3 |
| **实测硬件** | **RTX 5060 Ti 16 GiB VRAM + Intel Core Ultra 7 255H + 32 GiB RAM**（WSL2 Ubuntu 22.04.5） |
| 模型 | Qwen3.8-27B GGUF，**IQ3 为主推荐**（`--kv-dtype q8_0`），IQ4 为可选对照（`q5_0`） |
| 实测峰值 | Task 2（32 轮工具调用，达 262,058/262,144 tokens）：VRAM 峰值 **15,617 MiB**，host RAM 峰值 **13,483 MiB**，decode **31.74 tok/s**，MTP 接受率 **64.70%** |
| 论文指标复现 | LongMemEval-S 32K active vs full 256K：**85.6% vs 86.6%**；AgentLongBench：**60.9% vs 59.5%** |

**这直接推翻了报告的两个隐含前提**：① 不需要 96GB RTX PRO 6000，16GB 卡就能跑；② 不需要 128GB host DRAM，32GB 就够（256K 档）。

### C.2 但它缺三样，且每样都砍掉一块实验

| 缺失项 | README 原文 | 后果 |
|---|---|---|
| **re-RoPE** | 未出现该术语。口径是 "Attention kernels and original positions stay unchanged. Reselection transfers only blocks that changed." | 见 C.3——这可能是**好消息** |
| **Mean-K 检索** | 未出现。默认是 `query replay auto` + `query policy user` + **128-token blocks**（论文是 32） | IFR / CASA 的索引实验**必须自己实现 baseline**，且 block size 口径不同 |
| **raw KV block → NVMe offload** | 只有 opt-in 的**非活跃 session 快照**（`--kvmem-session-nvme-gb`，源码版才有，rc3 预编译早于该功能）；native Windows CUDA build 里 legacy raw-block NVMe tier 被禁用 | **10M workspace / 324.2 GiB 那组实验当前做不了** |

### C.3 一个能省掉最贵那一步的洞察：re-RoPE 在 ≤256K 下根本不需要

llama.cpp 版**不做位置重映射**，保留原始绝对位置。为什么这样能工作？

> **只要 workspace 长度仍在模型的位置外推范围内，就不需要压缩位置，于是 re-RoPE 完全不必要。**

自洽证据：该版本的实测上限恰好是 **256K**（`-c 262144`），README 明写"超过 256K 质量仍属实验性"——256K 正是 YaRN 外推的合理边界。超出后位置跑出训练分布，**才**必须压缩回 bounded execution view，那时 re-RoPE / Δ-QRoPE 才成为必需。

| workspace | 需要 re-RoPE？ | 现有开源代码可直接跑？ | 报告方案四的 Δ-QRoPE kernel |
|---|---|---|---|
| **≤ 256K** | **不需要** | ✅ | **可以不做**（省掉最难、最容易翻车的一步） |
| 1M–4M | 需要 | ❌ 要自己加 | 必需 |
| 10M | 需要 + NVMe tier | ❌ 要加两层 | 必需 |

**而论文两个核心 benchmark（LongMemEval-S 85.6 vs 86.6、AgentLongBench 60.9 vs 59.5）恰好就在 256K 区间测的** ⇒ 把它们钉为 fidelity 真值时，可以完全绕开 CUDA kernel 工作。

> ⚠️ 这条需要在报告中被优先质疑：如果 Qwen3.8 的位置外推能力弱于假设（比如实际有效窗口 <256K），那么 ≤256K 的结论也不能免于 re-RoPE。**建议用 E1 的 ‖K‖ 衰减曲线与位置外推质量曲线顺带标定这一点。**

### C.4 实测容量系数（比论文的保守，做预算用这个）

```
llama.cpp Task 2 实测：262,058 tokens → host RAM 峰值 13,483 MiB
⇒ 每 token host 开销 ≈ 13,483 MiB × 1024² / 262,058 ≈ 53,944 B ≈ 52.7 KiB/token（含运行时开销）
论文 qw3 口径（纯 KV, FP8）= 324.2 GiB / 10M = 34.8 KB/token
```

| workspace | host RAM 需求（按 52.7 KiB/token） |
|---|---|
| 256 K | ≈ 13.5 GiB（= 实测值） |
| 1 M | ≈ 50 GiB |
| 4 M | ≈ 201 GiB |
| 10 M | ≈ 503 GiB ⇒ 必须 NVMe tier，当前未实现 |

**两者差 1.5×**，来源可能是 q8_0 的 per-block fp16 scale、host 侧副本、运行时其他开销。**做容量预算一律用 52.7。**

### C.5 对方案排序的影响

- **方案四 CASA 的 K-Freeze 内核部分（Phase 1 的 fused kernel）可推迟到第二年**。第一年钉在 ≤256K，用 llama.cpp 版的"保留原位置"路径即可，把人日投到 Phase 2（保真阶梯）与 Phase 3（索引裁决）。
- **方案一 IFR 的 fidelity 线（Phase 1）成本大幅下降**——不再需要 dump 10M workspace 的 Mean-K，256K 档的 Full Context 本身就是真值。
- **方案二 LADDER 的 P0 证伪实验（de-RoPE 后平均 vs 直接平均）可以在没有 GPU 的机器上做**（只需一个能吐 `past_key_values` 的小模型），成本从"2 人日 + GPU"降到"2 人日 + 0"。
- **E8（525 配对 rollout）与 E9（10M/NVMe）维持"不建议自费 / 第二年"的判断。**

---

*本报告由 WorkBuddy 智能设计助手组织 10 个并行研究 agent 产出，并经第二轮圆桌（共享简报 + 投票 + 互相质证）收敛至方案四；附录 C 为 2026-10-05 的代码可得性复核。所有引用均指向公开来源，所有推算均标明推导过程。欢迎挑错——尤其是附录 B 的七项与附录 C.3 的那条推论。*

*配套执行手册：`实验执行手册-设备与步骤.md`（设备清单、云 GPU 价格、逐实验命令与判据）。*
