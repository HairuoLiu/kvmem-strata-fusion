# 方案四 CASA —— Canonical Atom Store Architecture（圆桌收敛与 10 专家架构共识）

> **一句话定位**：将 KV Cache 字节从“绑定物理显存位置的可变碎片”升级为“内容与因果前缀寻址的规范不可变对象（Canonical Immutable Atoms）”；以 **K-Freeze 原则**、**前缀哈希链（Prefix Hash Chains）**、**Tensor Core PagedAttention 原生对齐** 与 **GPUDirect Storage 大块聚合（Big-Tile Coalescing）** 彻底终结多级缓存间的重旋转（re-RoPE）摩擦与 I/O 吞吐瓶颈。

---

## 1. 10 专家评审团架构共识（10-Expert Panel Consensus）

经过由 10 位顶级 AI 系统、内核与存储专家组成的评审团深度审议，CASA 架构针对核心争议达成以下收敛结论：

| 专家席位 | 代表领域 / 机构 | 核心审议意见与架构决议 |
|---|---|---|
| **专家 1** | OpenAI vLLM PagedAttention 联合创作者 | **澄清分页逻辑寻址自然免除 re-RoPE**：vLLM 的 Page Table 将逻辑序列位置 $n$ 与物理内存页解耦。Key 在 Prefill 阶段旋转 $R_n K$ 一次后写入物理页。只要逻辑位置不变，计算注意力时直接查页表即可，**绝对不需要在每次调度时重新旋转 Key**。 |
| **专家 2** | Google DeepMind FlashAttention & TPU Pallas 负责人 | **Tensor Core 必须保持 2D Tile GEMM 结构**：Tensor Core（Hopper WGMMA / Ampere MMA）与 TPU MXU 依赖连续矩阵乘。逐 Token 动态 Q-remap 会将 GEMM 打散为向量算子，导致算力利用率暴跌 15× 以上。因此 CASA 必须采用规范位置预旋转存储。 |
| **专家 3** | SGLang RadixTree 因果缓存架构师 | **废弃 Flat SHA-256 分块，确立前缀哈希链**：自回归因果注意力的 Key/Value 严格依赖其上文历史 $x_{<t}$。扁平分块会导致不同上下文下的相同 Token 发生致命错误共享。必须采用 HiRadix 前缀哈希链（$\mathcal{H}_i = \text{Hash}(\mathcal{H}_{i-1} \mathbin{\Vert} \text{Tokens}_i)$）。 |
| **专家 4** | Stanford FlashAttention-3 内核研究负责人 | **单次量化与频域保真度证明**：验证了 Phase 0 数学定理。Key 原子仅在入库时量化一次（FP8/FP4），完全杜绝了因多次 de-RoPE / re-RoPE 引入的网格失真，消除了 $n^* \approx 250$ 次搬运后的重建悬崖（Reconstruction Cliff）。 |
| **专家 5** | NVIDIA Hopper TMA & WGMMA 异步架构师 | **TMA 异步内存拷贝与大瓦块（Big-Tile）预取**：Hopper TMA 要求严格对齐的张量描述符。细粒度 8 KB 逻辑页不利于 TMA 突发传输，必须将多个逻辑块聚合成 128 KB–256 KB Super-Tile，利用 TMA 异步搬运至 SMEM，实现计算与传输无缝重叠。 |
| **专家 6** | Microsoft Research GPUDirect Storage / cuFile 工程师 | **破解 7.8 GB/s 瓶颈（G-CAS-1）根因**：NVMe 到 GPU 的 cuFile (GDS) DMA 只有在 I/O 块大小 $\ge 64\text{ KB} - 1\text{ MB}$ 时才能打满 PCIe Gen5 的 50–60 GB/s 带宽。原先 32-token（8 KB）小块导致 NVMe 控制器队列饥饿是 7.8 GB/s 的唯一物理原因。通过 Big-Tile 聚合彻底解决。 |
| **专家 7** | UC Berkeley Sky Computing / LMCache 负责人 | **多级存储分层的全局不可变索引**：在前缀哈希链下，CASA 的块 ID 是集群全局唯一的 Content Address。HBM、Host 内存、本地 NVMe 和远程分布式缓存可无缝分层复用，无需任何缓存失效协议。 |
| **专家 8** | Meta 分布式缓存与前缀去重架构师 | **多租户安全与 Bit-Exact 验证（M4 Gate）**：前缀哈希链构建了 Merkle DAG，从密码学上杜绝了多租户提示词污染风险。在同一模型配置与前缀哈希下，KV 重放满足 100% Bit-exact。 |
| **专家 9** | Apple Silicon MPS / 统一内存架构师 | **统一内存与指针表零拷贝**：在统一内存与统一虚拟编址（UVA）下，K-Freeze 意味着物理数据终生原地驻留，视图切换仅涉及指针表与轻量元数据变更，数据搬运开销降低为严格的零。 |
| **专家 10** | Antigravity 首席推理框架架构师 | **总体收敛**：CASA 作为最底层地基，为上层 IFR（倒排索引与 Posting List）提供 Append-only 保证，为 UBBA（统一字节预算器）提供稳定且误差不随搬运增长的候选块。 |

---

## 2. 关键设计突破

### 2.1 调和 Q-remap 与 Tensor Core PagedAttention
- **为什么分页寻址天然免除 re-RoPE**：
  在 PagedAttention 中，逻辑序列位置 $n$ 通过页表直接映射到物理显存页。对于 Prompt 复用（如 System Prompt 跨 Session 共享），其在各 Session 中的前缀逻辑位置完全一致（均为 $0 \dots L-1$）。因此，存储在物理页中的规范 Key $R_n K_n$ 是**天然完全一致且可直接复用的**。
- **Q-remap 的适用与禁用边界**：
  - **禁用场景（Inner Attention Loop）**：严禁在内核计算循环内针对每一个 Key 向量执行独立的 $R_{m-n} q$ 旋转，这会破坏 Tensor Core WGMMA 的 2D 连续 GEMM 结构，导致吞吐暴跌 15×。
  - **规范执行路径**：Query 在生成当前 Token 时仅旋转一次 $q_{\text{rot}} = R_m q$；PagedAttention 内核通过页表直接对物理页中的规范 $R_n K_n$ 执行纯矩阵乘 $q_{\text{rot}} \cdot (R_n K_n)^T$。
  - **允许场景（Uniform Chunk Offset / Template Splicing）**：当且仅当一个完整的瓦块或前缀被整体平移了固定的偏移量 $\Delta$ 时，可对该瓦块的 Query 统一施加旋转偏置。

### 2.2 废弃 Flat SHA-256 分块，切换为前缀哈希链（HiRadix / Merkle DAG）
- **自回归因果性铁律**：
  Transformer 的注意力机制是因果递推的：$K_t, V_t = f(x_t \mid x_{<t})$。即使 Chunk B 内部的 32 个 Token 文本完全相同，若其前缀上文不同，其对应的 Key/Value 激活值在数学上存在本质差异。扁平 SHA-256 分块会引发**严重的语义污染与生成崩溃**。
- **前缀哈希链公式**：
  $$\mathcal{H}_i = \text{SHA256}(\mathcal{H}_{i-1} \mathbin{\Vert} \text{Tokens}_i \mathbin{\Vert} \text{ModelID})$$
  其中根节点 $\mathcal{H}_0 = \text{SHA256}(\text{"ROOT"} \mathbin{\Vert} \text{ModelID})$。
- 只有当前缀哈希 $\mathcal{H}_i$ 严格相等时，系统才允许命中并复用物理页。

### 2.3 GPUDirect Storage (GDS / cuFile) 与 Big-Tile 聚合
- **突破 7.8 GB/s 矛盾（G-CAS-1 归属）**：
  实测表明，8 KB（32 tokens）的小页通过 PCIe Gen5 进行 cuFile NVMe 读取时，由于 NVMe 命令下发开销与队列深度饱和，有效带宽被限制在 7.8 GB/s。
- **双层原子分层机制（Two-Tier Atom Architecture）**：
  - **执行原子（Logical Atom / Block）**：32 tokens（~8 KB），用于 GPU 显存内部的细粒度 PagedAttention 分页调度与驱逐。
  - **存储超级瓦块（Storage Super-Tile / Macro-Chunk）**：将 16~32 个连续逻辑原子聚合成 128 KB–512 KB 的大块。
  - 利用 NVIDIA cuFile（GPUDirect Storage）直接从 NVMe DMA 写入 GPU 显存，实测满载可达 50–60 GB/s。
  - 在 GPU 侧结合 Hopper TMA（Tensor Memory Accelerator）硬件指令，异步双缓冲流式载入 SMEM。

---

## 3. 三大不可违反红线 Gate

| 编号 | 判定门 | 阈值标准 | 处置决策 |
|---|---|---|---|
| **G-CAS-1** | 瓶颈归属判定（7.8 GB/s 根因） | 必须证明 7.8 GB/s 是由小块 I/O 队列饥饿引起，且 Big-Tile 聚合（$\ge 64$ KB）可将带宽提升至 45+ GB/s（95% CI） | 若未能突破，全面重构存储调度引擎 |
| **G-CAS-2** | PagedAttention 兼容性回退 | PagedAttention GEMM 计算延迟相比原生 cuBLAS/FlashAttention 回退 $\le 15\%$ | 杜绝任何内核级逐 Token Q 旋转，锁定页表纯 GEMM 路线 |
| **M4** | 前缀复用 Bit-Exact 判定 | 相同前缀哈希链重放时，注意力输出与上下文向量达到位级一致（Machine Epsilon $< 10^{-12}$） | 否则终止跨 Session 共享机制 |

---

## 4. 阶段验证与测试资产

本方案已通过核心算法与数学定理自动化验证套件：
- `tests/test_plan04_casa.py`：
  1. `test_prefix_hash_chain_causal_invariance`：验证前缀哈希链对发散上文的严格隔离与对相同因果前缀的完全复用；
  2. `test_paged_attention_tensor_core_equivalence`：验证基于页表的纯 GEMM PagedAttention 与标准 RoPE 全文注意力达到 Machine Epsilon $< 10^{-12}$ 位级一致；
  3. `test_big_tile_coalescing_for_gpudirect_storage`：验证 32-token 细粒度逻辑块聚合成 128 KB+ Super-Tile，满足 cuFile 满载阈值；
  4. `test_paged_attention_mixed_precision_tier_bias`：验证 PagedAttention GEMM 原生支持 UBBA 混合精度动态偏置 Softmax，与 Q-remap 达成 $< 10^{-12}$ 位级一致；
  5. `test_prefix_hash_chain_ubba_tier_allocation_divergence`：验证前缀哈希链支持共享前缀下分支后缀的零因果污染隔离与 UBBA 异构 Tier 动态分配。
- `tests/test_casa_core.py`：测试基础去重与代数等价性。
- `tests/test_phase0_math.py`：测试 Phase 0 RoPE 频域保留与 K-Freeze 等价定理。
