# 对方更新（4c619d9 + c528981）全面复核：他们说了什么，以及他们可能没考虑到什么

> 触发：用户要求查看对方对我们的修正 + 他们"别的意见"，并**尽可能全面地思考专家没考虑到的可能性**。
> 复核对象：origin/main 上 793ba59 之后的两个提交——
> - `4c619d9` feat(phase2-3)：新增 2 个 benchmark + C++ 子质心排序 + stopping 改配对 McNemar
> - `c528981` fix(ifr)：wire `use_normalized=True` 默认 + 标定 `theta=0.868`（这是**对我们 comment 的修正**）
> 我方参照：`docs/real_kv_audit_og_v2_report.md`、`docs/og_review_c5e76c8.md`、本机实测。

---

## 0. 这次更新到底是什么

**"对我们的修正" = c528981**（已在上一轮复核）：采纳我们 review 的发现 A（死代码）与 C（阈值口径），把 `use_normalized` 默认改 True 并接线、阈值 0.85→0.868。

**"他们别的意见" = 4c619d9**，三条新主张，**没有新文档，观点全在代码和 benchmark 里**：

| # | 主张 | 载体 |
|---|---|---|
| 1 | **解耦多质心（K=2/K=4）检索优于"naive Strata 块均值"** | `benchmarks/benchmark_decoupled_retrieval.py` |
| 2 | **统计契约改用配对 McNemar 精确方差**（响应 critique v3 §8.2） | `kvmem_fusion/stopping.py`、`tests/test_stopping_rules.py` |
| 3 | **Mac 32GB 上可跑 70B 级逐层流式 + 256KB Super-Tile KV 卸载**，且"recall@64 ≥ 0.92" | `benchmarks/run_mac_layer_streaming_sim.py`、`csrc/ifr_lse.{cpp,hpp}` |

---

## 1. 我先做的两项实证核实（都已跑过）

### 核实 1：Mac 层流式演示在 macOS 上是**静默空转**
`run_mac_layer_streaming_sim.py` 的核心是 `os.posix_madvise(..., os.POSIX_MADV_WILLNEED)`。本机实测（macOS 27.0.1 arm64）：

```
has os.posix_madvise: False
has POSIX_MADV_WILLNEED: False
posix_madvise call FAILED -> AttributeError: module 'os' has no attribute 'posix_madvise'
```

**macOS/Darwin 根本没有 `posix_madvise`**。而代码把两处调用都包在 `except Exception: pass` 里 → 异常被静默吞掉，脚本照样打印 `[SUCCESS] ... verified on local hardware`。即：这个"在 Mac 上验证通过"的演示，**核心机制从未执行**。

### 核实 2：他们的新 benchmark 导入的是我们**已撤回的 v1**
```python
from benchmarks.real_kv_audit_og import ...   # v1（我们已自我撤回）
```
而不是 `real_kv_audit_og_v2`。两个模块在 origin/main 上都存在，他们选了 v1。这不是笔误——见下面 A1，v1 的方法学恰恰是他们这个实验的**核心混淆源**。

---

## 2. 他们可能没考虑到的（按严重度分级）

### A 级 · 结论可能被混淆（直接影响主张 1 是否成立）

**A1. 相关性空间错配——他们把我们已撤回的 v1 bug 又做了一遍**
在他们 benchmark 里：
- `truth = de-RoPE 后的块均值`；`q = truth[qb]` → **query 在去 RoPE 空间**
- `armA = 旋转后 K 的均值`（原生空间）
- `armB = de-RoPE → 均值 → re-RoPE 到块中点`（原生空间）← **这正是我们 v1 那个有位置错配的 Arm Bmid**
- `sub2/sub4 = de-RoPE 后的子均值`（**去 RoPE 空间**）

结果：**两个基线（A、Bmid）都与去 RoPE 的 query 空间错配；而子质心臂（K=2/K=4）是空间匹配的。** 于是"子质心 vs Strata_Naive_A"的 Gain 里，混进了"空间是否匹配"这个因素，而不全是子质心的功劳。
> 正确的基线是我们 v2 的位置匹配版 **B = 0.8913**（去 RoPE→均值→不 re-RoPE），全在去 RoPE 空间比。我们的 v2 已经给出单体 B=0.8913；真正要问的是"K=2/K=4 能否超过 0.8913"，而不是"能否超过被压低的 0.64"。

**A2. "Strata_Naive_A" 这个标签是错的（三重混淆）**
Strata **根本没有 Mean-K 索引**——它把 KV 原样回 GPU、按前缀复用，不做块均值池化。他们把"均值池化 + v1 空间错配"压出来的低分贴成"Strata"，混了三件事：① Strata 的原位复用、② KVMem 的 Mean-K 池化、③ v1 的空间错配。**这会让读者误以为 Strata 的检索质量只有 0.64**（与我们上一轮"vs 原版 Strata"的结论直接冲突，必须纠正）。

**A3. 部署路径与验证路径不在同一个空间**
C++ `rank_subcentroids_unnormalized()` 的打分是 `dot(query, sub) * scale_`——**没有 L2 归一化**；而"recall@64 ≥ 0.92"这个数是在 Python 侧用 `cos_rank`（**归一化余弦**）测出来的。
> 未归一化点积会被 **massive-activation**（σ≈8.03 是 L2 幅度、非语义）主导：幅度大的质心无条件胜出。0.92 是余弦空间的数，不能迁移到未归一化的 C++ 路径。这正是我们反复强调的"必须声明索引所在空间"。

### B 级 · 实验设计缺陷

**B1. query 是块自己的质心 → 自我检索**
`q_blocks = rng.choice(n_blocks, ...)`，`q = truth[qb]`。query 直接取自候选块自身的质心，**top-1 答案平凡地就是它自己**。真实推理时 query 来自当前 live token，不是某个存储块的质心。这让所有臂的 recall 虚高、且削弱区分度。

**B2. max-pooling 过 K 个子质心是**次序统计偏差
块分 = `max_s (q · sub_{b,s})`。**K 越大，max 的期望值越高**（多个抽样取最大）。所以"K=4 优于 K=2"里有一部分纯粹是"多抽了几次"的假象，而非真实保真提升。
> 解纠缠办法：加一个**随机切分 null 对照**——把块随机切成 K 组（不含语义），若 K=4 随机切分也能涨，就证明是 max-pooling 假象。

**B3. 只测单层（mid_layer）**
`mid_layer = n_layers // 2`，仅 1 层。我们 v2 的逐层位置先验（α 由 0.1→0.9）和**他们自己**测出的 20× 层差（L0 0.034 vs L16 0.702）都说明：结论必须逐层给，单层的数不能外推。

**B4. 选择率远离真实工况**
top-64 / 128 块 = **50% 选择率**，接近天花板、对方法差异极不敏感。KVMem 实际工作在 ~3% 选择率（32 tok/质心 @3.125%）。真实难点区间完全没测。

**B5. 单模型**
只有 Qwen2.5-0.5B-Instruct。我们跨模型（Qwen3-0.6B）复现时发现**量级差很大**（B/A 1.40× vs 1.07×；重定位 −31pp vs −9pp）。单模型结论不可外推。

**B6. 子质心是固定位置切分，且没做等预算对比**
`sub2 = [0:16]/[16:32]`、`sub4 = 4×8` 连续切分——**非内容自适应**（语义边界不一定在 16）。而且 K=2/K=4 让索引体积 ×2/×4，却没与"同等索引预算下缩小块"比较。我们的**质心定律**要求：保真只由 tokens/centroid 决定，比较必须**等预算**（我们实测 B32/m4 0.711 > B8/m1 0.654）。他们的对比缺少这个基线。

### C 级 · 完全缺失的物理因素（我们 v2 的核心发现，他们一条都没建模）

**C1. 重定位惩罚——完全没建模（最严重）**
我们实测：Qwen2.5 L12 recall 从 Δ=0 的 **0.877** 跌到 Δ=2048 的 **0.563**（−31pp）。**压缩不是免费的，而压缩正是 KVMem 的灵魂。** 他们的解耦检索全部是**原位**假设；一旦进入 compaction（KVMem 的前提），这条代价就会出现，而他们的实验里根本没有 Δ 维度。

**C2. 子质心的 re-RoPE 没处理**
块被搬走后，位置变了。他们给 armB 做了 re-RoPE（`rope_at(..., mid, ...)`），但**子质心（sub2/sub4）完全没有 re-RoPE**——它们永久停留在去 RoPE 空间。真实系统里每个子质心都需要各自的位置处理，这块是空的。

**C3. anti-collapse doublet 与 decoupled 子质心（K=2）是否重复？**
他们现在**同时**有两套机制：anti-collapse 分裂出 2 个质心（needle + residual），decoupled 子质心又是 K=2。**两者都产出 2 个质心——是同一个东西的两种实现，还是可叠加？** 没人讨论过，也没有消融实验。这是明显的冗余/冲突风险。

**C4. INT2（LADDER）压缩 × de-RoPE/re-RoPE 往返的量化误差**
re-RoPE 是 `rope_inverse → rope_forward` 的往返变换。在 INT2 量化域上做往返，误差如何放大？**无人测过**。这可能吃掉 de-RoPE 带来的全部收益。

**C5. "max 相似度"作为块相关性代理，有效性未验证**
真实注意力是 softmax **加总**，不是取 max。max-pooling 只是代理。我们 v2 的 partial-Spearman 已经显示：**浅层内容排序与真实注意力相关性很差（L0 ρ=−0.091）**——那么 max 代理在浅层可能尤其失效。这个 proxy 从未被对照真实注意力质量验证。

### D 级 · Mac 模拟器不构成证据（主张 3 不成立）

| 问题 | 具体 |
|---|---|
| **D1 编造的 recall** | docstring/hpp 写"recall@64 ≥ 0.92"，但质心是 `rng.randn(2, HEAD_DIM)` **随机高斯**、块是 `b"X"*n` 哑元，**recall 从未被计算**。数写在注释里，不是测出来的。 |
| **D2 静默空转** | `posix_madvise` 在 macOS 不存在（已实测 AttributeError），被 `except: pass` 吞掉 → 核心机制从未执行仍打印 SUCCESS。 |
| **D3 文件别名** | 临时文件 seek 到 `min(44.8GB-1, 1GB)` = **1GB**，却声称 44.8GB；offset 用 `% 1GB` 取模别名；`layer_bytes=560MB` → 实际只存在 **~1.8 层**。 |
| **D4 KV 尺寸低估 ~640×** | 假设 1 层 / 1 KV head / dim 64 / FP32 → 512 B/token。真实 70B（80 层 × 8 KV head × dim 128 × FP16 × K,V）= **327,680 B/token**。**640 倍低估**。 |
| **D5 硬编码结论** | "Peak Resident RAM < 4.5 GB"、"recall ≥ 0.92"、"< 0.1ms"全是 **print 字符串**，无 RSS 插桩、无计时依据。 |
| **D6 无真实计算** | "80 层流式延迟"循环只发 madvise，没有一次 matmul/attention。测到的是**系统调用开销**。 |
| **D7 投机硬件** | "Apple M6 / 32GB"——无法核实。 |

### E 级 · 统计设计问题

**E1. 双重计数：精确配对方差 × ICC 设计效应**
`stopping.py` 新增了正确的配对 McNemar 方差 `((b+c) - (b-c)²/n)/n²`，但**又**乘了 `sqrt(Deff)`（Deff = 1+(m−1)·ICC = 1.9）。设计效应是用来修正**非配对**近似的；既然已用精确配对方差，再乘 Deff 属于**重复膨胀**。
> 而且 `N=471` 是用**不含 Deff** 的公式算的。若分析里真要 Deff=1.9，有效样本量需求约为 **471×1.9 ≈ 894**。**样本量目标与分析口径不一致。**

**E2. 关键输入是假设值，不是实测**
`discordant_rate=0.15`、`delta_mde=0.05`、`icc=0.3` 全部是默认值/假设。我们 v2 的数据可以直接估出真实的 discordant rate 与 ICC，但他们没有回填。

**E3. 连续指标被二分**
recall@k 是连续量，McNemar 要求把每条任务二分成"成功/失败"，**损失统计功效**。更合适的是连续配对检验（paired t / Wilcoxon / bootstrap CI）。

**E4. 无多重比较校正**
4 个臂 × 多层 × 多模型同时检验，没有 Bonferroni/Holm 或 FDR 控制 → 假阳性膨胀。

**E5. 只有优效检验，无非劣效/等价检验**
对系统级改动，真正该问的常是"压缩后**不差于**基线超过某边际"（TOST / 非劣效）。目前只有"是否更好"，无法回答"是否够用"。

### F 级 · **双方都没做过**的真空区（这是最值得投入的地方）

| # | 空白 | 为什么重要 |
|---|---|---|
| F1 | **端到端质量（T2/T3）**：无人跑过 needle-in-a-haystack 或真实任务质量 | recall@k 是代理；我们自己也还停在 T1-partial |
| F2 | **跨架构 MLA（DeepSeek 系）** | head_dim / KV head 数全变，现有结论（**包括我们的**）都只适用于 GQA-Qwen |
| F3 | **真实长上下文 32K–1M** | 我们的实验 ≤4096；他们的 32K 是哑元数据。真实区间无人测 |
| F4 | **QW3 闭源（G0）** | 目标模型无法验证，所有结论都停留在开源代理模型上 |
| F5 | **并发与调度**（Strata 的 cache-aware、delay hit） | 双方都没碰；真实服务的吞吐瓶颈在这里 |
| F6 | **索引增量维护成本** | 上下文增长/淘汰时索引重建代价，无人建模 |
| F7 | **合并不可逆性** | 块合并（compaction）后能否还原？影响可否安全部署 |

---

## 3. 该肯定的部分（避免只挑错）

- **c528981 是干净的工程回应**：确实修了死代码、采纳了我们的 0.868，可验证。
- **配对 McNemar 方差公式本身是对的**（`((b+c)-(b-c)²/n)/n²` 是标准式），方向正确，响应了 critique v3 §8.2。只是又叠了 Deff（E1）。
- **"多加质心"的方向与我们的质心定律一致**——我们独立证实过"等预算下加质心优于缩小块"。所以他们的**架构直觉是对的**，错的是**验证方式**（A1/A2/B2）。

---

## 4. 建议回他们的可执行验证清单

1. **换基线**：把 `Strata_Naive_A` / `Bmid` 换成**全去 RoPE 空间的位置匹配版**（我们 v2 的 B=0.8913），重测 K=2/K=4 的真实增量。
2. **改 import**：`real_kv_audit_og` → `real_kv_audit_og_v2`（否则永远在复现已撤回的结论）。
3. **加随机切分 null**：随机切 K 组，分离 max-pooling 次序统计假象。
4. **加 Δ 维度**：把我们的重定位扫描接进去，看 K=2/K=4 在 Δ>0 时是否还成立（这是 compaction 的前提）。
5. **统一空间**：C++ 路径要么归一化，要么重新在**未归一化空间**测 recall，别把余弦空间的 0.92 贴到未归一化 kernel 上。
6. **扩维**：≥2 模型 × 多层 × 选择率扫到 3%，别停在单层 50%。
7. **统计**：去掉 Deff 与配对方差的重复膨胀；用实测 discordant rate / ICC 回填；加多重比较校正；补非劣效检验。
8. **Mac 演示**：要么补真实 RSS 插桩 + 真实矩阵计算 + 正确的 madvise（macOS 需用 `madvise` 于已 mmap 的地址，或改用 `fcntl` 的 `F_RDADVISE` / `F_NOCACHE`），要么**撤掉 recall ≥ 0.92 与"verified on local hardware"的断言**。

---

## 5. 结论

- 他们的**修正部分（c528981）合格**，采纳了我们的两个发现。
- 他们的**新主张（4c619d9）方向对、验证错**："多加质心"与我们的质心定律一致，但基准被 v1 空间错配污染、对比未等预算、max-pooling 未解纠缠、且完全没建模重定位与 re-RoPE——**在 KVMem 的真实工况（compaction）下，这些结论目前不能成立**。
- **Mac 70B 演示不构成证据**：核心 syscall 在 macOS 上不存在并被静默吞掉，recall 与内存数字是硬编码字符串，KV 尺寸低估约 640 倍。
- **最值得投入的是 F 级真空区**：端到端质量、MLA 跨架构、真实长上下文——这些是双方都没做的，也是我们拉开差距的地方。
