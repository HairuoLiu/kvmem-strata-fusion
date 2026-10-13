# OG 核验 `aa07e56`：数字可复现，但"真值"不是注意力

> 对象：对方 `aa07e56 fix(benchmark): correct ground truth to token-level mean-of-cos,
> confirming monotonic gain of decoupled sub-centroids (0.9004 -> 0.9224 -> 0.9382)`
> 方法：在相同语料、相同 4096 token、相同 layer 12、相同 64 个 query 位置（seed 42）下，
> 先原样复跑他们的脚本，再替换真值定义与分组方式做敏感性分析。
> 环境：Qwen2.5-0.5B-Instruct fp32 eager，torch 2.14.1 / transformers 5.19.0，macOS arm64。
> 结论一句话：**他们的数字是真的，但"mean-of-cos 真值"与模型真实注意力的块排序相关性只有 ρ=+0.05；
> 换一个真值，臂的优劣顺序会整体翻转。** 所以 `ground_truth_definition` 里的
> `[exact token-level attention]` 这个括注必须删掉。

---

## 1. 先确认好的部分：数字原样复现，三项缺陷真修了

原样重跑 `benchmarks/benchmark_decoupled_retrieval.py`（输出写到 /tmp，不污染仓库），
与提交的 `docs/decoupled_retrieval_benchmark.json` **逐位一致**：

```
Strata_Naive_A        recall@64 = 0.6450   (CI 0.6316–0.6584)
Monolithic_Bmid       recall@64 = 0.6042   (CI 0.5884–0.6201)
Monolithic_B_matched  recall@64 = 0.9004   (CI 0.8918–0.9090)
Decoupled_K2_Mean     recall@64 = 0.9224   (CI 0.9155–0.9292)
Decoupled_K2_Max      recall@64 = 0.8660   (CI 0.8572–0.8748)
Decoupled_K4_Mean     recall@64 = 0.9382   (CI 0.9328–0.9437)
Decoupled_K4_Max      recall@64 = 0.8369   (CI 0.8291–0.8448)
Random_Baseline       recall@64 = 0.4910   (CI 0.4812–0.5007)
```

即：0.6450 / 0.6042 / 0.9004 / 0.9224 / 0.8660 / 0.9382 / 0.8369 / 0.4910 全部对上。
**没有硬编码，是真跑出来的。** 这一点必须先说清楚。

对照我们 v3 清单第一部分，这次确实修好了三条：

| 我们提的 | 状态 | 证据 |
|---|---|---|
| D-A 真值 = 单体均值 → `B_matched = 1.0` 恒等式 | **已修** | `B_matched = 0.9004 (se=0.0044)`，非退化 |
| 产物无 Random 基线 | **已补** | `Random_Baseline = 0.4910`，与 64/128 选择率的理论值 0.5 一致 |
| Arm A/Bmid 与 query 的空间失配（A/Bmid 用旋转空间质心 vs 去旋转 query） | **已修** | A 现在用 `q_rot_n × Mr_n`，B/Bmid/子质心用 `q_de_n ×` 去旋转表示，两边各自同空间 |
| D-B1 query 取自定义块自身的质心 | **部分修** | 改为真实 token 位置 `p`（`q_de = K_de[:, p, :]`），不再是"块均值自我检索" |

---

## 2. 核心问题：把 mean-of-cos 叫做 "exact token-level attention" 是不对的

`docs/decoupled_retrieval_benchmark.json` 里写着：

```json
"ground_truth_definition": "mean_t cos(q_de, k_de(t)) [exact token-level attention]"
```

这句话有三处不成立：

1. **mean-of-cos ≠ attention。** 注意力权重是 `softmax(q·k/√d)`，不是余弦的平均。
   softmax 由最大 logit 主导，所以"模型真正看重哪些块"的排序更接近 **max 型**规则，
   而不是 mean 型规则。用 mean 型真值去评比，本身就偏向 mean 型臂。
2. **query 用的是 key 向量，不是 query 向量。** `q_de = K_de[:, p, :]` 是第 p 个 token 的
   **K**；真实检索里的 query 来自 `q_proj`（Qwen2.5-0.5B 还有 `q_norm`，且 14 个 q-head 共享 2 个 kv-head）。
   用 K 当 query 是"内容相似度"探针，不是检索查询。
   （说明：我方 v2 的 P1 也用去旋转 key 作 query，但我方在文件头明确标注为 proxy，
   并另开 P2 用模型自己的 Q 向量测对齐——见 `benchmarks/real_kv_audit_og_v2.py` 第 27–28 行与第 42–44 行。
   对方 import 的正是这个文件，却只取了 proxy 那一半，还把括注写成了 "exact"。）
3. **没有因果掩码。** 位置 p 的 query 在解码时只能看到 `t ≤ p` 的块；
   他们的真值把 128 个块（含未来块）一起排序。

### 2.1 实测：换真值，结论翻转

同一批数据、同一批臂，只换真值定义（recall@64）：

| 臂 | gt1 mean-of-cos<br>key-as-query（他们的） | gt2 softmax 质量<br>key-as-query | gt5 softmax 质量<br>真实 Q、非因果 | gt3 softmax 质量<br>真实 Q、因果 |
|---|---|---|---|---|
| A_naive（旋转均值） | 0.6450 | **0.8425** | **0.7129** | 0.5156 |
| Bmid（v1 伪影） | 0.6042 | 0.5295 | 0.4805 | 0.5686 |
| B_matched（去旋转均值） | **0.9004** | 0.6299 | 0.5840 | 0.5178 |
| K2_mean | 0.9224 | 0.6299 | 0.5771 | 0.5132 |
| K4_mean | **0.9382** | 0.6333 | 0.5728 | 0.5125 |
| K4_max | 0.8369 | 0.6255 | 0.5801 | 0.5149 |
| Random | 0.4910 | 0.5044 | 0.4961 | 0.5002 |

读法：

- 在他们的真值下，`B > A` 差 **+39.6%**，子质心再 +2.2pp / +3.8pp。
- 换成 softmax 质量真值（gt2 / gt5），**符号反转**：`A (0.8425 / 0.7129) > B (0.6299 / 0.5840)`，
  子质心增益完全消失（K4 ≈ B）。
- 换成真实 Q + 因果 softmax（gt3），所有臂都掉到期随机附近（0.51–0.57 vs Random 0.50）。

### 2.2 实测：他们的真值与真实注意力几乎不相关

块排序的 Spearman 相关（16 个 query × 128 块）：

| | gt1 mean-of-cos | gt2 softmax/key-q | gt3 真实Q/因果 | gt5 真实Q/非因果 |
|---|---|---|---|---|
| **gt1** | 1.000 | +0.4149 | **+0.0495** | +0.1601 |
| gt2 | | 1.000 | +0.1705 | +0.5715 |
| gt3 | | | 1.000 | **−0.0060** |
| gt5 | | | | 1.000 |

- `gt1 vs gt3 = +0.05`：**他们的"真值"排序与模型真实注意力排序基本无关。**
- `gt3 vs gt5 = −0.01`：加不加因果掩码，真实注意力的块排序完全变成另一回事。
- 结论：真值定义不是细节，**真值定义就是结论本身**。

---

## 3. 用我方 v2 的 P2 协议测：所有臂对真实注意力都对得很差

用模型自己的 `q_proj` query（含 `q_norm`、GQA 展开、`1/√d`、因果掩码）算每块真实注意力质量，
再算各臂打分与它的相关性（layer 12，64 个 live query）：

| 指标 | 均值 | 含义 |
|---|---|---|
| `rho_pos`（块序号 vs 注意力质量） | **−0.5412** | 注意力质量高度集中在**早期块**（attention sink），与序号负相关 |
| A_naive 原始 | +0.3067 | 朴素旋转均值反而是对齐最好的 |
| A_naive 去位置（partial） | +0.2269 | |
| A_naive 仅老块（去掉最近 16 块） | +0.3623 | |
| B_matched 原始 | +0.1279 | 去旋转均值对齐更差 |
| B_matched 去位置 | +0.0920 | |
| K4_mean 原始 | +0.1135 | 子质心没有改善对齐 |
| K4_mean 去位置 | +0.0675 | |
| Bmid（v1 伪影）原始 | **+0.3578** | 最高——但这是伪相关：re-RoPE 到块中点恰好编码了位置，而真实注意力被 sink/位置主导 |
| Random | −0.0557 | |

**解读（对双方都成立，包括我们自己）**：在 layer 12 这个配置下，
没有任何块均值索引（朴素的、去旋转的、子质心的）能跟上模型真实注意力；
朴素的跟得最好，而去旋转臂跟得更差。Bmid 的高相关是位置造成的假相关。
这正好说明：**"recall vs 内容相似度真值"与"对齐真实注意力"是两个不同的目标，
必须在报告里分开陈述，不能用一个括注糊过去。**

附带说明：`rho_pos = −0.54`（sink 主导）是**这一层这一个配置**的性质；
我方 v2 已实测逐层位置先验差异很大（α 0.1→0.9，L0 的 ρ=−0.091），
所以这条结论不能外推到其他层——但那恰恰是他们必须补多层实验的理由。

---

## 4. 两个对照实验：一个支持对方，一个支持我们，一个否掉我自己

### 4.1 预算匹配对照（等索引字节）——**支持对方的去旋转主张**

K4 每 32 token 存 4 个向量。等预算的朴素对照是"8-token 旋转均值 × 4"（同样 4 个向量/32 token）：

| 臂 | 索引字节/32token | gt1 recall@64 |
|---|---|---|
| A_naive（32-token 旋转均值） | 1 向量 | 0.6450 |
| A_blk8_mean（8-token 旋转均值，max/mean 池化） | **4 向量（与 K4 相同）** | 0.6521 |
| K4_mean（8-token 去旋转子质心） | **4 向量** | **0.9382** |

只把粒度从 32 降到 8（不去旋转），增益 **+0.7pp**；同样字节下做去旋转 + 子质心，增益 **+29.3pp**。
**所以去旋转的收益不是"粒度更细"造成的，这一条对方站得住。**
（注意 `A_blk8_max/mean` 两个数几乎相同，说明朴素臂内部池化方式不敏感。）

### 4.2 分组方式对照——**否掉我自己在 v3 P1 里提的"内容自适应子质心"**

同样是 4 个向量/32 token，只改"这 4 个向量怎么分"：

| 分组方式 | gt1 recall@64 | 相对 K4 |
|---|---|---|
| 连续切分（对方现状，K4） | **0.9382** | — |
| 块内随机打乱后切分 | 0.9114 | −2.7pp |
| 块内 k-means（Lloyd 25 轮，余弦） | 0.8845 | **−5.4pp** |

**连续（保局部性）分组最好，k-means 反而更差。** 我在 v3 第一部分 P1 里建议的
"内容自适应子质心"在这组数据上被实测否掉，应当撤回——相邻 token 的语义连贯性
比"块内聚类更紧"更有价值。请以这条实测为准，不要浪费 2–3 天在 k-means 上。

### 4.3 自我包含偏置——**可以放过**

把 query 自身那个 token 从它所在块的真值里剔掉（gt1b）：`B_matched 0.9004 → 0.8987`，
`K4_mean 0.9382 → 0.9365`，差异 ≤0.2pp。**我此前担心的"自我检索抬高 recall"在数值上可忽略，
这条不再列为必须修项**，但报告里写一句"真值含 query 自身 token，影响 <0.2pp"会更严谨。

---

## 5. 生产路径与基准不同步：一个具体的、可执行的发现

`kvmem_fusion/ifr.py` 的 `IVFMeanKIndex.retrieve()`（第 369–399 行）对分裂块的处理是：
把每个子质心都作为独立候选加入 `cand_keys`，统一打分后**按 block id 去重、保留排名最高的那个**。
这在数学上等价于 **max-pooling over sub-centroids**。

而基准的 headline 用的是 **mean-pooling**。他们自己的数字说明这两者不等价：

| 规则 | K=2 | K=4 |
|---|---|---|
| mean（基准 headline） | 0.9224 | 0.9382 |
| max（生产路径实际行为） | 0.8660 | 0.8369 |
| 差 | **−5.6pp** | **−10.1pp** |

也就是说：**按现在的生产代码跑，拿到的是 0.8660 / 0.8369，不是 0.9224 / 0.9382。**
且 K=4 的 max 规则比 K=2 的 max 规则更差（0.8369 < 0.8660），与 D-C 的假阳性分析一致。
需要二选一：(a) 生产改 mean-pooling（存全部子质心、按块求均值再排序，代价可忽略）；
(b) 基准改用 max 并接受 -5.6pp/-10.1pp。不能一边报 mean 的收益、一边部署 max 的规则。

---

## 6. 仍然存在的陈旧与缺失

| # | 位置 | 问题 | 处置 |
|---|---|---|---|
| 1 | `README.md` 第 157–159 行 | 仍写 `Strata_Naive_A = 0.5999`、`Monolithic_Bmid = 0.6104`、`Decoupled_K2 = 0.8816 ~ 0.9275`，与 `aa07e56` 产物（0.6450 / 0.6042 / 0.9224）**不一致**；`0.9275` 这个旧数从 `808be84` 一直残留到现在 | 同步为 0.6450 / 0.6042 / 0.9224，删 0.9275，并补 `Monolithic_B_matched = 0.9004`、`K2_Max/K4_Max`、`Random = 0.4910` |
| 2 | 产物 JSON `ground_truth_definition` | `[exact token-level attention]` 括注不成立（§2） | 改为 `mean_t cos(q_de, k_de(t)) [content-similarity proxy; NOT attention]` |
| 3 | 提交信息 | `confirming monotonic gain of decoupled sub-centroids (0.9004 -> 0.9224 -> 0.9382)` 只报 mean 链，未提同一张表里 `K4_Max (0.8369) < K2_Max (0.8660)`，也未提 `B_matched` 已比 `Strata_Naive_A` 高 39.6%（增益主要来自去旋转，不是子质心） | 提交信息补全：去旋转 +39.6%，子质心 +2.2pp/+3.8pp（2×/4× 索引字节），max 规则 −5.6pp/−10.1pp |
| 4 | 实验规模 | 仍只有 **layer 12 单层、Qwen2.5-0.5B 单模型、4096 token、128 块、top-64 = 50% 选择率、n_query=64** | 按 v3 第二部分协议补：≥6 层、≥2 模型、选择率降到 3%、Δ 扫描 |
| 5 | 真值无因果掩码 | 位置 p 的 query 看不到未来块 | 至少加一版 causal 真值并列报告 |

### 6.1 一条对对方有利的实测（应当写进报告）

选择率稀疏化后，去旋转的优势**变大**而不是变小（gt1 真值，recall@k）：

| 臂 | k=64 (50%) | k=32 (25%) | k=16 (12.5%) | k=8 (6.3%) | k=4 (3.1%) |
|---|---|---|---|---|---|
| A_naive | 0.645 | 0.441 | 0.335 | 0.303 | 0.298 |
| B_matched | 0.909 | 0.845 | 0.772 | 0.697 | 0.665 |
| K4_mean | 0.943 | 0.881 | 0.841 | 0.809 | 0.766 |
| Random | 0.501 | 0.259 | 0.133 | 0.066 | 0.020 |

在 KVMem 真实工况（~3% 选择率）下 `B − A` 从 +0.26 拉到 +0.37、`K4 − B` 从 +0.03 拉到 +0.10。
**所以"50% 选择率太容易"这个批评只影响绝对区分度，不影响去旋转结论的方向——这条对方可以放心用。**

---

## 7. 复现方式

```bash
# 1) 原样复跑他们的脚本（验证无硬编码）
python - <<'PY'
import sys; sys.path.insert(0, "<repo>")
from benchmarks.benchmark_decoupled_retrieval import run_decoupled_benchmark
run_decoupled_benchmark(out_json="/tmp/repro_aa07e56.json")
PY

# 2) 真值敏感性 + 预算匹配 + 分组对照（本文件全部数字的来源）
#    gt_probe.py：5 个真值 × 12 个臂 × k∈{64,32,16,8,4}
#    p2_align.py ：v2 P2 协议（raw / partial / excl-recent Spearman vs 真实因果注意力质量）
```

依赖：`torch 2.14.1`、`transformers 5.19.0`、Qwen2.5-0.5B-Instruct（fp32 eager）。
注意 `dtype=`（不是旧版 `torch_dtype=`）；Qwen2.5-0.5B 是 14 q-head / **2 kv-head** / head_dim 64，
任何 einsum 都要小心 GQA 展开维度。

---

## 8. 优先级（给对方的下一步）

| 优先级 | 动作 | 量级 |
|---|---|---|
| **P0-a** | 删掉 `[exact token-level attention]` 括注，改名为 content-similarity proxy | 5 分钟 |
| **P0-b** | README 同步为新数，删 0.9275 | 30 分钟 |
| **P0-c** | 统一生产与基准的池化规则（mean 或 max，二选一） | 半天 |
| **P1-a** | 加 causal 真值 + 真实 `q_proj` query 的一臂，并列报告 | 1 天 |
| **P1-b** | 预算匹配臂 `A_blk8` 常驻进基准（等字节对照） | 半天 |
| **P2** | ≥6 层 × ≥2 模型 × 选择率 3% × Δ 扫描 | 1 周 |
| **P3** | 保留连续切分，**撤回 k-means 子质心路线** | 0 天（省 2–3 天） |
