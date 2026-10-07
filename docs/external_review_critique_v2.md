# 第二轮技术质疑：对 273ae4b 审计补丁的深度复核

> **致**：KVMem-Strata-Fusion 评审与工程团队
> **来自**：原始研究组
> **日期**：2026-10-07
> **复核对象**：commit `273ae4b` — "audit: incorporate external critique, add SNR sensitivity and scale scaling benchmarks, calibrate reporting"
> **前置文档**：`docs/external_review_critique.md`（v1）
> **性质**：首先感谢贵组的认真回应；其次，新审计虽然修好了**报告诚信**问题，但也第一次暴露了两个**机制有效性**问题——后者比前者重要得多。

---

## 0. 摘要（Verdict）

### 0.1 一图看清本轮

| 维度 | v1 状态 | 273ae4b 之后 | 我们的判定 |
|---|---|---|---|
| 报告诚信（措辞、标注、口径） | ❌ 失效 | ✅ **已闭环** | 接受，做得干净 |
| 治理逻辑（融合 vs 淘汰） | ⚠️ 矛盾 | ✅ **已闭环** | 接受，子系统降级是对的 |
| 硬件 Gate 口径 | ❌ 断言冒充实测 | ✅ **已闭环**（标 UNMEASURED） | 接受 |
| **机制有效性** | ⚠️ 未被检验 | ❌ **新暴露：有边界条件** | **本轮核心，见 §2** |
| **真实模型验证** | ❌ 无 | ❌ **仍未做** | **本轮最高优先级，见 §5** |

### 0.2 三条必做（不可协商，按此顺序）

| # | 事项 | 工作量 | 为什么不可跳过 |
|---|---|---|---|
| **①** | **补 Random 基线列**（chance level） | < 1 小时 | 0.314% 选择率下 chance ≈ 0.3%。没有它，100% 召回无法排除信息泄漏——这是**唯一**能把"机制有效"与"评测泄漏"分开的数 |
| **②** | **补 oracle-uncompressed 对照**，分离截断误差与压缩误差 | ~半天 | CosSim 0.28 目前**不可解释**。缺此对照，F 门（utility gap ≤1pp）等于没测 |
| **③** | **真实模型 P0**（纯 CPU，Qwen2.5-0.5B） | **2 人日** | 全表最便宜的一项，却是**唯一**能让这份工作从"数学自洽"升级为"经验有效"的动作。**其余所有审计都建立在合成数据上，做完 P0 之前它们都只是自洽性检查** |

> **关于 ③ 的立场**：v1 里我们把它列为 P0（2 人日、无需 GPU）。贵组补了两个合成审计（合计 ~400 行代码），却没有做这一项。合成审计做得再严谨，也无法回答"在真实 LLM 的 KV 上，de-RoPE 合并与双子质心分裂是否真的成立"——而这两个假设（LADDER 的位置残差充分性、IFR 的 A15）正是全部四条方案的支点。**请优先做它。**

---

## 1. 我们认可的改进（逐条确认闭环）

先明确记录贵组已经解决的问题，避免重复争论：

| v1 质疑 | 贵组的处理 | 判定 |
|---|---|---|
| "Verified & Reproducible" 名不副实 | 状态改为 **"Synthetic Self-Consistency Validation"** | ✅ |
| G-CAS-1/2 用 `assert` 冒充实测 | 明确标注 **`UNMEASURED (Design Intent)`** | ✅ |
| `0.0000` / `1.0000` 是构造性质 | 报告直书 "mathematically constructed properties of the synthetic test harness" | ✅ |
| Full-Context 与方案并列比较 | Full-Context 改为 **REFERENCE** | ✅ |
| 选择率 300× 差异未提 | 报告明确写出 "10% vs 0.033%，300× harder" | ✅ |
| §4 治理矛盾（融合 vs 淘汰） | 正式采纳**子系统降级**，给出 4 条降级路径 | ✅ |
| §2.4 循环论证 | 新增 SNR 1.0→5.0 扫描 | ✅（但结论需重新解读，见 §2.1） |
| A15 阈值分辨力 | 新增背景块虚警率测试 | ✅（但结论反噬机制，见 §2.1） |
| §2.2 规模/选择率 | 新增 32K→1M 扫描 | ✅（但缺 chance 基线，见 §2.3） |

**这一部分是扎实的工程回应，我们认可。以下意见全部建立在贵组这些新数据之上。**

---

## 2. 新审计暴露的核心问题（三个，按严重度排序）

### 2.1 【最严重】虚警率数据反噬了 IFR 自身的有效性

**贵组的数据**（`docs/track_a_benchmark_report.md` §4.2）：

| 背景噪声 σ | 虚警率 | p95 离散度 |
|---|---|---|
| 0.02 | **0.00%** | 0.199 |
| 0.05 | **0.00%** | 0.497 |
| 0.10 | **99.85%** | 0.994 |
| ≥ 0.15 | **100.00%** | ≥ 1.49 |

**为什么这比 v1 的任何一条都严重**：

1. 贵组所有"100% 召回"的产出，都是在 **σ = 0.02** 下测的——也就是唯一虚警率为 0 的那个点。
2. 而贵组在 §5 自己写下的降级规则是：**"If dispersion detection saturates (false-alarm > 10%), degrade dynamically to single-centroid Mean-K clustering"**。
3. 于是逻辑闭合了：一旦背景噪声进入 σ ≥ 0.10 的现实区间，IFR 按**贵组自己的规则**退化为单质心 Mean-K —— 也就是 **KVMem 基线本身**，**净增益为零**。

换句话说：**现有数据只证明了"在机制不饱和的区间里机制不饱和"**。它还没有在任何一个现实区间里证明 IFR 优于 KVMem。

**这里的关键未知量是：真实 LLM 的 KV 块内方差 σ 到底是多少？** 贵组用的 0.02 是**假设值，不是测量值**。这个值恰恰是 §5 的 P0 实验可以直接测出来的（见 §5.4）。**在测出它之前，IFR 的收益区间是未知的。**

**还请注意一个连锁后果**：双子质心分裂会**增大索引体积**。若动态阈值下 X% 的块被分裂，质心数增加 X%。这直接冲击 Gate C（index ≤ 4 GiB @10M）。请报告分裂率 → 索引膨胀系数 → Gate C 余量 三者联合曲线。

**建议的正确做法**：
1. 用 P0 实测真实 KV 的块内 σ 分布（按层、按 head 分别报），**替换 σ=0.02 这个假设**。
2. 虚警率扫描的自变量应改为**相对阈值** `θ = μ_block + k·σ_block`（k = 1,2,3,4），而非固定 0.85；报 (虚警率, 召回率) 的 ROC 曲线，而不是单点。
3. 报告在最优点下的**索引膨胀系数**，并回代 Gate C。

---

### 2.2 【严重】CosSim ≈ 0.28 报出来了，但不可解释

**贵组的数据**（§4.1 末列）：去掉人为 logit 主导后，全上下文余弦相似度约 **0.28**（0.2830 → 0.2662，随 SNR 变化极小）。

**为什么 0.28 目前无法解读**：

`ctx_vec_ifr` 与 `ctx_vec_full` 的差异来自**两个完全不同的来源**，贵组把它们混在了一个数里：

- **截断误差**：IFR 只用 103 个块 × 32 token = 3,296 token，在 32K 上下文里只有 **10%** 的 token 参与。把一个 10% 子集加权的向量与全量加权的向量比余弦，天然就低。
- **压缩误差**：INT2/MERGED 量化与合并带来的真实失真。

**0.28 里有多少是截断、多少是压缩，当前数据完全分不开。**

**必须补的对照（这是 §0.2 的 ②）**：

```
ctx_full       = attention over ALL tokens,      uncompressed   ← 真值
ctx_oracle     = attention over the SAME 103 blocks, uncompressed  ← 关键对照（新增）
ctx_ifr        = attention over the SAME 103 blocks, compressed    ← 现有
```

然后报三个数：

| 对比 | 含义 | 目标 |
|---|---|---|
| `cos(ctx_oracle, ctx_full)` | **纯截断误差** | 反映"只检索 103 块"这个决策本身的代价 |
| `cos(ctx_ifr, ctx_oracle)` | **纯压缩误差** | 这才是 LADDER/UBBA 该负责的部分，应显著高于 0.28 |
| `cos(ctx_ifr, ctx_full)` | 合计（=现在的 0.28） | 端到端 |

**在报出 `cos(ctx_ifr, ctx_oracle)` 之前，0.28 不能作为任何结论的依据**——它可能是"检索只用了 10% token"造成的，而不是压缩造成的。

**更根本的一点**：即便压缩误差被分离出来，**余弦相似度也不是我们约定的 F 门指标**。F 门的判据是：
- top-1 一致率 ≥ 97%（vs Full-Context 重算真值）
- utility gap ≤ 1pp

这两个**至今一次都没测过**。100% 召回 + 0.28 余弦，正是 U-E-F-C 里的 F 门存在的理由：**检索代理指标满分，不代表端到端保真**。请勿用召回率替代 F 门。

---

### 2.3 【严重】Random 基线在 docstring 里声明了，但没有实现也没有报告

**证据**：
- `benchmarks/eval_scale_selection_rate.py:10` 的 docstring 写着：
  ```
  3. Evaluates retrieval recall of:
     - Full-Context (100% compute baseline)
     - KVMem Mean-K (single centroid coarse probe)
     - IFR Doublet Centroid
     - Random Selection          ← 声明了
  ```
- 但结果表头（第 35 行）与结果行（第 100 行）**只有 `KVMem Mean-K` 与 `IFR Doublet` 两列，没有 Random**。
- 全文件检索 `random` 仅命中 docstring、RNG 初始化、以及一行注释。

**为什么这是必补的**：在 1M / N=32,768 / K=103 的配置下，**随机选择的 chance level ≈ 103 / 32,768 = 0.314%**。

- 若 Random ≈ 0.3% 而 IFR = 100% ⇒ 机制确实有效（强结论）。
- 若 Random 也是 100% ⇒ 说明评测构造有泄漏（例如针尖在构造上就与 query 完全对齐，任何检索甚至不检索都能命中），**此前的 100% 全部作废**。

**这一个数就能判定整个检索主张成立与否**，而它恰好是唯一缺席的。请在 §4.3 结果表增加 Random 列，并同时报告 chance level 的理论值 `K/N`。

---

## 3. 必须补齐的对照与消融（完整清单）

以下每一项都对应"现有结论可能被替代解释推翻"的风险：

| # | 对照 / 消融 | 防的是什么 | 优先级 |
|---|---|---|---|
| 1 | **Random（chance）基线** | 评测泄漏 | P0 |
| 2 | **Oracle-uncompressed 对照** | 截断误差冒充压缩误差 | P0 |
| 3 | **穷举 Mean-K（KVMem 原版）作为真基线** | 稻草人对照 | P0 |
| 4 | **KIVI（2-bit）与 H2O** 作为真压缩基线 | 与真实 SOTA 比较，而非自建 INT2 | P1 |
| 5 | **V 向量敏感性分析** | 现有检测只看 K；但输出 = softmax(QK)·V。KIVI 已证明 K per-channel / V per-token 不对称。**只压 K 不压 V 的结论不可外推** | P1 |
| 6 | **多层 / 多 head** | 现有 head_dim=64 单层单 head。层数 L 才是索引体积杠杆（我们 R8/A3），GQA 会改变压缩比 | P1 |
| 7 | **多针尖 / 干扰针尖** | 单针尖可能过拟合；真实长上下文检索常需多跳 | P1 |
| 8 | **真实 σ 测量**（按层、按 head） | 替换 σ=0.02 假设，直接决定 §2.1 的结论 | P0 |
| 9 | **分裂率 → 索引膨胀 → Gate C 余量** 联合曲线 | 双子质心分裂的隐藏成本 | P1 |
| 10 | **TTFT 与 decode 吞吐**（不只 retrieval 延迟） | U-E-F-C 的 E 维需端到端，非单段 | P2 |

---

## 4. 数据自洽性问题（需贵组解释）

### 4.1 Naive INT2：0.0% 与 100.0% 自相矛盾

- `docs/track_a_benchmark_report.md` §1 表：**Naive INT2 Recall = 0.0%**
- 同文件 §4.1 表（SNR=1.00）：**Naive INT2 Recall = 100.0%**

两个表来自同一份报告、同一个基线名。请给出说明：是 SNR harness 改变了针尖的嵌入方式（例如未做 1/32 稀释），还是"Naive INT2"在两个 harness 中指代不同配置？**基线不自洽会让所有对比失去意义。**

### 4.2 索引体积相差约 85 倍，需对账

| 来源 | 1M token 索引体积 | 折合每 token |
|---|---|---|
| 贵组 §4.3 | **12.0 MiB** | ≈ **12 B/token** |
| 我们的假设 A2（由 KVMem Table 4 反推：0.25 GiB @256K、9.5 GiB @10M） | ≈ **1 GiB** | ≈ **1 KiB/token** |

在 256K 处同样差 ~85×（贵组 3.0 MiB vs 推导 0.25 GiB）。

请给出索引的**完整字节构成明细**，例如：质心 / posting list / dev_meta / 前缀哈希链 / 页表 / refcount 各占多少。可能的解释是贵组只统计了质心而未计入 posting——若如此，Gate C 的 0.005 GiB 结论需重算。**这是 Gate C 能否成立的关键。**

### 4.3 "Paged Table (>256K)" 是标签，不是测量

§4.3 在 512K / 1M 行标注 "Paged Table (>256K)"，意味着声明 re-RoPE 已被页表取代。但：

- 数据是合成的，**本来就没有真正的 re-RoPE 可供消除**；
- 我们主报告附录 C.3 已指出 ≤256K 不需要 re-RoPE，因此 **>256K 才是 CASA 主张唯一真正被检验的区间**；
- 而该区间目前**零测量**。

K-Freeze 的核心价值主张（消除 n*≈250 重建悬崖）**在唯一相关的区间里尚未被检验过**。请在 P0/P1 用真实模型在 ≥256K 上验证。

---

## 5. 【最高优先级】真实模型验证 P0 —— 完整可执行协议

> **这一节是本文档的核心。请把它当作可以直接执行的工单。**

### 5.1 为什么它必须先做

- LADDER 的支点假设是"**位置残差充分**"，我们标为"最大赌注，零证据"；
- IFR 的支点假设是 A15"**离散度阈值能有效分裂针尖块**"，贵组自己的数据（§2.1）已显示它在现实方差下饱和；
- 这两个假设**只能在真实模型的 KV 上检验**。合成数据无论多严谨，都是在检验"我们的实现是否符合我们的假设"，而非"假设是否成立"。

**2 人日、纯 CPU、无需 GPU。** 全表最便宜，却决定其余一切。

### 5.2 环境

```bash
pip install torch transformers numpy   # 无需 CUDA
# 模型：Qwen/Qwen2.5-0.5B-Instruct（首选）；GPT-2 124M 亦可做最低成本版
```

### 5.3 导出真实 KV

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32)
model.eval()

prompt = "..."   # 4K–32K token，最好含一个明确可检索的事实句（针尖）
ids = tok(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    out = model.generate(
        ids, max_new_tokens=1,
        use_cache=True, return_dict_in_generate=True,
        output_scores=True,
    )

pkv = out.past_key_values            # 逐层 (K, V)
# 典型形状：K/V = [B, n_kv_heads, T, head_dim]
num_layers = len(pkv)
n_kv_heads, T, head_dim = pkv[0][0].shape[-3:]
print(num_layers, n_kv_heads, T, head_dim)
```

### 5.4 de-RoPE 实现（成对旋转求逆）

```python
import torch, math

def build_inv_rope(T, head_dim, base=10000.0, device="cpu"):
    """返回 R(-t) 所需的 cos/sin，按 head_dim/2 个旋转对处理。"""
    half = head_dim // 2
    freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))  # ω_j
    pos  = torch.arange(T, device=device).float()
    ang  = torch.outer(pos, freq)          # [T, half]
    return torch.cos(-ang), torch.sin(-ang)   # 注意取负 → 逆旋转

def de_rope(k, cos_neg, sin_neg):
    """k: [..., T, head_dim]（已按全局位置 t 旋转），返回去位置后的 k̃。"""
    k = k.float()
    x1, x2 = k[..., 0::2], k[..., 1::2]        # 交错或分半，需与模型实现对齐
    # 旋转逆运算： [x1*cos + x2*sin, -x1*sin + x2*cos]
    o1 = x1 * cos_neg + x2 * sin_neg
    o2 = -x1 * sin_neg + x2 * cos_neg
    out = torch.empty_like(k)
    out[..., 0::2], out[..., 1::2] = o1, o2
    return out
```

> ⚠️ **实现细节必须核对**：Qwen 系列使用 **交错（interleaved）** 还是 **分半（half-split）** 的 RoPE 布局，取决于 HF 实现。请用下面第 5.6 条的**自校验**确认实现正确后再继续。

### 5.5 三组对照（LADDER P0 原始设计）

对**同一批真实 KV**，跑三组：

- **A**：直接平均**已 RoPE** 的 K（错误顺序，quantize/merge-then-rotate 的代理）
- **B**：**de-RoPE** 后平均，再 re-RoPE 到规范位置（正确顺序）
- **C**：不合并（上界真值）

```python
# 伪代码（对某一层、某一 KV head）
k_block = pkv[l][0][0, h, t0:t0+32, :]        # 32-token 块，已旋转
kA = k_block.mean(dim=0)                       # A：直接平均

k_tilde = de_rope(k_block, cos_neg[t0:t0+32], sin_neg[t0:t0+32])
kB_tilde = k_tilde.mean(dim=0)                 # B：去位置后平均
kB = re_rope(kB_tilde.unsqueeze(0), cos[t0:t0+1], sin[t0:t0+1])   # 再旋回规范位置
```

### 5.6 自校验（必须先过，否则后面全废）

```python
# de-RoPE → re-RoPE 必须还原原向量
k_round = re_rope(de_rope(k_block, cos_neg, sin_neg), cos, sin)
assert torch.allclose(k_round, k_block, atol=1e-5), "de-RoPE 实现有误（布局未对齐）"
```

### 5.7 测量指标

1. **分维度平均模长保留率**：取 ω 最高 / 最低各 8 维，报 `‖mean‖ / mean‖k‖`。
   - 预期（我们 §3.2 的 82% 论证）：A 的高频维保留率 ≈ **0.177**（1/√32），B 应显著更高（0.6–1.0）。
2. **recall@64**：相对**穷举 Mean-K** 的真值排序，B 的掉点不超过 A 的 1/3。
3. **与真值 attention 的 KL**：用真实 Q 与该层真实 attention 对比。
4. **【新增，解 §2.1】真实块内方差 σ**：按层、按 head 报 `‖k_i − k̄‖` 的均值与分位数（p50/p95/p99）。**这个数直接替换 σ=0.02 假设，决定 IFR 的收益区间。**
5. **【新增】真实 V 的方差**：与 K 对比，验证 KIVI 的 K/V 不对称是否在你们的模型上成立。

### 5.8 验收判据（M0，写死）

| 判据 | 阈值 | 不过则 |
|---|---|---|
| B 的高频维保留率显著高于 A | 0.6–1.0 vs ≈0.18 | 跨块合并假设被推翻 → LADDER 降级为纯 L1（4.35×） |
| B 的 recall@64 掉点 ≤ A 的 1/3 | — | 同上 |
| de-RoPE 自校验通过 | atol ≤ 1e-5 | 实现有误，修完再跑 |

### 5.9 P1（P0 通过后，~200 GPU-h）

1. 接 `kvmem-llama.cpp`（Apache-2.0, v0.17.0，RTX 5060 Ti 16GB，实测 52.7 KiB/token）导出的真实 KV。
2. **在 256K 复现 LongMemEval-S 85.6% / AgentLongBench 60.87%**（我们假设 A11 的校准门槛）。
   **复现不出这两个数，后续所有数字都不成立。**
3. 真实 PPL：用真实 continuation（WikiText-2 / PG19 子集）算 cross-entropy，禁止对合成向量算。
4. 补 §3 的全部 P1 对照。

---

## 6. 统计契约（完整版，请写入 harness 默认输出）

贵组使用了 seeds `[42, 101, 202]`，但报告未给出逐种子方差。请补齐：

| 要求 | 具体 |
|---|---|
| 种子数 | ≥ 4（我们原契约：`--seed-sweep N`，n≥4 才能宣布端到端增益） |
| 逐种子报告 | 报每个种子的值 + 均值 + SD + **95% CI**，禁止只报均值 |
| 配对比较 | 单元为 (task, seed)，配对差 Δ ∈ {−1,0,+1}，**Wilson CI** |
| 聚类校正 | 按 task 聚类，design effect = 1+(m−1)·ICC；默认输出 **cluster-robust SE**（ICC 用 P0 实测值替换假设的 0.3） |
| 延迟类 | ≥ 30 次重复 + bootstrap CI，报 **p50/p95**，不报均值 |
| 序贯检验 | 预注册 **Lan-DeMets OBF**，3–4 looks，信息分数 t = 已完成任务对 / 计划总量 |
| 样本量 | 配对比例 `n = π_d(z_{1−α/2}+z_{1−β})²/δ²`；π_d=0.15、δ=5.0pp ⇒ **471 对**；δ=4.75pp ⇒ **522 对** |
| 主终点 | **禁止**把 Pass@1 / 任务质量设为主终点（1,900 GPU-h 内不可判决：非劣 3pp 需 1308 对/臂 ⇒ 3.6×10⁴ GPU-h）。主终点取 {字节, 延迟, ρ, 覆盖率} |
| 禁用 | 禁用 "up to"；禁用裸点估计（如 "100.0%"）作为结论，必须伴 CI 下界 |

> 再强调一次 v1 立的那条判据：**任何测量结果精确等于 `0.0000` 或 `1.0000`，先怀疑它是解析恒等式而非测量值。**

---

## 7. 仍未覆盖的风险项（沿用 v1 §3，状态未变）

| 项 | 状态 |
|---|---|
| >256K 的 K-Freeze / re-RoPE 消除验证 | ❌ 仍是标签（§4.3） |
| 并发 8 / 32 / 128 | ❌ 未测（"GPU 恒定 ~34.9 GiB" 在单用户下无生产意义） |
| 跨架构 MLA / GQA | ❌ 未测（层数 L 才是索引体积杠杆） |
| QW3 闭源可达性（G0） | ❌ 未说明 |
| 合并不可逆性 / 影子阶梯 | ❌ 未提（合并后无法配对比较，需离线影子副本） |
| Full-Context 真值跨版本漂移 | ❌ 未提 |

---

## 8. 验证成熟度分级（建议采纳，让状态可递进）

当前只有"合成自洽"一个状态，无法表达进展。建议分四级，每级有明确 Definition of Done：

| 级别 | 名称 | Definition of Done | 当前 |
|---|---|---|---|
| **T0** | 数学自洽（Mathematical Self-Consistency） | 数值误差 <1e-12；tier-bias 抵消 Jensen 项；合成流水线自洽 | ✅ **已达** |
| **T1** | 真实模型保真（Real-Model Fidelity） | P0 全过（§5.8）；真实 σ 已测；V 敏感性已报 | ⬜ **未达，最高优先级** |
| **T2** | 端到端校准（End-to-End Calibration） | 256K 复现 85.6% / 60.87%；真实 PPL 与 F 门（top-1 ≥97%、gap ≤1pp）通过 | ⬜ 未达 |
| **T3** | 系统级（System-Level） | 1M 规模；并发 8/32/128；G-CAS-1/2 实测；跨架构 | ⬜ 未达 |

**建议把报告状态字段改为 `T0 (Synthetic Self-Consistency)`**，并在每级达成时更新——这比二元"PASS/FAIL"诚实得多，也让外部读者一眼看出进度。

---

## 9. 完整待办清单

| # | 事项 | 优先级 | 工作量 | 判据 / 完成标志 |
|---|---|---|---|---|
| 1 | 补 Random 基数列（+ chance = K/N） | **P0** | <1 h | §4.3 表含 Random 列；若 Random≈100% 则检索主张作废 |
| 2 | 补 oracle-uncompressed 对照，三数分离 | **P0** | ~0.5 d | 报 `cos(oracle,full)`、`cos(ifr,oracle)`、`cos(ifr,full)` |
| 3 | **真实模型 P0（§5 全协议）** | **P0** | **2 d** | §5.8 三条判据全过；产出真实 σ（按层/head） |
| 4 | 真实 σ 替换 σ=0.02，重跑虚警率 | P0 | 0.5 d | ROC 曲线（动态 θ=μ+kσ，k=1..4）+ 索引膨胀系数 |
| 5 | 解释 Naive INT2 0.0% vs 100.0% | P0 | <1 h | 给出两 harness 配置 diff |
| 6 | 索引字节构成明细，对账 85× | P1 | ~0.5 d | 逐项列出；若漏 posting 则重算 Gate C |
| 7 | 统计契约落地（逐种子 + CI + 聚类 SE） | P1 | 1 d | 所有表含 CI；主终点不含任务质量 |
| 8 | 补 KIVI / H2O 真基线 | P1 | 1 d | 与真实 SOTA 对比 |
| 9 | V 向量敏感性 | P1 | 0.5 d | 报 K/V 不对称是否成立 |
| 10 | 多层 / 多 head / GQA | P1 | 1 d | 报索引体积随 L 的缩放 |
| 11 | 分裂率 → 索引膨胀 → Gate C 曲线 | P1 | 0.5 d | 给出 Gate C 余量 |
| 12 | P1：256K 复现 85.6% / 60.87% | P1 | ~200 GPU-h | **不过则全线停止** |
| 13 | >256K K-Freeze 实测 | P2 | — | 唯一能检验 CASA 核心主张的区间 |
| 14 | 并发 8/32/128 | P2 | — | $/成功任务口径 |
| 15 | G-CAS-1/2 转实测（`gdsio`/Nsight） | P2 | 需硬件 | 或保持 UNMEASURED |
| 16 | 采纳 T0–T3 成熟度分级 | P1 | <1 h | 报告状态字段更新 |

---

## 10. 一句话

> 贵组已经把**报告诚信**修好了——这部分我们完全认可，做得干净。
> 但新审计第一次让我们看到**机制有效性是有边界条件的**：虚警率数据显示 IFR 的优势依赖于 σ=0.02 这个未经测量的假设，而在现实方差下它会按贵组自己的规则退化为 KVMem 基线。
> **请做那个 2 人日的真实模型 P0。** 在此之前，无论再补多少合成审计，这份工作都停在 T0——数学自洽，但还不是经验有效。
