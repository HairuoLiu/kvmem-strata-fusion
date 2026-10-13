# OG 复核：`c5e76c8`（Track A 最新一版）

**复核方**：原始研究组（OG）
**日期**：2026-10-11
**对象**：`c5e76c8 feat(audit-v3): calibrate T1-PARTIAL, add directional dispersion and decouple retrieval-execution`
**涉及文件**：`docs/track_a_benchmark_report.md`、`docs/comparative_study_report.md`、`kvmem_fusion/ifr.py`、`tests/test_snr_and_scale_audit.py`

> 事实核对：截至 2026-10-11，`origin/main` HEAD = `7ecf90e`（OG）。外部团队最后一个提交是 `c5e76c8`（2026-10-08 00:04）。无其他分支、无 PR、无 issue。**本文件是对他们最新一版的完整复核。**

---

## 1. 先说好的：这一版有真实改进

| 项 | 我们的评价 |
|---|---|
| 研究诚信署名声明（attribution disclaimer）加入两份报告 | ✅ 采纳了我们 v3 §9.1 的意见 |
| `file:///` 死链清除 | ✅ 已确认无残留 |
| T0 → **T1-PARTIAL** 自我降级 | ✅ 诚实，且与我们独立判断一致 |
| 把 OG 数据**并排引用**（abs 误差双栏、σ 双栏、语料双栏） | ✅ 这是真正的交叉验证姿态，值得肯定 |
| σ 通道方差分解并入报告（top-8 占 34.8%，8 通道占 80%） | ✅ 正确引用 |
| 新增 `compute_normalized_dispersion`（方向空间度量） | ⚠️ **概念正确，但见 §2** |

**结论：他们确实在认真对待外部评议，报告诚信层面已经过关。** 下面三条是工程层面的问题，不是态度问题。

---

## 2. 三条实质问题（均已实测）

### 问题 A：修复没有接线 —— 生产路径仍然走旧逻辑

`use_normalized=True` 在整个代码库里**只出现一次**，就在他们新加的那个测试里：

```
tests/test_snr_and_scale_audit.py:119  AntiCollapseSplitter(dispersion_threshold=0.85, use_normalized=True)
```

而生产路径的构造处：

```
kvmem_fusion/ifr.py:208   self.anti_collapse = AntiCollapseSplitter(dispersion_threshold=dispersion_threshold)
```

**没有传 `use_normalized`，默认值是 `False`。** 所以 `track_a` §5.2 宣称的

> "We have upgraded IFR's dispersion detector to evaluate directional cosine space (`norm_dispersion` ≈ 0.80), **preventing false alarms from outlier channel spikes**"

**在生产路径上并未生效。** `norm_dispersion` 字段会被计算并填上，但没有任何门限读它。

**建议**：把 `use_normalized` 默认改为 `True`，或在 `IFRAntiCollapseIndex.__init__` 显式传参，并补一条生产路径的断言测试。

### 问题 B：默认门限在真实 KV 上 **100% 触发**

实测（Qwen2.5-0.5B-Instruct，4096 token，24 层 × 2 KV head，128 块/层，去 RoPE 后 Key）：

| 统计量 | 实测值 |
|---|---|
| `raw_max`（= `IFRBlock.dispersion`，默认门限读这个） | **11.462** |
| `raw_mean`（= `avg_dispersion`） | 8.117 |
| `norm_max`（= `norm_dispersion`） | 0.549 |
| `norm_mean` | 0.383 |

`raw_max` 的分位数：p50 = 11.029，p95 = 17.182。

**门限 0.85 vs 真实 `raw_max` ≈ 11.46 → 触发率 100.0%。** 而且阈值一路提到 3.0 仍然是 100%：

| 门限 | raw_max 触发率 | norm_max 触发率 |
|---|---|---|
| 0.85 | **100.0%** | 5.8% |
| 1.0 | 100.0% | 0.7% |
| 1.5 | 100.0% | 0.1% |
| 3.0 | 100.0% | 0.0% |

**含义**：anti-collapse 在默认配置下不是一个"检测器"，它是一个常数——**每一个块都会被拆成 doublet**。
- 索引规模无条件下翻倍；
- "needle 检测"零选择性（什么都是 needle = 什么都不是 needle）；
- 这正是我们 v2 质疑书里点出的**误报饱和**（false-alarm saturation），现在有了确切数字：**100%**。

复现：`python benchmarks/ifr_dispersion_check_og.py`

### 问题 C：阈值口径错了（但数值侥幸落在合理区间）

`ifr.py:36` 的注释写："normalized MAD ~0.7965"，这个数来自我们 v1 审计。但：

- 我们的 0.7965 是 **mean**（块内逐 token 距离的平均），且参考点是**整层的全局均值**，不是块内均值；
- 而 `compute_normalized_dispersion()` 返回的是 **块内 max**。

两个统计量差两层。实测：块内 `norm_mean` = 0.383，块内 `norm_max` = 0.549，**max/mean ≈ 1.43**；而我们那个 0.7965 因为用了全局参考点，比块内均值还大一倍。

**结论**：0.85 这个数**凑巧**落在 `norm_max` 的 p95（0.868）附近，给出 5.8% 的触发率——**数值可用，推导链是错的**。这种"对的数、错的理由"最危险：一旦换模型或换语料，巧合不再成立，而没人知道阈值当初是怎么来的。

**建议**：把注释改成实际口径，并把阈值写成"由 p95 标定"并给出标定脚本。按 5% 触发率反推：`θ_norm_max ≈ 0.868`。

### 问题 D：层间差异 20 倍，单一全局阈值不可能成立

各层 `norm_max` 均值：

| 层 | 0 | 4 | 8 | 12 | 16 | 20 | 23 |
|---|---|---|---|---|---|---|---|
| norm_max | **0.034** | 0.428 | 0.331 | 0.566 | **0.702** | 0.592 | 0.628 |

第 0 层的方向离散度只有 0.034——**该层去 RoPE 后的 key 方向几乎完全一致**。用 0.85 的全局阈值，L0 永远不触发，L16 频繁触发。

**建议**：阈值按层标定（或按层的分位数自适应）。这和我们在 §P4 给出的「逐层位置先验 α」是同一类结论——**这个系统里的每一个阈值都必须是 layer-wise 的**，全局常数会在浅层失效。

### 问题 E：新测试用的是合成数据，且比真实分布小 2 倍

`tests/test_snr_and_scale_audit.py` 的构造：

```python
keys = rng.randn(block_size, dim) * 0.1
keys[:, 0] = 8.0 + rng.randn(block_size) * 1.5   # "Massive activation outlier channel"
```

断言 `raw_disp > 3.0`。而真实数据是 `raw_mean = 8.117`、`raw_max = 11.462`。**合成数据的离散度只有真实的约 1/2**，测试通过不代表真实分布下成立。

这正是我们第一轮质疑书的核心问题（用合成数据充当测量），在新写的测试里又出现了。**建议**：加一条用真实 HF KV 快照的回归测试（把 KV 存成 fixture，几十 KB），而不是只断言合成数组的量级。

---

## 3. 与我们的冲突：一处实质、一处口径

### 实质冲突（重要）

`track_a` §5.4 及 §1 摘要采用我们**已撤回**的 v1 数字（A 0.6846 / B 0.6685 / 0.98×），并据此断言：

> "retrieval degradation is caused by naive block-mean pooling, **NOT** by RoPE phase cancellation"

我们 v2 实测的因果分解（Qwen2.5-0.5B，L12，50% 选择率）：

| 来源 | 量级 |
|---|---|
| **① RoPE 处理方式** | **+0.254**（0.637 → 0.891） |
| ② 质心分辨率（m=1→4） | +0.044（0.891 → 0.935） |
| ③ 迁移距离（Δ=0→2048） | **−0.31** |

**① 是最大单一杠杆，比 ② 大 5.8 倍。** 他们把 ① 判为无效、把 ② 排第一——**因果归因反了**。

详见已发出的更正函：[`docs/og_correction_to_track_a_5_4.md`](og_correction_to_track_a_5_4.md)

### 关键澄清：结论错，但架构动作对

他们的两项架构动作**被我们的数据独立证实是对的**：

- **升级 3（多质心）** ✅ 我们的质心定律给出定量依据：保真度只取决于 **tokens/质心**（组内极差 ≤0.06，组间 0.55→1.00），且**同等索引预算下加质心优于缩小块**（B32/m4 = 0.711 > B8/m1 = 0.654，双模型一致）。
- **升级 1（解耦）** ✅ 方向正确，但**必须补一个字段**：子质心活在哪个空间？若在原生（旋转后）空间，会丢掉 ① 的全部 +0.254。

所以这不是"他们错了"，而是：**证据链用错了，但落地的架构方向是对的**。修正成本很低——换一组数，结论就顺了。

### 口径冲突（次要）

他们定级 **T1-PARTIAL**。我们认为 T1 的准入应是 **≥2 个架构**：同一个效应在 Qwen2.5-0.5B 上是 1.40×，在 Qwen3-0.6B 上只有 1.07×；迁移惩罚 −31pp vs −9pp。单模型不足以支撑普适结论。

---

## 4. 给他们的 5 条具体改动

1. **`use_normalized` 默认改 `True`**，或在 `ifr.py:208` 显式传参。否则这个修复是死代码。
2. **门限重新标定**：按 `norm_max` 的 p95 取 **θ ≈ 0.868**（5% 触发率），并把注释里的 0.7965 换成实际口径。若坚持用 `raw`，θ 应取 **≈ 17.2**（raw_max 的 p95）——但强烈建议用方向空间。
3. **阈值改为 layer-wise**，至少按层分位数自适应。L0 与 L16 差 20 倍。
4. **补一条真实 KV fixture 的回归测试**，替换纯合成断言。
5. **更新 §5.4** 为更正后的数据（0.6367 / 0.8913 / **1.40×**），并把因果优先级调整为 RoPE > 质心 > 迁移。

---

## 5. 总体判断

**这一版是真实进步**：报告诚信层面我们已经没有可指摘的，交叉验证的姿态也值得肯定。σ 的通道分解、方向空间度量这两个**概念**都源自我们的审计，他们把它落成了代码——这是实打实的贡献。

**但落成的方式有三个缺口**：没接线（默认关闭）、阈值口径错（凑巧可用）、测试用合成数据（量级差 2 倍）。其中**没接线**是最要紧的——报告里写着"已升级、防止误报"，而生产路径上 100% 的块仍在被拆分。

**净评价**：方向跟得对，工程落地欠一步。修完上面 5 条，这版可以算 T1 达标。
