# 核验报告：对方 `aa49a7b` 是否把我们提的都改了？

> 核验对象：commit `aa49a7b` "fix(challenge): resolve double-counted Deff, add C++ cosine sub-centroids, fix macOS madvise, enforce T0.1 guard"
> 提交信息自陈修了 4 项。**本文逐条实测核对，结论：修好了 3 项，第 4 项是半成品，其余 ~16 项一项未动。**
> 方法：全部结论用命令产出，可复现；凡是我方检索出错的地方都已在文末记录。

---

## 0. 一句话结论

**没有全改。** 他们修的 3 项都是"**敲键盘就能改**"的纯代码问题；而**需要真正跑实验**的项（复现 0.9350、换 v2、Δ 重定位扫描、多模型、多层、选择率、端到端）——**一项都没做，本次提交没有任何新增实验文件**。

更值得注意的是：他们加的 C++ 余弦函数**只定义没调用**，是我们早就指出过的同一类错误（`use_normalized` 当初就是这样变成死代码的）。**同一个坑掉两次。**

---

## 1. 本次改了什么

`git show --name-status aa49a7b`：6 个文件，**全部是修改（M），没有任何新增文件**。

```
benchmarks/run_mac_layer_streaming_sim.py | 29 +++---
csrc/ifr_lse.cpp                          | 54 +++++++
csrc/ifr_lse.hpp                          | 11 +++
kvmem_fusion/stopping.py                  |  9 ++-
tests/test_snr_and_scale_audit.py         | 15 ++--
tests/test_stopping_rules.py              |  7 +--
```

**没有新增任何 benchmark / 实验脚本** → 意味着没有任何新数据被产生。

---

## 2. 逐条核验结果

### ✅ 真修好的（3 项，均已实测）

| 项 | 他们的改法 | 我的实测验证 |
|---|---|---|
| **T4.1 Deff 双重计数** | 把 `se_adj = se_raw * sqrt(Deff)` 移入 `else` 分支，配对方差不再乘 Deff；并把测试期望从 `[0.065,0.075]` 改为 `[0.048,0.054]`（=√0.0026=0.05099），`z>2.5` 改为 `z>3.5` | 改动正确；`pytest tests/test_stopping_rules.py` 通过。**这条改得干净。** |
| **T0.1 静默通过守卫** | `if not os.path.exists(json_path): pytest.fail(...)`，断言移出条件块 | **实测**：移走审计 JSON 后运行 → `1 failed: Required real Qwen dispersion audit artifact missing`。**现在真的会失败了。**（此前是 `1 passed`）|
| **T3.3 macOS madvise** | 从 `os.posix_madvise`（macOS 不存在）改为 `self.mm.madvise(mmap.MADV_WILLNEED, offset, length)`，并 `hasattr` 守卫；fd 改 `O_RDWR`、真 mmap | **实测**：`mm.madvise(mmap.MADV_WILLNEED, 0, 4096)` 与 `MADV_DONTNEED` 在 macOS 27.0.1 arm64 上**均调用成功**。系统调用不再是静默空转。|

### ⚠️ 半成品（1 项，看起来修了实际没生效）

| 项 | 问题 | 证据 |
|---|---|---|
| **T3.2 C++ 余弦 / 部署与验证同空间** | 他们新增了 `LSEEngine::rank_subcentroids_cosine()`（内部正确做了 query 与 sub-vector 的双 L2 归一化）。**但全仓检索该函数：只有定义（ifr_lse.cpp:227）和声明（ifr_lse.hpp:73），没有任何调用点。** | `git grep -n "rank_subcentroids_cosine" origin/main` → 仅 2 处（定义+声明）。<br>**实际部署路径仍是未归一化的 `rank_subcentroids_unnormalized`。0.92/0.935 依旧不能迁移到部署路径。** |

> **这就是当初 `use_normalized=True` 变成死代码的同一个错误类型：写了Ability，没接线。** 建议他们自查：加新函数后必须 grep 调用点，并在测试里断言生产路径调用了它。

### ❌ 一项未动（需要跑实验的，全部保留）

| 我们提的 | 现状核验 | 命令/证据 |
|---|---|---|
| **T1.0 基线自相矛盾**（报告 0.6367 vs README 0.6860） | **仍在**：README 第 156 行 `Strata_Naive_A = 0.6860`、157 行 `Decoupled_K2 = 0.9275`，与第 20 行引用 OG v2 的 `0.6367 → 0.8913` 并存 | `git show origin/main:README.md \| grep -nE '0\.6860\|0\.9275\|0\.6367'` |
| **T1.1 换 v2 import** | **仍在用已撤回 v1** | `benchmark_decoupled_retrieval.py` 第 13 行：`from benchmarks.real_kv_audit_og import ...` |
| **T1.1 复现 0.9350** | **仍是孤儿数字**，无结果产物 | 0.9350 仅存在于 `docs/track_a_benchmark_report.md`；已提交 JSON 仍只有 OG 的 3 个 + 语料缓存 |
| **T1.3 随机切分 null（解 max-pooling 假象）** | 未做 | 无新增实验文件 |
| **T1.4 Δ 重定位扫描**（KVMem 命门） | 未做 | 无新增实验文件 |
| **T1.5 子质心 re-RoPE** | 未做 | `ifr.py` 本次未改 |
| **T2.1 多模型** | 未做 | benchmark 仍单一 `MODEL_ID` |
| **T2.2 多层 / layer-wise θ** | 未做 | `ifr.py` 未改，θ 仍是单一全局 0.868 |
| **T2.3 选择率扫到 3%** | 未做 | 仍 top-64/128 = 50% |
| **T2.4 上下文长度 8K–128K** | 未做 | 仍 4096 |
| **T2.5 MLA 跨架构** | 未做 | — |
| **T3.1 端到端质量** | 未做 | 无 needle-in-haystack / LongMemEval 产物 |
| **T3.4 KV 尺寸低估 640×** | **未改** | `bytes_per_block = BLOCK_SIZE * HEAD_DIM * 4 * 2` 原样保留（第 135 行） |
| **T3.5 anti-collapse vs K=2 消融** | 未做 | — |
| **T4.2 用实测 ICC / discordant rate** | 未改 | `compute_mcnemar_sample_size` 仍默认 0.15/0.05，ICC 默认 0.3 |
| **T4.3–4.6** 连续配对检验 / 多重比较 / 非劣效 / query 构造 | 未做 | — |

### Mac 模拟器：只修了 syscall，编造断言一行没删

改后仍存在的原文：

```
7:  3. Decoupled Sub-Centroid (K=2) Retrieval Preservation (recall@64 >= 0.92).
43: self.file_size = min(self.total_bytes, 1024 * 1024 * 1024)      # 仍封顶 1GB（声称 44.8GB）
102: Takes < 0.1ms for 10K blocks, achieving recall@64 >= 0.92.      # 质心仍是 rng.randn 随机高斯
118: print("System: Apple M6 / 32GB Unified Memory / Fast NVMe SSD Tiering")   # 投机硬件
135: bytes_per_block = BLOCK_SIZE * HEAD_DIM * 4 * 2                 # KV 低估 ~640×
177: print("-> Peak Resident RAM Footprint ... < 4.5 GB")            # 硬编码字符串，无 RSS 插桩
180: print("[SUCCESS] ... successfully verified on local hardware.") # 无真实 matmul，仍声称验证通过
```

`recall@64 >= 0.92` **依然写在注释和字符串里，且从未被计算**（质心是随机 `randn`，无法产生真实 recall）。

---

## 3. 值得注意的模式（这条最重要）

把我们的清单按"**改动成本**"分类，他们的完成率是：

| 类别 | 项数 | 完成 |
|---|---|---|
| 纯代码编辑（不用跑实验） | 4 | **3 ✅ + 1 ⚠️(死代码)** |
| 需要跑模型 / 出数据的实验 | ~16 | **0 ❌** |

**结论**：不是理解问题，是**没有跑实验**。代码层面的反馈他们接受很快也很认真（Deff、守卫、madvise 三条改得都对）；但凡是要在 GPU/真实模型上出数的项，一个都没启动。

给对方的建议：把 Tier 1（尤其 T1.0 / T1.1 换 v2 / T1.4 Δ 扫描）当成必做项排期，**这三条直接决定他们 §5.4 的 +0.044 和"可用于 compaction"是否成立**，而不是继续补代码细节。

---

## 4. 附：本次我方检索踩坑记录（自曝，便于对方复核）

- 用 `grep "a\|b"`（BRE 转义分支）在 zsh/BSD grep 下对仓库对象做检索会出现**假阴性**；本次两处空结果均为该原因导致，改用 `grep -E` 后恢复正常。
- **由此纠正**：`0.9350` 确实存在于报告正文（并非不存在），准确表述应是"**仅存在于报告正文、无结果产物**"。
- 本文件所有"存在/不存在"类结论均已用 `grep -E` 复核过。
