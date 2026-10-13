# OG 审计 `8d88c3b`：三方横向对比——只有一方真的被测量了

> 对象：对方 `8d88c3b feat(benchmark): add tripartite horizontal comparison (Ollama vs Native
> Strata vs KVMem-Strata-Fusion) with live Metal measurements`
> 方法：全部数字在本机（Apple M6 / Mac18,5 / 32 GB）独立复现或溯源。
> 结论一句话：**这是一个"1 个实测 + 2 个断言"的表格，不是三方对比测试。**
> 被实测的只有官方 Ollama（基线）；对方自己系统的每一格都是硬编码字符串，
> 其中三格（35.97 GB/s、0.0698 ms、16.4 ms）分别属于**测错了对象**、**不可复现**、**张冠李戴**。

---

## 1. 先确认好的部分（确实是真的）

| 主张 | 我方复现 | 判定 |
|---|---|---|
| Ollama `qwen2.5:7b` Q4_K_M：434.66 / 26.30 tok/s | 477.38 / 26.27 tok/s | ✅ **真测量**。gen 差 0.15%，prompt 差 9.8%（单次运行的冷/热载差异：他们 load 5.56 s，我 1.55 s） |
| 文档另给 `qwen2.5:0.5b`：8478.50 / 236.02 tok/s | 8528.31 / 225.67 tok/s | ✅ **真测过**（但**不在 JSON 产物里**，见 §4.9） |
| 宿主机 "Apple M6" | `sysctl machdep.cpu.brand_string` = `Apple M6`，`hw.model` = `Mac18,5` | ✅ **正确**（此前误判为投机硬件，在此更正） |
| C++ 不变量（RoPE 代数、LSE、子质心余弦） | 自建 `clang++ -std=c++20 -O2` 编译运行，输出 `ALL BIT-LEVEL INVARIANTS ... VERIFIED SUCCESSFULLY [OK]` | ✅ 真的通过 |

**必须说清楚：这不是一次造假。是一次"把没测的说成测了的"扩写。** 基线那一格是真的，另外两格不是。

---

## 2. 核心问题：三格里只有一格是测量

`benchmarks/benchmark_tripartite_comparison.py` 的全部"实测"行为只有一件事：
向 `127.0.0.1:11434` 发一次 HTTP 请求。除此之外：

| 系统 | 测量方式 | 实际状态 |
|---|---|---|
| [1] 官方 Ollama | HTTP `/api/generate`，读 `prompt_eval_duration` / `eval_duration` | **真测**（4K 名义，实际 990 token） |
| [2] 原版 Strata | 无 | **全表硬编码字符串**。且 Strata 是 SGLang 的补丁（arXiv 2508.18572），Mac 上根本不存在可独立运行的 "Native Strata" 二进制——**这个脚本在原理上永远测不了它** |
| [3] KVMem-Strata-Fusion（他们自己） | 无 | **全表硬编码字符串**，数字全部搬运自其他文档 |

**用户问"是不是真的测试"——答案是：他们只测了对手（基线），没有测自己。**
提交信息里的 "with live Metal measurements" 指的是 Ollama 那一次 HTTP 调用。

---

## 3. 三个被标记为"实测"的数字，逐个溯源

### 3.1 `0.0698 ms ~ 0.0911 ms ... SLA < 0.1ms，提速 180×` → 不可复现，且被他们自己的 bench 打脸

`csrc/main_bench.cpp:128-131` 的判定逻辑是 `target_met = (fast_stats.mean_us < 100.0)`。
我在同一台机器上连续跑两次：

```
运行 A：General API 251.02 us | Fast Path 106.77 us → LATENCY SLA STATUS: FAILED (106.77 us = 0.11 ms)
运行 B：General API 238.05 us | Fast Path  75.13 us → LATENCY SLA STATUS: PASSED [OK] (75.13 us = 0.08 ms)
```

- **SLA 判定在同一台机器上随运行翻转**，75–107 µs 跨在 100 µs 线上。发布的 "0.0698 ms ~ 0.0911 ms" 是从这个分布里挑的一组数，**不能宣称 "SLA < 0.1ms"**。
- 只有 **Fast Path** 偶尔达标；**General API 稳定在 238–251 µs（超 SLA 2.5×）**，表格完全没提。
- `0.0698` 的源头是 `docs/comparative_study_report.md:24` 的 "10K 块背包求解实测 69.75 µs"——那是**背包求解器在合成候选上的单进程 CPU 耗时**，不是"控制面调度延迟"，更不是"纳秒级引擎"。
- "提速 180×" 的分母 16.4 ms 是**他们自己 Python 原型的耗时**（`plan-03-ubba/research-report.md:286`："无任何 JIT 加速的纯 Python 代码 ... 耗时仅 16.4 ms"）。也就是说：**他们把自己的数字放进了"原版 Strata"列**（见 §3.3）。

### 3.2 `实测 35.97 GB/s，跑满硬件吞吐` → 测错了对象，而且复现不出来

`csrc/main_bench.cpp:418-420`：

```cpp
const size_t total_tokens = num_pages * block_size;              // 512 * 32 = 16384
const double total_bytes = total_tokens * head_dim * sizeof(float) * 2;   // 算术算出来的 8 MiB
const double effective_bandwidth_gb = (total_bytes / (paged_stats.mean_us * 1e-6)) / (1024^3);
```

这除的是 `compute_paged_attention_tile()` 的耗时——一个在 **DRAM 里合成 page table 上的 CPU tile 运算**：

1. **不是 I/O 带宽。** `grep -E "ifstream|fopen|mmap|O_DIRECT|pread|cuFile" csrc/*` 全部无命中，只有 `main_bench.cpp:429` 和 `casa_paged_table.hpp:64` 两行**打印字符串**里写着 "GPUDirect Storage (GDS / cuFile)"。没有任何存储 I/O 发生。
2. **不是 PCIe 带宽。** 同上。
3. **复现不出 35.97。** 我两次运行得到 **49.35 GB/s** 和 **48.11 GB/s**，与他们发布的 35.97 差 27–37%。
4. **"跑满硬件吞吐" 也不成立。** 36–49 GB/s 是 8 MiB 合成 tile 的 DRAM 读取速率，远未跑满 Apple Silicon 统一内存带宽。
5. **规模不匹配。** tile 只覆盖 512 页 × 32 token = **16K token**，而表格同时在讲 10M / 128K / 256K。
6. **他们自己的外部复核早已否掉这个量级**：`docs/external_review_critique.md` 对 "G-CAS-1 PASS（≥45 GB/s）" 的判定是 "❌ **无效判定** | 用断言常量冒充实测（§2.3）"。把一个已被自家复核判无效的数字升级成 "实测"，是倒退。

### 3.3 `16.4 ms (Python 调度器 / 原位回填冲突)` 放在"原版 Strata"列 → 张冠李戴

`grep -rn "16.4 ms"` 全仓唯一来源是 `plan-03-ubba/research-report.md:286`：

> 在 $N = 10{,}000$ 候选块规模下，全流程（特征提取、保真剪枝、排序、贪心覆盖、结果封装）耗时仅 **16.4 ms**（无任何 JIT 加速的**纯 Python 代码**）

这是**对方自己 UBBA 方案 Python 原型的耗时**。Strata 的控制面延迟在本仓库从未被任何人测量过。
**把自己的数字填进对手那一列，是这次最严重的诚信问题**——它同时制造了"我们快 180×"的相对优势和"Strata 很慢"的绝对指控，而两者都没有测量依据。

---

## 4. 其余问题（按严重度）

| # | 问题 | 证据 |
|---|---|---|
| 4.1 | **`get_peak_rss_mb()` 定义了但从未调用**（死代码）。整个脚本没有任何内存测量，却在矩阵里断言 "常驻 < 4.0GB" | `grep -n get_peak_rss_mb` 只有第 34 行定义；且 Ollama 是独立进程，`RUSAGE_SELF` 原理上也测不到它 |
| 4.2 | **md 文档 §3 的 "4.8 GB → 16.8 GB (64K) → OOM" 无任何测量**，且算术错误：Qwen2.5-7B（28 层 / 4 KV head / d128 / fp16）在 64K 的 KV = **3.50 GiB**，加 4.7 GB 权重 ≈ 8.2 GB，32 GB 机器不会 OOM。且该模型 `context_length = 32768`，不调 `num_ctx` 根本喂不进 64K——**该断言在现有配置下不可测** | `/api/tags` + KV 算术 |
| 4.3 | **"4K prompt" 实际是 990 token**（变量名叫 `prompt_4k`，80 × 句子 = 990 token，产物 `prompt_tokens: 990`）。且 Ollama 默认 `num_ctx = 2048`，真正的 4K prompt 会被静默截断——他们从未设置 `num_ctx` | 产物 JSON + 我方复现 |
| 4.4 | **两个 `measured_*` 字段里装的是句子不是数字**：`"measured_generation_tok_s": "长上下文下吞吐比官方基线稳定 2.5×~3.8×"`、`"measured_prompt_eval_tok_s": "消除 I/O 瓶颈，锁死硬件带宽理论上限"`。**2.5×~3.8× 在全仓无任何来源** | 脚本第 160–161 行 |
| 4.5 | **召回率那一列把四个不同实验缝在一行**：recall 来自 Qwen2.5-**0.5B** / layer 12 / 4096 token / 50% 选择率；吞吐来自 qwen2.5:**7B** / 990 token；内存来自 70B **模拟**；足迹来自 10M token **推导**。一个"横向对比"行里四个模型四种条件 | 产物 JSON vs `decoupled_retrieval_benchmark.json` |
| 4.6 | **召回列继承我们已文档化的全部问题**：① `Strata_Naive_A` 是 KVMem 自己的 Mean-K 池化臂，被再次标成 "原版 Strata"（原版 Strata 没有 Mean-K 索引，已第三次指出）；② md §2 写 "严格在真实逐 Token 注意力真值下评测"——正是我们在 `aa07e56` 核验中实测否掉的措辞（该真值与真实注意力相关性 ρ=+0.05） | `docs/og_verify_aa07e56_gt_sensitivity.md` |
| 4.7 | **"已实测 128K~256K 极长上下文" 无产物**。全仓唯一的检索产物是 4096 token；JSON 里命中的 "128K/256K" 只是这句断言本身 | `grep -rn "131072\|128K\|256K" docs/*.json` |
| 4.8 | **KV 算术与所测模型不一致且不声明假设**：Qwen2.5-7B fp16 全网 @32K = **1.75 GiB**（他们写 1.54 GB，−14%）；@10M = **534 GiB**（他们写 324 GiB，−40%，且在其自己的 `comparative_study_report.md` 里标着 "(推导)"） | 架构取自公开 `Qwen/Qwen2.5-7B-Instruct/config.json`：28 层 / 28 q-head / **4 KV head** / head_dim 128 → KV = 57344 B/token |
| 4.9 | **md §3 引用了一个不含所引数据的产物**：声称 "数据产物验证：tripartite_comparison_benchmark.json" 给出 0.5B 的 8478.50/236.02，但 JSON 里**只有 7B 那一次运行**。数字本身是真的（我方复现 8528/226），属于**引用断裂**而非编造 | JSON 全文 |
| 4.10 | **`7.85× ~ 8.0× 压缩** 被他们自己的外部复核判为 "⚠️ 仅可作合成验证 \| 规模低于问题域两个数量级"，却进了头条对比表当作事实 | `docs/external_review_critique.md:35` |
| 4.11 | **失败即静默替换**：Ollama 连不上时，脚本写入 `850.0 / 125.0 / 1.2` 到 `ollama_live_measurement`，**没有 `fallback: true` 标志**，产物无法区分真假。本次没用上（有真数），但这个格式永久保留这个坑 | 脚本第 95–101 行 |
| 4.12 | **`build/` 被 gitignore 且不存在**，文档里的复现命令 `./build/main_bench` 对任何拉取仓库的人都会失败，且没有任何 bench 输出日志被提交 | `.gitignore` + `ls build` |
| 4.13 | **四位小数是假精度**：`434.6612` 来自单次运行，而我复现的 prompt 吞吐与其差 9.8% | 我方复现 |

---

## 5. 复现方式

```bash
# Ollama 基线（他们唯一真测的部分）
curl -s http://127.0.0.1:11434/api/tags                      # 确认 qwen2.5:7b / 0.5b 在位
python3 - <<'PY'                                             # 7B：434.66/26.30 → 复现 477.38/26.27
import json,time,urllib.request
p={"model":"qwen2.5:7b","prompt":"Analyze the fundamental tradeoffs in long-context transformer architectures. "*80,
   "stream":False,"options":{"num_predict":64,"temperature":0.0}}
r=urllib.request.Request("http://127.0.0.1:11434/api/generate",data=json.dumps(p).encode(),
                         headers={"Content-Type":"application/json"})
d=json.loads(urllib.request.urlopen(r,timeout=180).read())
print(d["prompt_eval_count"], d["prompt_eval_count"]/(d["prompt_eval_duration"]/1e9),
      d["eval_count"], d["eval_count"]/(d["eval_duration"]/1e9))
PY

# C++ bench（自建，不依赖他们 gitignore 掉的 build/）
clang++ -std=c++20 -O2 -o /tmp/main_bench \
    csrc/main_bench.cpp csrc/ifr_lse.cpp csrc/ubba_solver.cpp && /tmp/main_bench
# 连跑两次，观察 "LATENCY SLA STATUS" 与 "Effective Bandwidth" 的抖动
```

---

## 6. 该怎么改（按优先级）

| 优先级 | 动作 | 量级 |
|---|---|---|
| **P0-a** | 把表格改名：`tripartite_measured_baseline_only`，或把 [2][3] 两列的每一格标注 `unmeasured: true` | 10 分钟 |
| **P0-b** **（诚信）** | 把 "16.4 ms" 从 Strata 列撤出，或标注 "本组 Python 原型耗时，非 Strata 测量值" | 10 分钟 |
| **P0-c** | "实测 35.97 GB/s" 改为 "DRAM 合成 tile 算位宽 48–49 GB/s（本机复现），非存储 I/O 带宽"；如需真实带宽，先在 Mac 上做一次 NVMe/统一内存顺序读实测 | 半天 |
| **P0-d** | SLA 改为报告分布（n≥30 次的均值±sd 与达标率），不再用单次挑选值；同时报告 General API 的 238–251 µs | 半天 |
| **P1-a** | `measured_*` 字段只允许放数字；所有非实测格子改放 `null` + `note` | 1 小时 |
| **P1-b** | 删掉从未调用的 `get_peak_rss_mb()`，或真正对 Ollama 子进程做 RSS 采样（`ps`/`sample`），否则删掉所有内存断言 | 半天 |
| **P1-c** | 把 "4K" 改成实际的 990 token，并显式设置 `num_ctx` 后再谈长上下文 | 10 分钟 |
| **P2** | 真要三方对比，唯一可行路径：同一模型、同一 prompt、同一生成长度，在**同一台机器**上分别跑 ① Ollama ② SGLang+Strata 补丁 ③ 自家 Fusion，各取端到端 tok/s 与峰值 RSS。Mac 上跑不了 SGLang，就该明说 "Strata 列不可在本机测量，仅引用论文数" | 1 周起 |

---

## 7. 一句话给对方

> **这次真正被测量的只有官方 Ollama，而且它测得很干净、可复现——这部分值得保留。**
> 但另外两列是硬编码断言，其中 "16.4 ms" 是你们自己的 Python 原型数字被放进了 Strata 列，
> "35.97 GB/s" 是 8 MiB 合成内存 tile 的算位宽被写成了存储 I/O 带宽（且本机复现是 48–49 GB/s），
> "SLA < 0.1ms" 被你们自己的 bench 在两次运行中的一次判为 FAILED。
> 在把这三格改成 `unmeasured` 或补上真测之前，这个表格不能作为横向对比的证据。
