# OG 审计 `8d88c3b` + `7efc5b5`：三方对比只有一方被测量，且他们自己的日志推翻了表格

> 对象：
> - `8d88c3b feat(benchmark): add tripartite horizontal comparison (Ollama vs Native Strata vs KVMem-Strata-Fusion) with live Metal measurements`
> - `7efc5b5 docs(logs): upload raw benchmark execution logs for independent reviewer verification`（对方上传的 4 份原始日志）
>
> 方法：全部数字在本机（Apple M6 / Mac18,5 / 32 GB）独立复现或在其自有文档内溯源。
> 结论一句话：**这是一个"1 个实测 + 2 个断言"的表格。被实测的只有官方 Ollama（基线）；
> 对方自己系统的每一格都是硬编码字符串；而他们上传的 4 份日志里，有 3 处直接推翻表格里的结论。**

---

## 0. 先说好的部分（这部分是真功夫，应当保留）

| 项 | 核验结果 |
|---|---|
| Ollama `qwen2.5:7b` 434.66 / 26.30 tok/s | ✅ 真测量。我方复现 477.38 / 26.27（gen 差 0.15%） |
| Ollama `qwen2.5:0.5b` 8478.50 / 236.02 tok/s | ✅ 真测过。我方复现 8528.31 / 225.67 |
| 宿主机 "Apple M6" | ✅ 正确（`machdep.cpu.brand_string` = Apple M6，`hw.model` = Mac18,5）。此前误判为投机硬件，在此更正 |
| 上传原始日志供独立复核 | ✅ **姿态正确，是本轮最该肯定的动作** |
| `decoupled_retrieval_run.log` | ✅ 与 `decoupled_retrieval_benchmark.json` **逐位一致**（0.6450/0.6042/0.9004/0.9224/0.8660/0.9382/0.8369/0.4910），我方独立复跑同样逐位一致 |
| UBBA 规模扫描（1K→50K） | ✅ 干净可复现：7.05 → 349.75 µs，吞吐稳定在 140–152 M blocks/s |
| C++ 位级不变量 | ✅ 自建编译运行，`ALL BIT-LEVEL INVARIANTS ... VERIFIED SUCCESSFULLY [OK]` |

**必须讲清楚：这不是造假，是一次"把没测的说成测了的"扩写，外加两处数字搬栏。**
基线那一格测得很干净；问题全部集中在另外两格，以及用基线数字反推自己系统优势的方式。

---

## 1. 核心问题：三格里只有一格是测量

`benchmarks/benchmark_tripartite_comparison.py` 的全部"实测"行为只有一件事：向 `127.0.0.1:11434` 发一次 HTTP 请求。

| 系统 | 测量方式 | 实际状态 |
|---|---|---|
| [1] 官方 Ollama | HTTP `/api/generate` | **真测**（名义 4K，实际 990 token） |
| [2] 原版 Strata | 无 | **全表硬编码字符串**。且 Strata 是 SGLang 的补丁（arXiv 2508.18572），Mac 上不存在可独立运行的 "Native Strata" 二进制——**这个脚本在原理上永远测不了它** |
| [3] KVMem-Strata-Fusion（他们自己） | 无 | **全表硬编码字符串**，数字全部搬运自其他文档 |

提交信息 "with live Metal measurements" 指的是 Ollama 那一次 HTTP 调用。**他们测了对手，没有测自己。**

---

## 2. 数字搬栏：两个数从"我们自己的基线"被搬进了"Strata"列

这是本轮最硬的证据，因为它**不需要任何外部知识，只用他们自己的两份文档互相比对**。

他们自己更早的 `docs/comparative_study_report.md`（第 39–47 行），表头是
`Full-Context (基准) | Naive Flat (INT2) | 原生 KVMem (基线) | 本方案 (Fusion 实测)`：

```
| 控制面调度延迟 (10K) | N/A | N/A | 16.4 ms (Python)   | 0.0698 ms (C++ 实测) |
| G-CAS-1 I/O 带宽     | N/A | N/A | 7.8 GB/s (实测基线)| UNMEASURED (设计>=45) |
```

即：**"16.4 ms (Python)" 和 "7.8 GB/s" 原本都在"原生 KVMem (基线)"列——是他们自己项目的基线数字**
（16.4 ms 的出处是 `plan-03-ubba/research-report.md:286`："无任何 JIT 加速的**纯 Python 代码** ... 耗时仅 16.4 ms"）。

到了 `8d88c3b` 的三方表里，这两个数字被搬进了 **"原版 Strata (OSDI '26)"** 列：

```python
"control_plane_latency": "16.4 ms (Python 调度器 / 原位回填冲突)",   # ← Strata 列
"io_bandwidth_util":      "7.8 GB/s (PCIe 5.0 利用率仅 22%...)",      # ← Strata 列
```

后果：
- **"Strata 控制面 16.4 ms" 从未被任何人测量过**，它是对方自己 Python 原型的耗时。
- 表格因此同时制造了"我们快 180×"（16.4 ms ÷ 0.0911 ms）的相对优势和"Strata 很慢"的绝对指控，**两者都没有测量依据**。
- 同一份 `comparative_study_report.md` 里，他们自己把 Fusion 的带宽标为 **`UNMEASURED (设计>=45)`**；三方表却写成 **"实测 35.97 GB/s，跑满硬件吞吐"**。**从 UNMEASURED 变成"实测"，是文档诚信的倒退。**

---

## 3. "实测 35.97 GB/s" 测错了对象

`csrc/main_bench.cpp:418-420`：

```cpp
const size_t total_tokens = num_pages * block_size;                        // 512*32 = 16384
const double total_bytes = total_tokens * head_dim * sizeof(float) * 2;    // 算术算出的 8 MiB
const double effective_bandwidth_gb = (total_bytes / (paged_stats.mean_us * 1e-6)) / (1024^3);
```

除以的是 `compute_paged_attention_tile()` 的耗时——在 **DRAM 内合成 page table 上的 CPU tile 运算**：

1. **不是 I/O 带宽**：`grep -E "ifstream|fopen|mmap|O_DIRECT|pread|cuFile" csrc/*` 全部无命中；只有 `main_bench.cpp:429` 与 `casa_paged_table.hpp:64` 两行**打印字符串**里写着 "GPUDirect Storage (GDS / cuFile)"。没有任何存储 I/O 发生。
2. **不是 PCIe 带宽**：同上。
3. **复现不出 35.97**：他们日志是 35.79 GB/s；我两次运行是 **49.35 / 48.11 GB/s**。相差 27–37%，说明它是抖动值，不是硬件属性。
4. **"跑满硬件吞吐"不成立**：36–49 GB/s 是 8 MiB 合成 tile 的 DRAM 读取速率。
5. **规模不匹配**：tile 只覆盖 512 页 × 32 token = **16K token**，而表格同排在讲 10M / 128K / 256K。

---

## 4. "SLA < 0.1ms" 被他们自己的日志和我的复现共同否掉

`csrc/main_bench.cpp:128-131` 判定式是 `target_met = (fast_stats.mean_us < 100.0)`。三次运行：

| 来源 | General API | Fast Path | SLA 判定 |
|---|---|---|---|
| 他们上传的日志 | 167.12 µs（Max 510.42） | **69.85 µs** | PASSED |
| 我方复现 · 运行 A | 251.02 µs | **106.77 µs** | **FAILED (0.11 ms)** |
| 我方复现 · 运行 B | 238.05 µs | **75.13 µs** | PASSED |

- **同一台机器上判定会翻转**，Fast Path 落在 70–107 µs，跨在 100 µs 线上。发布的 "0.0698 ms ~ 0.0911 ms" 是从这个分布里挑的一组数，**不能宣称 "SLA < 0.1ms"**。
- 只有 Fast Path 偶尔达标；**General API 稳定在 167–251 µs（超 SLA 1.7–2.5×）**，表格完全没提。
- `0.0698` 的源头是 `comparative_study_report.md:24` 的 "10K 块背包求解实测 69.75 µs"——那是**背包求解器在合成候选上的单进程 CPU 耗时**，不是"控制面调度延迟"，更不是"纳秒级引擎"。
- 另一处佐证：`mac_layer_streaming_sim_run.log` 里 "Sub-Centroid Coarse Ranking Latency (Top-64): **5.595 ms**"——同一检索动作在 Python sim 里是 5.6 ms，在 C++ bench 里是 0.07 ms，**相差 80×**。两个数描述的不是同一件事。

---

## 5. 他们上传的日志里，三处直接推翻表格结论

### 5.1 反坍缩检测器在 100% 的块上触发——零选择性

`csrc_main_bench_run.log` §4：

```
Evaluating 2000 blocks (64000 tokens), Threshold = 0.85
Detected Splits: 2000 (Expected ~400)
```

benchmark 每隔 5 个块注入一个 needle（`b % 5 == 0`，应检出 400 个），检测器却在 **2000/2000 = 100%** 的块上报 split，包括 1600 个**没有 needle** 的背景块。即：**精度 20% 但召回 100%，等于没有判别力。**

生产后果（`kvmem_fusion/ifr.py` 的 `inspect_and_split`）：如果这个阈值行为成立，**每个块都会被拆成 2 个子质心 → 索引体积翻倍**，与表格里 "7.85× 压缩" 直接冲突。这也说明 θ=0.868 是在真实 Qwen 数据上校准的，搬到合成高斯数据上不成立——**要么换数据，要么明说该阈值不可迁移**。

### 5.2 保真崖在自己的日志里被击穿

同一份日志 §3：

```
Top-64 Unnormalized Ranking & Discarded Mass:
    Retained Candidates: 64
    Discarded Attention Mass (rho): 0.9648
    Fidelity Cliff (< 0.365): EXCEEDED
```

他们自己的门限定义是 `UEFCEvaluationGate.RHO_MAX = 0.36`（`ifr.py` 第 418 行）。日志实测 ρ=0.9648，**超限 2.7 倍，判定 EXCEEDED**。也就是说：从 20K 候选里选 top-64 会丢掉 96.5% 的注意力质量。表格对此只字未提。

### 5.3 Mac sim 投影 27.1 秒/token，表格却声称比基线快 2.5×~3.8×

`mac_layer_streaming_sim_run.log`：

```
-> Measured 8-Layer Stream Latency:             2710.674 ms (338.83 ms/layer)
-> Projected 80-Layer Pipelined Stream Latency: 27106.737 ms
-> Total Projected Decode Step Overhead:         27112.331 ms
```

**每个 decode step 27.1 秒**，日志自己标着 "Projected"。而同一套表格里：

- Ollama 实测 26.30 tok/s = **0.038 s/token**
- 表格声称本系统 "长上下文下吞吐比官方基线稳定 **2.5×~3.8×**"

**27.11 ÷ 0.038 = 713。他们自己的日志说慢 713 倍，表格说快 2.5–3.8 倍。**
这是同一批上传文件内部最直接的量的矛盾，不需要任何复现就能判定。

（附带：该 sim 的 "Virtual Weight Size: 43.75 GB mapped via sparse mmap" 是稀疏文件映射，**没有加载任何 70B 权重、没有 matmul、没有生成过一个 token**；"Real Measured Peak RSS: 3947.25 MB" 测的是这个映射进程的 RSS。日志末尾 "[VERIFIED] System I/O Paging & RSS Memory Budget verified" 只验证了 mmap+RSS 行为，**不能支撑表格里 "32GB 机器跑 70B ✅ 成功"**。）

---

## 6. 日志与产物对不上：三套 Ollama 数字，两个都不是同一个运行

| 来源 | 模型 | prompt tok/s | gen tok/s |
|---|---|---|---|
| 提交的 JSON 产物 | qwen2.5:**7b** | 434.66 | 26.30 |
| 上传的 `tripartite_comparison_run.log` | qwen2.5:**0.5b** | **6310.68** | 228.22 |
| md 文档 §3 声称（引用上述 JSON） | qwen2.5:0.5b | **8478.50** | 236.02 |
| 我方复现 | qwen2.5:0.5b | **8528.31** | 225.67 |

- **日志是 0.5B，产物是 7B——两者不是同一次运行，互为缺口**：JSON 没有日志，日志没有 JSON。
- md 文档 §3 写 "数据产物验证：`tripartite_comparison_benchmark.json`" 给出 0.5B 的 8478.50/236.02，**但 JSON 里只有 7B 那一次**。数字本身是真的（我方复到 8528/226），属于**引用断裂**。
- 同一个模型、同一个 prompt，prompt 吞吐在 **6310 / 8478 / 8528** 之间，**极差 35%**。原因清晰：他们那次 load_duration 5.56 s（冷），我那次 1.55 s（热）。**结论：单次运行的 Ollama prompt 吞吐不是可引用的测量值**，至少要 n≥5 次报均值±sd。表格和文档却把单次值写到小数点后两位四位。

---

## 7. 其余问题

| # | 问题 | 证据 |
|---|---|---|
| 7.1 | **`get_peak_rss_mb()` 定义了但从未调用**（死代码）。整个脚本没有任何内存测量，却在矩阵中断言 "常驻 < 4.0GB" | `grep -n get_peak_rss_mb` 仅第 34 行定义；且 Ollama 是独立进程，`RUSAGE_SELF` 原理上也测不到它 |
| 7.2 | **md §3 "4.8 GB → 16.8 GB (64K) → OOM" 无测量且算术错误**：Qwen2.5-7B（28 层/4 KV head/d128/fp16，已从公开 `config.json` 核实）在 64K 的 KV = **3.50 GiB**，加 4.7 GB 权重 ≈ 8.2 GB，32 GB 不会 OOM。且该模型 `context_length = 32768`，不调 `num_ctx` 根本喂不进 64K——**该断言在现有配置下不可测** | `/api/tags` + 算术 |
| 7.3 | **"4K prompt" 实际 990 token**（变量名 `prompt_4k`，80 × 句子 = 990，日志与产物均为 990）。且 Ollama 默认 `num_ctx = 2048`，真正的 4K prompt 会被静默截断——他们从未设置 `num_ctx` | 日志 + 产物 |
| 7.4 | **两个 `measured_*` 字段装的是句子不是数字**：`"measured_generation_tok_s": "长上下文下吞吐比官方基线稳定 2.5×~3.8×"`、`"measured_prompt_eval_tok_s": "消除 I/O 瓶颈，锁死硬件带宽理论上限"`。**2.5×~3.8× 在全仓无来源，且被 5.3 的 713× 直接反驳** | 脚本第 160–161 行 |
| 7.5 | **一行"横向对比"缝了四个不同实验**：recall 来自 0.5B / layer 12 / 4096 token / 50% 选择率；吞吐来自 7B（或 0.5B）/ 990 token；内存来自 70B **模拟**；足迹来自 10M token **推导** | 产物 JSON vs 各日志 |
| 7.6 | **召回列继承已文档化的全部问题**：① `Strata_Naive_A` 是 KVMem 自己的 Mean-K 池化臂，被第 N 次标成 "原版 Strata"（原版 Strata 没有 Mean-K 索引）；② md §2 写 "严格在真实逐 Token 注意力真值下评测"——正是 `aa07e56` 核验中实测否掉的措辞（该真值与真实注意力块排序相关性 **ρ=+0.05**） | `docs/og_verify_aa07e56_gt_sensitivity.md` |
| 7.7 | **"已实测 128K~256K 极长上下文" 无产物**。全仓唯一检索产物是 4096 token；JSON 里命中的 "128K/256K" 只是这句断言本身 | `grep -rn "131072\|128K\|256K" docs/*.json` |
| 7.8 | **`324.2 GiB @ 10M` 对应 34.8 KB/token，与仓库中任何模型的配置都不匹配**：Qwen2.5-7B fp16 = 57.3 KB/token → 534 GiB；7B int8 = 28.7 KB → 267 GiB；70B(80L×8H) fp16 = 327.7 KB → 3052 GiB。其自己的文档标着 "(推导)" 但未给公式 | 算术反推 + `comparative_study_report.md` §2.2 |
| 7.9 | **`7.85× ~ 8.0×` 被他们自己的外部复核判为 "⚠️ 仅可作合成验证 \| 规模低于问题域两个数量级"**，却进了头条对比表当作事实；且与 5.1 的"每块都拆"直接冲突 | `docs/external_review_critique.md:35` |
| 7.10 | **失败即静默替换**：Ollama 连不上时脚本写入 `850.0 / 125.0 / 1.2` 到 `ollama_live_measurement`，**无 `fallback: true` 标志**，产物无法区分真假。本次未触发，但格式永久保留该坑 | 脚本第 95–101 行 |
| 7.11 | **`build/` 被 gitignore 且不存在**，文档复现命令 `./build/main_bench` 对拉取仓库的人必然失败；本轮上传日志弥补了这一点，但应同时提交构建命令与输出 | `.gitignore` + `ls build` |

---

## 8. 复现方式

```bash
# Ollama 基线（唯一真测的部分；注意跑 5 次取均值，单次值有 ±15% 抖动）
curl -s http://127.0.0.1:11434/api/tags
python3 - <<'PY'
import json,urllib.request
p={"model":"qwen2.5:7b","prompt":"Analyze the fundamental tradeoffs in long-context transformer architectures. "*80,
   "stream":False,"options":{"num_predict":64,"temperature":0.0}}
r=urllib.request.Request("http://127.0.0.1:11434/api/generate",data=json.dumps(p).encode(),
                         headers={"Content-Type":"application/json"})
d=json.loads(urllib.request.urlopen(r,timeout=180).read())
print(d["prompt_eval_count"], d["eval_count"], d["load_duration"]/1e9)
PY

# C++ bench：自建（他们 gitignore 掉了 build/），连跑两次看 SLA 与带宽抖动
clang++ -std=c++20 -O2 -o /tmp/main_bench \
    csrc/main_bench.cpp csrc/ifr_lse.cpp csrc/ubba_solver.cpp && /tmp/main_bench
# 关注: "LATENCY SLA STATUS"（会翻转）、"Effective Bandwidth"（48-49 GB/s，非 35.97）、
#       "Detected Splits: 2000 (Expected ~400)"、"Fidelity Cliff ... EXCEEDED"
```

---

## 9. 该怎么改（按优先级）

| 优先级 | 动作 | 量级 |
|---|---|---|
| **P0-a（诚信）** | 把 "16.4 ms" 与 "7.8 GB/s" 从 Strata 列撤出——它们源自你们自己的 `comparative_study_report.md` 的"原生 KVMem (基线)"列。Strata 列要么放论文引用值并注明出处，要么放 `unmeasured` | 10 分钟 |
| **P0-b** | 表格改名 `baseline_only_measured`，或给 [2][3] 两列每格加 `unmeasured: true` | 10 分钟 |
| **P0-c** | 删除或改写 "比官方基线快 2.5×~3.8×"——你们自己的 sim 日志投影是 27.1 s/token（= 慢 713×）。要么给出端到端实测，要么标注 `unmeasured` | 1 小时 |
| **P0-d** | "实测 35.97 GB/s" 改为 "DRAM 合成 tile 算位宽 35.8–49.4 GB/s（本机两次复现），非存储 I/O 带宽"；要真实带宽就先做一次 NVMe/统一内存顺序读实测 | 半天 |
| **P1-a** | SLA 报分布（n≥30 的均值±sd+达标率），并同时报 General API 的 167–251 µs | 半天 |
| **P1-b** | 修反坍缩阈值：在合成数据上 2000/2000 触发 = 零选择性。换真实 Qwen 数据重校 θ，或明说该阈值不可迁移 | 1–2 天 |
| **P1-c** | 补报 `rho = 0.9648`（保真崖 EXCEEDED）及其对选择率的敏感性 | 半天 |
| **P1-d** | `measured_*` 字段只准放数字；非实测格放 `null + note` | 1 小时 |
| **P1-e** | Ollama 基线改为 n≥5 次报均值±sd；把 "4K" 改成实际的 990 token 并显式设 `num_ctx` | 1 小时 |
| **P2** | 真要三方对比，唯一可行路径：同一模型、同一 prompt、同一生成长度、同一台机器，分别跑 ① Ollama ② SGLang+Strata 补丁 ③ 自家 Fusion，各取端到端 tok/s 与峰值 RSS。Mac 上跑不了 SGLang，就该明说 "Strata 列不可本机测量，仅引用论文数" | 1 周起 |

---

## 10. 一句话给对方

> **上传原始日志是对的，这次真的该肯定——而且正是这些日志让我们能把问题定位到具体行。**
> Ollama 基线测得很干净，UBBA 规模扫描和检索实验的日志与产物逐位一致，这些都站得住。
> 但表格里另外两列是硬编码断言，其中 "16.4 ms / 7.8 GB/s" 出自你们自己文档的"原生 KVMem (基线)"列，
> 被搬进了"原版 Strata"列；"实测 35.97 GB/s" 是 8 MiB 合成内存 tile 的算位宽（我复现是 48–49 GB/s）；
> "SLA < 0.1ms" 在同一台机器上两次运行中有一次被判 FAILED；
> 而你们自己的 sim 日志投影 **27.1 秒/token**，与表格声称的"比基线快 2.5×~3.8×"相差 713 倍。
> 在把这些格改成 `unmeasured` 或补上真测之前，这个表格不能作为横向对比的证据。
