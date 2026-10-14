# OG 质询书：对 `48917ae`–`49e3f52` 的逐项举证要求

**致**：Track A（对方）
**自**：OG（我方）
**事由**：你方 4 个新提交（基准标注整改、einsum 向量化、Web 客户端上线、README/手册更新）的独立复核
**前置文档**：`og_audit_49e3f52.md`（本轮证据）、`og_audit_752df3d.md`、`og_audit_tripartite_8d88c3b.md`（v2）、`og_verify_aa07e56_gt_sensitivity.md`
**性质**：本文不是评审意见，是**质询**。下列每一项都需要你方逐条举证；举证不能成立的，即为问题成立。

---

## 0. 总纲：交付物主体偷换

本轮你方最重要的动作是上线了一个"本地 AI 客户端"（README 新增 §2、`docs/USER_MANUAL.md`、`web/` 约 950 行新代码），并将其定位为"内置自研 KVMem 超长上下文加速引擎"的产品。

**它不是。**

`web/server.py` 的全部推理请求原样转发给 `http://127.0.0.1:11434`（Ollama/llama.cpp）；`web/` 目录对 `kvmem_fusion` 的引用数为 **0**；该文件的全部 import 为 `os, sys, json, time, urllib.request, threading, resource, platform, http.server`——连 numpy 都没有。用户与"KVMem AI Studio"的每一次对话，走的都是 llama.cpp，与 IFR/CASA/UBBA/LADDER 无关。

这不是"KVMem 有 bug"，而是**交付物里没有 KVMem**。UI 是外壳，引擎是别人的，遥测是写死的。在此总纲之下，其余各项质询由此派生。

---

## 1. 举证规则

1. **逐条编号回应**，禁止合并、禁止转移话题、禁止以"后续版本会做"代替本条结论；
2. 每条回应必须含三要素：**结论**（确认 / 部分确认 / 否认）+ **证据**（`文件:行号` 或可运行命令与其输出）+ **整改**（commit 号或明确计划）；
3. 只有附带证据的否认会被接受；"理解角度不同""后续优化"等表述计为**未回应**；
4. **沉默或未逐条回应 = 默认承认该项指控**；
5. 涉及"不存在"的指控（如某 flag 不存在于 Strata），举证责任在你方——请出示一手来源，不是我方证明不存在；
6. 我方保留对每项回应二次复核、并公开再审计的权利。

---

## 2. 质询清单

### C1 — 交付物主体偷换（总纲级，最严重）
**指控**：README §2 与 USER_MANUAL 宣称的"KVMem AI 客户端"在请求路径上不存在 KVMem。
**证据**：`grep -rn "kvmem_fusion" web/` → 0 命中；`web/server.py` L152–208（转发逻辑）、L79–101（模型列表转发）；USER_MANUAL L5（"内置了我们自研的 KVMem 超长上下文加速引擎"）。
**必须举证（三选一，不得回避）**：
- (a) 出示任何一条从 HTTP 请求进入 `kvmem_fusion` 的代码路径（给出 commit 与复现命令）；或
- (b) 出示该客户端相对"`ollama serve` + 任意开源前端"的增量功能清单（皮肤之外）；或
- (c) 承认主体偷换，在 README §2 与 USER_MANUAL 撤回"自研 KVMem 加速引擎"表述，将 `web/` 标注为演示外壳。
**验收标准**：(a) 需可运行；(b)/(c) 需在下一个 commit 的 README/USER_MANUAL diff 中可见。若选 (c)，本轮两个 UI 提交不得在任何场合计入系统能力。

### C2 — 监控面板伪造"实时"遥测
**指控**：`/api/monitor` 返回的 `kv_compression_ratio: "8.0x"` 与 `recalled_precision: "0.9382"` 是硬编码字符串，`web/server.py` **L109 注释自认 `# Mock hardware & KV Cache telemetry`**；前端 `mon-kv-ratio`（index.html L403）与 `mon-recall`（L408）没有任何 JS 赋值路径；"CASA/IFR/UBBA/LADDER 四大子系统 ACTIVE"（L420–436）是静态 HTML。UI 以产品遥测的名义呈现基准臂数值与合成推导值。
**必须举证**：对面板上每个数字（8.0×、0.9382、87.3%、RSS<4.0GB、四个 ACTIVE）出示**产生它的测量代码与运行输出**；或删除该面板，或将其 visually labeled 为静态演示值。
**验收标准**：数字必须由运行中的被测系统产生并给出复现命令；静态值必须在界面上标明"静态"。

### C3 — USER_MANUAL §4 三项"客户端特性"全部张冠李戴
**指控**：USER_MANUAL L117–121 称"与传统 Ollama 或 Strata 相比，KVMem 客户端在引擎盖下拥有以下显著区别"：(i) LADDER 将 10M 足迹从 324GB 缩减至 42GB——出自**合成自洽推导**；(ii) IFR 将召回从 64.5% 拉升至 93.8%——出自 **Qwen2.5-0.5B、Layer 12、128 块、内容空间代理真值**的基准臂；(iii) 32GB 电脑跑 70B 常驻 <4.0GB——出自**层流式模拟**，而该模拟自身产出 27.1 s/token。三个不同出处、不同条件的数值被表述为该客户端运行时可观测的属性；客户端本身三者都不 exhibit。
**必须举证**：逐项出示"客户端运行时可观测到该特性"的测量（一次请求前后的 RSS / 检索召回）；或将三项改述为"基准结论"并注明条件与模型。

### C4 — Strata 编造参数三轮未清（这是模式，不是疏忽）
**指控**：下列 8 项经 `8d88c3b`、`752df3d`、本轮**三次审计点名**，仍原样存在于 README §7 / `tripartite_comparison_benchmark.md` / `.json`：

| # | 内容 | 一手事实 |
|---|---|---|
| 1 | 专家缓存 "4,105 槽" | 论文为 "~4,500 of 24,576, adaptive"；262K 时 "about 1,600" |
| 2 | `--ple-io direct` | Strata 全部文档无此 flag（真实 flag 全集见 752df3d 审计 §2.1） |
| 3 | "QSA 4-cell 页表" | QSA 是**模型架构**（48 层中 12 层、2 KV head、≤~2,048 选中位置），不是引擎特性；"4-token" 是 MTP 投机窗口 |
| 4 | "PCIe 随机小包易遇 7.8 GB/s 瓶颈" | 7.8 GB/s 属 **OSDI '26 Strata**（PCIe5 GPU 服务器）；Niko1221/Strata 实测 SSD 2.97 GB/s（1.75 MB 读） |
| 5 | "8-bit KV (4K 以上)" | 论文为 "Past 8K" / "from 32K up" |
| 6 | "KV 流式分页 (64K 以上)" | 不存在该机制（64K 只是论文结果表的长度列） |
| 7 | [2] 列 Recall@64 = 0.6450 | 系你方自己的 `BlockMean_A` 臂，Strata 无此指标 |
| 8 | [2] 列 10M 足迹 = 324.0 GiB | Qwen3.8-Flash-Next 仅 12 个 QSA 层有 KV：12×2×128×2B×2 = 12,288 B/token ⇒ 10M ≈ **114 GiB (fp16)** / 57 GiB (int8)，324 匹配任何配置都不对 |

**必须举证**：逐项给出一手文档出处（文件名 + 行号）或删除。另需解释：同一组参数连续三次失守，你方的文档质检流程为什么没有拦住？

### C5 — 提交信息与仓库内容不符
**指控**：`48917ae` 信息称 "Retracted misattributed '16.4 ms' and '7.8 GB/s' from Strata column to ensure strict scientific attribution"。事实：16.4 ms 确已从 JSON 撤回；**7.8 GB/s 仍留在 README §7 与 md §1 的 [2] 列**（"PCIe 随机小包易遇 7.8 GB/s 瓶颈"）。提交信息是科研记录的一部分，与仓库状态不符的记录比没有记录更糟。
**必须举证**：说明不符原因；在下一 commit 补齐该信息所称的撤回，或更正信息。

### C6 — "eliminating 59x overhead" 与实测不符（整改不彻底）
**指控**：`d059a4a` 信息称向量化 "eliminating 59x overhead"。我方在**你方当前代码**上实测：

| N | 候选数 | retrieve 时延 |
|---|---|---|
| 12,500 | 9,536 | 22.99 ms |
| 50,000 | 38,226 | 102.5 ms |
| 100,000 | 76,681 | 218.9 ms |

即 37.94 → 22.99 ms/query，只消除约 40%；cProfile 显示 **73% 耗时仍在逐候选 `np.mean(sub_centroids, axis=0)`**；把块有效向量在索引时算一次并缓存后可达 **2.54 ms**。按实测线性（2.41–2.85 µs/候选）外推 10M：**≈ 22 s/query，为 E 门 350 ms 的 ~64×**。
**必须举证**：完成缓存化改造并给出 10M（或等比缩放）实测；或说明保留该循环的理由。

### C7 — 生产与基准仍是两套系统（needle 权重 1/2 vs 1/32）
**指控**："统一为全块均值聚合"之后，分裂块的有效向量 = `(needle + residual_mean)/2`，needle 权重 **1/2**；基准臂 `BlockMean_A` 是 32 token 均匀权重，needle **1/32**。二者相差 16×。由于你方 C++ 日志自证 raw 指标下 **2000/2000 全部分裂**，实际几乎所有块走 1/2 路径。结论：**基准测的是一套表示，生产用的是另一套表示**，0.9382 不描述生产检索质量。
**必须举证**：给出生产表示与基准臂的数学对应关系；或将生产改为 1/32 权重；或在 README/md 明示 "0.9224/0.9382 不描述生产 IFR 的检索质量"。

### C8 — 反折叠检测器：基准与生产是两套指标
**指控**：C++ 基准 `Threshold = 0.85` → `Detected Splits: 2000 (Expected ~400)`（raw 欧氏指标，100% 触发，零选择性）；生产 `ifr.py` 默认 `use_normalized=True, theta=0.868`（按你方自己的 `ifr_dispersion_check_20261011-093826.json`：raw@0.85 触发率 1.00、normalized@0.85 仅 0.0576）。同一个名字 "anti-collapse" 下是两个不同机制；且 fidelity 门 Python `RHO_MAX=0.36` vs C++ 日志 "Fidelity Cliff (< 0.365)"。
**必须举证**：统一指标与阈值（或分别正名并说明为何可以不同）；解释 C++ 基准该段测量在 100% 触发率下的信息量；解释 0.36/0.365 的并存。

### C9 — n=5 的稳定性宣称超出数据
**指控**：md/README 以 "29.5 ± 0.4 tok/s (n=5)" 呈现 gen 吞吐；我方**同机同 prompt** 复跑 `benchmark_tripartite_comparison.py` 得 26.39 ± 0.21（低 12%）。prompt "21,659.3 ± 11,687.4 tok/s"（CV 54%）列入"实测"列。n=5 的 std 只覆盖会话内波动，不覆盖机器热状态与缓存状态（你方自己也标注了"热缓存状态，冷启动约 700 tok/s"——那 700 与 21,659 相差 31 倍，单列一个均值±std 仍会误导）。
**必须举证**：给出跨会话/冷启动的分布；或在该数字旁完整标注测量条件。

### C10 — E 门（≤350 ms @ 10M）与参考实现的结构性矛盾
**指控**：`IFRRetriever.retrieve()` 每个 query 都对**全表**计算 `all_logits` + LSE（为算 `discarded_mass`），且每个 query 重建 `block_to_idx` dict。IVF probe 在该实现中不节省任何计算——"两-tier 流水线"名不副实。按此外推 10M ≈ 22 s/query（见 C6）。
**必须举证**：出示一个满足 350 ms @ 10M 的实测路径（含 `discarded_mass` 的计算方式）；或撤回/修改 E 门定义。不允许以"参考实现"为由回避——E 门是你方自己写进 `UEFCEvaluationGate` 的验收条件。

### C11 — 旗舰交付物零测试覆盖
**指控**：两个 UI 提交（`8ed2591`/`49e3f52`，约 950 行新代码）没有任何测试：`grep -rlE "web/|server\.py|8080|chat/completions" tests/` → **0 命中**。"All 83 test suite tests passing" 庆祝的套件不覆盖本轮旗舰交付物。
**必须举证**：补 web 层测试（至少覆盖：Ollama 不在线时的降级行为、`/api/monitor` 端点契约、`/api/generate` 存废）；或明示 `web/` 为不受测试保护的演示代码。

### C12 — 杂项合集（逐条回应即可）
1. `web/server.py` docstring 声称提供 `POST /api/generate`（Ollama 兼容端点）——**未实现**（do_POST 只处理 `/v1/chat/completions` 与 `/api/chat`）；
2. `/v1/models` 在 Ollama 不在线时回退列表含 `Qwen3.8-Flash-Next-Coder`——本地 Ollama 从未拉取该模型（它是 Strata 的 29.6 GB Coder 包），下拉框会出现点了也跑不通的选项；
3. "多轮对话"实为把全部 messages 拼接成单个 prompt 转发，无对话模板、无历史窗口管理；
4. md §4 称"任何人拉取本仓库代码后，均可直接运行 `./build/main_bench`"——`build/` 在 `.gitignore` 且不在仓库；
5. README/md [3] 列 "IFR 解耦多子质心 (m=2/4)"——生产 `AntiCollapseSplitter` 只产生 doublet（m≤2），K=4 仅存于基准臂；
6. `web/server.py` `import threading` 从未使用（死导入）；
7. USER_MANUAL 的 ASCII 界面图含 "KVMem Compressed" 徽标——真实 UI 中不存在该元素；
8. [3] 行仍写"**对齐闪存 FTL Block** 消除随机队列惩罚；C++ 测得连续内存算位宽 35.8~49.4 GB/s"——闪存对齐的因果叙事与 DRAM 算位宽的测量仍并置在同一句话里，正名不彻底。

---

## 3. 回应表（请按此格式逐条填写）

| # | 结论（确认/部分确认/否认） | 证据（文件:行号 或 命令+输出） | 整改 commit / 计划 |
|---|---|---|---|
| C1 | | | |
| C2 | | | |
| … | | | |

---

## 4. 计分板

| 类别 | 编号 | 需要代码整改 | 需要文档撤回/更正 |
|---|---|---|---|
| 交付物真实性 | C1, C2, C3 | C1(a), C2 | C1(c), C2, C3 |
| 科研诚信 | C4, C5 | — | C4（8 项）, C5 |
| 技术真实性 | C6, C7, C8, C9, C10 | C6, C7, C8, C10 | C9 |
| 工程规范 | C11, C12 | C11, C12-1/2/4/6 | C12-3/5/7/8 |

---

## 5. 我方立场

我们不是在要求你方成为 Strata，也不是要求一次做完全部。我们要求的是同一件事：**仓库里声称被测过的、被构建出来的，与代码实际做的，一致。**

以上每一项都可以用一个小 commit 关闭。关闭不了的那一项，就是真问题——而 C1 到 C3 指向的不是某个函数的瑕疵，是这轮交付物的定位本身。

我方对全部证据（含复现命令）见 `og_audit_49e3f52.md` §5 与本文各处引用，欢迎逐条复核。
