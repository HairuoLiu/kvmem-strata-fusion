# OG 独立审计：`48917ae`–`49e3f52`（4 个新提交，含 README 大改与 Web 客户端上线）

**审计对象**：我方 `e4b7f6e` 之后对方推送的 4 个提交：
- `48917ae` docs(benchmark)：三系统矩阵加状态标注、n=5 统计、撤回误归属数字
- `d059a4a` perf(ifr)：einsum 向量化候选打分
- `8ed2591` feat(ui)：Web Dashboard + USER_MANUAL
- `49e3f52` feat(ui)："完整独立 Strata/Ollama 风格本地 AI 客户端" + 手册重写

**总结论**：对方对我方 P0/P1/P2 清单的响应**部分是真实的、可验证的**（状态标注、n=5 统计、35.97 GB/s 正名、27.1 s/tok 采纳、180× 删除、16.4 ms 从 JSON 撤回、einsum 向量化——全部复现通过）。但：① **新上线的"本地 AI 客户端"是一个 Ollama 代理外壳**，推理全走 llama.cpp，监控面板的两个核心数字（8.0×、0.9382）是代码注释自标 "Mock" 的硬编码，USER_MANUAL 却把基准臂数值写成客户端特性；② README §7 与 md 中**我点名的编造参数（4,105 槽、--ple-io direct、QSA 4-cell、7.8 GB/s、8-bit KV 4K 以上、KV 流式分页 64K 以上）原样保留**，其中"撤回 7.8 GB/s"只在 JSON 里做到了，提交信息与 README/md 不符；③ einsum 只消除了 59× 中的一部分，逐候选 `np.mean` 循环仍占 73% 耗时。

---

## 1. 确认修复（逐条复现通过）

### 1.1 三系统矩阵状态标注 ✅
README §7 / md §1 / JSON 三列分别标注 **[1] 实测基线 / [2] 官方标称规格 / [3] 微架构原型与模拟**；JSON 增加 `status` 字段（LIVE_MEASURED / OFFICIAL_PUBLISHED_SPEC / MICROBENCH_AND_SIMULATION）。这是我方 v2 审计的核心要求，执行到位。

### 1.2 Recall 归因（部分）✅🟡
- md §2 表格："[2] 原版 Strata (Arm A)" → "**[2] BlockMean_A**"，列头 "相对原版 Strata 增益" → "**相对朴素均值增益**"；
- README/md 说明列："相对 Strata 召回提升 +43.0%~+45.5%" → "**相对朴素均值提升**"；
- md §2 增加真值说明："内容空间代理余弦相似度，非真实动态因果 Softmax 注意力"；
- 🟡 **但 [2] 列仍显示 "0.6450 (BlockMean_A 均值池化实测)"**——把自己臂的值放在 Strata 列里，读者扫表格仍会读成 "Strata Recall = 0.6450"。正确写法：[2] 单元格应为 "N/A（Strata 无 KV 块检索召回指标）"。

### 1.3 "35.97 GB/s" 正名 ✅
README/md/JSON 中的 "跑满 35.97 GB/s 带宽" → "**C++ 测得连续内存算位宽 35.8~49.4 GB/s**"（区间含我方复现的 49.35/48.11 与他们日志的 35.79）。JSON 直接删掉了 `io_bandwidth_util` 字段。✅

### 1.4 "16.4 ms" 与 "180×" ✅（仅 JSON 与说明列）
- JSON [2] `control_plane_latency`："16.4 ms (Python 调度器 / 原位回填冲突)" → "C++ 异步图捕获与 CPU 专家线程池调度"；
- README/md 控制面行删除 "提速 180×"，改为 "单核微基准测得，Fast Path 在 100µs SLA 边界波动"，并给出 **Fast Path 69.8~106.8 µs / General 167~251 µs**（含我方测出的 FAILED 运行）——诚实。
- 🟡 但 md §1 [2] I/O 行仍写 "PCIe 随机小包易遇 **7.8 GB/s** 瓶颈"，README §7 同样保留（见 §3.2）。

### 1.5 Ollama n=5 统计 ✅（方法学真升级）
`benchmark_tripartite_comparison.py` 改为 n=5 repeats + mean±std + `num_ctx=4096`。我方复跑：**prompt 19,787.87 ± 10,747.49 tok/s**（他们 21,659.27 ± 11,687.43，同量级同方差量级）；**gen 26.39 ± 0.21**（他们 29.51 ± 0.44——比他们低 12%，见 §3.6）。md 诚实地标注 "热缓存状态，冷启动约 700 tok/s"。

### 1.6 采纳我方复现值与 27.1 s/tok ✅
- md §3 的 0.5B 行改用**我方复现值** 8,528.31 ± 350 / 225.67 ± 10；
- [3] 生成速度："**层流式模拟下 ~27.1 s/tok（用于离线超大模型推理，非实时交互场景）**"——我方 v2 审计发现的数字被正式采纳为对己不利的证据；
- [3] Prompt："**待整机端到端全量测量 (UNMEASURED)**"；
- "✅ 成功" → "✅ 架构可行"；"42.0 GiB" 标注 "推导目标"；[1] 10M 足迹的错误 "324.0 GiB" 删除。

### 1.7 `ifr.py` einsum 向量化 ✅（我方的 P2 #10 被采纳，但只做了一半）
`d059a4a` 将逐子质心 Python 循环改为单次 `compute_logits(query, cand_vectors_arr)` + `argpartition` + O(1) dict 查找——与我方建议一致，且语义等价（分裂块显式取 `np.mean(sub_centroids, axis=0)`，与旧 mean-of-logits 数学相同）。83 测试仍全过（我方复现 `83 passed in 3.44s`）。

---

## 2. 新上线产品：Web "AI 客户端" 是 Ollama 代理外壳（❌ 本轮最严重问题）

`web/server.py`（276 行）+ `web/index.html`（675 行）构成 README 新增的 "§2 一键启动本地 AI 客户端"。逐行核对：

### 2.1 推理链路与 KVMem 无关
- `POST /v1/chat/completions`、`POST /api/chat` → **原样转发 `http://127.0.0.1:11434/api/generate`（Ollama）**；流式响应是逐 chunk 转发；
- 全文件**没有任何一处**调用 `kvmem_fusion`（IFR/CASA/UBBA/LADDER）—— grep 可证；
- README 称 "启动与 Ollama / Strata 原生体验一致的本地桌面 Web 交互客户端"，实质是"给 Ollama 套了一层 Strata 风格皮肤的 OpenAI 兼容代理"。

### 2.2 监控面板是硬编码装饰，代码注释自认 "Mock"
`server.py` L109 注释原文：`# Mock hardware & KV Cache telemetry`。返回的 `kv_compression_ratio: "8.0x"`、`recalled_precision: "0.9382 (Recall@64)"` 都是**写死的字符串**。前端 `mon-kv-ratio`（8.0×）与 `mon-recall`（0.9382）两个元素**没有任何 JS 更新路径**（`fetchMonitor()` 只更新 RSS 和请求历史），永远显示初始 HTML 值；"CASA/IFR/UBBA/LADDER 四大子系统 ACTIVE" 是静态 HTML 表格。唯一真实的 `peak_rss_mb` 是这个 Python 代理进程自己的 RSS（几十 MB），与 "CASA 零拷贝层双缓冲流式" 无关。

### 2.3 USER_MANUAL 把基准数值写成客户端特性
`docs/USER_MANUAL.md` §4 "核心技术特性与优势"："与传统 Ollama 或 Strata 相比，KVMem 客户端在'引擎盖下'拥有以下显著区别：LADDER 将 10M 足迹从 324GB 缩减至 42GB；IFR 将召回从 64.5% 拉升至 93.8%；32GB 电脑跑 70B 常驻 < 4.0GB"——这三个数分别来自**合成推导、Qwen2.5-0.5B 代理真值基准臂、层流式模拟**，没有一个在这个客户端的代码路径里。§2 Monitor 节称 "实时展示 0.9382 高召回率"——实际是写死的 HTML。
（手册 FAQ Q1 反而无意中说了真话："本客户端后端默认连接本地推理算子守护进程…`./bin/ollama serve`"。）

### 2.4 其他实现层问题
- 模块 docstring 声称提供 `POST /api/generate`（Ollama 兼容端点），**do_POST 只处理 `/v1/chat/completions` 与 `/api/chat`，该端点未实现**；
- Ollama 不在线时 `/v1/models` 回退列表包含 `Qwen3.8-Flash-Next-Coder`——本地 Ollama 从未拉取过该模型（那是 Strata 的 29.6GB Coder 包），下拉框会出现一个点了也跑不通的选项；
- "多轮对话" 是把全部 messages 拼成单个 prompt 转发，无对话模板与历史窗口管理；
- USER_MANUAL 的 ASCII 界面图里有 "KVMem Compressed" 徽标，真实 UI 中不存在。

---

## 3. README/md 中我点名的编造参数原样保留（❌ P0 未清完）

`48917ae` 提交信息称 "Retracted misattributed '16.4 ms' and '7.8 GB/s from Strata column to ensure strict scientific attribution"——**该宣称只对 JSON 成立**。README §7 与 md §1 的 [2] 列现状：

| 内容 | 我方审计结论 | 现状 |
|---|---|---|
| 动态 VRAM 专家缓存 **(4,105 槽)** | 编造；论文为 "~4,500 of 24,576, adaptive"（262K 时 ~1,600） | ❌ 仍在 README+md+JSON |
| SSD 直接 I/O **(`--ple-io direct`)** | 编造；Strata 无此 flag | ❌ 仍在 README+md（JSON 已删该 flag 字样） |
| **QSA 4-cell 页表**细粒度调度 | 范畴错误（QSA 是模型架构 12/48 层）+ 数字张冠李戴（"4 token" 是 MTP 窗口） | ❌ 仍在 README+md+JSON（且从 "4-token indexer cell" 改成了同样错误的 "4-cell 页表"） |
| PCIe 随机小包易遇 **7.8 GB/s 瓶颈** | 属 OSDI '26 Strata（PCIe5 GPU 服务器），与 Niko1221/Strata（SSD 专家流式，实测 2.97 GB/s @1.75MB 读）无关 | ❌ 仍在 README+md；JSON 已删 |
| **8-bit KV (4K 以上)** | 论文为 "Past 8K" / "from 32K up"，非 4K | ❌ 仍在 |
| **KV 流式分页 (64K 以上)** | 不存在此机制（64K 只是论文结果表的长度列） | ❌ 仍在 |
| [2] Recall@64 = **0.6450** | 系他们自己的 BlockMean_A 臂 | 🟡 值旁加了臂名，但仍在 Strata 列 |
| [2] 10M 足迹 = **324.0 GiB** | 125B MoE（12 QSA 层×2 KV 头×128 维）fp16 应为 ~117 GiB、8-bit ~58.6 GiB，324 匹配任何配置都不对 | ❌ 仍在 README+md+JSON |
| [2] "Mac 端依赖 Unified Memory" / "~14 tok/s (Mac 32GB 流式)" | 新增的无来源 gloss；14 tok/s 本身可溯源（MAC.md M4 32GB Q2_0 14.1）但配置条件未注 | 🟡 |

---

## 4. einsum 修复只消除了一部分开销（❌ 量化）

我方在 `752df3d` 审计中测得：逐子质心循环 37.94 ms vs 等价单次 dot 0.64 ms（59×）。`d059a4a` 后我方在**他们自己的代码**上复测（12,500 块、9,536 候选、dim 64）：

| 路径 | 时延/query |
|---|---|
| `752df3d`（逐子质心 Python 循环） | 37.94 ms |
| **`d059a4a` 后（当前 HEAD）** | **22.99 ms** |
| 可达（块有效向量缓存后，不含全表扫描） | **2.54 ms** |

cProfile 显示当前 73% 的耗时在**逐候选 `np.mean(sub_centroids, axis=0)`**（9,536 次/query 的 numpy 调用）。修法很简单：在 `train_and_index`/`add_block` 时把每块的有效向量（分裂块 = 双子质心均值，否则 = mean_k）算一次并缓存，`retrieve` 只做 `np.stack` + 一次 dot。按此外推 10M 规模：当前 ≈ 4.6 s/query vs E 门 350 ms（**13× 超**；虽比 `752df3d` 的 23× 改善，但离达标仍远）。

**未处理的语义问题（我方 P2 #11）**：分裂块有效向量 = (needle + residual_mean)/2，needle 权重 1/2；基准臂 `BlockMean_A` 为 1/32。生产与基准对同一块的表示仍相差 16×，einsum 化没有触碰这一点。

---

## 5. 复现材料

| 复核项 | 命令/结果 |
|---|---|
| 测试套件 | `pytest tests/ -q` → `83 passed in 3.44s` |
| 他们 n=5 Ollama | 运行 `benchmarks/benchmark_tripartite_comparison.py` → prompt 19,787.87 ± 10,747.49；gen 26.39 ± 0.21（JSON 已用 git checkout 还原） |
| ifr 性能 | 见 §4 表（脚本逻辑同 `docs/og_repro_752df3d/ifr_perf2.py`，改为直接驱动 `IFRRetriever`） |
| Web 客户端性质 | `grep -n "kvmem_fusion" web/server.py` → 无命中；L109 `# Mock hardware & KV Cache telemetry`；`grep -n "mon-kv-ratio\|mon-recall" web/index.html` → 仅 HTML 初始值，无 JS 赋值 |
| Strata 参数 | `Niko1221/Strata` 一手文档（同 `752df3d` 审计） |

---

## 6. 给对方的下阶段清单

**P0**
1. README/md/JSON 三处同步删除：`4,105 槽`（改 "~4,500，自适应"）、`--ple-io direct`、`QSA 4-cell 页表`（改为模型架构注记）、`7.8 GB/s`、`8-bit KV (4K 以上)`（改 8K/32K）、`KV 流式分页 (64K 以上)`（删除）。
2. [2] 列 Recall 单元格改为 "N/A"，0.6450 只在 §2 己方臂表格出现；[2] 10M 足迹 324 GiB 按 125B MoE 实际 KV 维度重算或删除。
3. Web 客户端的定位与事实对齐：要么在 UI/手册明示"前端外壳 + Ollama 后端，监控面板数值为演示用静态值"，要么把 `/api/monitor` 接到真实测量。USER_MANUAL §4 的三个数必须改述为"基准/模拟结论"，不是客户端特性。
4. 补 `POST /api/generate` 或从 docstring 删除；`/v1/models` 回退列表删除未安装的 `Qwen3.8-Flash-Next-Coder`。

**P1**
5. `ifr.py` 缓存块有效向量（我方实测 22.99 → ~2.5 ms/query @12.5K 候选）。
6. 明确分裂块 needle 权重 1/2 与基准 1/32 的差异及理由。
7. gen 吞吐标注机队间波动（他们 29.5±0.4 vs 我方同机复跑 26.4±0.2，差 12%）。

**P2**
8. README §7 引言 "三方全维度实测" 改为 "一方实测、一方官方标称、一方原型模拟"。

---

## 7. 对我方的行动建议（问题 3 的回答底稿）

1. **本审计文档入库**（`docs/og_audit_49e3f52.md`），并在 `docs/README.md` 索引；同时更新 `og_track_a_v3_cleanup_compare_optimize.md` 的状态列（P0 清单 5 项中 2 项已清、3 项部分清）。
2. **采纳他们的 n=5 mean±std 协议**到我方所有 Ollama 基线测量（我方前两轮的单次值 434.66/477.38/537.55 应重测为 n=5 并报告 std）。
3. **考虑发布一页 "Strata 事实卡"**（MODELS.md/MAC.md/论文的关键数值 + 常见误引纠正），对方已连续三次在同一组参数上出错，一次性给准比逐轮点名更高效。
4. **评估是否需要一个我方的可运行 demo**：对方用 276 行代理 + 硬编码面板做出了"产品感"；我方若只有审计文档，在演示层面会持续被动。最低成本方案是把现有 `ifr.py` + 真实 Qwen 测量包一个只读监控页（数值全部来自实测，无静态装饰）。

---

*审计人：OG 组。所有数值可依 §5 复现。本文件审计 `e4b7f6e..49e3f52`；更早问题见 `og_audit_752df3d.md`、`og_audit_tripartite_8d88c3b.md`（v2）、`og_verify_aa07e56_gt_sensitivity.md`。*
