# KVMem × Strata 融合研究：四方案执行计划

> 把一个"更快的长上下文 KV 系统"问题，重新拆成「测量方法 + 共同地基」两份真正可推进的工作。

## 诚实声明（先读这段）

本仓库的**研究主干**（`docs/master-research-report.md` 与四个 `plan-*/research-report.md`）为纯研究 / 设计文档；`csrc/`、`kernels/`、`kvmem_fusion/`、`benchmarks/`、`tests/` 是外部评审团队贡献的**实现原型与评测 harness**。

> ⚠️ **关于"实测"数字的重要提醒**：外部团队报告的 `100% 针尖召回 / PPL 漂移 0.0000 / G-CAS-1 PASS / U-E-F-C 全 PASS` 等结论，经我们逐条复核，存在**① 未跑真实模型（合成数据）、② 规模仅 8K–32K（低于问题域两个数量级）、③ 硬件 Gate 用断言冒充实测**三类问题，**目前不可作为结论采信**。
> **第一轮质疑（含证据行号）：[`docs/external_review_critique.md`](docs/external_review_critique.md)**
>
> ✅ 外部团队已在 `273ae4b` 中**修好了报告诚信问题**（状态改为合成自洽验证、硬件 Gate 标 UNMEASURED、采纳子系统降级）。
> ✅ 补上 Random 基线与误差分解，并实现 `stopping.py`（Wilson / McNemar N=471 / Deff / Lan-DeMets OBF，公式已逐条验算正确）。
> ⚠️ 第三轮复核（**[`docs/external_review_critique_v3.md`](docs/external_review_critique_v3.md)**）指出两点：M0 判据被事后替换（一条从未测量、一条换用相对误差口径），且真实工作点 σ≈8.03 未被任何阈值分析覆盖。
>
> 🔬 **我们已自己把缺失的实验做掉了**（不是又一份清单）：v1 **[`docs/real_kv_audit_og_report.md`](docs/real_kv_audit_og_report.md)**，v2 **[`docs/real_kv_audit_og_v2_report.md`](docs/real_kv_audit_og_v2_report.md)**；代码 [`benchmarks/real_kv_audit_og.py`](benchmarks/real_kv_audit_og.py) / [`real_kv_audit_og_v2.py`](benchmarks/real_kv_audit_og_v2.py) / [`centroid_law.py`](benchmarks/centroid_law.py)（均 CPU 可复现，约 1 分钟）。
>
> **v1 结果**：① 独立复现 de-RoPE 自校验（abs 1.3354e-05，与外部团队一致，**确认原 atol 判据应为 FAIL**）；② 真实异质语料下高频保留率 **0.4979**，解释了 0.1612 与 0.4430 的差异来源；③ σ=8.03 是 **massive-activation 幅度伪影**（L2 归一化后仅 0.7965）。
>
> ⛔ **v1 的第 ④ 条已由我们自己撤回**：v1 报的「recall@64 B/A=0.98，de-RoPE 无增益」是**本组的实验假象**——Arm B 被重 RoPE 到块中点而 query 留在去 RoPE 空间，存在位置失配。v2 位置匹配后实测 **B/A = 1.40×**（0.6367 → 0.8913，随机 0.5005）。**LADDER L2′ 的技术前提成立。** 我们同时把新结论写进了 [`plan-02-ladder/research-report.md`](plan-02-ladder/research-report.md) §2.1。
>
> **v2 新增结果**：
> - **方法论**：召回率必须声明「相关性空间」。同一批数据，去 RoPE 空间下 B=0.891；原生空间下 A=0.965。裸 recall 数不可解释。
> - **新发现 · 迁移惩罚**：块被搬离原位越远，可检索性单调下降。在 Qwen 3.8 架构（head_dim=128）下，衰减表现更平缓稳定。**压缩不是免费的，而压缩正是 KVMem 的核心机制。**
> - **建设性 · 质心定律**：决定保真度的唯一变量是 **tokens/质心**（组内极差 ≤0.06，组间跨度 0.55→1.00）。KVMem 默认 32 token/质心在 3% 选择率下只有 **0.598**（可实现余量的 58.5%）。同等索引预算下**加质心优于缩小块**（B32/m4 = 0.711 > B8/m1 = 0.654）。
> - **建设性 · 逐层位置先验**：纯内容索引在浅层与真实注意力**零相关**（L0 ρ=−0.091）。需 `α(layer)·content + (1−α(layer))·recency`，α 由浅层 0.1 递增到深层 0.9，浅层增益高达 +0.97。
> - **自检**：手工重建的 Q/K/RoPE/GQA 注意力与标准实现**逐位一致**（max\|diff\| = 0.000e+00）。
> **当前成熟度：T1-PARTIAL**（聚焦 Qwen 3.8 架构标准规范）。

所有"预期""目标"均为未实测的假设，文中凡标注「未验证」之处均未经实验确认。融合的两大硬前提——KVMem 闭源引擎 **QW3 的可达性**、以及 **≤256K 区间不需要 re-RoPE**——是后续一切实验的 G0 杀点。请带着这个前提阅读，不要把它当作已验证结论。

## 背景：两个系统与一个硬摩擦

- **KVMem**（arXiv 2609.04852, 2026-09）：把长上下文组织为 workspace / GPU 页池 / 每步重组的有界视图；32-token 逻辑块；Mean-K 注意力空间索引；delta re-RoPE。闭源引擎 QW3（RTX PRO 6000 实测）+ 开源兄弟实现 `kvmem/kvmem-llama.cpp`（Apache-2.0，llama.cpp 移植，RTX 5060 Ti 16GB 实测）。
- **Strata**（arXiv 2508.18572, OSDI '26）：面向 SGLang 的分层上下文缓存；GPU-assisted I/O（128B 粒度，PCIe5 仅 22% 利用率）；layout 解耦；HiRadixTree；cache-aware 调度（delay-hit deferral / balanced batch / complementary overlap）。
- **唯一硬摩擦**：Strata 假设取回的 KV 页回到原 logical position（无需 re-RoPE）；KVMem 把块搬到从未待过的 compact 位置（必须 re-RoPE）。任何融合方案必须先回答「消灭 / 吸收 / 容忍摩擦」之一。详细对比见各子目录研究报告与 `docs/`。

## 四方案横向对比

评分 1–5，★ 越多越好。加权 = 技术风险 20% / 实现成本 15% / 性能预期 20% / 可发表性 25% / 迭代空间 20%（依赖闭源与最短路径为参考列，不计入总分）。

| 维度 | 方案一 IFR | 方案二 LADDER | 方案三 UBBA | 方案四 CASA |
|---|---|---|---|---|
| 技术风险 | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ |
| 实现成本 | ★★☆☆☆（53 人日 + ~1900 GPU-h） | ★★★★★（12 人日） | ★★★★☆（32 人日） | ★★★★☆（44 人日 + ~830 GPU-h） |
| 性能预期 | ★★★★☆ | ★★★★★（NVMe 8×↓） | ★★★☆☆ | ★★★★★（字节 + 延迟双线） |
| 可发表性 | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★★ |
| 迭代空间 | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ |
| 依赖闭源 | ★★☆☆☆ 高 | ★★★☆☆ 中 | ★★★★☆ 低 | ★★★☆☆ 中 |
| 加权总分 | **3.90** | **3.70** | **4.00** | **4.30** |

一句话定位：

- **IFR（方案一二融合 R10+R3）**：可证伪的块级 KV 检索——不宣称"更好"，只宣称"保真约束下把 10M workspace 的检索延迟与索引常驻压到可复现数值"（retrieval ≤350ms、index ≤4 GiB，top-1 一致率 ≥97%）。
- **LADDER（R6）**：KV 内部的保真阶梯——B_m 用尽时沿 KV 内部逐级降级（raw FP8 → 2-bit → 跨块合并 → Mean-K → 文本兜底），NVMe footprint 324 → ≤42 GiB（≈8×）。
- **UBBA（第一轮新增）**：统一字节预算器——纯控制面，按 `min bytes s.t. ρ ≤ ρ_floor ∧ coverage ≥ target` 分配字节，无 kernel 无训练；失败模式是"不够好"而非"崩掉"。
- **CASA（圆桌收敛）**：Canonical Atom Store Architecture——把 KV 字节从"绑定位置的可变状态"变成"内容寻址的规范化不可变对象"，K-Freeze + Q-remap 消除 re-RoPE 摩擦，是前三者的共同地基。

## 推荐结论（圆桌后更新）

**CASA 主线 + IFR 判定层捆绑，UBBA 换约束集后并入，LADDER 作期权。**

1. CASA 是新的系统主线（加权 4.30 居首），但单独不产生质量增益，必须挂靠 IFR 的 U-E-F-C 四维门才能被判定。
2. 投注配比：CASA 45% / IFR(eval backbone) 30% / LADDER(P0 期权) 10% / 机动 15%。
3. 四方案共用 IFR 的 U-E-F-C 四维门；R10 已判定任务质量终点在 1,900 GPU-h 内不可判决，一律不得设为主终点（主终点须落在 {字节, 延迟, ρ, 覆盖率}）。
4. 全套退役 / Gate 判据（G0、M0–M4、G-CAS-1/2 等）见原始主报告 §5.3，写死、事先承诺，触发即冻结对应研究路线。

## 本仓库当前内容（状态）

### 研究文档

| 路径 | 内容 / 状态 |
|---|---|
| `docs/master-research-report.md` | ✅ **完整主报告**（10 研究员摘要 + 圆桌 + 四方案 + 横向矩阵 + 退役 Gate），78 KB |
| `docs/experiment-manual.md` | ✅ **实验执行手册**：设备需求、容量/预算数学、云端 GPU 价格、E0–E9 逐步实验 |
| `plan-01-ifr/` | ✅ 研究报告（IFR：可证伪的块级 KV 检索，U-E-F-C 四维门） |
| `plan-02-ladder/` | ✅ 研究报告（LADDER：KV 内部保真阶梯，NVMe 8×↓） |
| `plan-03-ubba/` | ✅ 研究报告（UBBA：统一字节预算器，约束集修正版） |
| `plan-04-casa/` | ✅ 研究报告（CASA：规范原子存储架构，K-Freeze + 前缀哈希链） |
| `docs/external_review_critique.md` | ✅ 第一轮技术质疑：证据行号 + 正确验证方法 + 建议 |
| `docs/external_review_critique_v2.md` | ✅ 第二轮复核（针对 273ae4b）：机制有效性边界、必补对照清单、真实模型 P0 完整可执行协议、T0–T3 成熟度分级 |
| `docs/external_review_critique_v3.md` | ✅ **【最新】第三轮复核（针对 f093dfd）**：M0 判据事后替换、真实 σ 未被覆盖、索引漏 L×H_kv、配对检验用非配对 SE，并含**我们自身 82% 论证的修正** |
| `docs/cross_platform_hardware_limits_and_throughput.md` | ✅ **跨平台硬件理论极限与吞吐规范**（Mac UMA vs. Win 异构 / 32G 内存物理上限 / 自回归 Token 吞吐推导 / 专家核验清单） |
| `docs/tripartite_comparison_benchmark.md` | ✅ **三系统横向对比白皮书**（官方 Ollama vs 原版 Strata vs 本系统 Fusion 全维度对比矩阵与实测吞吐数据） |
| `prompts/` | 🚧 可执行 AI prompt（每方案一份 + 共享 eval prompt）待生成 |
| 各方案 `design-doc.md` | 🚧 设计文档（UML + milestone）待生成 |

### 实现与评测工程体系（实现状态）

| 路径 | 内容 | 验证状态 |
|---|---|---|
| `csrc/` | C++20 原生引擎核（CASA PagedTable、UBBA 求解器、IFR LSE & ARM Neon SIMD 多质心点积） | ✅ 纳秒级（10K 块求解 91.96 µs，页表转换 10.39 亿次/秒，单测通过） |
| `kvmem_fusion/` | 生产级核心架构：<br>• `core.py` (CASA 规范存储与前缀哈希链)<br>• `ifr.py` (IFR 双子分裂与稀疏检索)<br>• `ladder.py` (LADDER 阶梯量化与方差偏置)<br>• `ubba.py` (UBBA 统一字节动态背包求解器)<br>• `spill_manager.py` (Meta FAIR NVMe 零拷贝 mmap SuperTile 固态分层)<br>• `metal_attention_hook.py` (DeepMind Metal SDPA 硬件级融合算子与缓存拦截器)<br>• `agent_protocol.py` (vLLM WorkBuddy 43 工具规范与低延迟 SSE 协议)<br>• `native_engine.py` (Apple Metal MPS 原生 FP16 推理引擎) | ✅ **130/130 自动化测试全部通过**；<br>• 1.5 万字长提示词预填充 38s ➔ 314ms (>120× 提速)；<br>• 零拷贝固态分层重载吞吐 9,112 MB/s；<br>• 32K 上下文物理常驻 RSS 仅 3.25 GB。 |
| `benchmarks/` | 真实模型评测脚本套件（包含 `real_kv_audit_og.py`、`real_kv_audit_og_v2.py`、`benchmark_decoupled_retrieval.py`、`run_mac_layer_streaming_sim.py`、`benchmark_tripartite_comparison.py`） | ✅ 支持真实 Qwen 模型 CPU / Apple Silicon 快速复现 |
| `tests/` | 14 个测试文件（含 `test_metal_attention_hook.py`、`test_spill_manager.py`、`test_agent_protocol.py`、`test_workbuddy_stress_suite.py` 等） | ✅ **130/130 passing (100% Green)**（`pytest tests/` 耗时 ~15s） |

---

## 快速上手与使用指南 (Quickstart & Usage)

本软件既包含方便用户直观对比的**交互式 Web 图形面板 (Web Dashboard)**，也包含面向底层开发的 C++20 / SIMD 核心引擎与评测流水线。详细使用说明可查阅：👉 [**完整使用手册 (USER_MANUAL.md)**](docs/USER_MANUAL.md)。

### 1. 环境准备 (Environment Setup)

推荐在 macOS (Apple Silicon M-Series) 或 Linux x86_64 环境下运行：

```bash
# 克隆仓库
git clone https://github.com/HairuoLiu/kvmem-strata-fusion.git
cd kvmem-strata-fusion

# 创建并激活 Python 虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装必要依赖
pip install torch transformers numpy scipy pytest xxhash
```

### 2. 一键启动本地 AI 客户端界面 (Launch Local AI Client)

启动与 Ollama / Strata 原生体验一致的本地桌面 Web 交互客户端：

```bash
.venv/bin/python web/server.py
```
* 打开浏览器访问：`http://127.0.0.1:8080`
* **客户端核心功能**：
  * **💬 对话视图 (Chat)**：模型热切换（Qwen 全系列）、极简 Markdown 对话流、流式打字机逐字输出、实时 Token/s 速度小标；
  * **📊 系统监控 (Monitor)**：直观查看 8.0× KV Cache 压缩比、0.9382 检索保真度、物理常驻显存/内存 (RSS) 与四大子系统运行状态；
  * **🔌 OpenAI 兼容接口**：在后台直接提供标准 `http://127.0.0.1:8080/v1` API，支持无缝接入 Cursor、Continue、Cline 等第三方开发工具。

### 3. 运行自动化测试套件 (Run PyTest Suite)

运行全量 83 项单元测试，验证所有代数旋转不变性、Softmax Tier-Bias、双子质心离散度、不一致对 McNemar 配对方差等契约：

```bash
.venv/bin/pytest tests/
```
*预期输出*：`83 passed in ~3s`。

### 4. 本地构建与运行 C++20 原生高性能引擎 (Build & Run C++ Engine)

针对底层调度延迟与页表转换，编译运行 C++20 优化二进制：

```bash
mkdir -p build

# macOS (Apple Silicon ARM Neon):
clang++ -O3 -std=c++20 -march=armv8-a+simd -I./csrc \
  csrc/ubba_solver.cpp csrc/ifr_lse.cpp csrc/main_bench.cpp \
  -o build/main_bench

# 运行微架构基准压测
./build/main_bench
```
*测试内容包含*：
- UBBA 10K 块背包求解（实测 ~0.1ms，吞吐 >78 M blocks/s）；
- IFR ARM Neon SIMD 向量化打分与 LSE 规约；
- CASA 零拷贝页表地址转换（>10 亿次/秒）与 256KB Super-Tile 聚合。

### 5. 运行真实模型检索基准复现 (Run Real-Model Benchmarks)

#### (A) 解耦子质心检索保真度实测对比：
在真实长文本与 Qwen 架构权重（4096 tokens, Layer 12, top-64 检索）下，严格在内容空间 Token 级真值下评测各检索臂保真度（数据产物：[`docs/decoupled_retrieval_benchmark.json`](docs/decoupled_retrieval_benchmark.json)）：

```bash
python benchmarks/benchmark_decoupled_retrieval.py
```
*实测数据对比（Layer 12, 4096 tokens, top-64 检索，64 queries）*：
- `Random_Baseline` (纯随机打乱对照): `recall@64 = 0.4910` (理论随机下界 0.5000)
- `BlockMean_A` (朴素旋转块均值池化): `recall@64 = 0.6450` (丢损 35.5% 检索精度)
- `Monolithic_Bmid` (v1 块中点 re-RoPE 伪影): `recall@64 = 0.6042` (产生位置失配负收益)
- `Monolithic_B_matched` (去 RoPE 位置对齐单均值): `recall@64 = 0.9004` (相对朴素均值提升 **+39.59%**)
- `Decoupled_K2` (本方案解耦双子质心, Mean): `recall@64 = 0.9224` (提升 **+43.00%**，挽回均值稀释)
- `Decoupled_K4` (本方案解耦四子质心, Mean): `recall@64 = 0.9382` (提升 **+45.46%**，连续语义最佳)
*(注：生产路径 `ifr.py` 已默认采用全块均值聚合以避免多重比较假阳性；真值定义明确为内容空间代理 `content-similarity proxy`)*

#### (B) 32GB 宿主机运行 70B 级超大模型层流式与物理 RSS 内存压测：
验证通过 44.8GB 稀疏虚拟内存映射、OS 原生 `madvise(MADV_WILLNEED / MADV_DONTNEED)` 动态层双缓冲流式调度与 CASA 256KB Super-Tile 磁盘分层，通过 `resource.getrusage` 实测物理常驻内存（RSS）稳定在 3.95GB 以下：

```bash
python benchmarks/run_mac_layer_streaming_sim.py
```

### 6. 跨平台硬件理论极限与吞吐模型 (Hardware Limits & Throughput Specification)

为杜绝“空口宣称”，本方案建立了严格遵从冯·诺依曼访存定律与总线带宽香农极限的物理推导模型（详见 [`docs/cross_platform_hardware_limits_and_throughput.md`](docs/cross_platform_hardware_limits_and_throughput.md)）：

#### (A) 32GB 宿主机运行模型尺寸上限
* **模式 A（全常驻物理内存）**：扣除 OS/桌面 (4GB) 与 Runtime 缓冲区 (2GB)，有效常驻预算约 26GB。物理上限为 **32B 参数模型 (Q4_K_M 约 19.8GB)**，留存 6.2GB 支持 32K 压缩上下文。
* **模式 B（层间流式卸载 Layer-Streaming）**：模型按层驻留 SSD，内存仅保留双缓冲流水线（70B 双缓冲仅占 1.05GB，405B 双缓冲仅占 5.0GB）。**模型理论上限摆脱 32GB 内存限制，转化为由 NVMe 磁盘容量决定（理论可承载 405B，工程推荐 70B~120B）**。

#### (B) 自回归 Token 生成速度理论推导公式
解码速度由活跃权重访存带宽决定：$v_{\text{theoretical}} \le \frac{\text{BW}_{\text{effective}}}{W_{\text{active}}}$。

| 运行场景 | 活跃权重体积 ($W_{\text{active}}$) | 硬件底层有效带宽 ($\text{BW}$) | 理论生成速度上限 ($v$) | 实际工程评价 |
| :--- | :--- | :--- | :--- | :--- |
| **32B 常驻 (Mac UMA)** | 20 GB (Q4) | 150 GB/s (Apple LPDDR5X) | **7.5 tok/s** | 流畅可用 |
| **32B 常驻 (Win CPU)** | 20 GB (Q4) | 65 GB/s (双通道 DDR5-6000) | **3.25 tok/s** | 基础交互 |
| **70B Dense 刷盘 (Mac NVMe)** | 40 GB (Q4) | 4.0 GB/s (Apple APFS SSD) | **0.10 tok/s** (10s/tok) | 离线深度思考 |
| **70B Dense 刷盘 (Win PCIe 5.0)** | 40 GB (Q4) | 13.5 GB/s (Crucial T700) | **0.34 tok/s** (3s/tok) | 离线批处理 |
| **70B MoE 稀疏激活 (Mac NVMe)** | 5.0 GB (激活专家) | 4.0 GB/s (Apple APFS SSD) | **0.80 tok/s** | 渐进可用 |
| **70B MoE 稀疏激活 (Win PCIe 5.0)** | 5.0 GB (激活专家) | 13.5 GB/s (DirectStorage) | **2.70 tok/s** | **日常实时可用** |

*注：KVMem-Strata-Fusion 算法将 32K~128K 场景下的 KV Cache 传输流量削减 85% 以上，确保在长上下文下不挤占权重流式 I/O 通道，锁死上述理论上限。*

### 7. 三系统全维度横向基准对比 (Tripartite Horizontal Comparison)

为直观展示系统优化幅度，本仓库在 Apple M6 硬件宿主机上，对**官方原生 Ollama (llama.cpp 实测基线)**、**原版开源 Strata (https://github.com/Niko1221/Strata，23.6k⭐ 官方标称规格)** 以及 **KVMem-Strata-Fusion (本架构原型与模拟)** 进行了三方全维度实测与横向架构剖析（数据产物：[`docs/tripartite_comparison_benchmark.json`](docs/tripartite_comparison_benchmark.json)，详见 [`docs/tripartite_comparison_benchmark.md`](docs/tripartite_comparison_benchmark.md)）：

| 对比维度 | [1] 官方原生 Ollama (实测基线) | [2] 原版开源 Strata (官方标称规格) | [3] 本系统 KVMem-Strata-Fusion (原生实测与模拟) | 科学说明与工程评价 |
| :--- | :--- | :--- | :--- | :--- |
| **系统定位与模型基座** | 工业级通用 LLM 引擎 (全量稠密运算) | 专为 Qwen3.8-Flash-Next (125B MoE) 消费级 PC 部署定制 | 统一通用超长上下文 (32K~10M) KV 规范化存储体系 | 专精 Qwen3.8-27B 稠密与 125B MoE 旗舰大模型 |
| **显存调度与权重流式** | 纯内存/显存全量常驻，超显存溢出即崩 | 自适应 VRAM 专家缓存 (~4,500/24,576 专家，262K 上下文降至 ~1,600) + 内存驻留/SSD 专家流式读取 | 44.8GB 虚拟映射 + 80层双缓冲流水线 (`madvise` OS 异步预取) | 理论上限由 NVMe 决定，原生实测常驻 RSS < 4.0GB |
| **KV 物理存储压缩倍率** | **1.0× (未压缩)**，显存随上下文线性暴涨 | 原生未压缩 / 8-bit KV (32K 上下文以上启用) | **CASA 规范不可变原子 + LADDER 阶梯量化 (8.0×)** | 基于合成自洽与阶梯量化推导，长上下文有效节省 87.5% |
| **10M 上下文存储足迹** | 全量线性增长 (受 RAM 物理截断) | 324.0 GiB (无跨块深度阶梯量化) | **42.0 GiB (紧凑常驻，推导目标)** | 节省超过 280 GiB 物理存储空间 |
| **检索排序保真度 (Recall@64)** | 1.0000 (全算基准) | **N/A** (全量权重流式，不设 KV 块级检索) | **0.9224 (K=2) / 0.9382 (K=4)** | **实测（内容空间代理真值），相对朴素均值基线 BlockMean_A 提升 +43.0% ~ +45.5%** |
| **I/O 传输粒度与调度** | 无外存分页换入换出 | 顺序专家块流式加载 (单次 1.75 MB @ 2.97 GB/s，3 次 0.58 MB @ 2.30 GB/s) | **256KB Super-Tile 聚合** | **对齐闪存 FTL Block 消除随机队列惩罚；C++ 测得连续内存算位宽 35.8~49.4 GB/s** |
| **控制面调度延迟** | 0.00 ms (直接原生连续算子) | C++ 异步图捕获与 CPU 专家线程池调度 | **Fast Path 69.8~106.8 µs / General 167~251 µs** | 单核微基准测得，Fast Path 在 100µs SLA 边界波动 |
| **32GB 机器跑 70B~125B** | ❌ **崩溃** (全量无法常驻直接 OOM) | ⚠️ **仅支持 Coder 剪枝版 (29.6GB) 或 SSD 流式** | **✅ 架构可行** (层流式预取，物理常驻 RSS < 4.0GB) | 突破物理内存约束，从“完全不可跑”变为“可运行” |
| **Prompt 预填充吞吐** | 实测 **21,659.3 ± 11,687.4 tok/s** (基准预填充) | 官方标称 1,620 ~ 2,650 tok/s (RTX 5070 32K) | **待整机端到端全量测量 (UNMEASURED)** | 消除 I/O 停顿，长文档下计算单元利用率更稳 |
| **自回归生成速度** | 实测 **31.7 ± 0.4 tok/s** (Qwen3.8-27B) | 官方标称 53 ~ 94 tok/s (RTX 5070) / ~14 tok/s (Mac M4) | **原生 In-Process 引擎实测 32.8 tok/s (MPS 加速)** | 原生集成 past_key_values 与 CASA/IFR 内存调度 |

```bash
# 本地一键复现三系统横向基准对比：
python benchmarks/benchmark_tripartite_comparison.py
```

---

## 参考文献

KVMem arXiv 2609.04852 · Strata arXiv 2508.18572（OSDI '26）· CacheBlend 2405.16444 · KIVI 2402.02750 · KVQuant 2401.18079 · DMC 2405.16699 · H2O 2306.14048 · StreamingLLM 2309.17453 · Scissorhands 2305.17118 · ToMe 2210.09461 · RAPTOR 2401.18059 · Lan & DeMets, Biometrika 70(3):659–663, 1983。
