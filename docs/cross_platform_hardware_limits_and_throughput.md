# 跨平台硬件理论极限与吞吐推导规范 (Cross-Platform Limits & Throughput Specification)

> **文档属性**：系统级理论推导与工程边界白皮书  
> **适用目标**：提交外部专家评审团（External Review Panel / OG Review Team）交叉审计与物理自洽性核验  
> **制定原则**：严格遵从研究诚信准则，**严格区分物理理论上限（Theoretical Bound）与本地原型实测（Empirical Measurement）**。

---

## 1. 硬件平台与体系结构差异建模 (Mac UMA vs. Windows Heterogeneous)

在大模型长上下文推理系统中，内存带宽（Memory Bandwidth）与 I/O 吞吐是决定解码延迟的绝对物理瓶颈。

### 1.1 系统架构与总线对比

| 体系结构属性 | Mac (Apple Silicon, 如 M-Series) | Windows PC (典型 x86_64 + 独显 / APU) | 对 KVMem-Strata-Fusion 的直接影响 |
| :--- | :--- | :--- | :--- |
| **内存组织** | **统一物理内存 (UMA)**<br>CPU、GPU、NE 共享统一 LPDDR5X 总线 | **分离异构存储 (Host RAM + VRAM)**<br>主机内存 (DDR4/5) 与 GPU 显存 (GDDR6X) 物理隔离 | Mac 支持统一指针直接寻址（零拷贝 Zero-Copy）；Windows 跨介质必须经 PCIe 复制。 |
| **内存总线带宽** | **150 ~ 800+ GB/s**<br>Base ~150 GB/s, Pro ~200-300 GB/s, Max ~400-800 GB/s | **50 ~ 80 GB/s (Host RAM)**<br>双通道 DDR5-6000 理论 96 GB/s，有效 ~65 GB/s；显存内部 500~1000 GB/s | 纯 CPU 解码时 Mac 带宽大幅占优；若 Windows 具备大显存独显，显存内运算极快但受限于显存容量。 |
| **PCIe 通道瓶颈** | 无独立 PCIe 显卡总线开销 | **PCIe 4.0 x16: 31.5 GB/s**<br>**PCIe 5.0 x16: 63.0 GB/s** | 超显存流式换页必须经过 PCIe 瓶颈，存在 Host $\to$ Device 传输延迟。 |
| **NVMe 闪存 I/O** | **APFS + Mach VM / mmap**<br>系统级 `madvise(MADV_WILLNEED)`，顺序读速约 3.5 ~ 5.5 GB/s | **Win32 IOCP + DirectStorage**<br>支持 GPU DirectStorage (NVMe $\to$ VRAM 旁路)，PCIe 5.0 SSD 读速可达 **12.0 ~ 14.5 GB/s** | **Windows 高端 NVMe 物理读取吞吐是 Mac 板载闪存的 2.5 ~ 3.5 倍**。 |
| **计算后端** | ARM Neon SIMD + Metal (MPS) | x86 AVX-512 / AMX + NVIDIA CUDA / Triton | Windows NVIDIA Tensor Core 浮点峰值算力大幅领先。 |

> **带宽口径声明**：M1 基座实测为 **68.25 GB/s**（LPDDR4X），M1 Pro ≈ 200 GB/s、M1 Max ≈ 400 GB/s、M2 Ultra ≈ 800 GB/s；
> "Base ~150 GB/s" 不对应任何量产芯片。§3.2 场景 1 采用的 150 GB/s 属 **M1 Pro 级假设**，引用该结论时必须同时声明机型
> （若按 M 系列基座 68–100 GB/s 计，32B 常驻为 **3.4 ~ 5.0 tok/s**，而非 7.5）。

---

## 2. 32GB 物理内存宿主机：模型运行尺寸的理论极限

针对本地 32GB RAM 配置，必须严格界定两种运行模式的理论边界：

### 2.1 模式 A：模型与 KV Cache 全量常驻内存 (In-Memory Resident)

* **物理内存预算方程**：
  $$\text{RAM}_{\text{total}} = 32\text{ GB} = \text{RAM}_{\text{OS}} + \text{RAM}_{\text{Runtime}} + \text{RAM}_{\text{Weights}} + \text{RAM}_{\text{KV\_Cache}} + \text{RAM}_{\text{Act}}$$
  * 操作系统与图形桌面保留（Windows 11 / macOS）：$\approx 4.0\text{ GB}$
  * 运行时堆栈与中间激活值缓冲区：$\approx 2.0\text{ GB}$
  * **用户实际可用常驻预算**：$\approx 26.0\text{ GB}$

* **常驻规模物理判定矩阵**：

| 模型参数量 | 4-bit 权重 $W$ | KV 余量 | KV 字节/token<br>(FP16, GQA-8, d=128) | 可承载上下文<br>(FP16 KV) | 可承载上下文<br>(int8 KV) | 判定结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **7B / 8B** | $\approx 4.8\text{ GB}$ | $\approx 21.2\text{ GB}$ | 128 KB | **173.7K tokens** | 347.3K tokens | 达 256K **需 KV 量化** |
| **14B** | $\approx 9.2\text{ GB}$ | $\approx 16.8\text{ GB}$ | 160 KB | **110.1K tokens** | 220.2K tokens | 达 128K **需 KV 量化** |
| **32B** | $\approx 19.8\text{ GB}$ | $\approx 6.2\text{ GB}$ | 256 KB | **25.4K tokens** | 50.8K tokens | **常驻实用上限**；达 32K 需 KV 量化 |
| **70B / 72B** | $\approx 42.0\text{ GB}$ | 负数 ($-16\text{ GB}$) | 320 KB | $0$ (直接 OOM) | $0$ (直接 OOM) | **不可常驻** (除非 2-bit 极端劣化至 22GB) |

> **核算口径（必须随表声明）**：KV 字节/token = 层数 × KV head 数 × head_dim × 2 (K,V) × dtype 字节。
> 例：8B（32 层, 8 KV head, d=128, FP16）= 32×8×128×2×2 = 131,072 B = **128 KB/token**。
> 早期版本写作 "$>256\text{K}$ / $\approx128\text{K}$ / $\approx32\text{K}$" 但**未声明该假设**，核算后分别需要
> **1.47× / 1.16× / 1.26×** 的 KV 压缩才成立。

---

### 2.2 模式 B：层间流式卸载模式 (Layer-by-Layer Streaming / Over-Memory)

* **系统工作机制**：
  * 模型全量权重存储于 NVMe SSD，按 Transformer 层划分。
  * 内存中仅维护**双缓冲预取窗口 (Double-Buffering Ping-Pong Window)**：计算当前层 Layer $i$ 的同时，异步预取下一层 Layer $i+1$，计算完成后立即向 OS 发送 `MADV_DONTNEED` 释放物理页。
  * KV Cache 历史冷块打包为 CASA 256KB Super-Tile 驻留 NVMe，Pinned 内存中仅保留 IFR 双子质心索引与活跃热块。

* **常驻物理内存占用方程**：
  $$\text{RAM}_{\text{Streaming}} = \text{RAM}_{\text{OS}} + 2 \times \text{Size}(\text{Layer}_i) + \text{Size}(\text{Centroid\_Index}) + \text{RAM}_{\text{Act\_Buffer}}$$
  * 70B 模型单层 4-bit 权重：$\approx 525\text{ MB}$ $\implies$ 双缓冲常驻仅需 $\approx 1.05\text{ GB}$。
  * 405B 模型单层 4-bit 权重：$\approx 1.6\text{ GB}$（405B×0.5B ÷ **126 层**） $\implies$ 双缓冲常驻约 $\approx 3.2\text{ GB}$。
    * 注：Llama-3.1-405B 为 126 层；若按 80 层估算会得 2.5 GB/层（偏高 55%）。
  * 32K 上下文 IFR 质心索引：$1024 \text{ 块} \times 2 \times 64 \times 4\text{ Bytes} \times 80\text{ 层} \times 8\text{ KV head} \approx \mathbf{320\text{ MB}}$（head_dim=128 时 $\approx 640\text{ MB}$）。
    * ⚠️ **单层单 head 口径下的 512 KB 少算 640 倍**——该错误曾在本仓 Mac 仿真脚本中出现并被 OG 清单 T3.4 点名，勿在白皮书复现。

* **理论极限结论**：
  在层间流式换页模式下，**模型大小不再受 32GB 宿主内存容量截断，其容量上限完全转化为 NVMe SSD 的可用物理磁盘空间**。
  * 只要 SSD 具备 $\ge 250\text{ GB}$ 空闲空间，**理论物理极限甚至可加载 405B 模型**；
  * **工程推荐兼顾实用性上限**：**70B ~ 120B 参数量级**。

---

## 3. 自回归 Token 生成速度的理论推导 (Theoretical Token Generation Speed)

自回归解码（Autoregressive Decoding, Batch Size = 1）受冯·诺依曼访存带宽瓶颈（Memory Wall / IO Wall）支配。生成每一个新 Token 必须完整扫描一遍模型的活跃权重。

### 3.1 理论吞吐基础方程
$$T_{\text{step}} = \frac{W_{\text{active}}}{\text{BW}_{\text{effective}}} + T_{\text{compute\_overhead}} + T_{\text{KV\_IO}}$$
在超内存层流式架构下，访存 I/O 耗时远大于纯 Tensor 计算耗时（$T_{\text{compute}} \ll T_{\text{I/O}}$），因此生成速度理论上界为：
$$v_{\text{theoretical}} \le \frac{\text{BW}_{\text{effective}}}{W_{\text{active}}} \quad (\text{Tokens / Second})$$

---

### 3.2 典型硬件场景理论极限对照

#### 场景 1：32B 模型全常驻内存 (权重体积 $W \approx 20\text{ GB}$)
* **Mac 统一内存 (UMA，以 150 GB/s 计)**：
  $$v_{\text{mac}} = \frac{150\text{ GB/s}}{20\text{ GB}} = \mathbf{7.5\text{ tok/s}}$$
* **Windows 纯 CPU + DDR5-6000 内存 (有效带宽以 65 GB/s 计)**：
  $$v_{\text{win\_cpu}} = \frac{65\text{ GB/s}}{20\text{ GB}} = \mathbf{3.25\text{ tok/s}}$$
* **Windows RTX 4090 独显显存 (有效带宽以 1000 GB/s 计)**：
  $$v_{\text{win\_gpu}} = \frac{1000\text{ GB/s}}{20\text{ GB}} = \mathbf{50.0\text{ tok/s}}$$

---

#### 场景 2：70B Dense 模型由 NVMe SSD 层流式加载 (权重体积 $W \approx 40\text{ GB}$)
* **Mac 板载 NVMe SSD (连续读取带宽 $\approx 4.0\text{ GB/s}$)**：
  $$v_{\text{mac\_ssd}} = \frac{4.0\text{ GB/s}}{40\text{ GB}} = \mathbf{0.10\text{ tok/s}} \quad (\approx 10.0\text{ 秒 / token})$$
* **Windows PCIe 4.0 NVMe SSD (连续读取带宽 $\approx 7.0\text{ GB/s}$)**：
  $$v_{\text{win\_pcie4}} = \frac{7.0\text{ GB/s}}{40\text{ GB}} = \mathbf{0.175\text{ tok/s}} \quad (\approx 5.7\text{ 秒 / token})$$
* **Windows PCIe 5.0 NVMe SSD (连续读取带宽 $\approx 13.5\text{ GB/s}$)**：
  $$v_{\text{win\_pcie5}} = \frac{13.5\text{ GB/s}}{40\text{ GB}} = \mathbf{0.338\text{ tok/s}} \quad (\approx 2.95\text{ 秒 / token})$$

---

#### 场景 3：70B 级别 MoE 稀疏激活模型 (每 Token 仅激活 $W_{\text{active}} \approx 5.0\text{ GB}$)
* **Mac 板载 NVMe SSD (连续读取带宽 $\approx 4.0\text{ GB/s}$)**：
  $$v_{\text{mac\_moe}} = \frac{4.0\text{ GB/s}}{5.0\text{ GB}} = \mathbf{0.80\text{ tok/s}}$$
* **Windows PCIe 5.0 NVMe SSD (连续读取带宽 $\approx 13.5\text{ GB/s}$)**：
  $$v_{\text{win\_pcie5\_moe}} = \frac{13.5\text{ GB/s}}{5.0\text{ GB}} = \mathbf{2.70\text{ tok/s}} \quad (\mathbf{达到日常流式阅读可用阈值})$$

---

> **补充约束（此前遗漏）**：解码每 token 除读取活跃权重外，还必须取回被选中的 KV 块。
> 70B / 32K 上下文 / top-64：$64\text{ 块} \times 32\text{ token} \times 320\text{ KB/token} = 640\text{ MB/token}$，
> Mac NVMe 下另需 $\approx 160\text{ ms/token}$。
> 本仓 Mac 仿真中的 `Retrieval Time 2.76 ms` **仅扫描质心索引，并未真实读取那 64 个 KV 块**，
> 因此其 "Total Decode Step Overhead 280 ms" 同时漏掉了权重读取（$\approx 10\text{ s/token}$）与 KV 取回（$\approx 160\text{ ms/token}$）两项主导成本。

---

## 4. 本地原型验证与专家审计准则 (Audit Verification Protocol)

为杜绝"纸面空算"，本仓库提供可独立复现的测试 Harness 与微架构仿真脚本。**但必须严格区分：以下输出是"脚本打印值"，不是端到端实测。**

1. **层流式预取 Harness（代码路径可执行性验证）**：
   ```bash
   ./.venv/bin/python benchmarks/run_mac_layer_streaming_sim.py
   ```
   *脚本打印值（非实测）*：80 层 `madvise` 换页调用耗时约 **228 ~ 280 ms**（本机复现 277.997 ms，随运行波动）；
   `Peak Resident RAM < 4.5 GB` 为脚本内**写死的字符串，无 RSS 插桩**。
   ⚠️ 该 Harness 的权重是 1 GB 稀疏文件、KV 是哑元字节、质心是随机 `randn`，**全程无 matmul / attention**，
   既未读取 40 GB 权重（§3.2 判 ≈10 s/token），也未读取任何 KV 块（≈160 ms/token）。
   **不可据此宣称"可行性"**——吞吐可行性只能由 §3.2 的理论边界判定。
2. **C++20 控制面微基准**：
   ```bash
   ./build/main_bench
   ```
   *脚本打印值*：10K 块背包求解 **91.96 µs**，页表转换吞吐 **10.39 亿次/秒**（仅控制面，非端到端）。

### 外部专家交叉验证清单 (Reviewer Audit Checklist)
- [ ] **物理带宽自洽性**：检验公式 $v = \frac{\text{BW}}{W}$ 是否严格遵从底层总线带宽香农极限；
- [ ] **常驻内存预算**：核查 OS 4GB + 框架 2GB 扣除后，物理页是否发生系统级 Swap Thrashing；
- [ ] **CASA 256KB Super-Tile**：核对 NVMe I/O 是否对齐物理闪存 Block 粒度，消除小随机写惩罚；
- [ ] **KV 每 token 字节数核算**：是否按 层数 × KV head 数 × head_dim × 2 × dtype 计算，并据此反推可承载上下文（防 640× 少算）；
- [ ] **索引体积是否乘了 层数 × KV head 数**；
- [ ] **每 token 权重读取量 × (1/BW)** 是否与文中 tok/s 一致；
- [ ] **"实测"与"脚本打印值"是否分开标注**，是否有 RSS 插桩；
- [ ] **仿真是否包含真实 matmul 与真实 KV 块读取**（否则不能用于可行性判定）。
