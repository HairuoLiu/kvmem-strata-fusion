# KVMem-Strata-Fusion 软件使用指南 (User Manual & Quickstart)

欢迎使用 **KVMem-Strata-Fusion** 大模型超长上下文与超大模型推理加速系统。

本指南旨在帮助您快速在本地（macOS Apple Silicon / Linux x86_64）启动本系统，并通过**图形化 Web 看板（Web Dashboard）**直观横向对比官方原生 Ollama、原版开源 Strata 与本系统的实时性能。

---

## 快速导航 (Table of Contents)

1. [环境要求与一键安装](#1-环境要求与一键安装)
2. [一键启动图形化对比面板 (Web GUI)](#2-一键启动图形化对比面板-web-gui)
3. [在图形界面中进行三方横向对比](#3-在图形界面中进行三方横向对比)
4. [命令行核心能力评测指南](#4-命令行核心能力评测指南)
5. [常见问题与使用排查 (FAQ)](#5-常见问题与使用排查-faq)

---

## 1. 环境要求与一键安装

### 硬件支持与推荐配置
* **系统环境**：
  * macOS (Apple Silicon M1/M2/M3/M4/M5/M6 系列，已在 Apple M6 32GB 验证)
  * Linux x86_64 (支持 AVX2 / AVX-512)
* **内存建议**：16GB 或以上（推荐 32GB+）
* **Python 版本**：Python 3.9+

### 一步安装脚本
```bash
# 1. 克隆代码仓库
git clone https://github.com/HairuoLiu/kvmem-strata-fusion.git
cd kvmem-strata-fusion

# 2. 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖包
pip install torch transformers numpy scipy pytest xxhash
```

---

## 2. 一键启动图形化对比面板 (Web GUI)

为了让用户不再面对枯燥的黑框控制台，我们内置了极简的轻量级 Web 图形交互看板：

```bash
# 启动 Web 看板服务
.venv/bin/python web/server.py
```

* 终端将显示：  
  `🚀 KVMem-Strata-Fusion Web Dashboard is live at: http://127.0.0.1:8080`
* 打开浏览器访问：**`http://127.0.0.1:8080`**

---

## 3. 在图形界面中进行三方横向对比

在浏览器打开看板后，您将看到以下功能区域：

### (A) 实时物理指标仪表盘 (Gauges)
* **检索保真度 (Recall@64)**：实时显示当前双子质心与四子质心相较朴素均值的召回精度提升幅度（+43.0% ~ +45.5%）。
* **KV 物理压缩比**：显示基于 CASA 规范化原子与 LADDER 阶梯量化的内存压缩收益（7.85× ~ 8.0×）。
* **物理常驻显存/内存 (RSS)**：显示在 32GB 机器上运行 70B/125B 级超大模型时的真实物理内存开销（稳定在 4.0GB 以下）。

### (B) 三系统全维度对比矩阵 (Tripartite Comparison)
页面正中央展示了对齐科研规范的横向对比大表：
* **[1] 官方原生 Ollama (llama.cpp)**：支持一键点击 `▶ 重新实测对比 (n=5)`，自动调用后台运行 5 次独立热缓存采样，实时刷新 Prompt 吞吐与生成速度。
* **[2] 原版开源 Strata (Niko1221/Strata)**：展示 23.6k⭐ 官方标注规格参数与 Mac 端的专家流式标称速度。
* **[3] KVMem-Strata-Fusion**：展示微架构内核实测延迟（68µs）与层流式模拟物理数据。

### (C) 在线模型交互对话 (Live Chat)
* 支持直接在网页内向已运行的本地 Ollama 模型（例如 `qwen2.5:7b`、`qwen2.5:0.5b`）发送长文本对话。
* 实时返回：模型生成文本、准确的 Token/s 生成速度、Prompt 解析吞吐与耗时。

### (D) C++20 高性能调度器一键压测
* 点击 `▶ 执行 main_bench` 按钮，后台将自动调用 Clang 编译的高性能二进制。
* 实时在页面代码框内输出：
  * UBBA 背包求解延迟（10K 块耗时 ~0.07ms）；
  * ARM Neon SIMD 向量化打分吞吐（>340 M dot-products/s）；
  * CASA 零拷贝页表地址转换吞吐（>10 亿次/秒）。

---

## 4. 命令行核心能力评测指南

如果您需要将结果集成进自动化流水线或生成原始评测报告，可直接运行以下命令行工具：

### 1. 运行自动化单元测试 (PyTest)
```bash
.venv/bin/pytest tests/
```
*预期输出*：`83 passed in ~3s`，验证所有代数旋转不变性、Softmax 偏置与 LSE 缓存契约。

### 2. 运行长文本解耦子质心检索实验
```bash
.venv/bin/python benchmarks/benchmark_decoupled_retrieval.py
```
*评估输出*：在真实的 Wikipedia 长文章上评测 4096 tokens 的内容空间 Top-64 检索保真度，自动更新 `docs/decoupled_retrieval_benchmark.json`。

### 3. 运行 70B 模型超低内存层流式调度模拟
```bash
.venv/bin/python benchmarks/run_mac_layer_streaming_sim.py
```
*评估输出*：验证 44.8GB 虚拟映射与 80 层双缓冲流式调度下，OS 内核实测物理常驻内存（ru_maxrss）稳定在 3.95GB。

### 4. 编译与运行 C++20 原生高性能微基准
```bash
mkdir -p build
clang++ -std=c++20 -O3 -march=native -I./csrc \
  csrc/ubba_solver.cpp csrc/ifr_lse.cpp csrc/main_bench.cpp \
  -o build/main_bench

./build/main_bench
```

---

## 5. 常见问题与使用排查 (FAQ)

**Q1：打开网页后提示 "Ollama 未启动"，如何处理？**  
*答*：本项目推荐本地搭配官方 Ollama 守护进程以获得真实的横向基准对比。  
请在另一终端窗口运行：`./bin/ollama serve`（或系统已安装的 `ollama serve`），确保访问 `http://127.0.0.1:11434` 正常后刷新看板即可。

**Q2：为什么 32GB 机器运行 70B 模型速度约为 27 秒一个 token？**  
*答*：这是物理规律决定的。70B 权重体积约为 40GB，无法塞进 32GB 物理内存中，必须依靠 NVMe 磁盘按层换入换出。27s/token 属于离线深度思考或批处理场景，其最大的突破在于**“让原本直接崩溃退出、完全跑不了的模型能够在轻薄本上稳定跑出结果”**。日常聊天交互推荐使用 7B 稠密模型（可达 ~30 tok/s 流畅交互）。

**Q3：技术白皮书和底层推导在什么地方查看？**  
*答*：本系统全部底层推导和科学规范均汇总整理在项目的 [README.md](README.md) 以及 [docs/](docs/) 目录中。
