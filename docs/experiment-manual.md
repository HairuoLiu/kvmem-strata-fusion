# KVMem × Strata 实验执行手册：设备清单与详细步骤

> 生成日期：2026-10-05
> 配套文档：`kvmem-strata-融合研究方案.md`（四份方案与判据）
> 本手册回答三件事：**需要什么设备 → 怎么搭环境 → 每一步怎么跑、怎么判**

---

## 0. 先说三件你必须知道的事（读完再决定要不要投钱）

### 0.1 你手上的设备跑不了主线实验

你的机器是 **Mac mini M6（32GB 统一内存）+ Samsung T7 2TB**。判定：

| 关键依赖 | 你的 Mac | 结论 |
|---|---|---|
| KVMem-llama.cpp 是否支持 Metal | README 明确写 **"Metal integration is not included"** | ❌ 跑不了 |
| 有没有 NVIDIA GPU | 无（且 macOS 全系无 NVIDIA 驱动，eGPU 也不可能） | ❌ |
| 32GB 统一内存够不够 | 256K workspace 实测 host RAM 峰值 ~13.5 GiB ⇒ 内存其实够 | ✅ 但没用 |

**结论：主线实验必须租 GPU 云。** 好消息是门槛比报告里写的低得多——见 0.2。

### 0.2 报告里最大的风险（QW3 闭源）已被部分解除

第一轮报告把"QW3 不可得"列为 G0 杀点（7/10 份报告的 Top-3 风险）。实际情况：

| 仓库 | 状态 | 硬件 |
|---|---|---|
| `kvmem/kvmem-qw3` | CUDA 原生，Q8 为主，RTX PRO 6000 测试 | 论文口径，闭源性高 |
| **`kvmem/kvmem-llama.cpp`** | **开源 Apache-2.0，源码 v0.17.0，预编译 v0.16.0-rc3** | **RTX 5060 Ti 16GB 实测** |

llama.cpp 版的 README 给出实测数据：RTX 5060 Ti 16GB + 32GB RAM，跑 Qwen3.8-27B GGUF IQ3，256K workspace，Task 2（32 轮工具调用）达到 262,058 / 262,144 tokens，host RAM 峰值 13,483 MiB，VRAM 峰值 15,617 MiB，decode 31.74 tok/s，MTP 接受率 64.70%。

**但它缺三样关键东西**（这决定了你能做哪些实验）：

| 缺失项 | README 原文口径 | 影响 |
|---|---|---|
| **re-RoPE** | 未实现；"Attention kernels and original positions stay unchanged" | 见 0.3——这可能是好事 |
| **Mean-K 检索** | 未实现；默认是 `query replay auto` + `query policy user` + **128-token blocks** | IFR/CASA 的索引实验必须自己实现 |
| **raw KV block → NVMe offload** | 只有 opt-in 的**非活跃 session 快照**；native Windows CUDA build 里 legacy raw-block NVMe tier 被禁用 | **10M workspace / 324 GiB 那组实验现在做不了** |

### 0.3 一个能帮你砍掉最贵那一步的洞察

报告里"融合的硬摩擦"是：KVMem 必须把块搬到新的 compact position ⇒ 必须 re-RoPE。而 llama.cpp 版**根本不重映射**，它保留原始绝对位置。为什么这样可以工作？

> **只要 workspace 长度仍在模型训练/外推的位置范围内，就不需要压缩位置，于是 re-RoPE 完全不必要。**

证据链自洽：llama.cpp 版的实测上限正好是 **256K**（`-c 262144`），且 README 明确写"超过 256K 质量仍属实验性"——256K 正是 YaRN 外推的合理边界。超过之后位置跑出训练分布，**才**必须压缩回 bounded view，那时 re-RoPE / Δ-QRoPE 才成为必需。

**这条给你一个便宜得多的入口：**

| workspace 规模 | 是否需要 re-RoPE | 是否能用现有开源代码 | 方案四的 Δ-QRoPE kernel 要不要做 |
|---|---|---|---|
| **≤ 256K** | **不需要** | ✅ llama.cpp 版直接可跑 | **可以不做**（省掉最难的一步） |
| 1M–4M | 需要 | ❌ 要自己加 | 必需 |
| 10M | 需要 + NVMe tier | ❌ 要自己加两层 | 必需 |

**建议：把第一年全部实验钉在 ≤256K 这个已验证区间。** 这样你跳过了 CUDA kernel 这一最贵、最容易翻车的环节，而论文的两个核心 benchmark（LongMemEval-S 85.6% vs Full 86.6%、AgentLongBench 60.9% vs 59.5%）**恰好就在这个区间测的**——它们可直接作为 fidelity 真值。

---

## 1. 设备需求总表

### 1.1 按实验分档

| 实验 | 能否本地(Mac) | 最低 GPU | 最低 host RAM | 磁盘 | GPU-h 估 |
|---|---|---|---|---|---|
| **E0** G0 复核 + 代码可得性 | ✅ 纯网络 | 无 | 无 | 无 | **0** |
| **E1** LADDER P0 证伪（de-RoPE 平均） | ✅ **MLX/PT-MPS 可跑** | 无 | 16 GB | 20 GB | **0** |
| **E1b** R10 统计复现 + stopping rule | ✅ 纯代码 | 无 | 无 | 无 | **0** |
| **E2** G-CAS-1 检索瓶颈分解 | ❌ | 16 GB VRAM | 32 GB | 60 GB | ~80 |
| **E3** G-CAS-2 Δ-QRoPE 回退 | ❌ | 16 GB VRAM | 32 GB | 60 GB | ~120 |
| **E4** LADDER P1–P2 保真阶梯 | ❌ | 16 GB VRAM | 64 GB | 100 GB | ~250 |
| **E5** 索引裁决（MVR / LPI / Mean-K） | ❌ | 24 GB VRAM | 64 GB | 150 GB | ~300 |
| **E6** IFR fidelity + eval backbone | ❌ | 16 GB VRAM | 32 GB | 100 GB | ~300 |
| **E7** CASA canonical 字节层 | ❌ | 24 GB VRAM | 64 GB | 200 GB | ~150 |
| **E8** 525 配对 utility rollout | ❌ | 多卡 | 64 GB×N | 300 GB | ~1,500–3,600 |
| **E9** 10M / NVMe tier | ❌ | 96 GB VRAM | 128 GB+ | **4 TB NVMe** | ~400 |

### 1.2 内存/容量换算（用实测数，别用论文的）

llama.cpp 版 Task 2 实测：262,058 tokens → host RAM 峰值 13,483 MiB

```
每 token host 开销 ≈ 13,483 MiB × 1024² / 262,058 ≈ 53,944 B ≈ 52.7 KiB/token（含运行时开销）
论文 qw3 口径（纯 KV, FP8）≈ 324.2 GiB / 10M = 34.8 KB/token
```

用 **52.7 KiB/token** 做容量规划（保守）：

| workspace | host RAM 需求 |
|---|---|
| 256 K | ≈ 13.5 GiB（实测值） |
| 1 M | ≈ 50 GiB |
| 4 M | ≈ 201 GiB |
| 10 M | ≈ 503 GiB ⇒ **必须 NVMe tier，而现在没实现** |

> ⚠️ 论文口径 34.8 KB/token 与实测 52.7 KiB/token 差 1.5×。差异来源可能是 q8_0 的 per-block fp16 scale、host 侧副本、以及运行时其他开销。**做容量预算时一律用实测的 52.7**，别用论文数。

### 1.3 云 GPU 真实价格（2026 年 9–10 月观测）

来源：Vast.ai 官方定价页（每小时更新）、IntuitionLabs 2026-09 provider 汇总表、VPSRated 2026 对比。

| GPU | 显存 | Vast.ai（最低可租） | RunPod（on-demand） | Lambda | 备注 |
|---|---|---|---|---|---|
| **RTX PRO 6000** | **96 GB** | **$0.73–1.00 /GPU-h** | — | — | **论文同款 GPU，性价比最优** |
| RTX 5090 | 32 GB | $0.39 | — | — | 便宜但只有 32GB |
| H100 SXM | 80 GB | $1.73 | $2.99 | $3.99 | 生态最成熟 |
| H200 SXM | 141 GB | $2.63 | $4.39 | — | 大显存 |
| A100 SXM4 | 80 GB | $0.31 | $1.39 | — | 最便宜的 80GB |

> ⚠️ Vast.ai 的"from"价是 **spot/可中断** 市场最低报价，无可用性保证；RunPod/Lambda 是 on-demand。同一张 H100 在 Vast.ai 上 on-demand 约 $1.87、spot 可低至 $0.34，**5× 价差**。价格每小时变动，**下单前自己去核实**。
>
> 另需计入：实例磁盘费（Vast.ai 按 GB/小时另收）、出网流量费、以及空闲未关机的时间。

### 1.4 三档预算方案

| 档位 | 做什么 | GPU-h | 费用估算（RTX PRO 6000 @$1） | 说明 |
|---|---|---|---|---|
| **A 零预算** | E0 + E1 + E1b | 0 | **$0** | 全部在你的 Mac 上跑，验证方向对不对 |
| **B 试错档** | + E2 + E3 + E6 | ~500 | **~$500**（+磁盘约 $50） | 拿到瓶颈归属 + fidelity 真值，决定是否继续 |
| **C 全量档** | + E4 + E5 + E7 | ~1,200 | **~$1,200**（+磁盘约 $150） | 走完 CASA Phase 0–3 与 IFR 主线 |
| **D 不建议自费** | E8（525 配对） | 1,500–3,600 | **$1.5k–3.6k+** | 且 R10 已判定：非劣 3pp 需 1308 对/臂 ⇒ 3.6×10⁴ GPU-h 才真能判质量。**这笔钱花下去大概率买不到结论** |

**建议路径：先做 A（$0），跑通再上 B（~$500），B 的结果决定要不要上 C。E8 不要自费，去申算力或找合作。**

---

## 2. 环境准备

### 2.1 本地（Mac mini M6）—— 只做 E0/E1/E1b

```bash
# 建隔离环境，别污染系统 Python
/Users/hai/.workbuddy/binaries/python/versions/3.13.12/bin/python3 -m venv ~/venv-kvmem
source ~/venv-kvmem/bin/activate
pip install torch numpy scipy statsmodels transformers safetensors

# E1 需要能跑一个小模型出 KV：Qwen3-1.7B/4B 在 32GB 统一内存上没问题
pip install mlx mlx-lm          # Apple 官方 MLX，M 系列最快
```

> 说明：MPS 后端拿 `past_key_values` 最省事；MLX 更快但要自己写前向钩子。**第一次做建议用 PyTorch + MPS**，代码路径最短。

### 2.2 云端（推荐：Vast.ai 租 RTX PRO 6000 96GB）

选实例时**必须同时满足三项**，缺一项就跑不动：

1. **GPU**: RTX PRO 6000（96GB）或 H100（80GB）或 A100（80GB）
2. **host RAM**: 256K workspace 需 ≥32GB；1M 需 ≥64GB；做 E5/E7 建议 ≥64GB
3. **磁盘**: ≥150GB（Qwen3.8-27B GGUF IQ3 约 11–13GB，加上数据集、dump、日志）

```bash
# 连上实例后（Ubuntu 22.04 基线）
sudo apt update && sudo apt install -y build-essential cmake git python3-pip nvtop htop iotop sysstat
nvidia-smi                      # 确认驱动与 GPU
free -h                         # 确认 host RAM
df -h                           # 确认磁盘
nvidia-smi --query-gpu=name,memory.total --format=csv

# CUDA Toolkit：README 明确要求 CUDA 13.2 Update 2 (nvcc 13.2.86) 或更新
# 且明确警告：不要用 nvcc 13.2.51（会产生 garbage output）
nvcc --version
```

### 2.3 拉代码与构建（KVMem-llama.cpp）

```bash
git clone --recurse-submodules https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp
git checkout master             # 注意：master pin llama.cpp v0.5.0 (7fe450e19)
git submodule update --init

# 应用补丁（重复运行安全；不要混用 0001-0004 与 cumulative patch）
scripts/apply-patches.sh

# 构建。默认 CMAKE_CUDA_ARCHITECTURES=120a-real（对应测试过的 RTX 5060 Ti）
# 用别的 GPU 必须显式指定架构，否则跑不起来
export CMAKE_CUDA_ARCHITECTURES="90-real"   # H100；A100 用 80-real；RTX PRO 6000(Blackwell) 用 120-real
scripts/build-cuda.sh                        # 产出 build/bin/llama-kvmem-server
```

```bash
# 冒烟测试（用小模型，别一上来跑 27B）
python scripts/test_server_compat.py --server ./build/bin/llama-kvmem-server
python scripts/test_server_environment.py --server ./build/bin/llama-kvmem-server --output /tmp/env

# 健康检查 + 列设备
./build/bin/llama-kvmem-server --list-devices
```

> ⚠️ README 原话：**"A successful build, health check or small Q8 model test does not validate IQ3 inference."** 小模型跑通不等于 IQ3 能跑，必须真模型验。

### 2.4 拉模型与起服务

```bash
# Qwen3.8-27B GGUF IQ3（推荐主模型）+ 视觉 projector
# 从 HuggingFace 拉（约 11–13GB），放到有足够空间的盘
```

README 给的 IQ3 recipe（Linux launcher）：

```bash
MMPROJ=/path/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf scripts/start-iq3.sh
# 先看它准备用什么参数（不实际启动）：
MODEL=/path/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  MMPROJ=/path/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf scripts/start-iq3.sh --dry-run
```

等价的原始 server flags：

```bash
./build/bin/llama-kvmem-server \
  -m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --mmproj mmproj-Qwen3.8-27B-Q5_K-MIX.gguf --no-mmproj-offload --image-max-tokens 512 \
  -c 262144 -n 16384 \
  --kvmem-budget 36864 --kvmem-gen-reserve 16384 \
  --kv-dtype q8_0 \
  --spec-type draft-mtp \
  --enable-thinking --reasoning-budget 4096 \
  --host 127.0.0.1 --port 18200 \
  --threads 8 --threads-batch 8 --batch-size 512 --ubatch-size 128 \
  --flash-attn on --gpu-layers all --parallel 1
```

**两个必须记住的硬限制**：

- 单次生成不能超过 `--kvmem-gen-reserve`（IQ3 是 16384、IQ4 是 12288，含 thinking）。Agent 场景要在 system prompt 里显式限制输出长度。
- `-ngl auto` / `-1` **未实现会报错**，只能写非负层数或 `all`（-2）。

### 2.5 观测开关（所有实验都要开）

```bash
KVMEM_PERF=1          # 性能计数器（独立于 full tracing）
--kvmem-trace         # 或 KVMEM_TRACE=1：stderr 输出原始 KVMEM_* 诊断记录
-lv 4 --kvmem-trace   # bug report 级别，抓 stdout+stderr
```

> ⚠️ tracing 有开销。**对比实验必须保证两臂同设置**，否则测出来的是 tracing 成本不是系统成本。

---

## 3. 实验逐个：步骤与判据

### E0 · G0 复核：代码可得性与接口盘点（0 GPU-h，1–2 人日）

**这是所有事的第一步，且不要钱。**

```bash
# 1) 确认 llama.cpp 版能构建、能起服务、能跑通 256K
#    见 §2.3 / §2.4，记录每一步的实际输出

# 2) 盘点可用的观测面（决定后续实验能不能埋点）
grep -rn "KVMEM_" --include=*.hpp --include=*.cpp kvmem/include kvmem/src | head -50
grep -rn "kvmem" scripts/test_*.py | head -30

# 3) 确认 qw3 是否真的拿不到（决定要不要走开源重实现路线）
git clone --depth 1 https://github.com/kvmem/kvmem-qw3.git && ls kvmem-qw3
# 看 include/qw3/kvmem_store.hpp、nvme_kv_tier.hpp、rope_block_remap_* 是否完整

# 4) 看 modification-plan.md，确认从 qw3 迁什么（P0 可原样迁，P1-P3 需重写）
cat docs/modification-plan.md
```

**判据（G0）**：

- ✅ 拿到 `block-score` / 检索决策的观测接口，且能 dump 每块的 `block_id / tier / bytes / load_latency / hit-miss` ⇒ 继续
- ⚠️ 只有端到端延迟，没有块级观测 ⇒ 必须自己加埋点（约 3–5 人日），此时先做 E1/E1b（本地）再决定
- ❌ 连 256K 都跑不起来 ⇒ 全部暂停

### E1 · LADDER P0 证伪：**本地 Mac 就能做**（0 GPU-h，2 人日）

**目的**：证伪或证实 R6 最关键的赌注——"直接平均已 RoPE 的 K 会损失 ~82% 高频维振幅，必须先 de-RoPE 再合并"。这是方案二/方案四 L2′ 层的地基。

**为什么本地能做**：不需要 KVMem 系统，只需要一个能吐 `past_key_values` 的模型。

```python
# e1_derope_merge.py —— MPS 后端，Qwen3-1.7B 或 4B
import torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "Qwen/Qwen3-1.7B"          # 先小模型；结论稳健后再上 4B/8B
dev = "mps"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float16,
                                             attn_implementation="eager").to(dev).eval()

# 1) 造一段足够长的上下文，拿 KV
text = " ".join([f"sentence {i} about topic {i%37}." for i in range(400)])
ids = tok(text, return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    out = model(ids, use_cache=True)
pkv = out.past_key_values            # 每层 (K, V)，K 已施加 RoPE
K = pkv[0][0].squeeze(0)             # [n_kv_heads, seq, head_dim]
n_heads, seq, d = K.shape
print("K shape", K.shape, "norm", K.norm(dim=-1).mean().item())

# 2) 构造逆 RoPE：R(-pos)。RoPE 是成对旋转 (2i, 2i+1)，频率 theta_i = 1/10000^(2i/d)
def rope_freqs(d, base=10000.0):
    i = torch.arange(0, d, 2).float()
    return 1.0 / (base ** (i / d))                     # [d/2]
def derope(K, pos0=0):
    # K: [H, S, d] -> 剥掉位置 pos0..pos0+S-1 的旋转
    th = rope_freqs(d).to(K.device)                    # [d/2]
    p  = torch.arange(pos0, pos0+K.shape[1]).float().to(K.device)
    ang = torch.outer(p, th)                           # [S, d/2]
    cos, sin = torch.cos(ang), torch.sin(ang)          # [S, d/2]
    x  = K.float().view(*K.shape[:-1], d//2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    # 逆旋转
    y0 =  x0 * cos + x1 * sin
    y1 = -x0 * sin + x1 * cos
    return torch.stack([y0, y1], dim=-1).reshape(K.shape)

# 3) 三臂对比：块内 32 token 平均
B = 32
nb = seq // B
def avg_blocks(K_):
    return K_[:, :nb*B].view(n_heads, nb, B, d).mean(dim=2)   # [H, nb, d]

K_raw   = K                          # 臂 A：直接用已 RoPE 的 K
K_de    = derope(K)                  # 臂 B：先 de-RoPE
avg_raw = avg_blocks(K_raw)
avg_de  = avg_blocks(K_de)

# 4) 判据 1：‖K‖ 衰减
print("臂A ‖avg‖/‖K‖ =", (avg_raw.norm(dim=-1).mean()/K.norm(dim=-1).mean()).item())
print("臂B ‖avg‖/‖K‖ =", (avg_de.norm(dim=-1).mean()/K_de.norm(dim=-1).mean()).item())
# 期望（R6 预测）：臂A ≈ 0.18（损失 82%），臂B 显著更高

# 5) 判据 2：高频维 vs 低频维分开看（这才是关键证据）
def band_norms(x):                   # x: [H, nb, d]，按 RoPE 频率分带
    f = rope_freqs(d)
    lo = slice(0, d//4); hi = slice(3*d//4, d)
    return x[..., lo].norm(dim=-1).mean().item(), x[..., hi].norm(dim=-1).mean().item()
print("臂A lo/hi =", band_norms(avg_raw))
print("臂B lo/hi =", band_norms(avg_de))
# R6 预测：臂A 的高频带振幅崩塌最严重（跨 32 token ≈ 5 个整圈）
```

**再补一个更硬的判据 —— recall**：

```python
# 6) 判据 3：用真实的 q·k 打分看召回是否被翻坏
#    取某个真实 query 的最后一位 q，对两种摘要做 top-k，比对"对真实 K 的穷举 top-k"的重合率
#    这里 q 也要用同样的 de-RoPE 坐标系，否则不可比
```

**判据（M0-LADDER）**：

- ✅ 臂 B（de-RoPE 后平均）的 ‖K‖ 保留率显著高于臂 A，且高频带差异最大 ⇒ R6 的位置相位论证成立，L2′ 可继续
- ❌ 两臂无显著差异 ⇒ **方案二整体终止，方案四的 L2′ 层也要重写**（2 人日止损，这是全场性价比最高的一枪）

> ⚠️ 注意口径：论文用 32-token block，llama.cpp 版用 **128-token block**。128 token 跨的圈数更多，臂 A 崩塌应该**更严重**。建议两个 block size 都跑，这本身就是个漂亮的 ablation。

### E1b · R10 统计复现与 stopping rule 实现（0 GPU-h，1–2 人日）

**目的**：把 R10 的所有统计结论用代码重算一遍，并把序贯 stopping rule 写成可复用工具。这是"判定层"，后面每个实验都要用。

```python
# e1b_stats.py
import numpy as np
from scipy import stats

# 1) 复核 DeepSWE：28/64 vs 31/64
n = 64; a, b = 28, 31
p1, p2 = a/n, b/n
pool = (a+b)/(2*n)
se = np.sqrt(pool*(1-pool)*2/n)
z = (p2-p1)/se
print(f"pooled p={pool:.4f} SE={se:.4f} z={z:.3f} p_two_sided={2*(1-stats.norm.cdf(abs(z))):.3f}")
# 期望：z≈0.53, p≈0.59

# 2) Wald 95% CI
se_un = np.sqrt(p1*(1-p1)/n + p2*(1-p2)/n)
d = p2-p1
print(f"diff={d:.4f} CI=[{d-1.96*se_un:.4f}, {d+1.96*se_un:.4f}]")   # 期望 ±17.3pp

# 3) 配对 McNemar 所需样本量（不一致对率 pi_d=0.15）
pi_d = 0.15; delta = 0.05; za, zb = 1.96, 0.84
N = (za+zb)**2 * pi_d / delta**2
print("配对所需 n =", np.ceil(N))       # 期望 ~471（R10 用的口径）

# 4) 序贯 Lan-DeMets OBF 边界
def obf_alpha(t): return 2*(1-stats.norm.cdf(1.96/np.sqrt(t)))
for t in [0.25,0.50,0.75,1.00]:
    a_t = obf_alpha(t)
    print(f"t={t}: cum_alpha={a_t:.5f} z_boundary={stats.norm.ppf(1-a_t/2):.2f}")
# 期望：α = 0.00009/0.0056/0.0239/0.05，z = 3.92/2.77/2.26/1.96

# 5) 补 R10 自己承认漏掉的那条：task 内聚类校正
for icc in [0.0, 0.1, 0.3, 0.5]:
    deff = 1 + 3*icc                    # 4 seeds
    se_adj = se*np.sqrt(deff)
    print(f"ICC={icc}: design_effect={deff:.2f} SE_adj={se_adj:.4f} z={d/se_adj:.3f}")
```

**产出物**：一个 `stopping.py`，后面 E4/E5/E6 的所有对比实验都调它，禁止临时改 α。

### E2 · G-CAS-1 检索瓶颈分解（~80 GPU-h，8 人日）

**目的**：解决报告里那个没人能解的矛盾——`10.24 GB ÷ 1.311 s = 7.8 GB/s`，这个数只被"冷 NVMe 顺序读"解释，而论文说 index 常驻 host（DRAM 理论 0.05 s、PCIe5 有效 0.21 s）。

**设备**：16GB VRAM + 32GB RAM 起（256K 档）；要扫到 1M 需 64GB RAM

```bash
# 1) 五档 workspace 阶梯，每档 30 次重复
for C in 32768 65536 262144 524288 1048576; do
  for R in $(seq 1 30); do
    KVMEM_PERF=1 ./build/bin/llama-kvmem-server \
      -m model.gguf -c $C --kvmem-budget 36864 --kvmem-gen-reserve 16384 \
      --kv-dtype q8_0 --flash-attn on --gpu-layers all \
      --verbosity 3 >> /logs/gcas1_c${C}_r${R}.log 2>&1
  done
done

# 2) Nsight 分解：gather / H2D / scatter 各自占比
nsys profile -o /logs/nsys_c262144 --stats=true \
  ./build/bin/llama-kvmem-server -m model.gguf -c 262144 ...
ncu --set full -o /logs/ncu_c262144 ./build/bin/llama-kvmem-server ...   # 更细，慢

# 3) 纯带宽基线（关键对照，很多人漏掉）
#    测这台机器的：DRAM 带宽、PCIe H2D 有效带宽、NVMe 顺序读
sudo apt install -y fio mbw
mbw -n 10 2048                                     # DRAM
fio --name=seqread --rw=read --bs=1m --size=8g --numjobs=1 --direct=1    # NVMe
# PCIe H2D：写一个 cudaMemcpy 微基准，或 bandwidthTest
```

**判据**：

- 搬运时间占比 >70% 且实测纯带宽 <12 GB/s ⇒ 支持"带宽受限"（R8 口径）
- H2D 有效 >30 GB/s 且 cache-loading stall ≥70% ⇒ 支持"I/O 未打满、是调度/碎片问题"（R1 口径）
- **两组占比差的 95% CI 须 ≥15pp**，否则判"未分辨"，触发 G-CAS-1 Gate

> 这个结果**无论落在哪一边都可发表**，且是全流程最早的独立产出（Phase 0，8 人日）。

### E3 · G-CAS-2 Δ-QRoPE 回退实测（~120 GPU-h，8 人日）

**目的**：R4 自曝 flops 从"<3%"上修到 4.7%，且 Hopper WGMMA 流水未算、可能回退 >15%。这个 Gate 决定方案四的 K-Freeze 路线生死。

**⚠️ 先读 §0.3**：如果你的实验窗口钉在 ≤256K，**这一步可以直接跳过**——llama.cpp 版已经用"保留原始位置"绕开了 re-RoPE，且这是被实测验证的。只有在你要做 >256K 时才必须测。

```bash
# 三臂 × 3 档 × 20 重复
# 臂 1 (Full)      : -c 262144 --kvmem-budget 262144  （不压缩，全在 GPU）
# 臂 2 (现状)      : -c 262144 --kvmem-budget 36864   （KVMem 检索，保留原位置，无 re-RoPE）
# 臂 3 (fused)     : 你自己加的实现（Δ-QRoPE / K-tile prologue 旋转）
#
# 注意：臂 3 需要改代码。最小改法参考 docs/modification-plan.md 的 P2：
#   "P2 用 ggml_rope in-place block_kmean_* / softmax-over-pages"
#   R2 的改法：放在 FlashAttention 的 K-tile prologue 内一次性寄存器旋转，
#   并复用缓存的 FP8 scale（逐对旋转保范 ⇒ ‖R(Δ)k‖ = ‖k‖，scale 不变、无需重定标）

# 终点：TTFT。用 E1b 的 stopping rule 判
python stopping.py --arm-a arm2.json --arm-b arm3.json --delta 0.15 --metric ttft
```

**判据（G-CAS-2）**：

- ✅ fused 相对现状加速 95% CI 下界 >1.15，且相对 Full 回退上界 <5% ⇒ 方案四继续
- ❌ 下界 <0.85 ⇒ R4 的"回退 >15%"成立 ⇒ **K-Freeze 路线终止，预算转投方案二 LADDER**

### E4 · LADDER P1–P2 保真阶梯（~250 GPU-h，8 人日）

**前置**：E1 必须通过（de-RoPE 合并成立）。

```bash
# P1: L1 (2-bit KIVI 式)。llama.cpp 已支持 -ctk/-ctv，先试现成的量化档做下界对照
./build/bin/llama-kvmem-server -m model.gguf -ctk q8_0 -ctv q4_0 ...   # mixed 已验证可用
# 注意：q8_0/f16 这类 float+quant 组合会在加载前被拒；mixed 对只做了参数解析检查，
#       五个组合未做完整 inference/quality 验证 ⇒ 必须自己验

# P2: L2' (G=4 quad-merge + 质心 4-bit) —— 要自己写
#   铁律：永远 RoPE-then-quantize，禁止 quantize-then-rotate
#   加 tier-bias: b_t = -(rho_t * s)^2 / 2，s ≈ 1.85

# 评测：LongMemEval-S（~115K < 256K native window，Full Context 就是 fidelity 真值）
```

**判据**：

- **M1**：2-bit 下 LongMemEval-S 掉点 ≤0.5pt
- **M2**：ρ 实测 ≤0.32 且 @256K footprint 达标；**ρ >0.375 ⇒ 砍掉 L2′，接受 4.35×，并出"8× 不可达"的阴性结论**

### E5 · 索引裁决：MVR vs LPI vs Mean-K（~300 GPU-h，8 人日）

**⚠️ 前提**：llama.cpp 版**没有 Mean-K**，检索默认是 `query replay auto` + `query policy user` + 128-token blocks。所以三臂都得自己实现/或至少实现一个 Mean-K baseline。

```bash
# 1) 先 dump：把每个 block 的 K 落盘（这是三臂的共同输入）
#    用 qw3 的 block_kmean_* 语义在 llama.cpp 侧重写（modification-plan 的 P2 项）

# 2) 三臂离线对比（以穷举排序为 oracle）
#    - Mean-K  : 剥 RoPE → (layer, kv_head) 块内均值
#    - MVR     : k̄ + 2 条最大余弦残差行 + OPQ 压缩（R3）
#    - LPI     : p×r FP8 探针，p=8 r=16，由 d→64 小 MLP 蒸馏（R8，~150 GPU-h）

# 3) 关键 ablation：NIAH 插针集（治 R3 自曝的"秩反转"）
#    构造：块 A 含 1 个 cos≈1 的强相关 token；块 B 含 32 个 cos≈0.2 的弱相关 token
#    均值池化会判 B > A（0.2 > 1/32），但 max-like attention 下 A 该排前
```

**判据（M3）**：胜出方案在 ≤4.77 GiB（@10M 口径）下 recall@8 不低于 9.5 GiB Mean-K 减 1pp；且 top-8 mass 覆盖率掉 ≤1pp。

> 另需澄清 R8 挑出的口径问题：**R3 说 OPQ 把 9.5 → 1.19 GiB，这隐含 24× 压缩，不是 8×。** 实测时把字节比直接量出来，别用推导值。

### E6 · IFR fidelity + eval backbone（~300 GPU-h，8 人日）

```bash
# LongMemEval-S 上跑四臂：Full / Sliding / Compact+RAG / KVMem-retrieval
# fidelity 真值 = Full Context（因为 ~115K < 256K native window）
# 报：top-1 token 一致率 ≥97%、utility gap ≤1pp
python stopping.py --paired --alpha-spending obf --looks 4 --delta 0.01
```

**硬约束（R10 裁决）**：**不得把 Pass@1 设为主终点。** 主终点只能是 {字节, 延迟, ρ, 覆盖率}，质量报 interim 并标"未验证"。

### E7 · CASA canonical 字节层（~150 GPU-h，8 人日）

```bash
# 实现 Atom-IR：128B (token, head) FP8 原子
# ⚠️ R2 自曝：MXFP8 每 32 值 1B 块缩放 ⇒ 实为 132B，非 2 的幂，破坏 128B 扇区与 DMA 下限
#   修补：缩放旁路为 (block, head) 侧表（312,500×8×4B ≈ 10MB 常驻 host），原子回落严格 128B
# 原子 ID = FNV(model_ver, layer, head, token_hash, p_orig)
# 三种布局 = (layer, block, token, head, K/V) 五维仿射，编译进 scatter descriptor
```

**判据（M1）**：@1M workspace 检索延迟下降 ≥1.5× 且 LongMemEval-S 掉点 ≤0.5pt。

### E8 / E9 · 不建议自费的部分

- **E8（525 配对 rollout）**：R10 判定质量终点需 3.6×10⁴ GPU-h 才真能判。自费 $1.5k–3.6k 大概率只买到"CI 含 0"。**去申算力（学校/云厂商 credit）或找有集群的合作方。**
- **E9（10M / NVMe tier）**：raw KV block offload **当前开源实现里没有**。要自己写整层 + 需要 4TB NVMe 实例（云上很贵）。**列为第二年目标。**

---

## 4. 执行顺序与里程碑

```
第 0 周   E0（$0）              → G0：代码能跑吗？观测面够吗？
第 1–2 周 E1 + E1b（$0，本地）   → M0-LADDER：de-RoPE 合并成立吗？统计工具就绪
          ↑ 这一阶段零成本，跑不通就整个项目止损

第 3–4 周 E2（~$80）            → G-CAS-1：检索瓶颈到底在哪？【无论结果都可发表】
第 5–6 周 E6（~$300）           → fidelity 真值 + eval backbone 立住
          ↑ 到这里约 $380，已有一条独立可发表的产出

第 7–8 周 E3（~$120，若要做 >256K）
第 9–10 周 E4（~$250）
第 11–12 周 E5（~$300）+ E7（~$150）
          ↑ 全量约 $1,200
```

| 里程碑 | 时点 | 判据 | 不达标动作 |
|---|---|---|---|
| **G0** | 第 0 周 | 256K 跑通 + 块级观测可得 | 全部暂停 |
| **M0-LADDER** | 第 2 周 | 臂 B（de-RoPE）显著优于臂 A | 方案二终止；方案四 L2′ 重写 |
| **G-CAS-1** | 第 4 周 | 瓶颈归属 95% CI ≥15pp 可分辨 | 延迟目标作废，退化为纯存储/保真工作 |
| **G-CAS-2** | 第 8 周 | fused 回退 <5%、加速 CI 下界 >1.15 | K-Freeze 终止，转投 LADDER |
| **M1-IFR** | 第 6 周 | recall@8 ≥0.90 且 F gap ≤1pp | 保留 eval backbone，索引层停止 |
| **M2-CASA** | 第 10 周 | ρ ≤0.32 | 砍 L2′，接受 4.35× |

---

## 5. 一张图总结：该买什么

| 你的情况 | 建议 |
|---|---|
| 只想验证方向对不对 | **$0**：E0 + E1 + E1b 全在 Mac 上跑完 |
| 想拿到第一条可发表结果 | **~$80**：加 E2（瓶颈归属，无论结果都能写） |
| 想走完主线 | **~$1,200**：Vast.ai RTX PRO 6000（$0.73–1.00/GPU-h），约 1,200 GPU-h |
| 想判"任务成功率" | **别自费**：3.6×10⁴ GPU-h，去申算力 |
| 想做 10M / NVMe | **第二年**：开源实现里没有这层，要自己写 + 4TB NVMe 实例 |

**三个最容易翻车的点**：

1. **CUDA 版本**：README 明确要求 nvcc 13.2.86 或更新，**明确警告不要用 13.2.51**（产生 garbage output）。升级 Toolkit 后必须新建 build 目录重建。
2. **CUDA 架构**：默认 `120a-real`（RTX 5060 Ti）。换 GPU 必须显式设 `CMAKE_CUDA_ARCHITECTURES`，否则构建产物跑不起来。
3. **tracing 开销**：所有对比实验两臂必须同 `--verbosity` 同 `--kvmem-trace`，否则测到的是观测成本。

---

*本手册的价格为 2026 年 9–10 月观测值，云 GPU 价格每小时变动，下单前请自行核实。所有来自 README 的限制条款均引自 `github.com/kvmem/kvmem-llama.cpp`。*
