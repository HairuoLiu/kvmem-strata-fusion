# 勘误表：`docs/cross_platform_hardware_limits_and_throughput.md`（f79f2b7）

> 性质：**我方白皮书的自我勘误**。该文档已对对方团队与外部专家公开，以下问题必须在对方引用前修掉，否则会被反打。
> 方法：所有数字均由本机核算/实跑得出，附复现命令。

---

## 🔴 严重（会被外部专家直接驳回）

### E1. §2.2 质心索引少算 640 倍（我们批评过对方的同一个错，自己又犯了）

原文：

> 32K 上下文 IFR 质心索引：$1024 \text{ 块} \times 2 \times 64 \times 4\text{ Bytes} \approx 512\text{ KB}$

这 512 KB 只是**单层、单 KV head、dim=64、FP32** 的量。真实模型要乘 `层数 × KV head 数`：

```
one = 1024 × 2 × 64 × 4 B = 524,288 B = 512 KB     # 1 layer, 1 kv head
real (80 层 × 8 KV head)      = 320 MB             # factor 640×
real (80 层 × 8 KV head, d128) = 640 MB            # factor 1280×
```

**这正是我们在 `og_challenge_test_checklist.md` T3.4 里批评对方 Mac 模拟器"低估 640 倍"的同一处错误**——现在出现在我们自己的白皮书里。对方只要拿我们对他们的指控回指这份文档，我们就完全被动。

### E2. §4 把脚本的硬编码输出当成"实测输出"引用

原文称"为杜绝纸面空算…提供可独立复现的测试 Harness"，并给出：

> *实测输出*：80 层流式预取换页延迟为 **228.3 ms**，单步峰值常驻内存稳定在 **< 4.5 GB**，证实…可行性。

我实跑了该脚本（本机）：

```
$ python benchmarks/run_mac_layer_streaming_sim.py
    -> Total KV Cache on Disk: 16.00 MB
    -> Pinned DRAM Sub-Centroid Index Size: 512.00 KB (Tiny!)
    -> 80-Layer Pipelined Stream Latency:           277.997 ms
    -> Total Decode Step Overhead:                  280.759 ms
    -> Peak Resident RAM Footprint during step:     < 4.5 GB (Well within 32GB RAM!)
    [SUCCESS] Phase 2 & 3 Architecture Simulation successfully verified on local hardware.
```

问题：
1. **`< 4.5 GB` 是脚本里写死的 print 字符串**，没有任何 RSS 插桩——它不是测量值。
2. 我跑出来是 `277.997 ms`，白皮书写 `228.3 ms`：说明这是**某次运行的打印值被抄成了固定"实测输出"**，本身就不是稳定测量（只是 160 次 `madvise` 系统调用的开销）。
3. **"证实可行性"不成立**：脚本里权重是 1GB 稀疏文件、KV 是 `b"X"*16KB` 哑元、质心是 `rng.randn`，**全程没有任何一次 matmul / attention**。它没有读过那 40 GB 权重，也没有读过任何 KV 块。

### E3. §4 与 §3.2 自相矛盾（最致命的一条）

我们自己的 §3.2 已经算出：**70B Dense 在 Mac NVMe 上层流式 = 0.10 tok/s（≈10 秒/token）**。
但 §4 却说该仿真"证实 32GB 宿主机运行 70B 级流水线的可行性"，并给出 `Total Decode Step Overhead: 280.759 ms`。

$10\,\text{s} / 0.28\,\text{s} \approx 36\times$ ——**我们的仿真比我们自己的理论上界快了 36 倍**，因为它既没读 40 GB 权重，也没读 KV。

### E4. §2.1 常驻上下文人表全部偏乐观（1.16×–1.47×）

原文表格没有给出 KV 每 token 字节数这个关键假设。按 GQA-8 / head_dim=128 / FP16 的标准配置核算：

| 模型 | KV 字节/token | 预算余量 | **物理可承载（FP16 KV）** | 原文声称 | 倍数 |
|---|---|---|---|---|---|
| 8B | 128 KB | 21.2 GB | **173.7K tokens** | > 256K | **1.47× 夸大** |
| 14B | 160 KB | 16.8 GB | **110.1K tokens** | ≈ 128K | 1.16× 夸大 |
| 32B | 256 KB | 6.2 GB | **25.4K tokens** | ≈ 32K | 1.26× 夸大 |

若改用 int8 KV（需明确声明为压缩前提），则为 347K / 220K / 50.8K。**原文的 "> 256K" 只有在 KV 量化到 int8 以下才成立，但文中未声明任何压缩假设。**

---

## 🟡 需补充声明（不至于被驳，但会被追问）

### E5. §1.1 / §3.2 Mac 带宽基数"150 GB/s"缺机型依据

M1 基座实际是 **68.25 GB/s**（LPDDR4X），M1 Pro ≈ 200 GB/s，M1 Max ≈ 400 GB/s，M2 Ultra ≈ 800 GB/s。
"Base ~150 GB/s" 并不对应任何一款真实芯片；而 §3.2 场景 1 直接用它推出 7.5 tok/s。
**须声明具体机型**（本机为 Mac Mini）。若按 M 系列基座 68–100 GB/s 计，32B 常驻为 **3.4 ~ 5.0 tok/s**，而非 7.5。

### E6. §2.2 "405B 单层 ≈ 2.50 GB"层数假设错

405B × 0.5 B(4-bit) ≈ 202 GB；Llama-3.1-405B 为 **126 层** → 每层 ≈ **1.6 GB**（不是 2.5 GB，2.5 GB 对应 80 层）。双缓冲应为 3.2 GB 而非 5.0 GB。

### E7. §3.2 完全未计 KV 读取带宽

解码时除权重外还必须取回被选中的 KV 块。70B / 32K 上下文、top-64 块：

```
64 块 × 32 token × 320 KB/token = 640 MB / token
640 MB ÷ 4 GB/s ≈ 160 ms / token
```

而仿真里的 `Sub-Centroid Coarse Retrieval Time: 2.761 ms` **只扫了质心索引，没有真的去 NVMe 取那 64 个块**。所以"Total Decode Step Overhead 280 ms"漏掉了权重读取（10 s）和 KV 取回（160 ms）两项主导成本。

### E8. §4 专家核验清单漏掉了本文件自身的错

清单三项（带宽自洽 / 常驻内存 / Super-Tile 对齐）都不足以发现 E1–E4。应补：
- [ ] **KV 每 token 字节数核算**（层数 × KV head × dim × 2 × dtype），并据此反推可承载上下文
- [ ] **索引体积是否乘了 `层数 × KV head 数`**
- [ ] **每 token 权重读取量 × (1/BW)** 是否与给出的 tok/s 一致
- [ ] **"实测"项是否区分"理论推导"与"脚本打印值"**，是否有 RSS 插桩
- [ ] 仿真是否包含真实 matmul 与真实 KV 块读取

---

## 建议的替换文本（可直接贴回白皮书）

**§2.1 表格加一列"KV B/token（FP16, GQA-8, d128）"**，并把结论列改为：

| 模型 | KV B/token | 物理可承载（FP16 KV） | 物理可承载（int8 KV） | 结论 |
|---|---|---|---|---|
| 8B | 128 KB | 173.7K | 347.3K | 需 KV 量化才能到 256K |
| 14B | 160 KB | 110.1K | 220.2K | 需 KV 量化才能到 128K |
| 32B | 256 KB | 25.4K | 50.8K | 需 KV 量化才能到 32K |
| 70B | 320 KB | — | — | 不可常驻 |

**§2.2 质心索引**改为：`1024 块 × 2 × 64 × 4 B × 80 层 × 8 KV head ≈ 320 MB`（dim=128 时为 640 MB）。

**§4**把"实测输出"改名为"**仿真脚本打印值（仅覆盖 madvise 系统调用开销，不含权重/KV 读取与真实计算）**"，并删除"证实…可行性"一句，改为：

> 该 Harness 仅验证控制面调度的代码路径可执行；**吞吐可行性由 §3.2 的理论边界判定**（70B Dense + Mac NVMe = 0.10 tok/s）。两者不可混用。

---

## 结论

白皮书的**推导框架是对的**（冯·诺依曼瓶颈、UMA vs 异构、层流式双缓冲、$v \le BW/W$），问题全部集中在：
1. **KV 体积少算 640×**（E1，且正是我们批评对方的同一条）；
2. **把脚本硬编码打印值当成实测**（E2）；
3. **§4 与 §3.2 相差 36 倍的自相矛盾**（E3）；
4. **上下文表格 1.16×–1.47× 偏乐观且未声明 KV 假设**（E4）。

建议：先修 E1–E4 再让对方引用；否则对方会拿我们对他们的 T3.4 指控原样回敬。
