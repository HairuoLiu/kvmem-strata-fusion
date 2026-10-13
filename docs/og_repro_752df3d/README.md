# og_repro_752df3d — `752df3d` 审计复现脚本

配合 `docs/og_audit_752df3d.md` 使用。三个脚本均只读对方代码行为，不修改仓库任何文件。

| 脚本 | 对应审计章节 | 作用 | 运行方式 |
|---|---|---|---|
| `ifr_perf.py` | §6.1 | 证明 `ifr.py` 新聚合路径中 mean-of-logits ≡ logit-of-mean-of-sub-centroids（max\|Δ\| ≈ 1.8e-15），并测逐子质心循环与等价单向量点积的时延差 | `python ifr_perf.py` |
| `ifr_perf2.py` | §6.2 | 旧版（向量化 max-pooling）vs 新版（逐子质心 mean）vs 等价向量化均值的时延与 top-8 行为对比 | `python ifr_perf2.py` |
| `repro_ollama.py` | §5 | 复现对方 Ollama 7B 测量（需本地 Ollama 已加载 `qwen2.5:7b`），演示 prompt 吞吐的缓存敏感性 | `python repro_ollama.py` |

Strata 一手数值核（MODELS.md / MAC.md / 论文）不在脚本内，需从 https://github.com/Niko1221/Strata 获取 `docs/MODELS.md`、`docs/MAC.md` 后按审计文档 §2 表格逐字比对。
