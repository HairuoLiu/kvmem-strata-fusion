# KVMem 本地大模型服务 — 标准 OpenAI 接口对接规范指南

本文档汇总了将本本地模型服务接入任意第三方 AI 应用（如 **Chatbox**、**NextChat**、**Open WebUI**、**Cherry Studio**、**LobeChat**、**Dify**、**Cursor**、**Continue** 等）时的完整参数设置规范。

---

## 一、基础配置 (Base Configuration)

| 配置字段 | 推荐填写值 | 说明 |
| :--- | :--- | :--- |
| **接口类型 (Provider)** | `OpenAI` 或 `OpenAI API 兼容` | 本服务 100% 遵循 OpenAI 标准 REST / SSE 协议 |
| **接口地址 (Base URL)** | `http://127.0.0.1:8080/v1` | 大多数软件需要带 `/v1` 后缀 |
| **备用地址 (Host 简写)** | `http://127.0.0.1:8080` | 部分软件（如 NextChat）会自动追加 `/v1/chat/completions` |
| **局域网地址 (LAN URL)** | `http://[你的Mac局域网IP]:8080/v1` | 供同 Wi-Fi 下的 iPhone / iPad / 其他电脑调用 |
| **API Key (令牌/密钥)** | `sk-local` | 本地自研私有引擎，无外部扣费鉴权，填任意非空字符均可放行 |
| **当前推荐模型 (Model ID)**| `qwen3.8-27b` 或 `qwen3.8` | 27B 稠密大模型，兼具高质量推理与端侧统一显存轻量驻留 |
| **其他可用模型 (Models)** | `qwen3.8-flash-next`<br>`qwen3.8-coder-30b`<br>`gpt-4o`<br>`default` | 均已注册在 `/v1/models` 中，支持点击“获取模型列表”一键读取 |

---

## 二、输入与输出参数 (Input & Output Parameters)

| 参数名称 | 默认推荐值 | 调优建议 |
| :--- | :--- | :--- |
| **Temperature (采样温度)** | `0.7` | • **代码重构 / 逻辑严谨推理 / 事实问答**：建议设为 `0.1 ~ 0.3`<br>• **日常对话 / 创意写作 / 头脑风暴**：建议设为 `0.7 ~ 0.9` |
| **Top_P (核采样)** | `0.95` | 采样概率质量阈值，与 Temperature 配合使用 |
| **Max Tokens (单次最大输出)**| `2048` | 端侧单次生成推荐设为 `2048` ~ `4096`，可根据需要随时调大 |
| **Context Length (上下文窗口)**| `32768` (32K) ~ `131072` (128K) | 本地内核挂接了 **CASA (零重计算开销 Key 冻结) + IFR (反折叠质心压缩)**，显存占用仅为传统未压缩状态的 12.5% (节省 87.5%) |
| **Stream (流式打字输出)** | **务必开启 (True)** | 首字延迟 (TTFT) 仅几十毫秒，逐字流式打字体验最顺畅 |
| **Presence / Frequency Penalty** | `0.0` (留空) | 若偶发重复车轱辘话，可微调至 `0.1 ~ 0.2` |

---

## 三、高级设置与协议特性 (Advanced Settings)

### 1. 思考模式 / 思维链 (`<think>`) 处理
* **原理**：当前加载的模型为 **Qwen 3 原生 Reasoning（深度推理）架构**。在回答任何问题前，模型内部会自动自发生成一段包含在 `<think> ... </think>` 标签中的思维链推理过程。
* **在支持思考折叠的客户端**（如 *Cherry Studio*、*LobeChat*、最新版 *Chatbox*）：
  * 客户端会自动检测 `<think>` 标签并将其折叠为一个精巧的 **“思考中 / 思考过程”** 抽屉，体验与 DeepSeek-R1 / OpenAI o1 一致。
* **在不支持思考折叠的客户端**（如果客户端直接将 `<think>` 文本原样打印在屏幕上）：
  * 若您不想看到模型的思考内心独白，只需在客户端的 **系统提示词 (System Prompt)** 中添加一句指令：
    > `直接给出答案，不要输出 <think> 思考过程标签。` (或英文 `Directly answer the question without outputting <think> tags.`)

### 2. 工具调用 / Agent 扩展 (Function Calling)
* 本地 API 遵循 OpenAI Chat Completions 协议，接收 `messages` 数组结构（包含 `role: "system"`, `role: "user"`, `role: "assistant"`, `role: "tool"`）。
* 当您在第三方客户端（如 Dify、Open WebUI、Chatbox 插件库）开启网页搜索、计算器、代码执行等 Agent 插件时，客户端会自动完成多轮工具调用注入。

### 3. 跨域与网络 (CORS)
* 后端已开启全域宽松跨域支持：
  * `Access-Control-Allow-Origin: *`
  * `Access-Control-Allow-Headers: *`
  * `Access-Control-Allow-Methods: GET, POST, OPTIONS, PUT, DELETE`
* 任何基于本地或局域网浏览器运行的 Web 前端（如在另一台机器打开的 Web 界面）直连调用均不会触发浏览器 CORS 跨域拦截。

---

## 四、快捷验证与调用示例

### Python (OpenAI SDK)
```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-local"
)

stream = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "你好，请简单做个自我介绍"}],
    stream=True,
    temperature=0.7,
    max_tokens=1024
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### 终端 cURL 命令行
```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```
