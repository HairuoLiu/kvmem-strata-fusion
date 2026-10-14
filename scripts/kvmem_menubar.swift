import AppKit
import Foundation

class MenuBarController: NSObject {
    var statusItem: NSStatusItem!
    var timer: Timer?
    let workspaceDir = "/Users/hai/workspaces/kvmem-strata-fusion"
    
    var isRunning = false
    var statusString = "正在检测..."
    var speedString = "-- tok/s"
    var memString = "-- GB"
    var currentModel = "Qwen3-0.6B-Instruct (Local In-Process Baseline MPS)"
    
    let availableModels = [
        "Qwen3-0.6B-Instruct (Local In-Process Baseline MPS)",
        "Qwen3.8-27B-GSQ",
        "Qwen3.8-Flash-Next (125B MoE)"
    ]

    override init() {
        super.init()
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.title = "⚡ KVMem"
        }
        buildMenu()
        checkServerStatus()
        
        timer = Timer.scheduledTimer(withTimeInterval: 2.5, repeats: true) { [weak self] _ in
            self?.checkServerStatus()
        }
    }

    func checkServerStatus() {
        guard let url = URL(string: "http://127.0.0.1:8080/api/monitor") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 0.8

        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if let httpResp = response as? HTTPURLResponse, httpResp.statusCode == 200, let data = data {
                    self.isRunning = true
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        if let spd = json["speed_tok_s"] as? Double {
                            self.speedString = String(format: "%.1f tok/s", spd)
                        }
                        if let mem = json["process_rss_mb"] as? Double {
                            self.memString = String(format: "%.1f GB", mem / 1024.0)
                        }
                    }
                    self.statusString = "● 运行中 (端口 8080)"
                    let shortName = self.currentModel.contains("0.6B") ? "0.6B" : (self.currentModel.replacingOccurrences(of: "Qwen3.8-", with: "").components(separatedBy: " ").first ?? "27B")
                    self.statusItem.button?.title = "⚡ KVMem (\(shortName))"
                } else {
                    self.isRunning = false
                    self.statusString = "○ 已停止 (端口空闲)"
                    self.speedString = "-- tok/s"
                    self.memString = "-- GB"
                    self.statusItem.button?.title = "⚪ KVMem"
                }
                self.buildMenu()
            }
        }
        task.resume()
    }

    func buildMenu() {
        let menu = NSMenu()
        
        let headerItem = NSMenuItem(title: "KVMem 本地大模型引擎 (OpenAI API)", action: nil, keyEquivalent: "")
        headerItem.isEnabled = false
        menu.addItem(headerItem)

        let statusItem = NSMenuItem(title: self.statusString, action: nil, keyEquivalent: "")
        statusItem.isEnabled = false
        menu.addItem(statusItem)

        if isRunning {
            let metricItem = NSMenuItem(title: "显存占用: \(self.memString) · 吞吐: \(self.speedString)", action: nil, keyEquivalent: "")
            metricItem.isEnabled = false
            menu.addItem(metricItem)
        }

        menu.addItem(NSMenuItem.separator())

        // Model Switcher Submenu
        let modelMenuItem = NSMenuItem(title: "🧠 切换运行模型 ❯", action: nil, keyEquivalent: "")
        let modelSubMenu = NSMenu()
        for m in availableModels {
            let item = NSMenuItem(title: m, action: #selector(switchModelAction(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = m
            if m == currentModel {
                item.state = .on
            } else {
                item.state = .off
            }
            modelSubMenu.addItem(item)
        }
        modelMenuItem.submenu = modelSubMenu
        menu.addItem(modelMenuItem)

        menu.addItem(NSMenuItem.separator())

        // 1. Copy API URL
        let copyApiItem = NSMenuItem(title: "🔗 复制 API 接口地址 (http://127.0.0.1:8080/v1)", action: #selector(copyApi), keyEquivalent: "u")
        copyApiItem.target = self
        menu.addItem(copyApiItem)

        // 2. Copy Full Configuration Specification
        let copyFullConfigItem = NSMenuItem(title: "📋 复制完整 API 对接配置清单 (一键配置)", action: #selector(copyFullConfig), keyEquivalent: "c")
        copyFullConfigItem.target = self
        menu.addItem(copyFullConfigItem)

        menu.addItem(NSMenuItem.separator())

        if isRunning {
            let stopItem = NSMenuItem(title: "⏹ 停止服务并释放显存", action: #selector(stopServer), keyEquivalent: "s")
            stopItem.target = self
            menu.addItem(stopItem)
        } else {
            let startItem = NSMenuItem(title: "▶️ 启动后台服务", action: #selector(startServer), keyEquivalent: "r")
            startItem.target = self
            menu.addItem(startItem)
        }

        menu.addItem(NSMenuItem.separator())

        let quitItem = NSMenuItem(title: "❌ 退出菜单栏托盘", action: #selector(quitAll), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        self.statusItem.menu = menu
    }

    @objc func switchModelAction(_ sender: NSMenuItem) {
        guard let modelName = sender.representedObject as? String else { return }
        self.currentModel = modelName
        self.buildMenu()
        
        if let url = URL(string: "http://127.0.0.1:8080/api/model") {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try? JSONSerialization.data(withJSONObject: ["model": modelName])
            URLSession.shared.dataTask(with: req).resume()
        }
        showNotification(title: "KVMem 模型已切换", body: "当前模型: \(modelName)")
    }

    @objc func copyApi() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("http://127.0.0.1:8080/v1", forType: .string)
        showNotification(title: "已复制 API 地址", body: "http://127.0.0.1:8080/v1 已写入剪贴板")
    }

    @objc func copyFullConfig() {
        let configText = """
# ==============================================================================
# KVMem 本地大模型服务 — 标准 OpenAI 接口配置清单
# ==============================================================================

【1. 基础配置 (Base Config)】
• 接口类型 (Provider)    : OpenAI (选择标准 OpenAI 协议)
• 接口地址 (Base URL)    : http://127.0.0.1:8080/v1
• 备用地址 (Host 简写)   : http://127.0.0.1:8080
• API 密钥 (API Key)     : sk-local (本地私有服务，填任意非空字符串均可)
• 当前默认模型 (Model ID): \(currentModel)
• 可选模型列表           : qwen3, qwen3.8, qwen3.8-27b, qwen3.8-flash-next

【2. 输入与输出参数 (I/O Parameters)】
• 采样温度 (Temperature) : 0.7 (写代码/严格分析建议 0.2, 创意写作建议 0.7~0.8)
• 核采样 (Top_P)         : 0.95
• 最大单次输出 (Max Tok) : 2048 (端侧推荐 2048 ~ 4096)
• 上下文长度 (Context)   : 32768 (32K) ~ 131072 (128K)
• 流式传输 (Stream)      : 开启 (True / 打字机即时输出)
• 惩罚系数 (Penalty)     : 0.0 (留空或 0)

【3. 高级与思考模式设置 (Advanced Settings)】
• 思考链处理 (Reasoning) : 模型原生携带 <think>...</think> 思考链。
  - 若客户端支持“思考折叠”(如 Cherry Studio, LobeChat, Chatbox)，会自动收拢显示。
  - 若客户端不支持、希望直接回答，可在系统提示词(System Prompt)中加入：
    "直接回答问题，不要输出 <think> 思考过程标签。"
• 函数/工具调用 (Tools) : 兼容标准 OpenAI 客户端 Agent 与插件调用。
• 跨域访问 (CORS)       : 已全开 (*)，本地 Web 应用无需担心跨域报错。
"""
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(configText, forType: .string)
        showNotification(title: "已复制完整 API 参数", body: "所有对接参数（Base URL、Key、模型、I/O默认值）已写入剪贴板")
    }

    @objc func startServer() {
        runScript("scripts/start_api.sh")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
            self?.checkServerStatus()
        }
    }

    @objc func stopServer() {
        runScript("scripts/stop_api.sh")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in
            self?.checkServerStatus()
        }
    }

    @objc func quitAll() {
        NSApplication.shared.terminate(nil)
    }

    func runScript(_ relativePath: String) {
        let fullPath = "\(workspaceDir)/\(relativePath)"
        let task = Process()
        task.launchPath = "/bin/bash"
        task.arguments = [fullPath]
        task.currentDirectoryPath = workspaceDir
        try? task.run()
    }

    func showNotification(title: String, body: String) {
        let script = "display notification \"\(body)\" with title \"\(title)\""
        if let appleScript = NSAppleScript(source: script) {
            appleScript.executeAndReturnError(nil)
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let controller = MenuBarController()
app.run()
