#!/usr/bin/env python3
"""
Web GUI Server for KVMem-Strata-Fusion & Tripartite Benchmark Dashboard
Provides an interactive web interface at http://127.0.0.1:8080 to:
1. Run and visualize live Ollama inference vs Strata (spec) vs KVMem-Strata-Fusion
2. Run live Wikipedia long-context retrieval benchmark (Arm A vs Arm B vs K2 vs K4)
3. Run micro-architectural C++ benchmarks and view live hardware gauges
4. Interactive tripartite capability matrix inspector
"""

import os
import sys
import json
import time
import urllib.request
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8080
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_PYTHON = os.path.join(ROOT_DIR, ".venv", "bin", "python")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet standard console logging
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(os.path.join(ROOT_DIR, "web", "index.html"), "rb") as f:
                self.wfile.write(f.read())
            return

        if self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            # Check Ollama status
            ollama_up = False
            ollama_models = []
            try:
                with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0) as resp:
                    data = json.loads(resp.read().decode())
                    ollama_up = True
                    ollama_models = [m.get("name") for m in data.get("models", [])]
            except Exception:
                pass

            # Load latest benchmark JSONs if available
            tripartite_data = {}
            tripartite_file = os.path.join(ROOT_DIR, "docs", "tripartite_comparison_benchmark.json")
            if os.path.exists(tripartite_file):
                try:
                    with open(tripartite_file) as f:
                        tripartite_data = json.load(f)
                except Exception:
                    pass

            retrieval_data = {}
            retrieval_file = os.path.join(ROOT_DIR, "docs", "decoupled_retrieval_benchmark.json")
            if os.path.exists(retrieval_file):
                try:
                    with open(retrieval_file) as f:
                        retrieval_data = json.load(f)
                except Exception:
                    pass

            res = {
                "ollama_online": ollama_up,
                "ollama_models": ollama_models,
                "tripartite_benchmark": tripartite_data,
                "retrieval_benchmark": retrieval_data,
                "cpp_binary_ready": os.path.exists(os.path.join(ROOT_DIR, "build", "main_bench"))
            }
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            req_json = json.loads(post_data)
        except Exception:
            req_json = {}

        if self.path == "/api/run_tripartite":
            model_id = req_json.get("model", "qwen2.5:7b")
            cmd = [VENV_PYTHON, "benchmarks/benchmark_tripartite_comparison.py"]
            try:
                proc = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, timeout=120)
                tripartite_file = os.path.join(ROOT_DIR, "docs", "tripartite_comparison_benchmark.json")
                res_data = {}
                if os.path.exists(tripartite_file):
                    with open(tripartite_file) as f:
                        res_data = json.load(f)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": proc.returncode == 0,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "data": res_data
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
            return

        if self.path == "/api/run_cpp_bench":
            bin_path = os.path.join(ROOT_DIR, "build", "main_bench")
            if not os.path.exists(bin_path):
                # Build first
                subprocess.run(["mkdir", "-p", "build"], cwd=ROOT_DIR)
                build_cmd = ["clang++", "-std=c++20", "-O3", "-march=native", "-I./csrc",
                             "csrc/ubba_solver.cpp", "csrc/ifr_lse.cpp", "csrc/main_bench.cpp", "-o", "build/main_bench"]
                subprocess.run(build_cmd, cwd=ROOT_DIR, capture_output=True)

            try:
                proc = subprocess.run([bin_path], cwd=ROOT_DIR, capture_output=True, text=True, timeout=60)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": proc.returncode == 0,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
            return

        if self.path == "/api/run_chat":
            model_id = req_json.get("model", "qwen2.5:7b")
            prompt = req_json.get("prompt", "Hello!")
            # Call Ollama
            try:
                payload = json.dumps({
                    "model": model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 128}
                }).encode()
                req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=60) as resp:
                    out = json.loads(resp.read().decode())
                elapsed = time.perf_counter() - t0
                
                tok_s = (out.get("eval_count", 0) / (out.get("eval_duration", 1) / 1e9)) if out.get("eval_duration", 0) > 0 else 0
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "response": out.get("response", ""),
                    "tokens": out.get("eval_count", 0),
                    "prompt_tokens": out.get("prompt_eval_count", 0),
                    "tok_per_sec": tok_s,
                    "wall_time": elapsed
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
            return

        self.send_error(404, "Not Found")


def run_server(port=PORT):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print("=" * 70)
    print(f"🚀 KVMem-Strata-Fusion Web Dashboard is live at: http://127.0.0.1:{port}")
    print("=" * 70)
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
