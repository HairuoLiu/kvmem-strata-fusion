import json, time, urllib.request
prompt_4k = "Analyze the fundamental tradeoffs in long-context transformer architectures. " * 80
payload = {"model": "qwen2.5:7b", "prompt": prompt_4k, "stream": False,
           "options": {"num_predict": 64, "temperature": 0.0}}
req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=180) as resp:
    res = json.loads(resp.read().decode())
wall = time.perf_counter() - t0
print("prompt_tokens      =", res.get("prompt_eval_count"))
print("prompt_tok_per_sec =", round(res.get("prompt_eval_count",0)/(res.get("prompt_eval_duration",1)/1e9), 4))
print("gen_tokens         =", res.get("eval_count"))
print("gen_tok_per_sec    =", round(res.get("eval_count",0)/(res.get("eval_duration",1)/1e9), 4))
print("load_duration_s    =", round(res.get("load_duration",0)/1e9, 3))
print("total_duration_s   =", round(res.get("total_duration",0)/1e9, 3))
print("wall_time_s        =", round(wall, 3))
print("num_ctx in model   = 32768 (from /api/tags)")
print("--- 他们的产物 ---")
print("prompt_tokens=990  prompt_tok_per_sec=434.6612  gen_tokens=64  gen_tok_per_sec=26.3045  load=5.5599  total=10.2731")
