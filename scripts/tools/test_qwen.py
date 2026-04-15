import httpx

resp = httpx.post(
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    headers={"Authorization": "Bearer sk-5b5ea605ff294283890cc88ec7e892ea"},
    json={"model": "qwen3-235b-a22b", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 100},
    timeout=30,
)
print(resp.status_code)
print(resp.json()["choices"][0]["message"]["content"])
