import requests
import json
from typing import List, Tuple
from gpt_thongso import API_KEY, API_URL

def ask_gpt(model: str, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }
    try:
        resp = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        return payload["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {e}"