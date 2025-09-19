# src/core/llm_provider.py
# Lớp giao tiếp với API của OpenRouter, được tối ưu với caching và rate limiting.

import requests
import logging
import time
import json
import hashlib
from pathlib import Path
from collections import deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Điều tiết tần suất gọi API để tránh bị khóa.
    """
    def __init__(self, requests_per_minute: int = 20):
        self.requests_per_minute = requests_per_minute
        self.request_times = deque()

    def wait_if_needed(self):
        """Đợi nếu cần thiết trước khi thực hiện request tiếp theo."""
        now = time.time()
        # Xóa các request đã quá 1 phút
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                logger.info(f"⏳ Rate limit đạt ngưỡng, chờ {wait_time:.1f} giây...")
                time.sleep(wait_time)
        
        self.request_times.append(time.time())

class LLMProvider:
    """
    Lớp chính để tương tác với LLM, tích hợp caching và rate limiting.
    """
    def __init__(self, api_key: str, api_url: str, model: str, cache_dir: Path):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.rate_limiter = RateLimiter()
        
        # Cấu hình caching
        self.cache_dir = cache_dir / "llm_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache: Dict[str, str] = {}

        # Chế độ mock nếu không có API key
        self.is_mock = not api_key or "your_" in api_key
        if self.is_mock:
            logger.warning("LLMProvider đang chạy ở chế độ MOCK do không có API key.")
        else:
            logger.info(f"✅ LLMProvider đã khởi tạo với model: {self.model}")

    def _get_cache_key(self, system_prompt: str, user_prompt: str) -> str:
        """Tạo một key duy nhất cho request để caching."""
        content = f"model:{self.model}|sys:{system_prompt}|user:{user_prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[str]:
        """Lấy kết quả từ cache (memory trước, sau đó đến disk)."""
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.memory_cache[key] = data['response'] # Cập nhật memory cache
                return data['response']
        return None

    def _save_to_cache(self, key: str, response: str):
        """Lưu kết quả vào cả memory và disk cache."""
        self.memory_cache[key] = response
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({'response': response}, f, ensure_ascii=False, indent=2)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
        """
        Phương thức chính để sinh văn bản.
        Tích hợp sẵn mock mode, caching, rate limiting và retry.
        """
        if self.is_mock:
            return f"**[CHẾ ĐỘ MOCK]**\n**System Prompt:** {system_prompt}\n**User Prompt:** {user_prompt[:200]}..."

        cache_key = self._get_cache_key(system_prompt, user_prompt)
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            logger.info(f"🎯 Cache hit cho prompt: '{user_prompt[:50]}...'")
            return cached_response
        
        logger.info(f"🌐 Cache miss, đang gọi API cho prompt: '{user_prompt[:50]}...'")
        self.rate_limiter.wait_if_needed()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost", # Cần thiết cho OpenRouter
            "X-Title": "A2A School Platform"
        }
        json_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=json_data, timeout=90)
                response.raise_for_status()
                result_json = response.json()
                content = result_json['choices'][0]['message']['content']
                self._save_to_cache(cache_key, content)
                return content
            except requests.exceptions.RequestException as e:
                logger.warning(f"Lỗi API (lần {attempt + 1}/{max_retries}): {e}. Đang thử lại...")
                time.sleep(2 ** attempt) # Exponential backoff
        
        logger.error(f"Không thể hoàn thành request API sau {max_retries} lần thử.")
        return "[LỖI] Không thể kết nối đến LLM. Vui lòng thử lại sau."
