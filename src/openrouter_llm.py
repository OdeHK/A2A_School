import os
import requests
import json

class OpenRouterLLM:
    """
    Lớp vỏ bọc (wrapper) để gọi model qua OpenRouter API.
    """
    def __init__(self, model_name: str, api_key: str):
        self.model = model_name
        self.api_key = api_key
        self.api_url = "https://api.openrouter.ai/api/v1/chat/completions"
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/OdeHK/A2A_School",  # GitHub repo URL
            "X-Title": "A2A School Project"  # Tên ứng dụng
        }
        # --- KẾT THÚC PHẦN SỬA LỖI ---

    def check_internet_connection(self):
        """
        Kiểm tra kết nối internet bằng cách ping google.com
        """
        try:
            requests.get("https://www.google.com", timeout=5)
            return True
        except:
            return False

    def invoke(self, prompt: str, max_retries=3, **kwargs) -> str:
        """
        Gửi prompt đến OpenRouter với cơ chế retry.
        """
        import time
        from requests.exceptions import RequestException, HTTPError, ConnectionError
        
        # Kiểm tra kết nối internet
        if not self.check_internet_connection():
            raise ConnectionError("❌ Không thể kết nối internet. Vui lòng kiểm tra kết nối mạng của bạn.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60  # Tăng timeout lên 60 giây
                )
                
                if response.status_code == 503:
                    wait_time = min((attempt + 1) * 5, 20)  # 5, 10, 15, 20 giây, tối đa 20 giây
                    print(f"🔄 API không khả dụng. Thử lại sau {wait_time} giây...")
                elif response.status_code == 401:
                    raise ValueError("❌ API key không hợp lệ hoặc đã hết hạn. Vui lòng kiểm tra lại.")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                
                print(f"⚠️ Định dạng phản hồi không mong đợi: {json.dumps(result, indent=2)}")
                return "Xin lỗi, tôi đang gặp vấn đề trong việc xử lý. Vui lòng thử lại sau."

            except RequestException as e:
                error_detail = e.response.text if hasattr(e, 'response') and e.response else str(e)
                print(f"❌ Lỗi khi gọi OpenRouter API (lần {attempt + 1}/{max_retries}): {error_detail}")
                
                if attempt == max_retries - 1:  # Nếu là lần thử cuối cùng
                    if isinstance(e, HTTPError) and e.response.status_code == 503:
                        return "Xin lỗi, dịch vụ tạm thời không khả dụng. Vui lòng thử lại sau vài phút."
                    return "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."
                    
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)  # Đợi trước khi thử lại
                
        return "Không thể kết nối với dịch vụ sau nhiều lần thử. Vui lòng thử lại sau."