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
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"  # Fixed URL!
        
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
            response = requests.get("https://www.google.com", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"🔍 Kiểm tra kết nối internet thất bại: {e}")
            return False

    def invoke(self, prompt: str, max_retries=3, **kwargs) -> str:
        """
        Gửi prompt đến OpenRouter với cơ chế retry.
        """
        import time
        from requests.exceptions import RequestException, HTTPError, ConnectionError
        
        # Kiểm tra kết nối internet trước
        if not self.check_internet_connection():
            return "❌ Không thể kết nối internet. Vui lòng kiểm tra kết nối mạng của bạn và thử lại."

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
                print(f"🔄 Đang thử kết nối đến OpenRouter API (lần {attempt + 1}/{max_retries})...")
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30  # Giảm timeout xuống 30 giây
                )
                
                if response.status_code == 503:
                    wait_time = min((attempt + 1) * 5, 20)  # 5, 10, 15, 20 giây, tối đa 20 giây
                    print(f"🔄 API không khả dụng. Thử lại sau {wait_time} giây...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 401:
                    return "❌ API key không hợp lệ hoặc đã hết hạn. Vui lòng kiểm tra lại."
                    
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                
                print(f"⚠️ Định dạng phản hồi không mong đợi: {json.dumps(result, indent=2)}")
                return "Xin lỗi, tôi đang gặp vấn đề trong việc xử lý. Vui lòng thử lại sau."

            except (ConnectionError, requests.exceptions.ConnectionError) as e:
                print(f"❌ Lỗi kết nối đến OpenRouter API (lần {attempt + 1}/{max_retries}): {str(e)}")
                if "getaddrinfo failed" in str(e):
                    print("💡 Lỗi DNS resolution. Có thể do:")
                    print("   - Vấn đề với DNS server")
                    print("   - Firewall hoặc proxy chặn kết nối")
                    print("   - Không có kết nối internet ổn định")
                
                if attempt == max_retries - 1:
                    return "❌ Không thể kết nối đến OpenRouter API. Vui lòng kiểm tra kết nối mạng và thử lại sau."
                    
                wait_time = (attempt + 1) * 3
                print(f"⏳ Đợi {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
                
            except requests.exceptions.Timeout as e:
                print(f"⏰ Timeout khi gọi API (lần {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return "⏰ Kết nối quá chậm. Vui lòng thử lại sau."
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
                
            except requests.exceptions.HTTPError as e:
                print(f"❌ Lỗi HTTP (lần {attempt + 1}/{max_retries}): {e.response.status_code} - {e.response.text}")
                if e.response.status_code == 503:
                    if attempt == max_retries - 1:
                        return "🔧 Dịch vụ tạm thời không khả dụng. Vui lòng thử lại sau vài phút."
                    wait_time = min((attempt + 1) * 5, 30)
                    time.sleep(wait_time)
                else:
                    return f"❌ Lỗi API: {e.response.status_code} - {e.response.text}"
                    
            except Exception as e:
                print(f"❌ Lỗi không xác định (lần {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return f"❌ Đã xảy ra lỗi không xác định: {str(e)}"
                time.sleep(2)
                
        return "Không thể kết nối với dịch vụ sau nhiều lần thử. Vui lòng thử lại sau."