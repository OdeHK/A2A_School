import requests
import json
import time

class FallbackLLM:
    """
    LLM backup sử dụng Hugging Face Inference API miễn phí
    """
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Gửi prompt đến Hugging Face API
        """
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": kwargs.get("max_tokens", 500),
                    "temperature": kwargs.get("temperature", 0.7),
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "Xin lỗi, không thể tạo phản hồi.")
                return "Xin lỗi, không thể tạo phản hồi."
            else:
                return f"❌ Lỗi từ Hugging Face API: {response.status_code}"
                
        except Exception as e:
            return f"❌ Lỗi khi gọi Fallback API: {str(e)}"

class OfflineLLM:
    """
    LLM offline đơn giản khi không có kết nối mạng
    """
    def __init__(self):
        self.responses = {
            "quiz": "Xin lỗi, tôi không thể tạo quiz khi không có kết nối mạng. Vui lòng kiểm tra kết nối và thử lại.",
            "chat": "Xin lỗi, tôi đang không thể kết nối đến dịch vụ AI. Vui lòng kiểm tra kết nối mạng và thử lại sau.",
            "default": "Tôi hiện không thể xử lý yêu cầu do vấn đề kết nối. Vui lòng thử lại sau khi đã kiểm tra kết nối mạng."
        }
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Trả về phản hồi offline dựa trên loại prompt
        """
        prompt_lower = prompt.lower()
        
        if "quiz" in prompt_lower or "câu hỏi" in prompt_lower:
            return self.responses["quiz"]
        elif any(word in prompt_lower for word in ["chat", "nói chuyện", "trò chuyện"]):
            return self.responses["chat"]
        else:
            return self.responses["default"]
