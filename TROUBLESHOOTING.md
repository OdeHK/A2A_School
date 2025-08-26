# 🔧 Hướng dẫn xử lý lỗi kết nối API

## Các lỗi thường gặp và cách khắc phục

### 1. Lỗi `getaddrinfo failed` (DNS Resolution Error)
```
Lỗi: HTTPSConnectionPool(host='api.openrouter.ai', port=443): Max retries exceeded with url: /api/v1/chat/completions (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x...>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
```

**Nguyên nhân:**
- DNS server không thể phân giải tên miền `api.openrouter.ai`
- Firewall hoặc proxy chặn kết nối
- Vấn đề với kết nối internet

**Cách khắc phục:**
1. **Kiểm tra kết nối internet:**
   ```powershell
   ping google.com
   ```

2. **Kiểm tra DNS:**
   ```powershell
   nslookup api.openrouter.ai
   ```

3. **Thử đổi DNS server:**
   - Mở Control Panel > Network and Internet > Network Connections
   - Right-click adapter mạng > Properties
   - Chọn Internet Protocol Version 4 (TCP/IPv4) > Properties
   - Chọn "Use the following DNS server addresses:"
   - Primary: 8.8.8.8 (Google DNS)
   - Secondary: 8.8.4.4

4. **Tạm thời tắt Windows Firewall/Antivirus:**
   - Windows Security > Firewall & network protection
   - Tạm thời tắt để test

### 2. Lỗi Timeout
```
Lỗi: requests.exceptions.Timeout
```

**Cách khắc phục:**
- Kiểm tra tốc độ mạng
- Thử lại sau vài phút
- Hệ thống đã tự động giảm timeout xuống 30 giây

### 3. Lỗi API Key
```
Lỗi: 401 Unauthorized
```

**Cách khắc phục:**
1. Kiểm tra file `.streamlit/secrets.toml`
2. Đảm bảo API key đúng format
3. Kiểm tra hạn mức API key tại OpenRouter

### 4. Lỗi Rate Limit
```
Lỗi: 429 Too Many Requests
```

**Cách khắc phục:**
- Đợi vài phút rồi thử lại
- Hệ thống đã có retry logic tự động

## Các tính năng mới đã thêm:

### 1. **Fallback System** 🔄
- Khi OpenRouter API lỗi, tự động chuyển sang Hugging Face API
- Khi cả hai đều lỗi, chuyển sang chế độ offline với thông báo hữu ích

### 2. **Improved Error Handling** 🛡️
- Phân loại và xử lý cụ thể từng loại lỗi
- Thông báo lỗi chi tiết và gợi ý khắc phục
- Retry logic thông minh với exponential backoff

### 3. **Connection Testing** 🔍
- Kiểm tra kết nối internet trước khi gọi API
- Tự động phát hiện vấn đề DNS/network

### 4. **Better Logging** 📝
- Log chi tiết từng bước xử lý
- Hiển thị progress khi retry
- Gợi ý nguyên nhân lỗi

## Test hệ thống:

```python
# Test trong Python console
from src.openrouter_llm import OpenRouterLLM
from src.rag import RAGManager

# Tạo LLM instance
llm = OpenRouterLLM("microsoft/wizardlm-2-8x22b", "your_api_key")

# Test kết nối
print(f"Internet: {llm.check_internet_connection()}")

# Test API call
response = llm.invoke("Hello, how are you?")
print(response)
```

## Troubleshooting Commands:

```powershell
# Kiểm tra kết nối mạng
ping google.com
ping api.openrouter.ai

# Kiểm tra DNS
nslookup api.openrouter.ai

# Test với curl (nếu có)
curl -I https://api.openrouter.ai/api/v1/chat/completions

# Kiểm tra proxy/firewall
netsh winhttp show proxy
```

Nếu vẫn gặp vấn đề, hãy share log lỗi chi tiết để được hỗ trợ thêm! 🚀
