# 🔐 HƯỚNG DẪN BẢO MẬT API KEYS

## ⚠️ QUAN TRỌNG
**KHÔNG BAO GIỜ commit API keys vào Git!**

## 🛡️ Setup Bảo mật

### Bước 1: Tạo file .env
```bash
# Copy template
cp .env.example .env

# Chỉnh sửa .env với API key thật của bạn
OPENROUTER_API_KEY=sk-or-v1-your_real_api_key_here
```

### Bước 2: Setup cho Streamlit (nếu dùng)
```bash
# Tạo thư mục .streamlit
mkdir -p .streamlit

# Tạo file secrets.toml
echo 'OPENROUTER_API_KEY = "sk-or-v1-your_real_api_key_here"' > .streamlit/secrets.toml
```

### Bước 3: Xác nhận .gitignore
Đảm bảo các file sau trong .gitignore:
```
.env
.env.*
secrets.toml
.streamlit/secrets.toml
```

## 🔍 Kiểm tra trước khi commit

### Lệnh kiểm tra nhanh:
```bash
# Kiểm tra file nào sẽ được commit
git status

# Kiểm tra có API key bị lộ không
grep -r "sk-" --exclude-dir=.git .
```

### Nếu đã commit nhầm API key:
```bash
# XÓA khỏi history (NGUY HIỂM - backup trước!)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch .streamlit/secrets.toml' \
  --prune-empty --tag-name-filter cat -- --all

# Hoặc đơn giản hơn: tạo API key mới
```

## 🔄 Quy trình làm việc an toàn

1. **Clone repo mới:**
   ```bash
   git clone <repo>
   cd <repo>
   cp .env.example .env
   # Điền API key vào .env
   ```

2. **Trước khi commit:**
   ```bash
   git add .
   git status  # Kiểm tra kỹ!
   git commit -m "message"
   ```

3. **Kiểm tra định kỳ:**
   ```bash
   git log --oneline -10  # Xem commit gần đây
   ```

## 🚨 Nếu API key bị lộ

1. **Ngay lập tức:**
   - Vào OpenRouter dashboard
   - Revoke/Delete API key cũ
   - Tạo API key mới

2. **Clean Git history:**
   ```bash
   git rm --cached .streamlit/secrets.toml
   git commit -m "Remove leaked API key"
   ```

3. **Update .env với key mới**

## 🎯 Best Practices

- ✅ Luôn dùng environment variables
- ✅ Kiểm tra .gitignore trước khi commit
- ✅ Dùng API key khác nhau cho dev/prod
- ❌ Không hardcode API keys trong code
- ❌ Không commit .env files
- ❌ Không share API keys qua chat/email
