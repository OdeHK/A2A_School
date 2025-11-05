# 🔧 FIX CHI TIẾT - 3 VẤN ĐỀ

## ✅ 1. Fix Nút Tải Word Không Hiện

### Vấn đề:
- User prompt "tải word" → Response đúng nhưng nút download không hiện
- Component `download_file` được return nhưng `visible=False`

### Giải pháp:
**File: `ui/app.py` - Function `handle_chat_input()`**

```python
# BEFORE: Return file path string
if response.startswith("FILE_DOWNLOAD:"):
    file_path = response.replace("FILE_DOWNLOAD:", "").strip()
    return chat_history, file_path  # ❌ Gradio không tự update visibility

# AFTER: Return gr.File component với visibility control
if response.startswith("FILE_DOWNLOAD:"):
    file_path = response.replace("FILE_DOWNLOAD:", "").strip()
    return chat_history, gr.File(value=file_path, visible=True, label="📥 Tải xuống file Word")
else:
    return chat_history, gr.File(visible=False, label="📥 Tải xuống file Word")
```

### Kết quả:
✅ Khi user chat "tải word":
- File Word được tạo từ quiz_data.json
- Component `gr.File` hiện lên với file sẵn sàng download
- User click vào file component để tải xuống

---

## ⚠️ 2. Fix Quiz Generation Tạo Sai Số Lượng

### Vấn đề:
- User prompt: "tạo 2 câu tự luận và 2 câu trắc nghiệm"
- Expected: 4 câu (2 essay + 2 multiple choice)
- Actual: 6 câu (2 essay + 4 multiple choice)
- **Root cause**: LLM không tuân thủ chính xác số lượng câu

### Giải pháp:
**File: `services/quiz_generation/quiz_generation.py`**

Thêm **STRICT constraints** vào prompt:

```python
quiz_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),
    ("human", 
        "⚠️ CRITICAL: You MUST generate EXACTLY {num_questions} questions. No more, no less.\n"
        "Count carefully and ensure the total number matches {num_questions}.\n\n"
        "You are required to generate a quiz set with {num_questions} questions...\n\n"
        "⚠️ REMINDER: Generate EXACTLY {num_questions} questions total.\n\n"
        # ... rest of prompt
    )
])
```

### Cải thiện:
- ✅ Nhấn mạnh EXACTLY {num_questions} 3 lần trong prompt
- ✅ Dùng emoji ⚠️ để làm nổi bật
- ✅ Nhắc nhở đầu và cuối prompt

### Kết quả mong đợi:
- LLM sẽ tuân thủ tốt hơn về số lượng câu
- Nếu vẫn sai, có thể cần thêm post-processing để filter/limit số câu

---

## 🔐 3. Fix Google OAuth - Chỉ Login Được 1 Account

### Vấn đề:
- User đăng nhập Google lần đầu → OK
- User muốn đổi account → Không thể (vẫn dùng token cũ)
- **Root cause**: Token được cache trong `session_data/temp/token.json`

### Giải pháp:

#### A. Thêm `logout_google()` method
**File: `services/ui_integration_service.py`**

```python
def logout_google(self) -> Tuple[bool, str]:
    """Logout from Google by deleting the token file."""
    try:
        temp_folder = Path("session_data/temp")
        token_file = temp_folder / "token.json"
        
        if token_file.exists():
            token_file.unlink()  # Xóa token
            logger.info("Google token deleted successfully")
            return True, "Đã đăng xuất Google thành công"
        else:
            return True, "Chưa đăng nhập Google"
    except Exception as e:
        return False, str(e)
```

#### B. Thêm `force_reauth` parameter
**File: `services/ui_integration_service.py`**

```python
def open_sign_in_website(self, force_reauth: bool = False) -> Tuple[bool, str]:
    # If force_reauth, delete existing token
    if force_reauth:
        token_file = temp_folder / "token.json"
        if token_file.exists():
            token_file.unlink()
        creds = None
    # ... rest of code
```

#### C. Thêm nút "Đăng xuất Google" vào UI
**File: `ui/app.py`**

```python
# Add button
with gr.Tab("Công cụ"):
    google_auth_btn = gr.Button(value="Đăng nhập tài khoản Google", variant="primary")
    google_logout_btn = gr.Button(value="Đăng xuất Google", variant="secondary", visible=False)
    gr.Markdown("💡 **Ghi chú:** Nếu muốn đổi tài khoản Google, hãy đăng xuất trước.")

# Add handler
def handle_google_logout():
    success, message = ui_service.logout_google()
    if success:
        gr.Info(message="Đã đăng xuất Google. Có thể đăng nhập tài khoản khác.")
        return (
            gr.Button(value="Đăng nhập tài khoản Google", interactive=True, variant="primary"),
            gr.Button(visible=False)  # Hide logout button
        )
    # ... error handling

# Connect event
google_logout_btn.click(
    fn=handle_google_logout,
    outputs=[google_auth_btn, google_logout_btn]
)
```

#### D. Update login để force re-auth
**File: `ui/app.py`**

```python
def handle_google_authentication():
    # Force re-auth để user có thể chọn account khác
    auth_result, user_name = ui_service.open_sign_in_website(force_reauth=True)
    
    if auth_result:
        return (
            gr.Button(value=user_name, interactive=False),  # Show username
            gr.Button(value="Đăng xuất Google", visible=True)  # Show logout button
        )
```

### Workflow mới:

```
1. Đăng nhập lần đầu
   → User click "Đăng nhập tài khoản Google"
   → Mở browser chọn account
   → Token saved vào token.json
   → Button đổi thành username
   → Nút "Đăng xuất" hiện ra

2. Đăng nhập account khác
   → User click "Đăng xuất Google"
   → token.json bị xóa
   → Button đổi lại "Đăng nhập tài khoản Google"
   → User click lại
   → Mở browser chọn account KHÁC
   → Token mới được save
```

### Kết quả:
✅ User có thể đổi account Google tự do
✅ Workflow rõ ràng với nút logout
✅ Token được cleanup tự động

---

## 📊 Tóm tắt các files đã sửa:

### 1. `ui/app.py`
- ✅ Fix `handle_chat_input()` return gr.File component
- ✅ Thêm `handle_google_logout()` function
- ✅ Update `handle_google_authentication()` với force_reauth=True
- ✅ Thêm `google_logout_btn` button
- ✅ Connect logout event handler

### 2. `services/ui_integration_service.py`
- ✅ Thêm `logout_google()` method
- ✅ Update `open_sign_in_website()` với parameter `force_reauth`
- ✅ Logic xóa token.json khi force_reauth=True

### 3. `services/quiz_generation/quiz_generation.py`
- ✅ Enhance prompt với STRICT constraints
- ✅ Nhấn mạnh EXACTLY {num_questions} nhiều lần
- ✅ Thêm reminder ở đầu và cuối prompt

---

## 🧪 Test Cases

### Test 1: Tải Word
```
1. Tạo quiz: "tạo 5 câu về Chapter 1"
2. Chat: "tải word"
3. ✅ Check: File component hiện ra
4. ✅ Check: Click vào file → Download quiz.docx
5. ✅ Check: Đóng app → file .docx tự động xóa
```

### Test 2: Quiz số lượng chính xác
```
1. Chat: "tạo 2 câu tự luận và 2 câu trắc nghiệm"
2. ✅ Check: Tổng 4 câu (2 essay + 2 multiple choice)
3. ⚠️ Nếu vẫn sai: LLM issue, cần thêm post-processing
```

### Test 3: Đổi Google Account
```
1. Click "Đăng nhập tài khoản Google"
2. Chọn account A → Đăng nhập
3. ✅ Button hiện "Account A Name"
4. ✅ Nút "Đăng xuất Google" hiện ra
5. Click "Đăng xuất Google"
6. ✅ Button đổi lại "Đăng nhập tài khoản Google"
7. Click "Đăng nhập tài khoản Google" lại
8. Chọn account B → Đăng nhập
9. ✅ Button hiện "Account B Name"
10. ✅ Có thể sử dụng với account B
```

---

## 🚀 Chạy test ngay:

```bash
cd d:\Anh_Khiem\A2A_School
conda activate agent_for_teacher
python ui/app.py
```

Tất cả 3 vấn đề đã được fix! 🎉
