# RAG (Retrieval-Augmented Generation) Application

Ứng dụng AI hỗ trợ giảng viên phân tích tài liệu và trả lời câu hỏi dựa trên nội dung tài liệu đã upload.

## 🚀 Tính năng chính

Các tính năng đã hoàn thiện:
- **Upload và xử lý tài liệu PDF**: Hỗ trợ upload file PDF và tự động phân tích nội dung
- **Nhiều chiến lược chunking**: Chọn cách chia nhỏ tài liệu phù hợp
- **Tìm kiếm thông minh**: Sử dụng vector search để tìm thông tin liên quan
- **Giao diện thân thiện**: Interface Gradio dễ sử dụng
- **Tích hợp LLM**: Hỗ trợ NVIDIA và Google AI models

Các tính năng dự kiến:
- **Hỗ trợ nhiều định dạng tài liệu và web**: PDF, DOCX, HTML, và các định dạng khác
- **Cải thiện khả năng trích xuất dữ liệu ở nhiều định dạng khác nhau**: Sử dụng các công nghệ OCR để tăng độ chính xác khi trích xuất dữ liệu
- **Sử dụng Hybrid Search**: Kết hợp giữa vector search và keyword search để nâng cao độ chính xác
- **Thêm khả năng hiểu khái quát nội dung tài liệu**: Tóm tắt theo chương, theo phần
- **Thêm khả năng sử dụng các công cụ bên ngoài**: Agent có thể truy cập và sử dụng các API bên ngoài như: Google Forms, Google Classroom,...
- **Phân tích kết quả bài kiểm tra**: Phân tích các phần kiến thức học sinh còn yếu dựa trên kết quả bài kiểm tra

## Sơ đồ hoạt động
**Tính năng đăng nhập và tải lại danh sách tài liệu của người dùng**
```mermaid
sequenceDiagram
  actor A1 as User
  participant GR as Gradio (app.py)
  participant BE as UI Intergration Service
  participant DS as Database Service
  participant MA as Mongo Atlas
  title User Login and Document List Retrieval

  GR ->> A1: Show login prompt (username & password)
  A1 ->> GR: Submit credentials
  GR ->> BE: Request authentication (username, password)
  BE ->> DS: Request authentication
  DS ->> MA: Query user collection (MongoDB)
  MA -->> DS: Return user record / not found
  DS -->> BE: Authentication result (success / failure)
  BE -->> GR: Return authentication outcome
  Note left of GR: On success, Gradio stores username in `gr.State` and shows app UI

  alt Authentication success
    GR ->> BE: Request user's document list
    BE ->> DS: Request user's document list
    DS ->> MA: Query documents for user
    MA -->> DS: Return document list
    DS -->> BE: Return document list
    BE -->> GR: Provide document list (filenames shown in UI)
  else Authentication failed
    BE -->> GR: Return error message (invalid credentials)
  end
```




## 🏗️ Kiến trúc hệ thống

### Core Services
- **RagService**: Service chính orchestrate toàn bộ quy trình RAG
- **DocumentLoader**: Xử lý load tài liệu với nhiều strategy
- **DocumentChunker**: Chia nhỏ tài liệu theo các chiến lược khác nhau
- **VectorService**: Quản lý vector store và similarity search
- **UIIntegrationService**: Bridge giữa UI và core services



### 🛠️ Cài đặt

### Requirements (Conda environment)

Sử dụng file môi trường Conda: `agent_for_teacher_environment.yml`.
File này chứa cả thư viện cần thiết để chạy chương trình. Để tạo môi trường trên máy Windows:

```bash
conda env create -f .\agent_for_teacher_environment.yml
```

Sau khi tạo xong, kích hoạt môi trường:

```bash
conda activate agent_for_teacher
```

Ghi chú:
- File YAML đã liệt kê các gói cần thiết dưới phần `dependencies` và một số gói pip dưới mục `pip:`; chỉ cần chạy lệnh `conda env create` là đủ.
- Nếu bạn không sử dụng Conda, bạn có thể cài thủ công bằng `pip`: 

```bash
pip install langchain langchain-community langchain-nvidia-ai-endpoints
pip install langchain-google-genai langchain-chroma
pip install gradio pymupdf pydantic pydantic-settings
```

### Environment Variables
Tạo file `.env` từ template:
```bash
copy .env.example .env
```

Sau đó cập nhật các giá trị trong file `.env`:
```env
# API Keys
GOOGLE_API_KEY=your_google_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here

# MongoDB Configuration
MONGODB_URI=your_mongodb_connection_uri_here
MONGODB_DATABASE_NAME=agent_for_teacher

# Optional: Các cấu hình khác
VECTOR_DB_DIR=./vector_db
LOGS_DIR=./logs
```

**Lưu ý về MongoDB URI:**
- Để sử dụng MongoDB Atlas (cloud): `mongodb+srv://username:password@cluster.mongodb.net/`
- Để sử dụng MongoDB local: `mongodb://username:password@localhost:27017/`

## 🚀 Chạy ứng dụng
```bash
python ui/app.py
```

Sau khi chạy, mở browser và truy cập: `http://127.0.0.1:7860` 


## 📖 Hướng dẫn sử dụng
### 1. Đăng nhập vào ứng dụng với tại khoản được cung cấp

### 2. Upload tài liệu
- Click "Upload a File" để chọn file PDF
- File sẽ hiện trong danh sách "Nguồn dữ liệu đã tải"
- Xem trạng thái xử lý trong khung "Trạng thái xử lý"

### 3. Trò chuyện với AI
- Sau khi xử lý thành công, nhập câu hỏi vào ô chat
- AI sẽ trả lời dựa trên nội dung tài liệu đã xử lý

## 🔧 Kiến trúc Code

```
services/
├── document_processing/        # 📄 Document Processing Domain
│   ├── __init__.py
│   ├── document_loader.py     # Document loading strategies
│   ├── document_chunker.py    # Document chunking strategies
│   ├── document_management_service.py  # Document workflow orchestration
│   ├── document_repository.py # Document storage & metadata
│   └── toc_extractor.py       # Table of contents extraction
│
├── rag/                       # 🤖 RAG Operations Domain
│   ├── __init__.py
│   ├── rag_service.py         # Main RAG orchestrator
│   ├── embedding_service.py   # Embedding management & strategies
│   ├── vector_service.py      # Vector store management
│   └── llm_service.py         # LLM integration
│
├── models.py                  # 📝 Shared data models
├── quiz_generation.py         # 🧩 Quiz generation service
└── ui_integration_service.py  # 🎮 UI bridge service

ui/
├── app.py                     # Main Gradio interface

config/
├── settings.py                # Application configuration
└── constants.py               # System constants
```


