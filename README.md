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

## 🏗️ Kiến trúc hệ thống

### Core Services
- **RagService**: Service chính orchestrate toàn bộ quy trình RAG
- **DocumentLoader**: Xử lý load tài liệu với nhiều strategy
- **DocumentChunker**: Chia nhỏ tài liệu theo các chiến lược khác nhau
- **VectorService**: Quản lý vector store và similarity search
- **UIIntegrationService**: Bridge giữa UI và core services

### Chunking Strategies
1. **ONE_PAGE_PER_CHUNK**: Mỗi trang là một chunk
2. **RECURSIVE_CHARACTER_TEXT_SPLITTER**: Chia theo ký tự với overlap
3. **LLM_SPLITTER**: Sử dụng LLM để chia theo ngữ nghĩa

## 🛠️ Cài đặt

### Requirements
```bash
pip install langchain langchain-community langchain-nvidia-ai-endpoints
pip install langchain-google-genai langchain-chroma
pip install gradio pymupdf pydantic pydantic-settings
```

### Environment Variables
Tạo file `.env` với nội dung:
```env
GOOGLE_API_KEY=your_google_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here
```

## 🚀 Chạy ứng dụng
```bash
python ui/app.py
```

Sau khi chạy, mở browser và truy cập: `http://127.0.0.1:7860`

## 📖 Hướng dẫn sử dụng

### 1. Cấu hình (Sidebar) 
- **Loader**: Chọn phương thức load tài liệu (Base, OCR, Base+OCR) 
- **Chunker**: Chọn chiến lược chia nhỏ tài liệu

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
├── rag_service.py              # Main RAG orchestrator
├── ui_integration_service.py   # UI bridge service
├── document_loader.py          # Document loading strategies
├── document_chunker.py         # Document chunking strategies
├── vector_service.py           # Vector store management
├── embedding_service.py        # Embedding management
└── llm_service.py             # LLM integration

ui/
├── app.py                     # Main Gradio interface

config/
├── settings.py                # Application configuration
└── constants.py               # System constants
```

## 🔄 Quy trình xử lý

1. **Upload**: User upload file qua Gradio interface
2. **Load**: DocumentLoader đọc và parse file PDF  
3. **Chunk**: DocumentChunker chia tài liệu thành chunks nhỏ
4. **Embed**: VectorService tạo embeddings và lưu vào vector store
5. **Query**: User đặt câu hỏi
6. **Retrieve**: Tìm kiếm chunks liên quan trong vector store
7. **Response**: Trả về thông tin tìm được
