# 🎓 A2A School - Nền tảng AI Giáo dục Thông minh

## 📋 Tổng quan dự án

**A2A School Platform** là một hệ thống giáo dục thông minh sử dụng AI để xử lý tài liệu PDF, tạo câu hỏi trắc nghiệm, phân tích dữ liệu học tập và hỗ trợ Q&A tự động. Nền tảng này được xây dựng với kiến trúc modular, sử dụng các AI agents chuyên biệt và vector search để cung cấp trải nghiệm học tập cá nhân hóa.

## 🚀 Tính năng chính

### 📄 Xử lý tài liệu PDF thông minh
- **Trích xuất cấu trúc tự động**: Phân tích bookmark, metadata, và cấu trúc tài liệu
- **Chunking thông minh**: Chia nhỏ tài liệu theo ngữ cảnh và chương mục
- **Vector Search**: Tìm kiếm ngữ nghĩa với FAISS và sentence-transformers
- **Mapping trang chính xác**: Theo dõi vị trí nội dung trong tài liệu gốc

### 🤖 AI Agents chuyên biệt
- **ContextAwareQuizAgent**: Tạo câu hỏi trắc nghiệm dựa trên nội dung
- **SummarizationAgent**: Tóm tắt và tạo mục lục chuyên nghiệp
- **AnalysisAgent**: Phân tích dữ liệu học tập với Pandas và Plotly
- **CardManager**: Quản lý các AI agent cards tùy chỉnh

### 💬 Hệ thống Q&A thông minh
- **RAG (Retrieval-Augmented Generation)**: Kết hợp tìm kiếm và sinh text
- **Lọc theo chương**: Tìm kiếm trong phạm vi cụ thể
- **Context-aware**: Trả lời dựa trên ngữ cảnh tài liệu

### 📊 Phân tích và báo cáo
- **Thống kê mô tả**: Phân tích dữ liệu học tập cơ bản
- **Biểu đồ tương tác**: Sử dụng Plotly cho visualization
- **Theo dõi tiến độ**: Lưu trữ lịch sử quiz và kết quả

## 🏗️ Kiến trúc hệ thống

```
src/
├── core/                          # Thành phần cốt lõi
│   ├── data_structures.py         # Định nghĩa cấu trúc dữ liệu
│   ├── llm_provider.py           # Kết nối với LLM (OpenRouter)
│   ├── professional_pdf_processor.py  # Xử lý PDF chuyên nghiệp
│   ├── vector_store.py           # Vector search với FAISS
│   └── document_reader_optimized.py   # Đọc tài liệu tối ưu
├── agents/                        # AI Agents
│   ├── analysis_agent.py         # Phân tích dữ liệu
│   ├── context_aware_quiz_agent.py   # Tạo câu hỏi thông minh
│   ├── summarization_agent.py    # Tóm tắt nội dung
│   └── card_manager.py          # Quản lý agent cards
├── services/                      # Business Logic
│   ├── document_service.py       # Xử lý tài liệu end-to-end
│   ├── quiz_service.py          # Quản lý quiz và câu hỏi
│   └── analysis_service.py      # Phân tích dữ liệu
├── db/                           # Quản lý dữ liệu
│   └── database_manager.py      # SQLite database operations
└── ui/                          # Giao diện người dùng
    └── app_interface.py         # Gradio UI components
```

## 📁 Chi tiết thư mục và modules

### 🔧 `/src/core/` - Thành phần cốt lõi

#### `data_structures.py`
Định nghĩa các cấu trúc dữ liệu chính:
- **`ChunkType`**: Enum phân loại loại nội dung (PARAGRAPH, TABLE, HEADING, LIST, CODE)
- **`ChapterInfo`**: Thông tin chương mục (số thứ tự, tiêu đề)
- **`AgenticChunk`**: Đơn vị nội dung thông minh với metadata đầy đủ

#### `llm_provider.py`
Quản lý kết nối với Large Language Models:
- **`LLMProvider`**: Wrapper cho OpenRouter API
- **Methods**: `generate()`, `stream_generate()`, `get_available_models()`
- **Features**: Rate limiting, error handling, fallback mechanisms

#### `professional_pdf_processor.py`
Xử lý PDF chuyên nghiệp với PyMuPDF:
- **`ProfessionalPDFProcessor`**: Main processor class
- **Key methods**:
  - `extract_pdf_structure()`: Trích xuất cấu trúc hoàn chỉnh
  - `_extract_bookmarks()`: Phân tích bookmark hierarchy
  - `_generate_enhanced_toc()`: Tạo mục lục bằng LLM
  - `_extract_page_mapped_content()`: Mapping nội dung theo trang

#### `vector_store.py`
Quản lý vector embeddings và tìm kiếm:
- **`VectorStore`**: FAISS-based similarity search
- **Key methods**:
  - `build_index()`: Xây dựng chỉ mục vector
  - `search()`: Tìm kiếm ngữ nghĩa với filtering
  - `build_index_for_doc()`: Index cho từng tài liệu
- **Models**: Alibaba-NLP/gte-multilingual-base

### 🤖 `/src/agents/` - AI Agents

#### `context_aware_quiz_agent.py`
Agent tạo câu hỏi thông minh:
- **`ContextAwareQuizAgent`**: Main quiz generation class
- **Key methods**:
  - `generate_quiz_from_pdf_structure()`: Tạo quiz từ cấu trúc PDF
  - `generate_content_based_questions()`: Câu hỏi dựa trên nội dung
  - `_extract_key_concepts()`: Trích xuất khái niệm chính
- **Question types**: Multiple choice, True/False, Fill-in-the-blank

#### `summarization_agent.py`
Agent tóm tắt và tạo mục lục:
- **`SummarizationAgent`**: Content summarization
- **Key methods**:
  - `create_hierarchical_summary()`: Tóm tắt phân cấp
  - `generate_table_of_contents()`: Tạo mục lục chuyên nghiệp
  - `extract_key_points()`: Trích xuất điểm chính

#### `analysis_agent.py`
Agent phân tích dữ liệu học tập:
- **`AnalysisAgent`**: Data analysis với Pandas
- **Key methods**:
  - `load_data()`: Tải dữ liệu từ file
  - `get_basic_statistics()`: Thống kê mô tả
  - `generate_visualizations()`: Tạo biểu đồ Plotly

### 🛠️ `/src/services/` - Business Logic

#### `document_service.py`
Service xử lý tài liệu end-to-end:
- **`DocumentService`**: Orchestrate document processing
- **Key methods**:
  - `process_document()`: Xử lý PDF từ upload đến ready-to-query
  - `_create_agentic_chunks_with_pages()`: Tạo chunks thông minh
  - `get_context_for_query()`: RAG context retrieval
- **Workflow**: Upload → Extract → Chunk → Index → Store

#### `quiz_service.py`
Service quản lý quiz và câu hỏi:
- **`QuizService`**: Quiz lifecycle management
- **Key methods**:
  - `generate_quiz()`: Tạo quiz từ tài liệu
  - `generate_quiz_for_chapter()`: Quiz theo chương
  - `get_quiz_history()`: Lịch sử quiz
- **Features**: Scoring, progress tracking, difficulty adjustment

### 🗄️ `/src/db/` - Data Management

#### `database_manager.py`
Quản lý SQLite database:
- **`DatabaseManager`**: Database operations
- **Tables**: documents, quizzes, user_progress, agent_cards
- **Key methods**:
  - `add_or_update_document()`: CRUD tài liệu
  - `get_quiz_history()`: Lịch sử quiz
  - `save_quiz_result()`: Lưu kết quả

### 🎨 `/src/ui/` - User Interface

#### `app_interface.py`
Giao diện Gradio:
- **`A2ASchoolApp`**: Main application class
- **Tabs**:
  - **📄 Hỏi đáp PDF**: Upload, Q&A, summarization
  - **📝 Tạo Quiz**: Quiz generation và management
  - **📊 Phân tích**: Data analysis và visualization
  - **🤖 Agent Cards**: Custom AI agents
- **Key methods**:
  - `_handle_pdf_upload()`: Xử lý upload file
  - `_handle_chat()`: Xử lý Q&A
  - `launch()`: Khởi chạy ứng dụng

## ⚙️ Cài đặt và sử dụng

### 1. Yêu cầu hệ thống
- Python 3.8+
- 4GB RAM minimum (8GB recommended cho embedding models)
- CUDA optional (for GPU acceleration)

### 2. Cài đặt dependencies
```bash
# Clone repository
git clone <repository-url>
cd test_AI

# Tạo virtual environment
python -m venv .venv-1
.venv-1\Scripts\activate  # Windows
# source .venv-1/bin/activate  # Linux/Mac

# Cài đặt packages
pip install -r requirements.txt
```

### 3. Cấu hình environment
Tạo file `.env`:
```env
OPENROUTER_API_KEY=sk-or-your-api-key-here
MODEL_NAME=openai/gpt-4o-mini
EMBEDDING_MODEL=Alibaba-NLP/gte-multilingual-base
```

### 4. Chạy ứng dụng
```bash
python main.py
```
Truy cập: http://127.0.0.1:7860

## 🔧 Configuration

### Environment Variables
- **`OPENROUTER_API_KEY`**: API key cho LLM provider
- **`MODEL_NAME`**: Tên model LLM (default: openai/gpt-4o-mini)
- **`EMBEDDING_MODEL`**: Model embedding (default: gte-multilingual-base)
- **`DB_PATH`**: Đường dẫn database (default: data/a2a_school.db)

### File Structure
```
data/
├── a2a_school.db           # SQLite database
├── agent_cards/            # Custom agent definitions
└── uploads/                # Uploaded PDF files
```

## 🧪 Testing và Development

### Chạy tests
```bash
# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# Test specific module
python -c "from src.core.vector_store import VectorStore; print('✅ Import OK')"
```

### Debug mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

## 📊 Performance và Optimization

### Memory Usage
- **Embedding Model**: ~500MB (gte-multilingual-base)
- **FAISS Index**: ~10MB per 10k chunks
- **LLM API**: Stateless (external)

### Recommendations
- **CPU**: Multi-core recommended cho embedding generation
- **RAM**: 8GB+ cho large documents
- **Storage**: SSD recommended cho database performance

## 🤝 Contributing

### Code Style
- Follow PEP 8
- Use type hints
- Document functions với docstrings
- Meaningful variable names

### Pull Request Process
1. Fork repository
2. Create feature branch
3. Add tests cho new features
4. Update documentation
5. Submit PR với clear description

## 📝 License

MIT License - Xem file LICENSE để biết chi tiết.

## 🆘 Support và Troubleshooting

### Common Issues

1. **Import Error**: 
   ```bash
   # Fix missing dependencies
   pip install -r requirements.txt
   ```

2. **Memory Error với embedding**:
   ```python
   # Reduce batch size in vector_store.py
   BATCH_SIZE = 32  # Instead of 64
   ```

3. **API Rate Limit**:
   ```python
   # Implement exponential backoff
   # Already handled in llm_provider.py
   ```

### Contact
- GitHub Issues: [Project Issues]
- Email: [support@a2aschool.com]
- Documentation: [Wiki Pages]

---

## 🎯 Roadmap

### Version 2.0 (Planned)
- [ ] Multi-modal support (images, videos)
- [ ] Real-time collaboration
- [ ] Advanced analytics dashboard
- [ ] Mobile app companion
- [ ] Multi-language support expansion

### Version 1.1 (Current)
- [x] PDF processing với PyMuPDF
- [x] Vector search với FAISS
- [x] Context-aware quiz generation
- [x] Gradio web interface
- [x] SQLite data persistence