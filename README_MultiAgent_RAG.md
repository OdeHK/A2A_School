# Multi-Agent RAG System for School Management
# Hệ thống RAG đa agent cho quản lý trường học

## 🎯 Tổng quan

Hệ thống RAG đa agent được thiết kế đặc biệt cho môi trường giáo dục với 3 agents chuyên biệt:

### 🎓 Agent 1: Student Support Agent (Hỗ trợ học sinh)
**Vai trò:** Trợ lý học tập thông minh

**Chức năng chính:**
- 💡 Hỏi đáp kiến thức (giải bài tập toán, TA, lý, hóa, văn)
- 📚 Đề xuất tài liệu học phù hợp theo trình độ
- ⏰ Nhắc nhở lịch học, bài kiểm tra, deadline
- 📝 Lập kế hoạch học tập cá nhân

**Ví dụ sử dụng:**
```python
# Trả lời câu hỏi học thuật
answer = student_agent.answer_academic_question(
    "Eigenvalue của ma trận là gì?", 
    subject="toán"
)

# Đề xuất tài liệu
materials = student_agent.recommend_study_materials(
    subject="toán", 
    difficulty_level="medium"
)

# Thiết lập nhắc nhở
reminder = student_agent.set_study_reminder(
    student_id="HS001",
    reminder_type="exam",
    subject="toán", 
    datetime_str="2024-12-25T09:00:00",
    note="Kiểm tra chương ma trận"
)
```

### 👨‍🏫 Agent 2: Teacher Support Agent (Hỗ trợ giáo viên)
**Vai trò:** Trợ lý sư phạm chuyên nghiệp

**Chức năng chính:**
- 👥 Gợi ý cách chia nhóm học sinh (theo trình độ học tập/phong cách học)
- 📖 Đề xuất phương pháp dạy học hiệu quả
- 📋 Cung cấp tài liệu giảng dạy và giáo án
- 📅 Nhắc nhở lịch dạy, kiểm tra, họp

**Ví dụ sử dụng:**
```python
# Gợi ý chia nhóm
grouping = teacher_agent.suggest_student_grouping(
    student_data=[
        {"student_id": "HS001", "average_score": 8.5, "learning_style": "visual"},
        {"student_id": "HS002", "average_score": 6.2, "learning_style": "auditory"}
    ],
    grouping_criteria="academic_level"
)

# Đề xuất phương pháp dạy
method = teacher_agent.suggest_teaching_method(
    subject="toán",
    class_level="10", 
    topic="ma trận"
)

# Đề xuất tài liệu giảng dạy
materials = teacher_agent.suggest_teaching_materials(
    subject="toán",
    topic="ma trận"
)
```

### 📊 Agent 3: Data Analysis Agent (Phân tích dữ liệu)
**Vai trò:** Chuyên gia phân tích dữ liệu giáo dục

**Chức năng chính:**
- 📈 Phân tích kết quả kiểm tra của lớp
- 📋 Tổng hợp điểm số theo lớp, môn, khối
- ⚠️ Phát hiện học sinh cần hỗ trợ
- 🔮 Dự đoán xu hướng học tập của lớp, khối

**Ví dụ sử dụng:**
```python
# Phân tích hiệu suất lớp
analysis = data_agent.analyze_class_performance(class_data)

# Tổng hợp theo tiêu chí
aggregation = data_agent.aggregate_scores_by_criteria(
    grade_data, 
    group_by="class"
)

# Phát hiện học sinh cần hỗ trợ
at_risk = data_agent.identify_students_need_support(
    student_data,
    criteria={
        "low_average": 5.0,
        "failing_subjects": 2,
        "attendance_threshold": 80
    }
)

# Dự đoán xu hướng
trends = data_agent.predict_learning_trends(
    historical_data,
    prediction_period=6
)
```

## 🏗️ Kiến trúc hệ thống

```
Multi-Agent RAG System
├── BaseRAGAgent (Lớp cơ sở)
│   ├── Chunker (Phân đoạn văn bản)
│   ├── Embedder (Vector hóa)
│   ├── LLM (Mô hình ngôn ngữ)
│   └── Vector Store (Chroma DB)
│
├── StudentSupportAgent
│   ├── Knowledge Base theo môn học
│   ├── Study Schedules & Reminders
│   └── Personalized Recommendations
│
├── TeacherSupportAgent  
│   ├── Teaching Resources
│   ├── Grouping Algorithms
│   └── Teaching Schedules
│
├── DataAnalysisAgent
│   ├── Grade Database
│   ├── Statistical Analysis
│   └── Trend Prediction
│
└── MultiAgentRAGSystem (Router)
    └── Auto-routing based on query type
```

## 🚀 Cài đặt và chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Requirements chính:**
- langchain-community
- chromadb
- pandas
- sentence-transformers
- numpy
- scikit-learn

### 2. Chạy demo

```bash
cd src/
python demo_multi_agent_rag.py
```

### 3. Setup MCP Server (Tùy chọn)

```bash
# Tạo cấu hình MCP
python mcp_config.py

# Setup MCP servers
python setup_mcp_servers.py
```

## 📁 Cấu trúc files

```
src/
├── multi_agent_rag.py          # Core system với 3 agents
├── demo_multi_agent_rag.py     # Demo script đầy đủ
├── mcp_config.py               # Cấu hình MCP server
├── rag.py                      # RAG system gốc (tham khảo)
├── analysis_agent.py           # Agent gốc (tham khảo)
└── ...                         # Các files khác
```

## 🎮 Hướng dẫn sử dụng

### Khởi tạo hệ thống

```python
from multi_agent_rag import MultiAgentRAGSystem
from sentence_transformer_embedder import SentenceTransformerEmbedder
from openrouter_llm import OpenRouterLLM

# Setup components
chunker = MockChunker()  # Replace với chunker thực
embedder = SentenceTransformerEmbedder()
llm = OpenRouterLLM()

# Khởi tạo hệ thống
system = MultiAgentRAGSystem(chunker, embedder, llm)
```

### Setup dữ liệu cho từng agent

```python
# 1. Setup Student Agent
student_docs = {
    "toán": [Document(page_content="Nội dung toán học...")],
    "lý": [Document(page_content="Nội dung vật lý...")],
    # ... other subjects
}
system.student_agent.setup_knowledge_base(student_docs)

# 2. Setup Teacher Agent  
teaching_docs = [Document(page_content="Tài liệu sư phạm...")]
system.teacher_agent.setup_teaching_resources(teaching_docs)

# 3. Setup Data Agent
grade_docs = [Document(page_content="Dữ liệu điểm số...")]
system.data_agent.setup_grade_database(grade_docs)
```

### Routing tự động

```python
# Hệ thống tự động phát hiện loại query và chọn agent phù hợp
result = system.route_query("Làm sao giải phương trình bậc hai?")
# → Chọn StudentSupportAgent

result = system.route_query("Cách chia nhóm học sinh hiệu quả?") 
# → Chọn TeacherSupportAgent

result = system.route_query("Phân tích xu hướng điểm lớp 10A1")
# → Chọn DataAnalysisAgent
```

## 🔧 Tùy chỉnh và mở rộng

### Thêm môn học mới

```python
# Thêm môn học cho Student Agent
new_subject_docs = [Document(page_content="Nội dung môn mới...")]
system.student_agent.subject_knowledge["môn_mới"] = {
    'vector_store': vector_store,
    'retriever': retriever
}
```

### Tùy chỉnh thuật toán chia nhóm

```python
# Override method trong TeacherSupportAgent
class CustomTeacherAgent(TeacherSupportAgent):
    def suggest_student_grouping(self, student_data, criteria="custom"):
        # Implement custom grouping logic
        pass
```

### Thêm metrics phân tích mới

```python
# Extend DataAnalysisAgent
class ExtendedDataAgent(DataAnalysisAgent):
    def analyze_engagement_metrics(self, engagement_data):
        # Custom engagement analysis
        pass
```

## 🔌 Tích hợp MCP (Model Context Protocol)

### Agent Cards

Hệ thống cung cấp 3 agent cards để tích hợp với MCP:

1. **🎓 Student Support Card**
   - Server: `student_support_server`
   - Endpoints: `/ask`, `/recommend`, `/remind`, `/plan`

2. **👨‍🏫 Teacher Support Card**
   - Server: `teacher_support_server`  
   - Endpoints: `/group`, `/method`, `/materials`, `/schedule`

3. **📊 Data Analysis Card**
   - Server: `data_analysis_server`
   - Endpoints: `/analyze`, `/aggregate`, `/at-risk`, `/trends`

### Cấu hình MCP

```json
{
  "servers": {
    "student_support_server": {
      "command": "python",
      "args": ["mcp_server_student.py"],
      "capabilities": ["ask_questions", "recommend_materials", "set_reminders"]
    }
  }
}
```

## 📊 Ví dụ dữ liệu

### Dữ liệu học sinh
```json
{
  "student_id": "HS001",
  "name": "Nguyễn Văn A", 
  "average_score": 8.5,
  "learning_style": "visual",
  "subjects": ["toán", "lý", "hóa"]
}
```

### Dữ liệu điểm số
```json
{
  "student_id": "HS001",
  "subject": "toán",
  "score": 8.5,
  "exam_date": "2024-12-15",
  "class": "10A1",
  "attendance": 95
}
```

## 🧪 Testing

```bash
# Chạy all tests
python -m pytest tests/

# Test từng agent riêng
python test_student_agent.py
python test_teacher_agent.py  
python test_data_agent.py
```

## 🚦 Roadmap

### Version 1.1 (Q1 2025)
- [ ] Tích hợp với database thực (PostgreSQL/MySQL)
- [ ] API REST endpoints
- [ ] Web dashboard
- [ ] Multi-language support

### Version 1.2 (Q2 2025)  
- [ ] Real-time notifications
- [ ] Advanced analytics với ML models
- [ ] Parent portal integration
- [ ] Mobile app support

### Version 2.0 (Q3 2025)
- [ ] AI-powered curriculum recommendations
- [ ] Predictive analytics for student success
- [ ] Integration với Learning Management Systems
- [ ] Advanced personalization

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Support

- 📧 Email: support@school-rag.com
- 💬 Discord: [School RAG Community](https://discord.gg/school-rag)
- 📖 Docs: [https://docs.school-rag.com](https://docs.school-rag.com)

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - RAG framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [OpenRouter](https://openrouter.ai/) - LLM APIs

---

## 📋 TODO List

- [x] Core multi-agent RAG system
- [x] Student Support Agent
- [x] Teacher Support Agent  
- [x] Data Analysis Agent
- [x] Demo script
- [x] MCP configuration
- [ ] Unit tests
- [ ] API documentation
- [ ] Deployment guide
- [ ] Performance benchmarks

**Tạo bởi:** Hiep - Team 3
**Ngày:** August 25, 2025
**Version:** 1.0.0
