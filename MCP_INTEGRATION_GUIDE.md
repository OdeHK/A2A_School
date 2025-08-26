# Hướng dẫn tích hợp MCP với Multi-Agent RAG System

## 🎯 Tổng quan

Hệ thống Multi-Agent RAG đã tạo 3 agent cards sẵn sàng tích hợp với MCP (Model Context Protocol):

1. **🎓 Student Support Card** - Hỗ trợ học sinh
2. **👨‍🏫 Teacher Support Card** - Hỗ trợ giáo viên  
3. **📊 Data Analysis Card** - Phân tích dữ liệu

## 📁 Files đã tạo

```
src/
├── mcp_manifest.json       # Schema đầy đủ cho MCP
├── agent_cards.json        # Cấu hình agent cards
├── setup_mcp_servers.py    # Script setup MCP servers
├── multi_agent_rag.py      # Hệ thống core (đầy đủ)
├── simple_demo_rag.py      # Demo đơn giản (đã test)
└── mcp_config.py           # Cấu hình MCP
```

## 🚀 Cách tích hợp với MCP

### Bước 1: Setup MCP Servers

```bash
# Cài đặt dependencies (nếu chưa có)
pip install fastapi uvicorn python-multipart

# Chạy setup MCP servers
python setup_mcp_servers.py
```

### Bước 2: Tạo MCP Server Files

Tạo file `mcp_server_student.py`:

```python
#!/usr/bin/env python3
import asyncio
import json
from simple_demo_rag import SimpleMultiAgentRAGSystem, create_sample_data

async def main():
    # Khởi tạo hệ thống
    system = SimpleMultiAgentRAGSystem()
    student_docs, _, _ = create_sample_data()
    system.student_agent.setup_knowledge_base(student_docs)
    
    # MCP Protocol handling
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, input)
            request = json.loads(line)
            
            if request.get("method") == "ask_question":
                params = request.get("params", {})
                response = system.student_agent.answer_academic_question(
                    params.get("question", ""),
                    params.get("subject")
                )
                
                result = {
                    "id": request.get("id"),
                    "result": {"answer": response}
                }
            else:
                result = {
                    "id": request.get("id"),
                    "error": {"code": -32601, "message": "Method not found"}
                }
            
            print(json.dumps(result))
            
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            error_result = {
                "id": request.get("id", None),
                "error": {"code": -32603, "message": str(e)}
            }
            print(json.dumps(error_result))

if __name__ == "__main__":
    asyncio.run(main())
```

### Bước 3: Cấu hình trong VS Code

Thêm vào `settings.json` của VS Code:

```json
{
  "mcp.servers": {
    "student-support": {
      "command": "python",
      "args": ["D:/Project_school/A2A_School/src/mcp_server_student.py"],
      "env": {}
    },
    "teacher-support": {
      "command": "python", 
      "args": ["D:/Project_school/A2A_School/src/mcp_server_teacher.py"],
      "env": {}
    },
    "data-analysis": {
      "command": "python",
      "args": ["D:/Project_school/A2A_School/src/mcp_server_data.py"],
      "env": {}
    }
  }
}
```

## 🎮 Sử dụng Agent Cards

### 🎓 Student Support Card

**Capabilities:**
- Trả lời câu hỏi học thuật (Toán, Lý, Hóa, Anh, Văn)
- Đề xuất tài liệu học phù hợp
- Nhắc nhở lịch học và kiểm tra
- Lập kế hoạch học tập

**Example queries:**
```
- "Giải phương trình x² + 5x + 6 = 0"
- "Eigenvalue của ma trận là gì?"
- "Đề xuất tài liệu ôn tập Toán nâng cao"
- "Nhắc tôi kiểm tra Lý thứ 6 tuần sau"
```

### 👨‍🏫 Teacher Support Card

**Capabilities:**
- Gợi ý chia nhóm học sinh theo trình độ
- Đề xuất phương pháp giảng dạy
- Cung cấp tài liệu giảng dạy
- Quản lý lịch dạy

**Example queries:**
```
- "Chia 30 học sinh thành nhóm theo trình độ"
- "Phương pháp dạy hình học lớp 10"
- "Tài liệu giảng dạy chủ đề ma trận"
- "Lập lịch kiểm tra 15 phút"
```

### 📊 Data Analysis Card

**Capabilities:**
- Phân tích kết quả kiểm tra lớp
- Tổng hợp điểm số theo tiêu chí
- Phát hiện học sinh cần hỗ trợ
- Dự đoán xu hướng học tập

**Example queries:**
```
- "Phân tích điểm kiểm tra Toán lớp 10A1"
- "Tổng hợp điểm TB theo khối lớp 10"
- "Học sinh nào cần hỗ trợ đặc biệt?"
- "Xu hướng học lực 6 tháng tới"
```

## 📊 Data Format

### Input cho Student Agent
```json
{
  "question": "Eigenvalue là gì?",
  "subject": "toán",
  "student_id": "HS001"
}
```

### Input cho Teacher Agent
```json
{
  "student_data": [
    {
      "student_id": "HS001",
      "average_score": 8.5,
      "learning_style": "visual"
    }
  ],
  "criteria": "academic_level"
}
```

### Input cho Data Agent
```json
{
  "class_data": [
    {
      "student_id": "HS001",
      "subject": "toán", 
      "score": 8.5,
      "exam_date": "2024-12-15"
    }
  ]
}
```

## 🔧 Tùy chỉnh

### Thay đổi responses của LLM

Sửa file `simple_demo_rag.py`, class `SimpleLLM`:

```python
class SimpleLLM:
    def __init__(self):
        self.responses = {
            "toán": "Custom response cho toán học...",
            "lý": "Custom response cho vật lý...",
            # Thêm responses mới
        }
```

### Thêm môn học mới

```python
# Trong create_sample_data()
student_docs["sinh"] = [
    SimpleDocument("Tế bào là đơn vị cơ bản của sự sống"),
    SimpleDocument("DNA chứa thông tin di truyền")
]
```

### Tùy chỉnh thuật toán chia nhóm

```python
# Trong TeacherSupportAgent
def suggest_student_grouping(self, student_data, criteria="custom"):
    # Implement custom logic
    pass
```

## 🧪 Testing

### Test đơn lẻ từng agent:

```bash
# Test Student Agent
python -c "
from simple_demo_rag import *
system = SimpleMultiAgentRAGSystem()
docs, _, _ = create_sample_data()
system.student_agent.setup_knowledge_base(docs)
print(system.student_agent.answer_academic_question('Eigenvalue là gì?', 'toán'))
"
```

### Test routing:

```bash
python -c "
from simple_demo_rag import *
system = SimpleMultiAgentRAGSystem()
result = system.route_query('Làm sao giải phương trình?')
print(f'Agent: {result[\"agent\"]}')
"
```

## 🚦 Next Steps

### Version 1.1 - Tích hợp thực tế
- [ ] Thay SimpleLLM bằng OpenAI/Anthropic API
- [ ] Tích hợp Sentence Transformers cho embeddings
- [ ] Kết nối database thực (PostgreSQL/SQLite)
- [ ] API REST endpoints

### Version 1.2 - Advanced Features  
- [ ] Multi-language support
- [ ] Real-time notifications
- [ ] Web dashboard
- [ ] Mobile app

### Version 2.0 - AI Enhancement
- [ ] Fine-tuned models cho giáo dục
- [ ] Advanced analytics
- [ ] Personalization engine
- [ ] Integration với LMS

## ⚠️ Lưu ý quan trọng

1. **Dependencies**: Demo đơn giản chỉ cần pandas, version đầy đủ cần langchain
2. **Data Privacy**: Cần bảo mật thông tin học sinh khi deploy
3. **Scalability**: Test với dữ liệu nhỏ, cần optimize cho production
4. **Accuracy**: SimpleLLM chỉ cho demo, cần LLM thực cho độ chính xác cao

## 🆘 Troubleshooting

### Lỗi import langchain
```bash
# Sử dụng simple_demo_rag.py thay vì multi_agent_rag.py
python simple_demo_rag.py
```

### MCP server không kết nối
```bash
# Kiểm tra path đúng trong settings.json
# Đảm bảo Python executable path chính xác
```

### Agent không trả lời đúng
```bash
# Kiểm tra keywords trong _detect_user_type()
# Thêm keywords mới cho việc routing
```

## 📞 Support

- 📁 Files: `d:\Project_school\A2A_School\src\`
- 🎯 Demo: `python simple_demo_rag.py`
- 📖 Docs: `README_MultiAgent_RAG.md`

---

**Được tạo bởi:** Team 3 - Hiep  
**Ngày:** August 25, 2025  
**Status:** ✅ Ready for MCP Integration
