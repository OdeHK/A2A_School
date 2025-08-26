# backend_main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from io import BytesIO
import os
import traceback
from dotenv import load_dotenv # <-- 1. Import thư viện dotenv
from src.analysis_agent import AnalysisAgent # <-- Thêm import
from pydantic import BaseModel
# --- 2. Nạp các biến môi trường từ file .env ---
# Lệnh này phải được gọi trước khi truy cập os.environ
load_dotenv()
# ---------------------------------------------

# Import các module của bạn từ thư mục src
from src.document_reader import DocumentReader
from src.rag import RAGManager
from src.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.openrouter_llm import OpenRouterLLM
from src import database_manager
from src.simple_demo_rag import SimpleMultiAgentRAGSystem, create_sample_data
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(title="AI Backend Service")

# --- Khởi tạo các đối tượng AI (Singleton Pattern) ---
@app.on_event("startup")
async def startup_event():
    """Hàm này sẽ chạy một lần duy nhất khi server bắt đầu."""
    global document_reader, rag_manager, analysis_agent, llm, embedder, multi_agent_system
    
    try:
        # Load và validate environment variables
        model_name = os.getenv("OPENROUTER_MODEL")
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Nếu không tìm thấy trong .env, thử load từ secrets.toml
        if not api_key:
            try:
                import toml
                secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
                if os.path.exists(secrets_path):
                    with open(secrets_path, 'r') as f:
                        secrets = toml.load(f)
                        api_key = secrets.get('OPENROUTER_API_KEY')
                        print(f"✅ Đã load API key từ secrets.toml")
            except Exception as e:
                print(f"⚠️ Không thể đọc secrets.toml: {e}")
        
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY không được cấu hình trong file .env hoặc .streamlit/secrets.toml")
        
        # Validate API key format
        if not api_key.startswith('sk-or-v1-'):
            print(f"⚠️ API key có thể không đúng format OpenRouter: {api_key[:10]}...")
        else:
            print(f"✅ API key format hợp lệ: {api_key[:15]}...")
            
        if not model_name:
            model_name = "openai/gpt-oss-20b:free"  # Fallback to a stable model
            print(f"⚠️ OPENROUTER_MODEL không được cấu hình, sử dụng model mặc định: {model_name}")
        
        # Validate grades.csv exists
        grades_path = os.path.join(os.path.dirname(__file__), 'data', 'grades.csv')
        if not os.path.exists(grades_path):
            raise FileNotFoundError(f"File grades.csv không tồn tại tại: {grades_path}")
        
        # Initialize components with better error handling
        print("🔄 Đang khởi tạo OpenRouter LLM...")
        llm = OpenRouterLLM(model_name=model_name, api_key=api_key)
        
        print("🔄 Đang khởi tạo Sentence Transformer...")
        embedder = SentenceTransformerEmbedder()
        
        print("🔄 Đang khởi tạo Document Reader...")
        document_reader = DocumentReader()
        
        # Test OpenRouter connection
        test_prompt = "This is a test message. Reply with 'OK' if you receive this."
        try:
            response = llm.invoke(test_prompt)
            print("✅ Kết nối OpenRouter API thành công!")
        except Exception as e:
            print(f"⚠️ Cảnh báo: Không thể kết nối đến OpenRouter API: {str(e)}")
            print("⚠️ Hệ thống sẽ tiếp tục khởi động nhưng các chức năng LLM có thể không hoạt động.")
        
        # Initialize chunker CHỈ MỘT LẦN
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            chunker = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
        except ImportError:
            # Fallback to simple chunker if langchain not available
            class SimpleChunker:
                def __init__(self, chunk_size=1000):
                    self.chunk_size = chunk_size
                
                def split_text(self, text):
                    chunks = []
                    for i in range(0, len(text), self.chunk_size):
                        chunks.append(text[i:i + self.chunk_size])
                    return chunks
            
            chunker = SimpleChunker()
        
        print("🔄 Đang khởi tạo RAG Manager...")
        rag_manager = RAGManager(chunker=chunker, llm=llm, embedder=embedder)
        
        print("🔄 Đang khởi tạo Multi-Agent RAG System...")
        multi_agent_system = SimpleMultiAgentRAGSystem()
        
        # Setup sample data for multi-agent system
        student_docs, student_data, grade_data = create_sample_data()
        multi_agent_system.student_agent.setup_knowledge_base(student_docs)
        
        print("🔄 Đang khởi tạo Analysis Agent...")
        analysis_agent = AnalysisAgent(data_path=grades_path)
        
        # Verify DataFrame is loaded correctly
        if analysis_agent.df.empty:
            raise ValueError("Grades data is empty")
        if 'subject' not in analysis_agent.df.columns:
            raise ValueError(f"Required column 'subject' not found. Available columns: {', '.join(analysis_agent.df.columns)}")
        
        print("✅ Agent Phân tích đã sẵn sàng.")
        
        database_manager.create_tables()
        print("✅ Khởi tạo hệ thống thành công!")
        print("✅ Backend service đã sẵn sàng.")
        
    except Exception as e:
        print(f"❌ Lỗi khi khởi động hệ thống: {str(e)}")
        raise

# --- Endpoint Mới cho Agent Giáo viên ---
class SuggestionRequest(BaseModel):
    subject: str
    lesson_topic: str

@app.post("/teacher/generate-suggestions/")
async def handle_teaching_suggestions(request: SuggestionRequest):
    """
    Endpoint A2A: Lấy dữ liệu từ Agent Phân tích, sau đó dùng LLM để tạo gợi ý.
    """
    # 1. Gọi Agent Phân tích để lấy dữ liệu
    class_overview = analysis_agent.get_class_overview(request.subject)
    student_groups = analysis_agent.group_students_by_level(request.subject)

    if not class_overview or not student_groups:
        raise HTTPException(status_code=404, detail=f"Không có dữ liệu cho môn học {request.subject}")

    # 2. Xây dựng prompt cho LLM
    suggestion_prompt = f"""
    BẠN LÀ MỘT CHUYÊN GIA TƯ VẤN SƯ PHẠM.
    Dựa vào dữ liệu phân tích học lực của lớp và chủ đề bài học, hãy đưa ra đề xuất chi tiết cho giáo viên.

    **Chủ đề bài học sắp tới:** {request.lesson_topic}
    **Môn học:** {request.subject}

    **Dữ liệu Phân tích:**
    - Thống kê chung của lớp: {json.dumps(class_overview, indent=2, ensure_ascii=False)}
    - Phân nhóm học sinh theo trình độ: {json.dumps(student_groups, indent=2, ensure_ascii=False)}

    **Yêu cầu:**
    1.  **Gợi ý cách chia nhóm:** Đề xuất cách chia nhóm và hoạt động phù hợp cho từng nhóm (Giỏi, Khá, Trung bình, Yếu).
    2.  **Đề xuất phương pháp dạy:** Gợi ý 2-3 phương pháp giảng dạy cụ thể cho chủ đề "{request.lesson_topic}".
    3.  **Đề xuất tài liệu giảng dạy:** Gợi ý các loại tài liệu bổ sung mà giáo viên nên chuẩn bị.

    Hãy trình bày một cách rõ ràng và có tính hành động cao.
    """
    
    # 3. Gọi LLM và trả về kết quả
    try:
        suggestion = rag_manager.llm.invoke(suggestion_prompt)
        return {"suggestion": suggestion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Các API Endpoint (Giữ nguyên) ---
@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    try:
        file_bytes = BytesIO(await file.read())
        filetype = file.filename.split('.')[-1].lower()
        document_text = document_reader.read(file_bytes, filetype)
        if not document_text:
            raise HTTPException(status_code=400, detail="Không thể đọc nội dung file.")
        rag_manager.setup_with_text(document_text)
        print("✅ Đã xây dựng xong vector store và retriever.")
        return {"status": "success", "filename": file.filename, "message": "Tài liệu đã được xử lý."}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    question: str

@app.post("/query/")
async def handle_query(request: QueryRequest):
    if not rag_manager.retriever:
        raise HTTPException(status_code=400, detail="Chưa có tài liệu nào được xử lý.")
    answer = rag_manager.query(request.question)
    return {"answer": answer}

class QuizRequest(BaseModel):
    num_questions: int = 5
    subject: str = "Quiz chung"

@app.post("/generate-quiz/")
async def handle_quiz_generation(request: QuizRequest):
    try:
        if not rag_manager.retriever:
            raise HTTPException(
                status_code=400, 
                detail="Vui lòng tải lên và xử lý tài liệu trước khi tạo quiz."
            )
            
        quiz_data = rag_manager.generate_quiz(request.num_questions)
        
        # Kiểm tra nếu generate_quiz trả về error message (string)
        if isinstance(quiz_data, str):
            raise HTTPException(
                status_code=500,
                detail=quiz_data  # Sử dụng error message từ RAG
            )
            
        # Kiểm tra nếu không có dữ liệu quiz
        if not quiz_data:
            raise HTTPException(
                status_code=500,
                detail="Không thể tạo quiz. Vui lòng thử lại sau."
            )
            
        if not isinstance(quiz_data, list):
            raise HTTPException(
                status_code=500,
                detail=f"Định dạng quiz không hợp lệ: {type(quiz_data)}"
            )
            
        if len(quiz_data) == 0:
            raise HTTPException(
                status_code=500,
                detail="Không thể tạo câu hỏi từ tài liệu này. Vui lòng thử tài liệu khác."
            )
            
        # Validate quiz format
        for i, question in enumerate(quiz_data):
            if not all(key in question for key in ['question', 'choices', 'correct_answer']):
                raise HTTPException(
                    status_code=500,
                    detail=f"Câu hỏi {i+1} thiếu thông tin bắt buộc."
                )
        
        # Save quiz to database
        try:
            quiz_id = database_manager.save_quiz(quiz_data, subject=request.subject)
        except Exception as db_error:
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi khi lưu quiz vào database: {str(db_error)}"
            )
            
        return {
            "status": "success",
            "message": f"Đã tạo thành công {len(quiz_data)} câu hỏi",
            "quiz_id": quiz_id,
            "quiz_data": quiz_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Lỗi không mong muốn khi tạo quiz: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Đã xảy ra lỗi không mong muốn. Vui lòng thử lại sau."
        )

class ExpertChatRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/student/expert-chat/")
async def handle_expert_chat(request: ExpertChatRequest):
    if not rag_manager.retriever:
        raise HTTPException(status_code=400, detail="Chưa có tài liệu nào được xử lý.")
    answer = rag_manager.act_as_expert(request.question, request.chat_history)
    return {"answer": answer}

@app.get("/teacher/class-overview/{subject}")
async def get_class_overview(subject: str):
    """Endpoint để lấy thống kê tổng quan về lớp học."""
    class_data = analysis_agent.get_class_overview(subject)
    if not class_data:
        raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu cho môn học này")
    return class_data

# ==================== MULTI-AGENT RAG ENDPOINTS ====================

class StudentQuestionRequest(BaseModel):
    question: str
    subject: str = None
    student_id: str = "default_student"

class TeacherGroupingRequest(BaseModel):
    student_data: list
    criteria: str = "academic_level"

class TeacherMethodRequest(BaseModel):
    subject: str
    class_level: str
    topic: str

class DataAnalysisRequest(BaseModel):
    class_data: list

class StudentReminderRequest(BaseModel):
    student_id: str
    reminder_type: str
    subject: str
    datetime_str: str
    note: str = ""

@app.post("/agent/student/ask")
async def student_ask_question(request: StudentQuestionRequest):
    """Endpoint cho Student Support Agent - Trả lời câu hỏi học thuật"""
    try:
        answer = multi_agent_system.student_agent.answer_academic_question(
            request.question, 
            request.subject
        )
        return {
            "status": "success",
            "agent": "student_support",
            "answer": answer,
            "detected_subject": multi_agent_system.student_agent._detect_subject(request.question)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Student Agent: {str(e)}")

@app.post("/agent/student/recommend")
async def student_recommend_materials(subject: str, difficulty: str = "medium"):
    """Endpoint cho Student Support Agent - Đề xuất tài liệu học"""
    try:
        recommendations = multi_agent_system.student_agent.recommend_study_materials(
            subject, difficulty
        )
        return {
            "status": "success",
            "agent": "student_support",
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Student Agent: {str(e)}")

@app.post("/agent/student/reminder")
async def student_set_reminder(request: StudentReminderRequest):
    """Endpoint cho Student Support Agent - Thiết lập nhắc nhở"""
    try:
        result = multi_agent_system.student_agent.set_study_reminder(
            request.student_id,
            request.reminder_type,
            request.subject,
            request.datetime_str,
            request.note
        )
        return {
            "status": "success",
            "agent": "student_support",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Student Agent: {str(e)}")

@app.get("/agent/student/reminders/{student_id}")
async def student_get_reminders(student_id: str, days_ahead: int = 7):
    """Endpoint cho Student Support Agent - Lấy nhắc nhở sắp tới"""
    try:
        reminders = multi_agent_system.student_agent.get_upcoming_reminders(
            student_id, days_ahead
        )
        return {
            "status": "success",
            "agent": "student_support",
            "reminders": reminders
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Student Agent: {str(e)}")

@app.post("/agent/teacher/group")
async def teacher_suggest_grouping(request: TeacherGroupingRequest):
    """Endpoint cho Teacher Support Agent - Gợi ý chia nhóm học sinh"""
    try:
        grouping = multi_agent_system.teacher_agent.suggest_student_grouping(
            request.student_data,
            request.criteria
        )
        return {
            "status": "success",
            "agent": "teacher_support",
            "grouping": grouping
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Teacher Agent: {str(e)}")

@app.post("/agent/teacher/method")
async def teacher_suggest_method(request: TeacherMethodRequest):
    """Endpoint cho Teacher Support Agent - Đề xuất phương pháp dạy"""
    try:
        method = multi_agent_system.teacher_agent.suggest_teaching_method(
            request.subject,
            request.class_level,
            request.topic
        )
        return {
            "status": "success",
            "agent": "teacher_support",
            "method": method
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Teacher Agent: {str(e)}")

@app.post("/agent/teacher/materials")
async def teacher_suggest_materials(subject: str, topic: str, material_type: str = "all"):
    """Endpoint cho Teacher Support Agent - Đề xuất tài liệu giảng dạy"""
    try:
        materials = multi_agent_system.teacher_agent.suggest_teaching_materials(
            subject, topic, material_type
        )
        return {
            "status": "success",
            "agent": "teacher_support",
            "materials": materials
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Teacher Agent: {str(e)}")

@app.post("/agent/data/analyze")
async def data_analyze_class(request: DataAnalysisRequest):
    """Endpoint cho Data Analysis Agent - Phân tích hiệu suất lớp"""
    try:
        analysis = multi_agent_system.data_agent.analyze_class_performance(
            request.class_data
        )
        return {
            "status": "success",
            "agent": "data_analysis",
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Data Agent: {str(e)}")

@app.post("/agent/data/at-risk")
async def data_identify_at_risk(request: DataAnalysisRequest):
    """Endpoint cho Data Analysis Agent - Phát hiện học sinh cần hỗ trợ"""
    try:
        at_risk = multi_agent_system.data_agent.identify_students_need_support(
            request.class_data
        )
        return {
            "status": "success",
            "agent": "data_analysis",
            "at_risk_students": at_risk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Data Agent: {str(e)}")

@app.post("/agent/data/trends")
async def data_predict_trends(request: DataAnalysisRequest, prediction_period: int = 6):
    """Endpoint cho Data Analysis Agent - Dự đoán xu hướng học tập"""
    try:
        trends = multi_agent_system.data_agent.predict_learning_trends(
            request.class_data, prediction_period
        )
        return {
            "status": "success",
            "agent": "data_analysis",
            "trends": trends
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Data Agent: {str(e)}")

@app.get("/agent/route")
async def route_query(query: str):
    """Endpoint để routing tự động query đến agent phù hợp"""
    try:
        result = multi_agent_system.route_query(query)
        user_type = multi_agent_system._detect_user_type(query)
        
        return {
            "status": "success",
            "query": query,
            "detected_user_type": user_type,
            "selected_agent": result["agent"],
            "routing_info": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Routing: {str(e)}")

# ==================== END MULTI-AGENT RAG ENDPOINTS ====================
        
    distribution_data = analysis_agent.group_students_by_level(subject)
    return {
        "mean": class_data.get("mean", 0),
        "max": class_data.get("max", 0),
        "min": class_data.get("min", 0),
        "distribution": distribution_data
    }

class ResearchTopicRequest(BaseModel):
    num_topics: int = 3

@app.post("/student/suggest-topics/")
async def handle_research_topics(request: ResearchTopicRequest):
    if not rag_manager.retriever:
        raise HTTPException(status_code=400, detail="Chưa có tài liệu nào được xử lý.")
    topics = rag_manager.suggest_research_topics(request.num_topics)
    return {"topics": topics}

# --- Chạy server ---
if __name__ == "__main__":
    uvicorn.run("backend_main:app", host="0.0.0.0", port=8000, reload=True)
