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
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(title="AI Backend Service")

# --- Khởi tạo các đối tượng AI (Singleton Pattern) ---
@app.on_event("startup")
async def startup_event():
    """Hàm này sẽ chạy một lần duy nhất khi server bắt đầu."""
    global document_reader, rag_manager, analysis_agent, llm, embedder
    
    try:
        # Load và validate environment variables
        model_name = os.getenv("OPENROUTER_MODEL")
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY không được cấu hình trong file .env")
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
        
        # Initialize other components
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        print("🔄 Đang khởi tạo RAG Manager...")
        rag_manager = RAGManager(chunker=chunker, llm=llm, embedder=embedder)
        
        print("🔄 Đang khởi tạo Analysis Agent...")
        analysis_agent = AnalysisAgent(data_path=grades_path)
        
        print("✅ Khởi tạo hệ thống thành công!")
        
    except Exception as e:
        print(f"❌ Lỗi khi khởi động hệ thống: {str(e)}")
        raise
    
    # Initialize text splitter for chunking
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Initialize RAG manager with all required components
    rag_manager = RAGManager(chunker=chunker, llm=llm, embedder=embedder)
    
    # Initialize Analysis Agent with absolute path
    grades_path = os.path.join(os.path.dirname(__file__), 'data', 'grades.csv')
    analysis_agent = AnalysisAgent(data_path=grades_path)
    
    # Verify DataFrame is loaded correctly
    if analysis_agent.df.empty:
        raise ValueError("Grades data is empty")
    if 'subject' not in analysis_agent.df.columns:
        raise ValueError(f"Required column 'subject' not found. Available columns: {', '.join(analysis_agent.df.columns)}")
    
    print("✅ Agent Phân tích đã sẵn sàng.")
    
    database_manager.create_tables()
    print("✅ Backend service đã sẵn sàng.")

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
        
        if not quiz_data:
            raise HTTPException(
                status_code=500,
                detail="Không thể tạo quiz. Vui lòng thử lại sau."
            )
            
        if not isinstance(quiz_data, list):
            raise HTTPException(
                status_code=500,
                detail=f"Định dạng quiz không hợp lệ: {quiz_data}"
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
