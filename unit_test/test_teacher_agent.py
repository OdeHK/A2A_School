import os
import json
from services.document_processing.document_management_service import DocumentManagementService
from services.quiz_generation import QuizGenerationService
from services.teacher_agent import TeacherAgent
from services.rag.rag_service import RagService



def test_summarization():
    # Nạp thư viện tài liệu (giả sử)
    library = load_json_library("document_library.json")
    table_of_contents = load_json_library("table_of_contents.json")

    # Yêu cầu 1: Liên quan đến tóm tắt
    request_1 = "Tóm tắt sách Python rất là cơ bản - Võ Duy Tuấn, phần có tiêu đề 'Giới thiệu'."
    
    inputs_1 = {
        "user_request": request_1,
        "document_library": library,
        "table_of_contents": table_of_contents
    }

    print(f"\n===== Chạy với yêu cầu tóm tắt =====\nRequest: {request_1}")
    teacher_agent = TeacherAgent(
        rag_service=None,  # Thay thế bằng dịch vụ RAG thực tế
        quiz_generation_service=None,  # Thay thế bằng dịch vụ tạo quiz thực tế
        document_management_service=None,  # Thay thế bằng dịch vụ quản lý tài liệu thực tế
        llm_service=None  # Thay thế bằng dịch vụ LLM thực tế
    )
    result_1 = teacher_agent.workflow.invoke(inputs_1)
    print(f"\nFinal Answer:\n{result_1['answer']}")

    print("\n" + "="*40 + "\n")

def test_rag_qa():
    rag_service = RagService()
    teacher_agent = TeacherAgent(
        rag_service=rag_service,
        quiz_generation_service=None,  
        document_management_service=None,
        llm_service=None
    )

    # Tạo dữ liệu giả
    request = "Hãy cho tôi biết công thức của phương pháp chuẩn hóa Z-score."
    
    # Vector database được lưu trữ ở folder 'vector_db' và 
    # có chứa một chunk với nội dung liên quan đến Z-score
    inputs = {
        "user_request": request,
        "document_library": None,  # Không cần thư viện tài liệu cho RAG
        "table_of_contents": None
    }

    print(f"\n===== Chạy với yêu cầu RAG QA =====\nRequest: {request}")
    result = teacher_agent.workflow.invoke(inputs)
    print(f"\nFinal Answer:\n{result['answer']}")


def test_quiz_generation():
    # Tạm thời sử dụng session đã có sẵn
    document_management_service = DocumentManagementService()   
    document_management_service.load_session(session_id="session_20250918_203241_11f77b42")

    # Khởi tạo dịch vụ RAG. Lưu ý rằng dịch vụ RAG này đã có lưu trữ sẵn các chunk trong tài liệu đang thử
    rag_service = RagService()
    quiz_generation_service = QuizGenerationService(rag_service=rag_service)

    teacher_agent = TeacherAgent(
        rag_service=rag_service,
        quiz_generation_service=quiz_generation_service,
        document_management_service=document_management_service,
        llm_service=None
    )

    # Yêu cầu 3: Liên quan đến tạo quiz
    request_3 = "Tạo một bài quiz với 1 câu hỏi trong tài liệu 'Xử lý dữ liệu'."
    inputs_3 = {
        "user_request": request_3,
        "document_library": None,  # Không cần thư viện tài liệu cho tạo quiz
        "table_of_contents": None
    }

    print(f"\n===== Chạy với yêu cầu tạo quiz =====\nRequest: {request_3}")
    result_3 = teacher_agent.workflow.invoke(inputs_3)

    print(f"\nFinal Answer:\n{result_3['answer']}")



def load_json_library(json_path):
    """Helper function to load the initial data."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
if __name__ == "__main__":
    test_quiz_generation()