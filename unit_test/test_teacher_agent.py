import os
import json
from services.document_processing.document_management_service import DocumentManagementService
from services.quiz_generation import QuizGenerationService
from services.agent.agent_service import TeacherAgent, ParentGraphState
from services.rag.rag_service import RagService
from services.rag.llm_service import LLMService



def test_summarization():
    test_file = ".\\example_data\\python_rat_la_co_ban.pdf"
    document_management_service = DocumentManagementService()
    result = document_management_service.process_uploaded_document(
                file_path=test_file,
                extract_toc=True
            )
    # Danh sách các yêu cầu tóm tắt
    requests = [
        "Tóm tắt  sách Python rất là cơ bản - Võ Duy Tuấn, tiêu đề là 'Mục lục'",
        "Tóm tắt toàn bộ sách Python rất là cơ bản - Võ Duy Tuấn.",
        #"Tóm tắt nội dung phần 'Cài đặt' nằm trong Chương 1. Hello world của sách Python rất là cơ bản - Võ Duy Tuấn."
    ]

    # Khởi tạo agent (chỉ cần một lần)
    teacher_agent = TeacherAgent(
        rag_service=None,  # Thay thế bằng dịch vụ RAG thực tế
        quiz_generation_service=None,  # Thay thế bằng dịch vụ tạo quiz thực tế
        document_management_service=document_management_service,  # Thay thế bằng dịch vụ quản lý tài liệu thực tế
        llm_service=LLMService(llm_type="google_gen_ai")
    )

    # Duyệt qua các request
    for i, req in enumerate(requests, start=1):
        inputs = {"user_request": req}
        print(f"\n===== Chạy với yêu cầu tóm tắt {i} =====\nRequest: {req}")

        result = teacher_agent.workflow.invoke(inputs)
        print(f"\nFinal Answer:\n{result['answer']}")
        print("\n" + "="*40 + "\n")



def test_rag_qa():
    # Tạm thời sử dụng session đã có sẵn
    document_management_service = DocumentManagementService()
    document_management_service.load_session(session_id="session_20250925_095132_c2a6463a")
    rag_service = RagService()
    teacher_agent = TeacherAgent(
        rag_service=rag_service,
        quiz_generation_service=None,
        document_management_service=document_management_service,
        llm_service=LLMService(llm_type="nvidia")
    )

    # Tạo dữ liệu giả
    request = "Hãy cho tôi biết công thức của phương pháp chuẩn hóa Z-score."
    
    # Vector database được lưu trữ ở folder 'vector_db' và 
    # có chứa một chunk với nội dung liên quan đến Z-score
    inputs: ParentGraphState = {
        "user_request": request,
        "matched_document": {},  
        "table_of_contents": None,
        "answer": None,
        "route": ""
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
        llm_service=LLMService(llm_type="nvidia")
    )

    # Yêu cầu 3: Liên quan đến tạo quiz
    request_3 = "Tạo một bài quiz với 1 câu hỏi trong tài liệu 'Xử lý dữ liệu'."
    inputs_3: ParentGraphState = {
        "user_request": request_3,
        "matched_document": {}, 
        "table_of_contents": None,
        "answer": None,
        "route": ""
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
    test_rag_qa()