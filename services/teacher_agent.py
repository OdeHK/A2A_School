import json
import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from services.summarizer_agent import summarization_graph

from services.prompt import router_node_prompt
from langchain_core.output_parsers import StrOutputParser
from services.rag.rag_service import RagService
from services.quiz_generation import QuizGenerationService
from services.document_processing.document_management_service import DocumentManagementService
from services.rag.llm_service import LLMService

# --- Định nghĩa State cho Parent Graph ---
class ParentGraphState(TypedDict):
    user_request: str
    document_library: Optional[list]
    table_of_contents: Optional[list]
    answer: Optional[str]
    route: str 

# --- Các Node của Parent Graph ---
def router_node(state: ParentGraphState):
    """Phân loại yêu cầu và quyết định lộ trình."""
    print("--- 1. ROUTER: Phân loại yêu cầu ---")
    llm = LLMService(llm_type="google_gen_ai").get_llm()
    routing_chain =  router_node_prompt | llm | StrOutputParser()
    route = routing_chain.invoke({"user_request": state["user_request"]})
    print(f" -> Lộ trình được quyết định: '{route}'")
    return {"route": route}

def summarizer_node(state: ParentGraphState):
    """Thực thi subgraph tóm tắt."""
    print("--- 2a. EXECUTING: Subgraph Tóm tắt ---")
    inputs = {
        "user_request": state["user_request"],
        "document_library": state["document_library"],
        "table_of_contents": state["table_of_contents"]
    }
    # Giả sử kết quả cuối cùng của app tóm tắt nằm trong key 'summary'
    result = summarization_graph.invoke(inputs)
    final_summary = result.get('summary', "Không thể tạo tóm tắt.")
    return {"answer": final_summary}


def rag_qa_node(state: ParentGraphState):
    """Trả lời câu hỏi dựa trên tài liệu (RAG)."""
    print("--- 2c. EXECUTING: Subgraph RAG Q&A ---")
    query = state["user_request"]
    rag_service = RagService()
    response = rag_service.generate_rag_response(query)
    return {"answer": response}

def quiz_generation_node(state: ParentGraphState):
    """Sinh câu hỏi kiểm tra dựa trên tài liệu."""
    print("--- 2d. EXECUTING: Subgraph Quiz Generation ---")
    # Ở đây cần document_id, giả sử lấy từ document_library[0]
    if not state["document_library"]:
        return {"answer": "Không có tài liệu nào để sinh quiz."}
    document_management_service = DocumentManagementService()
    quiz_generation_service = QuizGenerationService()
    document_id = '' 

    response = generate_quiz_set(document_id, state["user_request"])
    return {"answer": response}

# --- Logic quyết định rẽ nhánh ---
def decide_route(state: ParentGraphState):
    """Hàm quyết định sẽ đi theo nhánh nào."""
    return state["route"]

def load_json_library(json_path):
    """Helper function to load the initial data."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
# --- Xây dựng và Compile Parent Graph ---



workflow = StateGraph(ParentGraphState)
workflow.add_node("router", router_node) 
workflow.add_node("summarizer", summarizer_node) 
workflow.add_node("rag_qa", rag_qa_node)
workflow.add_node("quiz_generation", quiz_generation_node)
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    decide_route,
    {
        "summarizer": "summarizer",
        "quiz_generation": "quiz_generation",
        "rag_qa": "rag_qa"
    }
)

workflow.add_edge("summarizer", END)
workflow.add_edge("quiz_generation", END)
workflow.add_edge("rag_qa", END)



# Compile đồ thị
parent_app = workflow.compile()


# --- Chạy Thử Parent Graph ---
if __name__ == "__main__":
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
    result_1 = parent_app.invoke(inputs_1)
    print(f"\nFinal Answer:\n{result_1['answer']}")

    print("\n" + "="*40 + "\n")
