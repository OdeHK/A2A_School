import json
import os
import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from services.prompt import router_node_prompt, find_document_node_prompt, summarize_content_node_prompt
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from services.rag.rag_service import RagService
from services.quiz_generation import QuizGenerationService
from services.document_processing.document_management_service import DocumentManagementService
from services.rag.llm_service import LLMService

# Logger toàn cục cho module này
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

    
   

# --- Định nghĩa State cho Parent Graph ---
class ParentGraphState(TypedDict):
    user_request: str
    matched_document : dict
    table_of_contents: Optional[list]
    answer: Optional[str]  # Add answer field for quiz and rag results
    route: str 

class TeacherAgent:
    """
    Teacher Agent that routes requests to appropriate subgraphs or services.
    
    This agent can handle requests for summarization, quiz generation, and RAG-based Q&A.
    """
    
    def __init__(
            self, 
            rag_service,
            quiz_generation_service,
            document_management_service,
            llm_service):
        """
        Initialize the Teacher Agent with required services.

        Args:
            rag_service: RAG service for document queries
            quiz_generation_service: Service for quiz generation
            document_management_service: Service for document management
            llm_service: LLM service for the agent
        """

        self.rag_service = rag_service
        self.quiz_generation_service = quiz_generation_service
        self.document_management_service = document_management_service
        self.llm_service = llm_service
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Create the workflow graph for the Teacher Agent."""

        def router_node(state: ParentGraphState):
            """Phân loại yêu cầu và quyết định lộ trình."""
            logger.info("--- 1. ROUTER: Phân loại yêu cầu ---")
            logger.info(f"User request: {state['user_request']}")
            llm = self.llm_service.get_llm()
            routing_chain =  router_node_prompt | llm | StrOutputParser()
            route = routing_chain.invoke({"user_request": state["user_request"]})
            logger.info(f" -> Lộ trình được quyết định: '{route}'")
            
            document_library = self.document_management_service.get_document_library()
            
            find_document_chain = find_document_node_prompt | llm | JsonOutputParser()
            library_str = json.dumps(document_library, indent=2)
            matched_document = find_document_chain.invoke({
                "library_str": library_str,
                "user_request": state["user_request"]
            })
            logger.info(f"Matched document: {matched_document}")
            print(f"Matched document: {matched_document}")
            if matched_document is None:
                logger.warning("Không tìm thấy tài liệu phù hợp trong thư viện.")
            return {"route": route,"matched_document":matched_document}

        def summarizer_node(state: ParentGraphState):
            """Thực thi subgraph tóm tắt."""
            logger.info("--- 2a. EXECUTING: Subgraph Tóm tắt ---")
            
            # Get content data instead of table of contents
            document_id = state["matched_document"]["document_id"]
            title = state["matched_document"]["title"][0]
            
            logger.info(f"Getting content for document_id: {document_id}, title: {title}")
            
            # Get content data which contains the actual content
            content_data = self.document_management_service.get_content_data(document_id)["content"]
            
            if not content_data:
                logger.warning(f"No content data found for document: {document_id}")
                return { "answer": "Không tìm thấy nội dung để tóm tắt."}
            
            # Find content by title
            extracted_content = None
            for content_item in content_data:
                if content_item.get("title") == title:
                    extracted_content = content_item.get("content")
                    break
            
            if not extracted_content:
                logger.warning(f"No content found for title: {title}")
                no_content_msg = f"Không tìm thấy nội dung cho '{title}'."
                return {"answer": no_content_msg}
            
            logger.info(f"Found content length: {len(extracted_content)} characters")
            
            llm = self.llm_service.get_llm()
            chain = summarize_content_node_prompt | llm
            summary = chain.invoke({"input_text": extracted_content})
            logger.info(f"Summary generated: {summary}")
            return {"answer": summary}

        def rag_qa_node(state: ParentGraphState):
            """Trả lời câu hỏi dựa trên tài liệu (RAG)."""
            logger.info("--- 2c. EXECUTING: Subgraph RAG Q&A ---")
            query = state["user_request"]
            logger.info(f"RAG Q&A query: {query}")
            try:
                if not query or not query.strip():
                    logger.warning("Câu hỏi không hợp lệ. Vui lòng nhập lại.")
                    return {"answer": "Câu hỏi không hợp lệ. Vui lòng nhập lại."}
                response = self.rag_service.generate_rag_response(query)
                logger.info(f"RAG Q&A response: {response}")
                return {"answer": response}
            except Exception as e:
                logger.error(f"Error in generate_rag_response tool: {str(e)}")
                return {"answer": f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"}

        def quiz_generation_node(state: ParentGraphState):
            """Sinh câu hỏi kiểm tra dựa trên tài liệu."""
            logger.info("--- 2d. EXECUTING: Subgraph Quiz Generation ---")
            user_request = state["user_request"]
            logger.info(f"Quiz generation user_request: {user_request}")
            # TODO: Vì hiện tại chỉ xử lý với một tài liệu duy nhất nên tạm thời lấy document_id đầu tiên trong session
            # Sau này cần bổ sung khả năng lấy document_id linh hoạt hơn
            try:
                document_id_dict = self.document_management_service.get_document_id_dict()
                first_document_id = next(iter(document_id_dict))
                document_id = first_document_id
                logger.info(f"Selected document_id: {document_id}")
                if not document_id or not user_request:
                    logger.warning("Cần cung cấp document_id và yêu cầu người dùng.")
                    return {"answer": "Cần cung cấp document_id và yêu cầu người dùng."}

                # Get table of contents
                toc_string = self.document_management_service.get_table_of_contents_as_string(document_id)
                logger.info(f"TOC string: {toc_string}")
                if not toc_string:
                    logger.warning(f"Không tìm thấy mục lục cho tài liệu: {document_id}")
                    return {"answer": f"Không tìm thấy mục lục cho tài liệu: {document_id}"}

                # Generate quiz
                result = self.quiz_generation_service.generate_quiz_set(
                    document_id=document_id,
                    user_request=user_request,
                    toc_data=toc_string
                )
                final_questions = result.get("final_questions", [])
                logger.info(f"Quiz generation result: {final_questions}")
                return {"answer": '\n'.join(final_questions) if final_questions else "Không thể tạo câu hỏi."}
            except Exception as e:
                logger.error(f"Error in generate_quiz_set tool: {str(e)}")
                return {"answer": f"Đã xảy ra lỗi khi tạo đề: {str(e)}"}

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
        logger.info("TeacherAgent workflow graph compiled.")
        return workflow.compile()



# --- Logic quyết định rẽ nhánh ---
def decide_route(state: ParentGraphState):
    """Hàm quyết định sẽ đi theo nhánh nào."""
    return state["route"]

