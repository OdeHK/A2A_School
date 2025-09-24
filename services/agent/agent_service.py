import json
import os
import logging
from typing import List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from services.summarizer_agent import summarization_graph
from services.prompt import router_node_prompt
from services.rag.rag_service import RagService
from services.rag.llm_service import LLMService
from services.quiz_generation import QuizGenerationService
from services.document_processing.document_management_service import DocumentManagementService


# Logger toàn cục cho module này
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Định nghĩa State cho Parent Graph ---
class ParentGraphState(TypedDict):
    user_request: str
    selected_document_id: Optional[str]
    document_library: Optional[list]
    table_of_contents: Optional[list]
    answer: Optional[str]
    route: str 

class TeacherAgent:
    """
    Teacher Agent that routes requests to appropriate subgraphs or services.
    
    This agent can handle requests for summarization, quiz generation, and RAG-based Q&A.
    """
    
    def __init__(
            self, 
            rag_service: RagService,
            quiz_generation_service: QuizGenerationService,
            document_management_service: DocumentManagementService,
            llm_service: LLMService):
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
            routing_chain =  router_node_prompt | self.llm_service.get_llm() | StrOutputParser()
            route = routing_chain.invoke({"user_request": state["user_request"]})
            logger.info(f" -> Lộ trình được quyết định: '{route}'")
            return {"route": route}

        def summarizer_node(state: ParentGraphState):
            """Thực thi subgraph tóm tắt."""
            logger.info("--- 2a. EXECUTING: Subgraph Tóm tắt ---")
            logger.info(f"Inputs: user_request={state['user_request']}, document_library={state['document_library']}, table_of_contents={state['table_of_contents']}")
            inputs = {
                "user_request": state["user_request"],
                "document_library": state["document_library"],
                "table_of_contents": state["table_of_contents"]
            }
            try:
                result = summarization_graph.invoke(inputs)
                final_summary = result.get('summary', "Không thể tạo tóm tắt.")
                logger.info(f"Summary result: {final_summary}")
                return {"answer": final_summary}
            except Exception as e:
                logger.error(f"Error in summarizer_node: {str(e)}")
                return {"answer": f"Đã xảy ra lỗi khi tóm tắt: {str(e)}"}

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

    def handle_chat_query(self, query: str, chat_history: Optional[List] = None, selected_document_id: Optional[str] = None):
        """
        Handle a chat query by routing to the appropriate subgraph.
        Args:
            query: The user's query string.
            chat_history: Optional list of previous chat messages.
            selected_document_id: Selected document ID from UI.
        
        Returns:
            A tuple of (answer string, updated chat history list).
        """
        
        try:
            if not query or not query.strip():
                return "Xin chào! Tôi có thể giúp gì cho bạn?", chat_history or []
            
            # Determine document_id to use
            if not selected_document_id:
                try:
                    document_id_dict = self.document_management_service.get_document_id_dict()
                    first_document_id = next(iter(document_id_dict))
                    document_id = first_document_id
                    logger.info(f"No selected_document_id provided, using first document: {document_id}")
                except (StopIteration, AttributeError) as e:
                    logger.warning(f"No documents available in document_id_dict: {str(e)}")
                    return "Không có tài liệu nào được tải lên. Vui lòng tải tài liệu trước khi đặt câu hỏi.", chat_history or []
            else:
                document_id = selected_document_id
                logger.info(f"Using provided selected_document_id: {document_id}")
            
            # Get document library and table of contents
            try:
                # TODO: get_document_library method will be implemented later in document_management_service
                document_library = None  # Placeholder until get_document_library is implemented
                
                # Get table of contents as string and convert to list format
                # TODO: Thống nhất format với Khôi để tránh xung đột trong summarization_agent
                toc_string = self.document_management_service.get_table_of_contents_as_string(document_id)
                if toc_string:
                    # Convert string to list by splitting on newlines and filtering empty lines
                    table_of_contents = [line.strip() for line in toc_string.split('\n') if line.strip()]
                else:
                    table_of_contents = []
                
                logger.info(f"Retrieved table_of_contents for document_id: {document_id}, items: {len(table_of_contents)}")
            except Exception as e:
                logger.error(f"Error retrieving document data: {str(e)}")
                return f"Không thể truy xuất dữ liệu tài liệu: {str(e)}", chat_history or []

            # Prepare state for workflow
            state: ParentGraphState = {
                "user_request": query,
                "selected_document_id": document_id,
                "document_library": document_library,
                "table_of_contents": table_of_contents,
                "answer": None,
                "route": ""
            }

            # Invoke the workflow
            result = self.workflow.invoke(state)
            
            # Get the answer from result
            answer = result.get("answer", "Không thể xử lý yêu cầu.")
            
            # Update chat history
            updated_history = chat_history or []
            updated_history.append((query, answer))
            
            return answer, updated_history
        
        except Exception as e:
            logger.error(f"Error handling chat query: {str(e)}")
            return "Đã xảy ra lỗi khi xử lý yêu cầu.", chat_history or []

# --- Logic quyết định rẽ nhánh ---
def decide_route(state: ParentGraphState):
    """Hàm quyết định sẽ đi theo nhánh nào."""
    return state["route"]

