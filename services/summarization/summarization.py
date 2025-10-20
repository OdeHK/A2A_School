from typing import TypedDict, Dict, Any, Optional
import logging
import json

from services.rag.rag_service import RagService
from services.document_processing.document_management_service import DocumentManagementService
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import JsonOutputParser
from services.summarization.prompt import find_document_node_prompt, summarize_content_node_prompt

logger = logging.getLogger(__name__)


# ====== Graph State =========
class SummarizationState(TypedDict):
    """State cho Summarization workflow"""
    user_request: str
    document_id: Optional[str]
    title: Optional[str]
    matched_document: Optional[dict]
    extracted_content: Optional[str]
    final_summary: Optional[str]
    context_for_llm: Optional[str]


class SummarizationService:
    """Service điều phối việc tóm tắt tài liệu sử dụng LangGraph"""

    def __init__(self, rag_service: RagService, document_management_service: DocumentManagementService):
        """
        Initialize SummarizationService
        
        Args:
            rag_service: RAG service for LLM access
            document_management_service: Service for document management
        """
        self.rag_service = rag_service
        self.document_management_service = document_management_service
        self.llm_service = rag_service.llm_service
        self.workflow = self._create_workflow()

    def generate_summary(self, 
                        user_request: str,
                        context_for_llm: Optional[str] = None,
                        matched_document: Optional[dict] = None) -> str:
        """
        Main entry point để tạo bản tóm tắt
        
        Args:
            user_request: Yêu cầu của người dùng
            context_for_llm: Context từ memory (optional)
            
        Returns:
            final_summary: Bản tóm tắt được sinh ra
        """
        logger.info("========================================")
        logger.info("SUMMARIZATION WORKFLOW START")
        logger.info(f"User Request: {user_request}")
        logger.info("========================================")
        
        initial_state: SummarizationState = {
            "user_request": user_request,
            "matched_document": matched_document,
            "extracted_content": None,
            "final_summary": None,
            "context_for_llm": context_for_llm
        }
        
        result = self.workflow.invoke(initial_state)
        
        logger.info("========================================")
        logger.info("SUMMARIZATION WORKFLOW COMPLETE")
        logger.info("========================================")
        
        return result.get("final_summary", "Không thể tạo bản tóm tắt."),result.get("matched_document","")

    def _create_workflow(self):
        """Create LangGraph workflow with properly configured nodes"""

        def find_content_node(state: SummarizationState) -> Dict[str, Any]:
            """Node để tìm tài liệu và trích xuất nội dung"""
            logger.info("=== FIND CONTENT NODE START ===")
            logger.info(f"User request: {state['user_request']}")
            
            llm = self.llm_service.get_llm()
            
            # -----------------------------
            # Step 1: Nếu chưa có matched_document, tìm trong thư viện
            # -----------------------------
            document_library = self.document_management_service.get_document_library()
            library_length = len(document_library)
            
            logger.info(f"Document library size: {library_length}")
            
            find_document_chain = find_document_node_prompt | llm | JsonOutputParser()
            library_str = json.dumps(document_library, indent=2)
            
            try:
                matched_document = find_document_chain.invoke({
                    "library_str": library_str,
                    "user_request": state["user_request"],
                    "library_length": library_length
                })
                
                logger.info(f"Matched document from library: {matched_document}")
                
            except Exception as e:
                logger.error(f"Error finding document: {e}")
                matched_document = None
        
            
            # -----------------------------
            # Step 2: Kiểm tra tài liệu có được tìm thấy không
            # -----------------------------
            if not matched_document:
                if not state.get("matched_document"):  # Kiểm tra state
                    return {
                    **state,
                    "matched_document": None,
                    "final_summary": "❓Bạn có thể giúp mình bằng cách nói rõ tên tài liệu cần tìm được không?"
                }
                else:
                    matched_document = state["matched_document"]  
                
            # -----------------------------
            # Step 3: Lấy document_id và title từ matched_document
            # -----------------------------
            document_id = matched_document.get("document_id")
            title = matched_document.get("title", [None])[0] if matched_document.get("title") else None
            
            if not document_id or not title:
                logger.warning("Document ID hoặc Title không hợp lệ")
                return {
                    **state,
                    "matched_document": matched_document,
                    "final_summary": "Không tìm thấy thông tin tài liệu hợp lệ."
                }
            
            logger.info(f"Getting content for document_id: {document_id}, title: {title}")
            
            # -----------------------------
            # Step 4: Lấy nội dung từ document_management_service
            # -----------------------------
            try:
                content_data = self.document_management_service.get_content_data(document_id)["content"]
                
                if not content_data:
                    logger.warning(f"No content data found for document: {document_id}")
                    return {
                        **state,
                        "matched_document": matched_document,
                        "final_summary": "Không tìm thấy nội dung để tóm tắt."
                    }
                
                # Find content by title
                extracted_content = None
                for content_item in content_data:
                    if content_item.get("title") == title:
                        extracted_content = content_item.get("content")
                        break
                
                if not extracted_content:
                    logger.warning(f"No content found for title: {title}")
                    return {
                        **state,
                        "matched_document": matched_document,
                        "final_summary": f"Không tìm thấy nội dung cho '{title}'."
                    }
                
                logger.info(f"Found content length: {len(extracted_content)} characters")
                
            except Exception as e:
                logger.error(f"Error extracting content: {e}")
                return {
                    **state,
                    "matched_document": matched_document,
                    "final_summary": f"Lỗi khi trích xuất nội dung: {str(e)}"
                }
            
            logger.info("=== FIND CONTENT NODE END ===")
            return {
                **state,
                "matched_document": matched_document,
                "extracted_content": extracted_content
            }

        def summarization_node(state: SummarizationState) -> Dict[str, Any]:
            """Node để tạo bản tóm tắt từ nội dung"""
            logger.info("=== SUMMARIZATION NODE START ===")
            
            # Kiểm tra nếu đã có final_summary từ find_content_node (lỗi)
            if state.get("final_summary"):
                logger.info("Final summary already set (error case), skipping summarization")
                return state
            
            extracted_content = state.get("extracted_content")
            
            if not extracted_content:
                logger.warning("No content to summarize")
                return {
                    **state,
                    "final_summary": "Không có nội dung để tóm tắt."
                }
            
            # -----------------------------
            # Generate summary using LLM
            # -----------------------------
            llm = self.llm_service.get_llm()
            
            # Prepare input
            llm_input = {
                "input_text": extracted_content
            }
            
            # Add context if available
            context_for_llm = state.get("context_for_llm")
            if context_for_llm:
                llm_input["context"] = context_for_llm
                logger.info("Added conversation context to LLM input")
            
            try:
                chain = summarize_content_node_prompt | llm
                summary = chain.invoke(llm_input)
                
                logger.info(f"Summary generated: {summary.content[:100]}...")
                
                final_summary = summary.content
                
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                final_summary = f"Lỗi khi tạo bản tóm tắt: {str(e)}"
            
            logger.info("=== SUMMARIZATION NODE END ===")
            return {
                **state,
                "final_summary": final_summary
            }
        def router_decision(state: SummarizationState) -> str:
            """Quyết định có tiếp tục tóm tắt hay kết thúc"""
            # Nếu không tìm thấy document hoặc đã có final_summary (lỗi)
            if state.get("matched_document") is None or state.get("final_summary") is not None:
                return END
            return "summarization"

        # -----------------------------
        # Build workflow
        # -----------------------------
        workflow = StateGraph(SummarizationState)
        
        # Add nodes
        workflow.add_node("find_content", find_content_node)
        workflow.add_node("summarization", summarization_node)
        
        # Define flow
        workflow.add_edge(START, "find_content")
        
        # Add conditional edge từ find_content
        workflow.add_conditional_edges(
            "find_content",
            router_decision,
            {
                "summarization": "summarization",
                END: END
            }
        )
        
        workflow.add_edge("summarization", END)
        
        return workflow.compile()
