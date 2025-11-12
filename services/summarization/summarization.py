from typing import TypedDict, Dict, Any, Optional
import logging
import json

from services.rag.rag_service import RagService
from services.document_processing.document_management_service import DocumentManagementService
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import JsonOutputParser
from prompts.summarization import find_titles_prompt, summarize_content_prompt
import random
logger = logging.getLogger(__name__)


# ====== Graph State =========
class SummarizationState(TypedDict):
    """State cho Summarization workflow"""
    user_request: Optional[str]
    username: Optional[str]
    document_id: Optional[str]
    titles: list[str]
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
                        username: Optional[str] = None,
                        document_id: Optional[str] = None,
                        titles: Optional[list[str]] = None) -> str:
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
            "username": username,
            "document_id": document_id,
            "titles": titles,
            "extracted_content": None,
            "final_summary": None,
            "context_for_llm": context_for_llm
        }
        
        result = self.workflow.invoke(initial_state)
        
        logger.info("========================================")
        logger.info("SUMMARIZATION WORKFLOW COMPLETE")
        logger.info("========================================")

        return result.get("final_summary", "Không thể tạo bản tóm tắt."), result.get("titles", "")

    def _create_workflow(self):
        """Create LangGraph workflow with properly configured nodes"""

        def find_content_node(state: SummarizationState) -> Dict[str, Any]:
            """Node để tìm tài liệu và trích xuất nội dung"""
            logger.info("=== FIND CONTENT NODE START ===")
            logger.info(f"User request: {state['user_request']}")
            username = state.get("username")
            document_id = state.get("document_id")
            context_for_llm = state.get("context_for_llm")
            llm = self.llm_service.llm
            try:
                content_data = self.document_management_service.get_content_data(
                    username=username, document_id=document_id
                )
                if not content_data:
                    logger.error(f"❌ No content_data returned for user={username}, document_id={document_id}")
                    return {"error": "No content found for the given document."}

                if "content" not in content_data:
                    logger.error(f"❌ 'content' key missing in content_data: {content_data.keys()}")
                    return {"error": "Invalid content format returned from document_management_service."}

                if not isinstance(content_data["content"], list):
                    logger.error("❌ content_data['content'] is not a list.")
                    return {"error": "Invalid content format: expected list of items."}

            except Exception as e:
                logger.exception("❌ Error while retrieving content_data:")
                raise
            try:
                titles_data = [item.get("title") for item in content_data["content"] if "title" in item]
                logger.info(f"✅ Extracted {len(titles_data)} titles from document.")
            except Exception as e:
                logger.exception("❌ Error while extracting titles from content_data:")
                raise
            
            find_document_chain = find_titles_prompt | llm | JsonOutputParser()
            titles_data_str = json.dumps(titles_data, indent=2, ensure_ascii=False)
            
            try:
                titles = find_document_chain.invoke({
                    "title_list": titles_data_str,
                    "user_request": state["user_request"],
                    
                })
                
                logger.info(f"Matched titles: {titles}")
                
            except Exception as e:
                logger.error(f"Error finding document: {e}")
                titles = None
        
            
            # -----------------------------
            # Step 2: Kiểm tra tài liệu có được tìm thấy không
            # -----------------------------
            if not titles:
                if context_for_llm and state.get("titles"):
                    titles = state.get("titles")
                    logger.info(f"Using titles from state (memory): {titles}")
                else:
                    titles = ["full_document"]
                    logger.info("Using full_document as default")
                
        
            # -----------------------------
            # Step 3: Lấy nội dung từ document_management_service
            # -----------------------------
            try:
                extracted_content = []
                for content_item in content_data["content"]:
                    if content_item["title"] in titles:
                        extracted_content.append(content_item["content"])
                extracted_content = "\n".join(extracted_content)
                if not extracted_content:
                    logger.warning(f"No content found for titles: {titles}")
                    return {
                        **state,
                        "final_summary": "Không tìm thấy nội dung để tóm tắt."
                    }
                
                logger.info(f"Found content length: {len(extracted_content)} characters")
                
            except Exception as e:
                logger.error(f"Error extracting content: {e}")
                return {
                    **state,
                    "final_summary": f"Lỗi khi trích xuất nội dung: {str(e)}"
                }
            
            logger.info("=== FIND CONTENT NODE END ===")
            return {
                **state,
                "titles": titles,
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
            llm = self.llm_service.llm
            
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
                chain = summarize_content_prompt | llm
                summary = chain.invoke(llm_input)
                
                logger.info(f"Summary generated: {summary.content[:100]}...")
                
                final_summary = summary.content
                extra_questions = [
                    #"✂️ Bạn có muốn tôi làm nó ngắn gọn hơn (ví dụ: chỉ 3 gạch đầu dòng) không?",
                    #"🎯 Bạn có muốn tôi tập trung vào một khía cạnh cụ thể nào khác của tài liệu (ví dụ: chỉ tóm tắt phần 'kết luận' hoặc 'phương pháp luận') không?",
                    "📏 Mức độ chi tiết này đã phù hợp với bạn chưa?",
                    "💬 Bạn có muốn biết thêm thông tin nào khác về tài liệu này không?",
                    "📝 Bản tóm tắt này đã đủ chi tiết cho bạn chưa?",
                   # "❓ Bạn có muốn tôi sinh một số câu hỏi liên quan đến tài liệu này không?"
                ]
                random_question = random.choice(extra_questions)
                state['user_request'] = random_question
                final_summary += f"\n\n\n{random_question}"

            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                final_summary = f"Lỗi khi tạo bản tóm tắt: {str(e)}"
            
            logger.info("=== SUMMARIZATION NODE END ===")
            return {
                **state,
                "final_summary": final_summary
            }
        

        # -----------------------------
        # Build workflow
        # -----------------------------
        workflow = StateGraph(SummarizationState)
        
        # Add nodes
        workflow.add_node("find_content", find_content_node)
        workflow.add_node("summarization", summarization_node)
        
        # Define flow
        workflow.add_edge(START, "find_content")
        workflow.add_edge("find_content", "summarization")
        workflow.add_edge("summarization", END)
        
        return workflow.compile()
