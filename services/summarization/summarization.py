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
    original_content: Optional[str]
    summary_content: Optional[str]
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
                        titles: Optional[list[str]] = None,
                        original_content: Optional[str] = None,
                        summary_content: Optional[str] = None) -> str:
        """
        Main entry point để tạo bản tóm tắt
        
        Args:
            user_request: Yêu cầu của người dùng
            context_for_llm: Context từ memory (optional)
            
        Returns:
            summary_content: Bản tóm tắt được sinh ra
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
            "original_content": original_content,
            "summary_content": summary_content,
            "context_for_llm": context_for_llm
        }
        
        result = self.workflow.invoke(initial_state)
        
        logger.info("========================================")
        logger.info("SUMMARIZATION WORKFLOW COMPLETE")
        logger.info("========================================")

        return result.get("summary_content", "Không thể tạo bản tóm tắt."), result.get("original_content", ""), result.get("titles", "")

    def _create_workflow(self):
        """Create LangGraph workflow with properly configured nodes"""

        def find_content_node(state: SummarizationState) -> Dict[str, Any]:
            """Node để tìm tài liệu và trích xuất nội dung"""
            logger.info("=== FIND CONTENT NODE START ===")
            logger.info(f"User request: {state['user_request']}")
            username = state.get("username")
            document_id = state.get("document_id")
            context_for_llm = state.get("context_for_llm")
            llm = self.llm_service.get_llm()
            try:
                content_data = self.document_management_service.get_content_data(
                    username=username, document_id=document_id
                )
                logger.info(f"✅ Retrieved content_data for user={username}, document_id={document_id}")
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
                titles.sort()
                
                logger.info(f"Matched titles: {titles}")
                
            except Exception as e:
                logger.error(f"Error finding document: {e}")
                titles = None
        
            
            # -----------------------------
            # Step 2: Kiểm tra tài liệu có được tìm thấy không
            # -----------------------------
            original_content = state.get("original_content", "")
            summary_content = state.get("summary_content", "")
            
            if not titles:
                if context_for_llm and state.get("titles"):
                    titles = state.get("titles")
                    logger.info(f"Using titles from state (memory): {titles}")
                else:
                    titles = ["full_document"]
                    logger.info("Using full_document as default")
            else:
                state_titles = state.get("titles") or []
                if any(title not in state_titles for title in titles):
                    original_content = ""
                    summary_content = ""
                    logger.info("Titles changed, resetting original_content and summary_content")
            # -----------------------------
            # Step 3: Lấy nội dung từ document_management_service
            # -----------------------------
            if not original_content:
                content_list = []
                for content_item in content_data["content"]:
                    if content_item["title"] in titles:
                        content_list.append(f"<{content_item['title']}>{content_item['content']}</{content_item['title']}>")
                original_content = "\n".join(content_list)
                logger.info(f"Found original content: {original_content}")
                if not original_content:
                    logger.warning(f"No content found for titles: {titles}")
                    return {
                        **state,
                        "summary_content": "Không tìm thấy nội dung để tóm tắt."
                    }
                
                logger.info(f"Found content length: {len(original_content)} characters")
                
            
            
            logger.info("=== FIND CONTENT NODE END ===")
            return {
                **state,
                "titles": titles,
                "summary_content": summary_content,
                "original_content": original_content
            }

        def summarization_node(state: SummarizationState) -> Dict[str, Any]:
            """Node để tạo bản tóm tắt từ nội dung"""
            logger.info("=== SUMMARIZATION NODE START ===")
            
            # -----------------------------
            # Generate summary using LLM
            # -----------------------------
            llm = self.llm_service.get_llm()
            
            # Prepare input
            llm_input = {
                "original_content": state.get("original_content", ""),
                "summary_content": state.get("summary_content", ""),
                "user_request": state["user_request"] if state.get("summary_content") else "",
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
                
                summary_content = summary.content
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
                summary_content += f"\n\n\n{random_question}"

            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary_content = f"Lỗi khi tạo bản tóm tắt: {str(e)}"
            
            logger.info("=== SUMMARIZATION NODE END ===")
            return {
                **state,
                "summary_content": summary_content
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
