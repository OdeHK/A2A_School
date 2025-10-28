import json
import os
import logging
import random
from typing import List, Tuple, TypedDict, Optional
from pathlib import Path
from apiclient import discovery
from httplib2 import Http
from oauth2client import client, file, tools
from langgraph.graph import StateGraph, END

from prompts.agent import router_prompt
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from services.quiz_generation.converter import QuizToGoogleFormConverter
from services.rag.rag_service import RagService
from services.quiz_generation.quiz_generation import QuizGenerationService
from services.summarization.summarization import SummarizationService
from services.document_processing.document_management_service import DocumentManagementService
from services.llm_service import LLMService
from services.agent.memory_manager import ShortTermMemory, MemoryEntry
from services.models import QuizQuestionOutput
from config.constants import StorageConstants

# Logger toàn cục cho module này
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)    
   

# --- Định nghĩa State cho Parent Graph ---
class ParentGraphState(TypedDict):
    user_request: str
    table_of_contents: Optional[list]
    answer: Optional[str]  # Add answer field for quiz and rag results
    route: str 
    username: str  # Add username field for user-specific operations 
    selected_document_id: str  # Add selected_document_id field for user-specific operations

class TeacherAgent:
    """
    Teacher Agent that routes requests to appropriate subgraphs or services.
    
    This agent can handle requests for summarization, quiz generation, and RAG-based Q&A.
    Enhanced with Short-Term Memory for context-aware interactions.
    """
    
    def __init__(
            self, 
            rag_service,
            quiz_generation_service,
            summarization_service,
            document_management_service,
            llm_service,
            database_service, 
            enable_memory: bool = True):
        """
        Initialize the Teacher Agent with required services.

        Args:
            rag_service: RAG service for document queries
            quiz_generation_service: Service for quiz generation
            summarization_service: Service for summarization
            document_management_service: Service for document management
            llm_service: LLM service for the agent
            enable_memory: Enable short-term memory (default: True)
        """

        self.rag_service = rag_service
        self.quiz_generation_service = quiz_generation_service
        self.summarization_service = summarization_service
        self.document_management_service = document_management_service
        self.llm_service = llm_service
        
        # Initialize memory systems
        self.enable_memory = enable_memory
        if enable_memory:
            self.memory = ShortTermMemory(
                max_entries=50,
                decay_minutes=30,
                min_importance_threshold=0.1
            )
            logger.info("Short-term memory initialized")
        else:
            self.memory = None
            logger.info("Memory disabled")
        
        self.database_service = database_service
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Create the workflow graph for the Teacher Agent."""

        def router_node(state: ParentGraphState):
            username = state["username"]
            """Phân loại yêu cầu và quyết định lộ trình."""
            logger.info("--- 1. ROUTER: Phân loại yêu cầu ---")
            logger.info(f"User {username} request: {state['user_request']}")
            
            llm = self.llm_service.get_llm()
            # -----------------------------
            # Step 2: Lấy ngữ cảnh hội thoại gần đây (nếu có)
            # -----------------------------
            context_for_llm = ""
            if self.enable_memory and self.memory:
                context_for_llm = self.memory.get_context_for_llm(
                    task_type="general"
                )
            
            # -----------------------------
            # Step 3: Xác định lộ trình dựa trên yêu cầu + ngữ cảnh
            # -----------------------------
            llm_input = {"user_request": state["user_request"], "chat_history": context_for_llm}
            routing_chain = router_prompt | llm | StrOutputParser()
            route = routing_chain.invoke(llm_input)

            logger.info(f" -> Lộ trình được quyết định: '{route}'")
            logger.info(f" -> Context for LLM : {context_for_llm}")

            # -----------------------------
            # Memory: Lưu lại quyết định định tuyến và truy vấn người dùng
            # -----------------------------
            if self.enable_memory and self.memory:
                self.memory.add_user_query(
                    query=state['user_request'],
                    task_type="general"
                )
                self.memory.add_agent_response(
                    response=f"Routing decision: {route}",
                    metadata={},
                    task_type="general"
                )

            return {
                "route": route,
            }
        def summarization_node(state: ParentGraphState):
            """Thực thi subgraph tóm tắt sử dụng SummarizationService."""
            logger.info("--- 2a. EXECUTING: Subgraph Tóm tắt ---")
            
            user_request = state["user_request"] 
            username = state["username"]
            selected_document_id = state["selected_document_id"]
            
            try:
                if not selected_document_id or not user_request:
                    logger.warning("Cần cung cấp document_id và yêu cầu người dùng.")
                    return {"answer": "Cần cung cấp document_id và yêu cầu người dùng."}
                titles =  None
                logger.info(f"Memory entries: {self.memory.entries}")
                if self.enable_memory and self.memory:
                    for entry in reversed(self.memory.entries):
                        if entry.task_type == "summary" and entry.metadata.get("document_id") == selected_document_id:
                            titles = entry.metadata.get("titles", None)
                            logger.info(f"Found matched_document from memory: {selected_document_id}")
                            break
                
                # Get conversation context from memory
                context_for_llm = ""
                if self.enable_memory and self.memory:
                    context_for_llm = self.memory.get_context_for_llm(
                        task_type="summary"
                    )
                
                summary, titles = self.summarization_service.generate_summary(
                    user_request=user_request,
                    context_for_llm=context_for_llm,
                    username=username,
                    document_id=selected_document_id,
                    titles=titles
                )
                
                # Memory: Add agent response với metadata từ summary result
                if self.enable_memory and self.memory:
                    # Lưu response vào memory để lần sau có thể reference
                    self.memory.add_agent_response(
                        response=summary,
                        metadata={
                            "document_id": selected_document_id,
                            "titles": titles
                        },
                        task_type="summary"
                    )

                return {"answer": summary}
                
            except Exception as e:
                logger.error(f"Error in summarization_node: {e}")
                error_msg = f"Lỗi khi tạo bản tóm tắt: {str(e)}"
                
                if self.enable_memory and self.memory:
                    self.memory.add_agent_response(
                        response=error_msg,
                        task_type="summary"
                    )
                
                return {"answer": error_msg}

        def rag_qa_node(state: ParentGraphState):
            """Trả lời câu hỏi dựa trên tài liệu (RAG) với metadata filtering."""
            logger.info("--- 2c. EXECUTING: Subgraph RAG Q&A ---")
            query = state["user_request"]
            username = state["username"]
            selected_document_id = state["selected_document_id"]

            logger.info(f"RAG Q&A query: {query}, username: {username}, document_id: {selected_document_id}")
            try:
                if not query or not query.strip():
                    logger.warning("Câu hỏi không hợp lệ. Vui lòng nhập lại.")
                    return {"answer": "Câu hỏi không hợp lệ. Vui lòng nhập lại."}
                
                # Prepare metadata filter for document-specific and user-specific queries
                metadata_filter = {
                    "$and": [
                        {"document_id": selected_document_id},
                        {"username": username}
                    ]
                }
                logger.info(f"Applying metadata filter: {metadata_filter}")
                
                # Generate RAG response with metadata filtering
                response = self.rag_service.generate_rag_response(query, filter=metadata_filter)
                logger.info(f"RAG Q&A response: {response}")
                
                # Memory: Add agent response
                if self.enable_memory and self.memory:
                    self.memory.add_agent_response(
                        response=response,
                        task_type="rag"
                    )
                
                return {"answer": response}
            except Exception as e:
                logger.error(f"Error in generate_rag_response tool: {str(e)}")
                return {"answer": f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"}

        def quiz_generation_node(state: ParentGraphState):
            """Sinh câu hỏi kiểm tra dựa trên tài liệu với metadata filtering."""
            logger.info("--- 2d. EXECUTING: Subgraph Quiz Generation ---")
            user_request = state["user_request"]
            username = state["username"]
            selected_document_id = state["selected_document_id"]

            logger.info(f"Quiz generation username: {username}, user_request: {user_request}, selected_document_id: {selected_document_id}")

            try:
                if not selected_document_id or not user_request:
                    logger.warning("Cần cung cấp document_id và yêu cầu người dùng.")
                    return {"answer": "Cần cung cấp document_id và yêu cầu người dùng."}

                # Get table of contents
                toc_data = self.document_management_service.get_table_of_contents(username=username, document_id=selected_document_id)
                logger.debug(f"TOC data: {toc_data}")
                if not toc_data:
                    logger.warning(f"Không tìm thấy mục lục cho tài liệu: {selected_document_id}")
                    return {"answer": f"Không tìm thấy mục lục cho tài liệu: {selected_document_id}"}

                # Generate quiz with username for metadata filtering
                result = self.quiz_generation_service.generate_quiz_set(
                    document_id=selected_document_id,
                    username=username,
                    user_request=user_request,
                    toc_data=toc_data
                )
                logger.info(f"Quiz generation result: {result}")

                # Memory: Add agent response
                if self.enable_memory and self.memory:
                    self.memory.add_agent_response(
                        response=result,
                        document_id=selected_document_id,
                        task_type="quiz"
                    )

                # Random hint messages for next step
                next_step_hint_list = [
                    "Bạn có muốn mình giúp bạn tạo form từ bộ câu hỏi này không? Mình có thể giúp bạn tạo Google Form từ bộ câu hỏi này.",
                    "Nếu bạn muốn tạo form từ bộ câu hỏi này, mình sẵn sàng giúp bạn.",
                    "Mình có thể giúp bạn tạo Google Form từ bộ câu hỏi này, bạn có muốn không?",
                ]
                next_step_hint = random.choice(next_step_hint_list)
                result += f"\n\n{next_step_hint}"
                return {"answer": result}
            except Exception as e:
                logger.error(f"Error in generate_quiz_set tool: {str(e)}")
                return {"answer": f"Đã xảy ra lỗi khi tạo đề: {str(e)}"}
        
        def create_form_node(state: ParentGraphState):
            """Node chính để tạo Google Form từ quiz data."""
            logger.info("--- 2e. EXECUTING: Subgraph Create Google Form ---")
            username = state["username"]
 
            # Kiểm tra điều kiện trước khi tạo form
            validation_result = self._check_google_authentication(username=username)
            if validation_result is not None:
                logger.info(f"Người dùng {username} chưa đăng nhập.")
                return {"answer": validation_result}
            
            quizset_data = self.database_service.get_quizset(username=username)
            if quizset_data is None:
                logger.info("Chưa có bộ câu hỏi để tạo form.")
                return {"answer": "Trước khi tạo bộ đề kiểm tra, mình sẽ giúp bạn tạo bộ câu hỏi nhé! Bạn muốn tạo bộ câu hỏi về chủ đề gì?"}

            # Tạo Google Form
            try:
                form_link = self._create_google_form(quizset_data=quizset_data, username=username)
                return {"answer": f"Mình đã tạo xong Google Form cho bạn rồi nhé! Đây là đường dẫn của form: \n{form_link}"}
            except Exception as e:
                error_msg = f"Error creating Google Form: {str(e)}"
                logger.error(error_msg)
                return {"answer": "Đã xảy ra lỗi, vui lòng thử lại sau."}


        # --- Xây dựng và Compile Parent Graph ---
        workflow = StateGraph(ParentGraphState)
        workflow.add_node("router", router_node) 
        workflow.add_node("summarization", summarization_node) 
        workflow.add_node("rag_qa", rag_qa_node)
        workflow.add_node("quiz_generation", quiz_generation_node)
        workflow.add_node("create_form", create_form_node)
        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            decide_route,
            {
                "summarization": "summarization",
                "quiz_generation": "quiz_generation",
                "create_form": "create_form",
                "rag_qa": "rag_qa",
                 None: END
            }
        )

        workflow.add_edge("summarization", END)
        workflow.add_edge("quiz_generation", END)
        workflow.add_edge("create_form", END)
        workflow.add_edge("rag_qa", END)

        # Compile đồ thị
        logger.info("TeacherAgent workflow graph compiled.")
        return workflow.compile()

    def _check_google_authentication(self, username: str) -> Optional[str]:
        """
        Kiểm tra trạng thái đăng nhập Google của người dùng.
        
        Returns:
            str: Thông báo lỗi nếu chưa đăng nhập, None nếu đã đăng nhập
        """

        token_path = Path(StorageConstants.get_user_token_path(username))
        
        store = file.Storage(token_path)
        try:
            creds = store.get()
        except Exception:
            creds = None
            
        if not creds:
            logger.warning("Người dùng chưa đăng nhập.")
            return "Bạn hãy đăng nhập vào tài khoản Google và cấp quyền cho ứng dụng nhé!"
        
        # TODO: Kiểm tra token hết hạn chưa, nếu hết hạn thì yêu cầu đăng nhập lại
        return None
        return None

    def _create_google_form(self, quizset_data: QuizQuestionOutput, username: str) -> str:
        """
        Tạo Google Form hoàn chỉnh từ quiz data.
        
        Returns:
            str: Link của Google Form đã tạo
            
        Raises:
            Exception: Nếu có lỗi trong quá trình tạo form
        """

        
        # 1. Chuyển đổi quiz data thành Google Form schema
        try:            
            converter = QuizToGoogleFormConverter()
            google_form_schema = converter.convert_quiz_to_google_form(quizset_data)
            logger.info(f"Converted quiz data to Google Form schema for user: {username}")
        except Exception as e:
            error_msg = f"Error converting quiz to Google Form schema: {str(e)}"
            logger.error(error_msg)
            raise Exception("Đã xảy ra lỗi khi chuyển đổi bộ câu hỏi sang dạng form. Vui lòng thử lại sau.")
        
        # 2. Xây dựng Google Forms service với authentication
        store = file.Storage(StorageConstants.get_user_token_path(username))
        creds = store.get()
        
        DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"
        form_service = discovery.build(
            "forms",
            "v1",
            http=creds.authorize(Http()),
            discoveryServiceUrl=DISCOVERY_DOC,
            static_discovery=False,
        )
        
        # 3. Tạo form ban đầu
        result = form_service.forms().create(body=google_form_schema.get("form_creation")).execute()
        form_id = result['formId']
        logger.info(f"Created form with ID: {form_id}")
        
        # 4. Thêm câu hỏi vào form
        form_service.forms().batchUpdate(
            formId=form_id, 
            body=google_form_schema.get("items_requests")
        ).execute()
        logger.info(f"Added questions to form ID: {form_id}")
        
        # 5. Publish form và cho phép nhận phản hồi
        publish_settings_body = {
            "publishSettings": {
                "publishState": {
                    "isPublished": True,
                    "isAcceptingResponses": True
                }
            }
        }
        
        form_service.forms().setPublishSettings(
            formId=form_id,
            body=publish_settings_body
        ).execute()
        logger.info(f"Published form ID: {form_id} and set to accept responses")
        
        # 6. Lấy link phản hồi của form
        form_result = form_service.forms().get(formId=form_id).execute()
        link_form = form_result['responderUri']
        logger.info(f"Created Google Form: {link_form}")
        
        return link_form

    def handle_chat_query(self, query: str, username: str, selected_document_id: str, chat_history: Optional[List] = None) -> str:
        """
        Handle a chat query by routing to the appropriate subgraph.
        Args:
            query: The user's query string.
            chat_history: Optional list of previous chat messages.
            username: Username for user-specific operations.
            selected_document_id: The ID of the document selected by the user.

        Returns:
            The response string from the agent.
        """
        
        try:
            # Note: User query is added in router_node with task_type based on routing decision
            
            # Prepare state for workflow
            state: ParentGraphState = {
                "user_request": query,
                "table_of_contents": None,
                "answer": None,
                "route": "",
                "username": username,
                "selected_document_id": selected_document_id
            }

            # Invoke the workflow
            result = self.workflow.invoke(state)
            
            # Get the answer from result
            answer = result.get("answer", "Không thể xử lý yêu cầu.")
            
            return answer

        except Exception as e:
            logger.error(f"Error handling chat query: {str(e)}")
            return "Đã xảy ra lỗi khi xử lý yêu cầu."
    
    def get_memory_context(self,  
                           document_id: Optional[str] = None,
                           task_type: Optional[str] = None) -> str:
        """
        Get memory context for LLM
        
        Args:
            max_tokens: Maximum tokens for context
            
        Returns:
            Context string
        """
        if not self.enable_memory or not self.memory:
            return ""

        return self.memory.get_context_for_llm(document_id=document_id, task_type=task_type)


# --- Logic quyết định rẽ nhánh ---
def decide_route(state: ParentGraphState):
    """Hàm quyết định sẽ đi theo nhánh nào."""
    return state["route"]

