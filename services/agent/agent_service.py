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

from services.document_processing import document_library
from services.prompt import router_node_prompt, find_document_node_prompt, summarize_content_node_prompt
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from services.quiz_generation.converter import QuizToGoogleFormConverter
from services.quiz_generation.quiz_to_word_converter import QuizToWordConverter
from services.rag.rag_service import RagService
from services.quiz_generation.quiz_generation import QuizGenerationService
from services.document_processing.document_management_service import DocumentManagementService
from services.rag.llm_service import LLMService

# Logger toàn cục cho module này
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)    
   

# --- Định nghĩa State cho Parent Graph ---
class ParentGraphState(TypedDict):
    user_request: str
    table_of_contents: Optional[list]
    answer: Optional[str]  # Add answer field for quiz and rag results
    word_file: Optional[str]  # Add word_file field for Word download
    route: str 
    username: str  # Add username field for user-specific operations 
    selected_document_id: str  # Add selected_document_id field for user-specific operations

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
            username = state["username"]
            """Phân loại yêu cầu và quyết định lộ trình."""
            logger.info("--- 1. ROUTER: Phân loại yêu cầu ---")
            logger.info(f"User {username} request: {state['user_request']}")
            
            llm = self.llm_service.get_llm()
            routing_chain =  router_node_prompt | llm | StrOutputParser()
            route = routing_chain.invoke({"user_request": state["user_request"]})
            logger.info(f" -> Lộ trình được quyết định: '{route}'")
            
            return {"route": route}

        def summarizer_node(state: ParentGraphState):
            """Thực thi subgraph tóm tắt."""
            logger.info("--- 2a. EXECUTING: Subgraph Tóm tắt ---")
            selected_document_id = state["selected_document_id"]
            username = state["username"]

            # Check which section to summarize
            document_library = self.document_management_service.get_document_library(username=username)
            llm = self.llm_service.get_llm()
            find_document_chain = find_document_node_prompt | llm | JsonOutputParser()
            library_str = json.dumps(document_library, indent=2)
            matched_document = find_document_chain.invoke({
                "library_str": library_str,
                "user_request": state["user_request"]
            })
            logger.info(f"Matched document: {matched_document}")
            
            # Validate matched_document structure
            if not matched_document or not isinstance(matched_document, dict) or "title" not in matched_document:
                logger.warning(f"Invalid matched document structure: {matched_document}")
                return {"answer": "Không tìm thấy nội dung bạn đề cập trong tài liệu."}

            # Get content data which contains the actual content
            content_result = self.document_management_service.get_content_data(username=username, document_id=selected_document_id)
            
            if not content_result or "content" not in content_result:
                logger.warning(f"No content data found for document: {selected_document_id}")
                return { "answer": "Không tìm thấy nội dung để tóm tắt."}
            
            content_data = content_result["content"]
            
            # Find content by title - handle both single title and list
            title = matched_document["title"]
            if isinstance(title, list) and len(title) > 0:
                title = title[0]
            elif not isinstance(title, str):
                logger.warning(f"Invalid title format: {title}")
                return {"answer": "Không tìm thấy nội dung để tóm tắt."}
            
            # Handle special case: full_document
            if title == "full_document":
                logger.info("User requested summary of full document")
                # Combine all content items into one text
                all_content = []
                for content_item in content_data:
                    item_title = content_item.get("title", "")
                    item_content = content_item.get("content", "")
                    if item_content:
                        all_content.append(f"**{item_title}**\n{item_content}")
                
                if not all_content:
                    return {"answer": "Không tìm thấy nội dung để tóm tắt."}
                
                # Combine and summarize (limit to avoid token overflow)
                combined_text = "\n\n".join(all_content[:20])  # First 20 sections
                logger.info(f"Combined content from {min(20, len(all_content))} sections, total {len(combined_text)} chars")
                
                llm = self.llm_service.get_llm()
                chain = summarize_content_node_prompt | llm
                summary = chain.invoke({"input_text": combined_text})
                logger.info(f"Full document summary generated")
                
                return {"answer": f"Tóm tắt toàn bộ tài liệu:\n\n{summary.content}"}
            
            # Normal case: find specific section
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
            return {"answer": summary.content}

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
                toc_string = self.document_management_service.get_table_of_contents_as_string(username=username, document_id=selected_document_id)
                logger.info(f"TOC string: {toc_string}")
                if not toc_string:
                    logger.warning(f"Không tìm thấy mục lục cho tài liệu: {selected_document_id}")
                    return {"answer": f"Không tìm thấy mục lục cho tài liệu: {selected_document_id}"}

                # Generate quiz with username for metadata filtering
                result = self.quiz_generation_service.generate_quiz_set(
                    document_id=selected_document_id,
                    username=username,
                    user_request=user_request,
                    toc_data=toc_string
                )
                logger.info(f"Quiz generation result: {result}")

                # Random hint messages for next step
                next_step_hint_list = [
                    "Bạn có muốn mình giúp bạn tạo form từ bộ câu hỏi này không? Mình có thể giúp bạn tạo Google Form từ bộ câu hỏi này. Hoặc nếu bạn muốn tải file Word, hãy nhắn 'tải word'.",
                    "Nếu bạn muốn tạo form từ bộ câu hỏi này, mình sẵn sàng giúp bạn. Bạn cũng có thể tải file Word bằng cách nhắn 'tải word'.",
                    "Mình có thể giúp bạn tạo Google Form từ bộ câu hỏi này, bạn có muốn không? Hoặc bạn có thể tải file Word bằng cách nhắn 'tải word'.",
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
            
            # Kiểm tra điều kiện trước khi tạo form
            validation_result = self._validate_form_creation_prerequisites()
            if validation_result is not None:
                return {"answer": validation_result}
            
            # Tạo Google Form
            try:
                form_link = self._create_google_form()
                return {"answer": f"Mình đã tạo xong Google Form cho bạn rồi nhé! Đây là đường dẫn của form: \n{form_link}"}
            except Exception as e:
                error_msg = f"Error creating Google Form: {str(e)}"
                logger.error(error_msg)
                return {"answer": "Đã xảy ra lỗi, vui lòng thử lại sau."}

        def download_word_node(state: ParentGraphState) -> ParentGraphState:
            """
            Tải file Word từ quiz_data.json
            """
            logger.info("=== download_word_node ===")
            
            # Check if quiz_data.json exists
            quiz_file = Path("session_data/temp/quiz_data.json")
            if not quiz_file.exists():
                return {
                    "answer": "Chưa có quiz nào được tạo. Vui lòng tạo quiz trước khi tải file Word."
                }
            
            try:
                # Create Word file from quiz
                word_converter = QuizToWordConverter()
                word_file_path = word_converter.create_word_from_quiz_file(str(quiz_file))
                
                if word_file_path and Path(word_file_path).exists():
                    return {
                        "answer": f"FILE_DOWNLOAD:{word_file_path}",  # Special format for UI to detect file download
                        "word_file": word_file_path  # Store file path in state
                    }
                else:
                    return {"answer": "Không thể tạo file Word. Vui lòng thử lại."}
                    
            except Exception as e:
                error_msg = f"Error creating Word file: {str(e)}"
                logger.error(error_msg)
                return {"answer": f"Đã xảy ra lỗi khi tạo file Word: {str(e)}"}


        # --- Xây dựng và Compile Parent Graph ---
        workflow = StateGraph(ParentGraphState)
        workflow.add_node("router", router_node) 
        workflow.add_node("summarizer", summarizer_node) 
        workflow.add_node("rag_qa", rag_qa_node)
        workflow.add_node("quiz_generation", quiz_generation_node)
        workflow.add_node("create_form", create_form_node)
        workflow.add_node("download_word", download_word_node)
        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            decide_route,
            {
                "summarizer": "summarizer",
                "quiz_generation": "quiz_generation",
                "create_form": "create_form",
                "rag_qa": "rag_qa",
                "download_word": "download_word"
            }
        )

        workflow.add_edge("summarizer", END)
        workflow.add_edge("quiz_generation", END)
        workflow.add_edge("create_form", END)
        workflow.add_edge("rag_qa", END)
        workflow.add_edge("download_word", END)

        # Compile đồ thị
        logger.info("TeacherAgent workflow graph compiled.")
        return workflow.compile()

    def _validate_form_creation_prerequisites(self) -> Optional[str]:
        """
        Kiểm tra các điều kiện cần thiết trước khi tạo Google Form.
        
        Returns:
            str: Thông báo lỗi nếu có, None nếu tất cả điều kiện đều thỏa mãn
        """
        temp_folder = Path("session_data/temp")
        quiz_data_path = temp_folder / "quiz_data.json"
        
        # Kiểm tra file quiz_data.json có tồn tại không
        if not quiz_data_path.exists():
            logger.warning("File quiz_data.json không tồn tại. Vui lòng tạo đề trước.")
            return "Trước khi tạo bộ đề kiểm tra, mình sẽ giúp bạn tạo bộ câu hỏi nhé! Bạn muốn tạo bộ câu hỏi về chủ đề gì?"
        
        # Kiểm tra người dùng đã đăng nhập chưa
        auth_error = self._check_google_authentication()
        if auth_error:
            return auth_error
            
        return None

    def _check_google_authentication(self) -> Optional[str]:
        """
        Kiểm tra trạng thái đăng nhập Google của người dùng.
        
        Returns:
            str: Thông báo lỗi nếu chưa đăng nhập, None nếu đã đăng nhập
        """
        temp_folder = Path("session_data/temp")
        token_path = temp_folder / "token.json"
        
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

    def _create_google_form(self) -> str:
        """
        Tạo Google Form hoàn chỉnh từ quiz data.
        
        Returns:
            str: Link của Google Form đã tạo
            
        Raises:
            Exception: Nếu có lỗi trong quá trình tạo form
        """
        temp_folder = Path("session_data/temp")
        quiz_data_path = temp_folder / "quiz_data.json"
        token_path = temp_folder / "token.json"
        
        # 1. Chuyển đổi quiz data thành Google Form schema
        try:
            converter = QuizToGoogleFormConverter()
            google_form_schema = converter.convert_file_to_google_form(input_file=str(quiz_data_path))
        except Exception as e:
            error_msg = f"Error converting quiz to Google Form schema: {str(e)}"
            logger.error(error_msg)
            raise Exception("Đã xảy ra lỗi khi chuyển đổi bộ câu hỏi sang dạng form. Vui lòng thử lại sau.")
        
        # 2. Xây dựng Google Forms service với authentication
        store = file.Storage(token_path)
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
        


# --- Logic quyết định rẽ nhánh ---
def decide_route(state: ParentGraphState):
    """Hàm quyết định sẽ đi theo nhánh nào."""
    return state["route"]

