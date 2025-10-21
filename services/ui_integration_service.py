import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from apiclient import discovery
from httplib2 import Http
from oauth2client import client, file, tools

from services.database_service import DatabaseService
from services.quiz_generation.quiz_generation import QuizGenerationService
from services.summarization.summarization import SummarizationService
from services.quiz_generation.converter import QuizToGoogleFormConverter
from services.rag.rag_service import RagService
from services.document_processing.document_chunker import ChunkingStrategyType
from services.document_processing.document_management_service import DocumentManagementService
from services.agent.agent_service import TeacherAgent

logger = logging.getLogger(__name__)


class UIIntegrationService:
    """
    Service to handle UI operations and integrate with RAG pipeline.
    """
    
    def __init__(self):
        """Initialize the UI integration service."""
        self.rag_service: Optional[RagService] = None
        self.doc_management_service: Optional[DocumentManagementService] = None
        self.quiz_generation_service: Optional[QuizGenerationService] = None
        self.summarization_service: Optional[SummarizationService] = None
        self.agent_service: Optional[TeacherAgent] = None
        self.processing_status: Dict[str, Any] = {}
        
        # Initialize services in correct order
        self._initialize_rag_service()
        self._initialize_database_service()
        self._initialize_document_management_service()
        self._initialize_quiz_generation_service()
        self._initialize_summarization_service()
        self._initialize_agent_service()

    def _initialize_rag_service(self, chunker_strategy: str = "ONE_PAGE") -> None:
        """
        Initialize or reinitialize the RAG service with specified configuration.
        
        Args:
            chunker_strategy: The chunking strategy to use
        """
        try:
            # Map UI strategy names to enum values
            strategy_mapping = {
                "ONE_PAGE": ChunkingStrategyType.ONE_PAGE_PER_CHUNK,
                "RECURSIVE_CHARACTER_TEXT_SPLITTER": ChunkingStrategyType.RECURSIVE_SPLIT,
                "LLM_SPLITTER": ChunkingStrategyType.LLM_SPLIT
            }
            
            strategy = strategy_mapping.get(chunker_strategy, ChunkingStrategyType.ONE_PAGE_PER_CHUNK)
            
            self.rag_service = RagService()
            self.rag_service.update_chunking_strategy(strategy)
            
            logger.info(f"RAG service initialized with strategy: {chunker_strategy}")
            
        except Exception as e:
            logger.error(f"Error initializing RAG service: {str(e)}")
            # Initialize with default settings as fallback
            self.rag_service = RagService()
            
    def _initialize_database_service(self):
        """
        Initialize or reinitialize the database service.
        """
        try:
            self.database_service = DatabaseService()
            logger.info("Database service initialized")
        except Exception as e:
            self.database_service = None
            logger.error(f"Error initializing database service: {str(e)}")
            raise e 

    def _initialize_document_management_service(self):
        """
        Initialize or reinitialize the document management service.
        """
        try:
            # TODO: Modify DocumentManagementService to accept loader and chunker strategies
            if not self.database_service:
                self._initialize_database_service()
            self.doc_management_service = DocumentManagementService(database_service=self.database_service)
            logger.info("Document management service initialized")
        except Exception as e:
            self.doc_management_service = None
            logger.error(f"Error initializing document management service: {str(e)}") 

    def _initialize_quiz_generation_service(self):
        """
        Initialize or reinitialize the quiz generation service.
        """
        try:
            # Ensure RAG service is initialized first
            if not self.rag_service:
                self._initialize_rag_service()
            
            # Check again after initialization
            if self.rag_service:
                self.quiz_generation_service = QuizGenerationService(rag_service=self.rag_service)
                logger.info("Quiz generation service initialized")
            else:
                logger.error("Failed to initialize RAG service, quiz generation service cannot be initialized")
                self.quiz_generation_service = None
        except Exception as e:
            logger.error(f"Error initializing quiz generation service: {str(e)}")
            self.quiz_generation_service = None

    def _initialize_summarization_service(self):
        """
        Initialize or reinitialize the summarization service.
        """
        try:
            # Ensure RAG service and document management service are initialized first
            if not self.rag_service:
                self._initialize_rag_service()
            
            if not self.doc_management_service:
                self._initialize_document_management_service()
            
            # Check again after initialization
            if self.rag_service and self.doc_management_service:
                self.summarization_service = SummarizationService(
                    rag_service=self.rag_service,
                    document_management_service=self.doc_management_service
                )
                logger.info("Summarization service initialized")
            else:
                logger.error("Failed to initialize required services, summarization service cannot be initialized")
                self.summarization_service = None
        except Exception as e:
            logger.error(f"Error initializing summarization service: {str(e)}")
            self.summarization_service = None

    def _initialize_agent_service(self):
        """
        Initialize the agent service with all required services.
        """
        try:
            # Ensure all required services are available
            if (self.rag_service and 
                self.quiz_generation_service and 
                self.summarization_service and 
                self.doc_management_service):
                self.agent_service = TeacherAgent(
                    rag_service=self.rag_service,
                    quiz_generation_service=self.quiz_generation_service,
                    summarization_service=self.summarization_service,
                    document_management_service=self.doc_management_service,
                    llm_service=self.rag_service.llm_service,
                    enable_memory=True
                )
                logger.info("Agent service initialized successfully")
            else:
                logger.warning("Cannot initialize agent service: Required services not available")
                self.agent_service = None
        except Exception as e:
            logger.error(f"Error initializing agent service: {str(e)}")
            self.agent_service = None

    def process_uploaded_document(self, uploaded_file_path: str, username: str): 
        """Handle file upload from Gradio interface using DocumentManagementService."""

        if not self.doc_management_service:
            return "Document management service not available", "Error"
        
        try:
            # Use the document management service to process the uploaded file
            # TODO: Determine which file types to support
            result = self.doc_management_service.process_uploaded_document(file_path=uploaded_file_path,
                                                                  username=username,
                                                                  rag_service=self.rag_service,
                                                                  extract_toc=True)
            

            return (f"✅ Đã xử lý thành công: {result.file_name}\n"
                   f"📄 Số trang: {result.metadata.page_count if result.metadata else 'N/A'}\n"
                   f"🔪 Số đoạn: {result.metadata.chunk_count if result.metadata else 'N/A'}\n")
        except Exception as e:
            return f"❌ Error: {str(e)}", "Error"
    
    def handle_url_input(self, url: str) :
        """
        Handle URL input (for future Google Drive integration).
        
        Args:
            url: The URL to add
            
        Returns:
            Tuple of (updated_file_list, cleared_url_input, status_message)
        """
        #TODO: Implement URL handling logic
        pass
    
    def update_chunker_strategy(self, strategy: str) -> str:
        """
        Update the chunking strategy and reinitialize services.
        
        Args:
            strategy: New chunking strategy
            
        Returns:
            Status message
        """
        try:
            # Reinitialize RAG service with new strategy
            self._initialize_rag_service(strategy)
            
            # Reinitialize quiz generation service
            self._initialize_quiz_generation_service()
            
            # Reinitialize summarization service
            self._initialize_summarization_service()
            
            # Reinitialize agent service
            self._initialize_agent_service()
            
            return f"✅ Chunking strategy updated to: {strategy}"
        except Exception as e:
            error_msg = f"Error updating chunker strategy: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def update_loader_strategy(self, loader: str) -> str:
        """
        Update the loader strategy.
        
        Args:
            loader: New loader strategy
            
        Returns:
            Status message
        """
        try:
            # For now, just log the change
            logger.info(f"Loader strategy changed to: {loader}")
            return f"✅ Loader strategy updated to: {loader}"
        except Exception as e:
            error_msg = f"Error updating loader strategy: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

    def handle_chat_query(self, query: str, chat_history: List, username: str, selected_document_id: str) -> str:
        """
        Handle chat queries using Agent Service.
        
        Args:
            query: User query
            chat_history: Current chat history
            username: Username for user-specific operations
            selected_document_id: Document ID selected by user for context
        Returns:
            Response string from the agent
        """
        try:
            if not query or not query.strip():
                return "🤖 Vui lòng nhập câu hỏi."
            
            # Check if agent service is ready
            if not self.agent_service:
                # Try to initialize if not ready
                self._initialize_agent_service()
                
                if not self.agent_service:
                    error_response = "🤖 Dịch vụ AI chưa sẵn sàng. Vui lòng thử lại sau."
                    return error_response
            
            # Use agent service to handle the chat
            response = self.agent_service.handle_chat_query(query=query, username=username, selected_document_id=selected_document_id, chat_history=chat_history)
            
            return response
            
        except Exception as e:
            error_msg = f"Error in chat query: {str(e)}"
            logger.error(error_msg)
            return f"🤖 Xin lỗi, đã có lỗi xảy ra: {error_msg}"

    def create_google_form_from_quiz(self) -> str:

        temp_folder = Path("session_data/temp")
        input_file = temp_folder / "quiz_data.json"
        converter = QuizToGoogleFormConverter()
        # TODO: Kiểm tra nội dung của file đã có chưa, nếu trống thì báo lỗi

        try: 
            google_form_schema = converter.convert_file_to_google_form(input_file=str(input_file))
        except Exception as e:
            error_msg = f"Error converting quiz to Google Form schema: {str(e)}"
            logger.error(error_msg)
            return f"Có lỗi xảy ra khi tạo Google Form. Bạn hãy thử tạo lại bộ đề kiểm tra nhé!"
        
        SCOPES = "https://www.googleapis.com/auth/forms.body"
        DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"
        try:
            store = file.Storage(temp_folder / "token.json")
            creds = None
            if not creds or creds.invalid:
                flow = client.flow_from_clientsecrets(temp_folder / "client_secret_vscode.json", SCOPES)
                creds = tools.run_flow(flow, store)

            form_service = discovery.build(
                "forms",
                "v1",
                http=creds.authorize(Http()),
                discoveryServiceUrl=DISCOVERY_DOC,
                static_discovery=False,
            )

            # Creates the initial form
            result = form_service.forms().create(body=google_form_schema.get("form_creation")).execute()
            logger.info(f"Created form with ID: {result['formId']}")
            
            # Adds the question to the form
            question_setting = (
                form_service.forms()
                .batchUpdate(formId=result["formId"], body=google_form_schema.get("items_requests"))
                .execute()
            )
            logger.info(f"Added questions to form ID: {result['formId']}")

            # Set form to be published and accepting responses
            publish_settings_body = {
                "publishSettings": {
                    "publishState": {
                    "isPublished": True,
                    "isAcceptingResponses": True
                    }
                }
            }
            published_settings = (
                form_service.forms().setPublishSettings(
                    formId=result["formId"],
                    body=publish_settings_body
                ).execute()
            ) 
            logger.info(f"Published form ID: {result['formId']} and set to accept responses")

            # Prints the result to show the question has been added
            form_result = form_service.forms().get(formId=result["formId"]).execute()
            link_form = form_result['responderUri']
            logger.info(f"Created Google Form: {link_form}")

            return (f"Mình đã tạo xong Google Form cho bạn rồi nhé! Đây là đường dẫn của form: \n{link_form}")
        except Exception as e:
            error_msg = f"Error creating Google Form: {str(e)}"
            logger.error(error_msg)
            return f"Có lỗi xảy ra khi đăng nhập vào tài khoản Google. Bạn hãy thử lại nhé!"

    # Sign in to Google account 
    def open_sign_in_website(self) -> Tuple[bool, str]:
        """
        Open the Google sign-in website for authentication.
        
        Returns:
            boolean: True if the user successfully signed in, False otherwise
            str: the Google account name if sign-in is successful,
        """
        try:
            temp_folder = Path("session_data/temp")
            SCOPES = [
                "https://www.googleapis.com/auth/forms.body",
                "https://www.googleapis.com/auth/userinfo.profile"
            ]
            store = file.Storage(temp_folder / "token.json")
            try:
                creds = store.get()
            except Exception:
                creds = None

            if not creds or creds.invalid:
                flow = client.flow_from_clientsecrets(temp_folder / "client_secret_vscode.json", SCOPES)
                creds = tools.run_flow(flow, store)
            
            # Lấy thông tin người dùng từ Google
            try:
                oauth2_service = discovery.build('oauth2', 'v2', http=creds.authorize(Http()))
                user_info = oauth2_service.userinfo().get().execute()
                user_name = user_info.get('name', 'Người dùng Google')
                user_email = user_info.get('email', '')
                
                logger.info(f"Google authentication successful for user: {user_name} ({user_email})")
                return (True, user_name)
                
            except Exception as e:
                logger.warning(f"Could not retrieve user info: {str(e)}")
                logger.info("Google authentication successful")
                return (True, "Người dùng Google")
                
        except Exception as e:
            error_msg = f"Error during Google authentication: {str(e)}"
            logger.error(error_msg)
            return (False, f"Lỗi đăng nhập Google: {error_msg}")


    def find_document_id_by_filename(self, username: str, selected_filename: str) -> Tuple[Optional[str], str]:
        """
        Find document_id based on username and selected filename.
        
        Args:
            username: The username to search documents for
            selected_filename: The filename selected by user from UI
            
        Returns:
            Tuple of (document_id or None, status_message)
        """
        try:
            if not selected_filename or not selected_filename.strip():
                return None, "Không có tài liệu nào được chọn"
            
            # Convert filename to document_id using document management service
            if not self.doc_management_service:
                logger.error("Document management service not available")
                return None, "❌ Dịch vụ quản lý tài liệu không khả dụng"
            
            document_id_dict = self.doc_management_service.get_document_id_dict(username=username)

            # Find document_id by matching filename
            selected_document_id = None
            for doc_id, filename in document_id_dict.items():
                if filename == selected_filename:
                    selected_document_id = doc_id
                    break
            
            if selected_document_id:
                logger.info(f"Found document: {selected_filename} -> document_id: {selected_document_id}")
                return selected_document_id, f"✅ Đã chọn tài liệu: {selected_filename}"
            else:
                logger.warning(f"Cannot find document_id for filename: {selected_filename}")
                return None, f"❌ Không tìm thấy ID cho tài liệu: {selected_filename}"
                
        except Exception as e:
            error_msg = f"Error finding document_id: {str(e)}"
            logger.error(error_msg)
            return None, f"❌ {error_msg}"
    
    def get_user_files(self, username: str) -> List[str]:
        """
        Get list of filenames for a specific user.
        
        Args:
            username: The username to get files for
            
        Returns:
            List of filenames
        """
        try:
            if not self.doc_management_service:
                logger.error("Document management service not available")
                return []
            
            document_id_dict = self.doc_management_service.get_document_id_dict(username=username)
            filenames = list(document_id_dict.values())
            logger.info(f"Found {len(filenames)} files for user {username}")
            return filenames
            
        except Exception as e:
            logger.error(f"Error getting user files: {str(e)}")
            return []
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """
        Authenticate user credentials using database service.
        
        Args:
            username: Username to authenticate
            password: Password to verify
            
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            if not self.database_service:
                logger.error("Database service not available for authentication")
                return False
            
            return self.database_service.authenticate_user(username, password)
            
        except Exception as e:
            logger.error(f"Error in authenticate_user: {str(e)}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current service status.
        
        Returns:
            Service status information
        """
        return {
            "rag_service_initialized": self.rag_service is not None,
            "doc_management_service_initialized": self.doc_management_service is not None,
            "quiz_generation_service_initialized": self.quiz_generation_service is not None,
            "summarization_service_initialized": self.summarization_service is not None,
            "agent_service_initialized": self.agent_service is not None,
            "documents_processed": len(self.processing_status)
            #"agent_service_status": self.agent_service.get_service_status() if self.agent_service else {},
        }
    def cleanup(self):
        """
        Cleanup all initialized service objects to release resources.
        """
        del self.rag_service
        del self.doc_management_service
        del self.quiz_generation_service
        del self.agent_service
        del self.database_service
        del self.processing_status
        logger.info("UIIntegrationService resources have been cleaned up.")
