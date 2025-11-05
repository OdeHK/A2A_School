"""
UI Integration Service for handling Gradio interface operations.
This service acts as a bridge between the UI and the core RAG services.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from apiclient import discovery
from httplib2 import Http
from oauth2client import client, file, tools

from services.database_service import DatabaseService
from services.quiz_generation.quiz_generation import QuizGenerationService
from services.quiz_generation.converter import QuizToGoogleFormConverter
from services.rag.rag_service import RagService
from services.document_processing.document_chunker import ChunkingStrategyType
from services.document_processing.document_management_service import DocumentManagementService
from services.agent.agent_service import TeacherAgent

# 🚀 Import Memory System
from services.memory.memory_service import HybridMemoryService

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
        self.agent_service: Optional[TeacherAgent] = None
        self.processing_status: Dict[str, Any] = {}
        
        # 🚀 FEATURE 1: Memory System
        self.memory_services: Dict[str, HybridMemoryService] = {}  # session_id -> memory
        
        # Initialize services in correct order
        self._initialize_rag_service()
        self._initialize_database_service()
        self._initialize_document_management_service()
        self._initialize_quiz_generation_service()
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
        If database connection fails, set to None and log warning (not critical).
        """
        try:
            self.database_service = DatabaseService()
            logger.info("Database service initialized successfully")
        except Exception as e:
            self.database_service = None
            logger.warning(f"Database service initialization failed: {str(e)}")
            logger.warning("App will run without database features (authentication, user data)")
            # Don't raise - allow app to run without database 

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
                logger.error("Cannot initialize quiz generation service: RAG service is None")
                self.quiz_generation_service = None
        except Exception as e:
            logger.error(f"Error initializing quiz generation service: {str(e)}")
            self.quiz_generation_service = None

    def _initialize_agent_service(self):
        """
        Initialize the agent service with all required services.
        """
        try:
            # Ensure all required services are available
            if self.rag_service and self.quiz_generation_service and self.doc_management_service:
                self.agent_service = TeacherAgent(
                    rag_service=self.rag_service,
                    quiz_generation_service=self.quiz_generation_service,
                    document_management_service=self.doc_management_service,
                    llm_service=self.rag_service.llm_service
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
            # Check if PDF and get page count
            import fitz
            from pathlib import Path
            
            file_ext = Path(uploaded_file_path).suffix.lower()
            
            # Auto-detect large PDF and use optimized processor
            if file_ext == '.pdf':
                try:
                    doc = fitz.open(uploaded_file_path)
                    page_count = doc.page_count
                    doc.close()
                    
                    # Use ultra-fast processor for large PDFs (1000+ pages)
                    if page_count >= 1000:
                        logger.info(f"🔥 Large PDF detected ({page_count} pages) - Using optimized processor!")
                        
                        # Progress callback for UI feedback
                        def progress_callback(current, total, message):
                            logger.info(f"Progress: {current}% - {message}")
                        
                        result = self.doc_management_service.process_ultra_large_pdf(
                            file_path=uploaded_file_path,
                            username=username,
                            rag_service=self.rag_service,
                            force_reprocess=False,
                            progress_callback=progress_callback
                        )
                        
                        return (f"⚡ Đã xử lý NHANH: {result.file_name}\n"
                               f"📄 Số trang: {result.metadata.page_count if result.metadata else 'N/A'}\n"
                               f"🔪 Số đoạn: {result.metadata.chunk_count if result.metadata else 'N/A'}\n"
                               f"⏱️ Thời gian: {result.metadata.processing_time:.1f}s\n"
                               f"🚀 Tốc độ: {result.metadata.page_count/result.metadata.processing_time:.1f} trang/giây\n")
                except Exception as pdf_error:
                    logger.warning(f"Cannot detect page count: {pdf_error}")
            
            # Use standard processor for small files
            result = self.doc_management_service.process_uploaded_document(
                file_path=uploaded_file_path,
                username=username,
                rag_service=self.rag_service,
                extract_toc=True
            )
            
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
        # Ensure temp folder exists
        temp_folder = Path("session_data/temp")
        temp_folder.mkdir(parents=True, exist_ok=True)
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
                import os
                secret_path = os.getenv("GOOGLE_CLIENT_SECRET_PATH", str(temp_folder / "client_secret_vscode.json"))
                flow = client.flow_from_clientsecrets(secret_path, SCOPES)
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
            form_service.forms().batchUpdate(
                formId=result["formId"], body=google_form_schema.get("items_requests")
            ).execute()
            logger.info(f"Added questions to form ID: {result['formId']}")

            # Set form to be published and accepting responses
            publish_settings_body = {
                "publishSettings": {
                    "publishState": {"isPublished": True, "isAcceptingResponses": True}
                }
            }
            form_service.forms().setPublishSettings(
                formId=result["formId"], body=publish_settings_body
            ).execute()
            logger.info(f"Published form ID: {result['formId']} and set to accept responses")

            # Prints the result to show the question has been added
            form_result = form_service.forms().get(formId=result["formId"]).execute()
            link_form = form_result["responderUri"]
            logger.info(f"Created Google Form: {link_form}")

            return (
                f"Mình đã tạo xong Google Form cho bạn rồi nhé! Đây là đường dẫn của form: \n{link_form}"
            )
        except Exception as e:
            error_msg = f"Error creating Google Form: {str(e)}"
            logger.error(error_msg)
            return (
                f"Có lỗi xảy ra khi đăng nhập vào tài khoản Google. Bạn hãy thử lại nhé!"
            )

    # Sign in to Google account 
    def logout_google(self) -> Tuple[bool, str]:
        """
        Logout from Google by deleting the token file.
        
        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            temp_folder = Path("session_data/temp")
            token_file = temp_folder / "token.json"
            
            if token_file.exists():
                token_file.unlink()
                logger.info("Google token deleted successfully")
                return True, "Đã đăng xuất Google thành công"
            else:
                return True, "Chưa đăng nhập Google"
                
        except Exception as e:
            error_msg = f"Error logging out from Google: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def open_sign_in_website(self, force_reauth: bool = False) -> Tuple[bool, str]:
        """
        Open the Google sign-in website for authentication.
        
        Args:
            force_reauth: If True, force re-authentication even if valid token exists
        
        Returns:
            boolean: True if the user successfully signed in, False otherwise
            str: the Google account name if sign-in is successful,
        """
        try:
            temp_folder = Path("session_data/temp")
            temp_folder.mkdir(parents=True, exist_ok=True)
            SCOPES = [
                "https://www.googleapis.com/auth/forms.body",
                "https://www.googleapis.com/auth/userinfo.profile"
            ]
            store = file.Storage(temp_folder / "token.json")
            
            # If force_reauth, delete existing token
            if force_reauth:
                token_file = temp_folder / "token.json"
                if token_file.exists():
                    token_file.unlink()
                    logger.info("Deleted existing token for re-authentication")
                creds = None
            else:
                try:
                    creds = store.get()
                except Exception:
                    creds = None

            if not creds or creds.invalid:
                # support overriding client secret path via env var
                import os
                secret_path = os.getenv("GOOGLE_CLIENT_SECRET_PATH", str(temp_folder / "client_secret_vscode.json"))
                flow = client.flow_from_clientsecrets(secret_path, SCOPES)
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
        Falls back to default credentials if database is not available.
        
        Args:
            username: Username to authenticate
            password: Password to verify
            
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Try database authentication first
            if self.database_service:
                return self.database_service.authenticate_user(username, password)
            
            # 🔧 FALLBACK: Use default credentials when database not available
            logger.warning("Database service not available for authentication")
            logger.warning("Using fallback authentication with default credentials")
            
            # Default credentials (ONLY for demo/development)
            default_credentials = {
                "admin": "admin",
                "demo": "demo123",
                "teacher": "teacher123",
                "student": "student123"
            }
            
            if username in default_credentials:
                is_valid = default_credentials[username] == password
                if is_valid:
                    logger.info(f"✅ Fallback authentication successful for user: {username}")
                else:
                    logger.warning(f"❌ Fallback authentication failed for user: {username}")
                return is_valid
            
            logger.warning(f"❌ User '{username}' not found in fallback credentials")
            return False
            
        except Exception as e:
            logger.error(f"Error in authenticate_user: {str(e)}")
            return False
    
    def register_user(self, username: str, password: str, email: str = None, full_name: str = None) -> tuple[bool, str]:
        """
        Register a new user.
        
        Args:
            username: Unique username
            password: User password
            email: User email (optional)
            full_name: User's full name (optional)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if not self.database_service:
                logger.error("Database service not available for registration")
                return False, "Hệ thống không khả dụng"
            
            return self.database_service.create_user(username, password, email, full_name)
            
        except Exception as e:
            logger.error(f"Error in register_user: {str(e)}")
            return False, f"Lỗi đăng ký: {str(e)}"
    
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
            "agent_service_initialized": self.agent_service is not None,
            "documents_processed": len(self.processing_status),
            "memory_sessions": len(self.memory_services)  # 🚀 Memory stats
            #"agent_service_status": self.agent_service.get_service_status() if self.agent_service else {},
        }
    
    # ========================================
    # 🚀 MEMORY SYSTEM METHODS
    # ========================================
    
    def get_or_create_memory(self, session_id: str) -> HybridMemoryService:
        """
        Get or create memory service for a session.
        
        Args:
            session_id: User session ID (username or UUID)
            
        Returns:
            HybridMemoryService instance
        """
        if session_id not in self.memory_services:
            # Get MongoDB collection for conversation history
            if self.database_service and self.database_service.db:
                conversation_collection = self.database_service.db['conversation_history']
            else:
                logger.warning("Database not available, memory will not be persisted")
                conversation_collection = None
            
            # Create new memory service
            self.memory_services[session_id] = HybridMemoryService(
                mongo_collection=conversation_collection,
                llm=self.rag_service.llm_service.llm if self.rag_service else None,
                session_id=session_id,
                buffer_size=10,
                summarize_threshold=10
            )
            logger.info(f"✅ Created memory service for session: {session_id}")
        
        return self.memory_services[session_id]
    
    def add_to_conversation_memory(self, session_id: str, user_message: str, ai_response: str):
        """
        Add user message and AI response to conversation memory.
        
        Args:
            session_id: User session ID
            user_message: User's message
            ai_response: AI's response
        """
        memory = self.get_or_create_memory(session_id)
        memory.add_user_message(user_message)
        memory.add_ai_message(ai_response)
        
        logger.debug(f"💬 Added to memory: U={len(user_message)} chars, AI={len(ai_response)} chars")
    
    def get_conversation_context(self, session_id: str, include_summary: bool = True) -> str:
        """
        Get conversation context for RAG queries.
        
        Args:
            session_id: User session ID
            include_summary: Include conversation summary
            
        Returns:
            Formatted conversation context
        """
        memory = self.get_or_create_memory(session_id)
        return memory.get_context(include_summary=include_summary)
    
    def clear_conversation_memory(self, session_id: str):
        """
        Clear conversation memory for a session.
        
        Args:
            session_id: User session ID
        """
        if session_id in self.memory_services:
            self.memory_services[session_id].clear_session()
            del self.memory_services[session_id]
            logger.info(f"🗑️ Cleared memory for session: {session_id}")
    
    def get_memory_statistics(self, session_id: str) -> Dict:
        """
        Get memory statistics for a session.
        
        Args:
            session_id: User session ID
            
        Returns:
            Dictionary with memory stats
        """
        if session_id in self.memory_services:
            return self.memory_services[session_id].get_statistics()
        return {
            "session_id": session_id,
            "total_messages": 0,
            "buffer_messages": 0,
            "has_summary": False
        }
    
    def cleanup(self):
        """
        Cleanup all initialized service objects to release resources.
        Also cleanup temporary Word files.
        """
        try:
            # Cleanup temporary Word files
            from pathlib import Path
            temp_folder = Path("session_data/temp")
            if temp_folder.exists():
                for word_file in temp_folder.glob("quiz_*.docx"):
                    try:
                        word_file.unlink()
                        logger.info(f"Cleaned up Word file: {word_file}")
                    except Exception as e:
                        logger.error(f"Error deleting Word file {word_file}: {e}")
        except Exception as e:
            logger.error(f"Error during Word file cleanup: {e}")
        
        # Cleanup service objects
        try:
            del self.rag_service
            del self.doc_management_service
            del self.quiz_generation_service
            del self.agent_service
            del self.database_service
            del self.processing_status
            logger.info("UIIntegrationService resources have been cleaned up.")
        except Exception as e:
            logger.error(f"Error during service cleanup: {e}")
