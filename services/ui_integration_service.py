"""
UI Integration Service for handling Gradio interface operations.
This service acts as a bridge between the UI and the core RAG services.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from services.quiz_generation import QuizGenerationService
from services.rag.rag_service import RagService
from services.document_processing.document_chunker import ChunkingStrategyType
from services.document_processing.document_management_service import DocumentManagementService
from services.agent.agent_service_old import AgentService

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
        self.agent_service: Optional[AgentService] = None
        self.current_files: List[str] = []
        self.processing_status: Dict[str, Any] = {}
        self.selected_document: Optional[str] = None  # Track selected document filename
        self.selected_document_id: Optional[str] = None  # Track selected document ID
        
        # Initialize services in correct order
        self._initialize_rag_service()
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

    def _initialize_document_management_service(self):
        """
        Initialize or reinitialize the document management service.
        """
        try:
            # TODO: Modify DocumentManagementService to accept loader and chunker strategies
            self.doc_management_service = DocumentManagementService()
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
                self.agent_service = AgentService(
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

    def process_uploaded_document(self, uploaded_file_path:str): 
        """Handle file upload from Gradio interface using DocumentManagementService."""

        if not self.doc_management_service:
            return "Document management service not available", "Error"
        
        try:
            # Use the document management service to process the uploaded file
            # TODO: Determine which file types to support
            result = self.doc_management_service.process_uploaded_document(file_path=uploaded_file_path,
                                                                  rag_service=self.rag_service,
                                                                  extract_toc=True)
            
            self.current_files.append(result.file_name)

            return (f"✅ Đã xử lý thành công: {result.file_name}\n"
                   f"📄 Số trang: {result.metadata.page_count if result.metadata else 'N/A'}\n"
                   f"🔪 Số đoạn: {result.metadata.chunk_count if result.metadata else 'N/A'}\n")
        except Exception as e:
            return f"❌ Error: {str(e)}", "Error"
        

    def get_current_files(self) -> List[str]:
        """Get the current list of files."""
        return self.current_files
    
    
    def handle_url_input(self, url: str) -> Tuple[List[str], str, str]:
        """
        Handle URL input (for future Google Drive integration).
        
        Args:
            url: The URL to add
            
        Returns:
            Tuple of (updated_file_list, cleared_url_input, status_message)
        """
        try:
            if not url or not url.strip():
                return self.current_files, "", "No URL provided"
            
            # For now, just add URL to the list (implement Google Drive integration later)
            if url not in self.current_files:
                self.current_files.append(url)
                logger.info(f"Added URL to list: {url}")
            
            return self.current_files, "", f"URL added: {url}"
            
        except Exception as e:
            error_msg = f"Error handling URL: {str(e)}"
            logger.error(error_msg)
            return self.current_files, "", error_msg
    
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
    
    def handle_chat_query(self, query: str, chat_history: List) -> List:
        """
        Handle chat queries using Agent Service.
        
        Args:
            query: User query
            chat_history: Current chat history
        Returns:
            List: Updated chat history
        """
        try:
            if not query or not query.strip():
                return chat_history
            
            # Check if agent service is ready
            if not self.agent_service:
                # Try to initialize if not ready
                self._initialize_agent_service()
                
                if not self.agent_service:
                    error_response = "🤖 Dịch vụ AI chưa sẵn sàng. Vui lòng thử lại sau."
                    chat_history.append((query, error_response))
                    return chat_history
            
            # Use agent service to handle the chat
            # TODO: Pass selected_document_id when agent_service is updated to support it
            response, updated_history = self.agent_service.handle_chat_query(query, chat_history)
            
            return updated_history
            
        except Exception as e:
            error_msg = f"Error in chat query: {str(e)}"
            logger.error(error_msg)
            chat_history.append((query, f"🤖 Xin lỗi, đã có lỗi xảy ra: {error_msg}"))
            return chat_history
    
    def set_selected_document(self, selected_filename: str) -> str:
        """
        Set the selected document and convert filename to document_id.
        
        Args:
            selected_filename: The filename selected by user from UI
            
        Returns:
            Status message
        """
        try:
            if not selected_filename or not selected_filename.strip():
                self.selected_document = None
                self.selected_document_id = None
                return "Không có tài liệu nào được chọn"
            
            # Store selected filename
            self.selected_document = selected_filename
            
            # Convert filename to document_id using document management service
            if self.doc_management_service:
                document_id_dict = self.doc_management_service.get_document_id_dict()
                
                # Find document_id by matching filename
                selected_document_id = None
                for doc_id, filename in document_id_dict.items():
                    if filename == selected_filename:
                        selected_document_id = doc_id
                        break
                
                if selected_document_id:
                    self.selected_document_id = selected_document_id
                    logger.info(f"Selected document: {selected_filename} -> document_id: {selected_document_id}")
                    return f"✅ Đã chọn tài liệu: {selected_filename}"
                else:
                    logger.warning(f"Cannot find document_id for filename: {selected_filename}")
                    return f"❌ Không tìm thấy ID cho tài liệu: {selected_filename}"
            else:
                logger.error("Document management service not available")
                return "❌ Dịch vụ quản lý tài liệu không khả dụng"
                
        except Exception as e:
            error_msg = f"Error setting selected document: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def get_selected_document_id(self) -> Optional[str]:
        """
        Get the current selected document ID.
        
        Returns:
            Selected document ID or None if no document is selected
        """
        return self.selected_document_id
    
    def get_selected_document_filename(self) -> Optional[str]:
        """
        Get the current selected document filename.
        
        Returns:
            Selected document filename or None if no document is selected
        """
        return self.selected_document
    
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
            "files_loaded": len(self.current_files),
            "documents_processed": len(self.processing_status),
            "agent_service_status": self.agent_service.get_service_status() if self.agent_service else {},
        }
