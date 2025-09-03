"""
UI Integration Service for handling Gradio interface operations.
This service acts as a bridge between the UI and the core RAG services.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import tempfile
import shutil

from services.rag_service import RagService
from services.document_chunker import ChunkingStrategyType
from services.document_loader import DocumentType
from config.settings import get_settings

logger = logging.getLogger(__name__)


class UIIntegrationService:
    """
    Service to handle UI operations and integrate with RAG pipeline.
    """
    
    def __init__(self):
        """Initialize the UI integration service."""
        self.rag_service: Optional[RagService] = None
        self.current_files: List[str] = []
        self.processing_status: Dict[str, Any] = {}
        self._initialize_rag_service()
    
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
    
    def handle_file_upload(self, uploaded_file_path:str) -> Tuple[List[str], str]:
        """
        Handle file upload from Gradio interface.
        
        Args:
            uploaded_file: Gradio file upload object
            
        Returns:
            Tuple of (updated_file_list, status_message)
        """
        try:
            if uploaded_file_path is None:
                return self.current_files, "No file uploaded"
            
            # Get the file path from Gradio upload
            file_path = str(uploaded_file_path)
            file_name = Path(file_path).name
            
            # Validate file
            if not Path(file_path).exists():
                return self.current_files, f"Error: File {file_name} not found"
            
            # Check file type
            if not file_name.lower().endswith('.pdf'):
                return self.current_files, f"Error: Only PDF files are supported. Got: {file_name}"
            
            # Add to current files list if not already present
            if file_path not in self.current_files:
                self.current_files.append(file_path)
                logger.info(f"Added file to list: {file_name}")
            
            return self.current_files, f"File {file_name} added successfully"
            
        except Exception as e:
            error_msg = f"Error handling file upload: {str(e)}"
            logger.error(error_msg)
            return self.current_files, error_msg
    
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
    
    def process_selected_document(self, file_path:str) -> str:
        """
        Process the selected document through RAG pipeline.
        
        Args:
            selected_items: List of selected file paths or URLs
            
        Returns:
            Status message
        """
        try:
            if not file_path:
                return "No document selected for processing"
            
            # Check if it's a file path or URL
            if file_path.startswith('http'):
                return "URL processing not yet implemented. Please upload a PDF file."
            
            
            if not Path(file_path).exists():
                return f"File does not exist: {file_path}"
            
            # Ensure RAG service is initialized
            if not self.rag_service:
                self._initialize_rag_service()
                
            # Double check initialization
            if not self.rag_service:
                return "❌ Failed to initialize RAG service"
            
            # Process through RAG pipeline
            logger.info(f"Processing document: {file_path}")
            result = self.rag_service.process_uploaded_document(file_path)
            
            # Store processing status
            self.processing_status[file_path] = result
            
            if result["status"] == "success":
                return (f"✅ Successfully processed {result['file_name']}\n"
                       f"📄 Pages: {result['document_count']}\n"
                       f"🔪 Chunks: {result['chunk_count']}\n"
                       f"📝 Ready for querying!")
            else:
                return f"❌ Error processing {file_path}: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def update_chunker_strategy(self, strategy: str) -> str:
        """
        Update the chunking strategy.
        
        Args:
            strategy: New chunking strategy
            
        Returns:
            Status message
        """
        try:
            self._initialize_rag_service(strategy)
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
    
    def handle_chat_query(self, query: str, chat_history: List) -> Tuple[List, str]:
        """
        Handle chat queries and retrieve relevant documents.
        
        Args:
            query: User query
            chat_history: Current chat history
            
        Returns:
            Tuple of (updated_chat_history, cleared_input)
        """
        try:
            if not query or not query.strip():
                return chat_history, ""
            
            # Check if RAG service is ready
            if not self.rag_service or not self.processing_status:
                response = ("🤖 Xin chào! Để tôi có thể trả lời câu hỏi của bạn, "
                           "vui lòng upload và xử lý tài liệu trước. "
                           "Tôi sẽ phân tích tài liệu và trả lời dựa trên nội dung đó.")
                chat_history.append((query, response))
                return chat_history, ""
            
            # Ensure RAG service is properly initialized
            if not self.rag_service:
                self._initialize_rag_service()
                
            # Double check initialization
            if not self.rag_service:
                response = "🤖 Xin lỗi, không thể khởi tạo RAG service"
                chat_history.append((query, response))
                return chat_history, ""
            
            # Retrieve relevant documents and generate response
            response = self.rag_service.generate_rag_response(query)
            if response == "":
                response = "🤖 Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi"
            else:
                chat_history.append((query, response))

            return chat_history, ""
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            response = f"🤖 Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi: {error_msg}"
            chat_history.append((query, response))
            return chat_history, ""
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current service status.
        
        Returns:
            Service status information
        """
        rag_status = self.rag_service.get_service_status() if self.rag_service else {}
        
        return {
            "rag_service_initialized": self.rag_service is not None,
            "files_loaded": len(self.current_files),
            "documents_processed": len(self.processing_status),
            "rag_service_status": rag_status
        }
