from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.rag.rag_service import RagService
"""
Document Processing Service - orchestrates document upload, processing, and metadata extraction.
This service handles the complete workflow from file upload to making documents ready for RAG.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from langchain.schema.document import Document

from services.models import (
    DocumentMetadata, 
    TableOfContents, 
    TocSection,
    ProcessingResult, 
    ProcessingStatus
)
from .document_repository import DocumentRepository
from .toc_extractor import TableOfContentsExtractor
from .document_loader import DocumentLoader, DocumentType
from .document_chunker import DocumentChunker, ChunkingStrategyType


logger = logging.getLogger(__name__)


class DocumentManagementService:
    """
    Service that orchestrates the complete document processing workflow.
    Handles upload, metadata extraction, ToC extraction, chunking, and RAG integration.
    """
    
    def __init__(
        self,
        repository: Optional[DocumentRepository] = None,
        toc_extractor: Optional[TableOfContentsExtractor] = None,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[DocumentChunker] = None
    ):
        """
        Initialize document processing service.
        
        Args:
            repository: Document repository for storage
            toc_extractor: Table of contents extractor
            loader: Document loader
            chunker: Document chunker
        """
        self.repository = repository or DocumentRepository()
        self.toc_extractor = toc_extractor or TableOfContentsExtractor()
        
        # Initialize loader with default PDF configuration
        self.loader = loader or DocumentLoader.create_with_config(
            document_type=DocumentType.PDF
        )
        
        # Initialize chunker with default strategy
        self.chunker = chunker or DocumentChunker.create_with_strategy_type(
            ChunkingStrategyType.ONE_PAGE_PER_CHUNK
        )
    
    def process_uploaded_document(
        self, 
        file_path: str, 
        rag_service: Optional["RagService"] = None,
        extract_toc: bool = True
    ) -> ProcessingResult:

        from services.rag.rag_service import RagService
        """
        Process an uploaded document through the complete pipeline.
        
        Args:
            file_path: Path to uploaded file
            rag_service: Optional RAG service for vector storage
            extract_toc: Whether to extract table of contents
            
        Returns:
            ProcessingResult with status and metadata
        """
        document_id = None
        
        try:
            logger.info(f"Starting document processing for: {file_path}")
            
            # Generate document ID and store file
            document_id = self._generate_document_id()
            stored_document_id = self.repository.store_uploaded_file(file_path, document_id)
            
            # Get stored file path
            stored_file_path = self.repository.get_document_file_path(stored_document_id)
            if not stored_file_path:
                raise ValueError("Failed to store uploaded file")
            
            # Create initial metadata
            file_path_obj = Path(file_path)
            metadata = DocumentMetadata(
                document_id=stored_document_id,
                file_name=file_path_obj.name,
                file_path=str(stored_file_path),
                file_size=file_path_obj.stat().st_size,
                upload_date=datetime.now(),
                processing_status=ProcessingStatus.PROCESSING
            )
            
            self.repository.save_document_metadata(metadata)
            
            # Load documents
            logger.info("Loading document pages...")
            docs = self.loader.lazy_load(str(stored_file_path))
            docs_list = list(docs)
            
            if not docs_list:
                raise ValueError("No documents were loaded from the file")
            
            logger.info(f"Loaded {len(docs_list)} document pages")
            
            # Extract table of contents if requested
            toc = None
            if extract_toc:
                logger.info("Extracting table of contents...")
                toc = self.toc_extractor.extract_table_of_contents(
                    str(stored_file_path), 
                    stored_document_id
                )
                self.repository.save_table_of_contents(stored_document_id, toc)
                logger.info(f"Extracted ToC with {len(toc.sections)} sections")
            
            # Chunk documents
            logger.info("Chunking documents...")
            chunks = self.chunker.chunk(iter(docs_list))
            
            if not chunks:
                raise ValueError("No chunks were created from the documents")
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Add chunks to RAG service if provided
            if rag_service:
                logger.info("Adding chunks to vector store...")
                rag_service.add_document_chunks_to_vector_store(chunks)
            
            # Update metadata with final status
            metadata.processing_status = ProcessingStatus.COMPLETED
            metadata.chunk_count = len(chunks)
            metadata.page_count = len(docs_list)
            self.repository.save_document_metadata(metadata)
            
            # Create success result
            result = ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                document_id=stored_document_id,
                file_name=file_path_obj.name,
                message=f"Successfully processed {file_path_obj.name}",
                metadata=metadata,
                table_of_contents=toc
            )
            
            logger.info(f"Successfully processed document {stored_document_id}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            
            # Update metadata with error status if document_id exists
            if document_id:
                try:
                    metadata = self.repository.get_document_metadata(document_id)
                    if metadata:
                        metadata.processing_status = ProcessingStatus.FAILED
                        metadata.error_message = str(e)
                        self.repository.save_document_metadata(metadata)
                except Exception as update_error:
                    logger.error(f"Failed to update error status: {update_error}")
            
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                document_id=document_id or "unknown",
                file_name=Path(file_path).name,
                message=error_msg,
                error=str(e)
            )
    
    def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Get document metadata by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document metadata or None if not found
        """
        return self.repository.get_document_metadata(document_id)
    
    def get_table_of_contents(self, document_id: str) -> Optional[TableOfContents]:
        """
        Get table of contents for document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Table of contents or None if not found
        """
        return self.repository.get_table_of_contents(document_id)
    
    def get_table_of_contents_as_string(self, document_id: str) -> Optional[str]:
        """
        Get table of contents for document formatted as string.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Table of contents formatted as string or None if not found
        """
        toc = self.repository.get_table_of_contents(document_id)
        if not toc:
            return None
        
        return self._format_toc_as_string(toc)
    
    def list_session_documents(self) -> List[DocumentMetadata]:
        """
        List all documents in current session.
        
        Returns:
            List of document metadata
        """
        return self.repository.list_session_documents()
    
    
    def get_document_id_dict(self) -> Dict[str, str]:
        """
        Get a dictionary mapping document IDs to file names for current session.
        Returns:
            Dictionary of document_id -> file_name
        """
        document_metadata_list = self.repository.list_session_documents()
        return {doc.document_id: doc.file_name for doc in document_metadata_list}
    
    def update_chunking_strategy(self, strategy_type: ChunkingStrategyType) -> None:
        """
        Update the chunking strategy.
        
        Args:
            strategy_type: New chunking strategy type
        """
        try:
            self.chunker = DocumentChunker.create_with_strategy_type(strategy_type)
            logger.info(f"Updated chunking strategy to: {strategy_type}")
        except Exception as e:
            logger.error(f"Error updating chunking strategy: {str(e)}")
            raise
    
    def get_current_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self.repository.get_current_session_id()
    
    def create_new_session(self) -> str:
        """Create new session."""
        return self.repository.create_new_session()
    
    def load_session(self, session_id: str) -> bool:
        """Load existing session."""
        return self.repository.load_session(session_id)
    
    def get_vector_store_path(self) -> Optional[str]:
        """Get vector store path for current session."""
        return self.repository.get_vector_store_path()
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        self.repository.cleanup_temp_files()
    
    def _generate_document_id(self) -> str:
        """Generate unique document ID."""
        return f"doc_{uuid.uuid4().hex[:8]}"
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current service status.
        
        Returns:
            Status information
        """
        return {
            "repository_initialized": self.repository is not None,
            "toc_extractor_initialized": self.toc_extractor is not None,
            "loader_initialized": self.loader is not None,
            "chunker_initialized": self.chunker is not None,
            "current_session": self.repository.get_current_session_id(),
            "chunker_strategy": (
                self.chunker.strategy.strategy_name 
                if hasattr(self.chunker, 'strategy') else "unknown"
            )
        }
    
    def _format_toc_as_string(self, toc: TableOfContents) -> str:
        """
        Format TableOfContents object as a readable string.
        
        Args:
            toc: TableOfContents object
            
        Returns:
            Formatted string representation
        """
        if not toc.sections:
            return "No table of contents available."
        
        result = []
        result.append(f"Table of Contents for Document: {toc.document_id}")
        result.append(f"Extraction Method: {toc.extraction_method}")
        result.append(f"Extracted on: {toc.extraction_date}")
        result.append("-" * 50)
        
        # Format sections recursively
        for index, section in enumerate(toc.sections):
            result.extend(self._format_section_as_string(section, section_index=str(index + 1)))
        return "\n".join(result)
    
    def _format_section_as_string(self, section: TocSection, section_index: str) -> List[str]:
        """
        Format a single TocSection as string lines.
        
        Args:
            section: TocSection to format
            indent_level: Current indentation level
            
        Returns:
            List of formatted string lines
        """

        page_info = f" (Page {section.page_number})" if section.page_number else ""
        line = f"{section_index} {section.section_title}{page_info}"
        
        result = [line]
        
        # Recursively format children
        for child_index, child in enumerate(section.children):
            result.extend(self._format_section_as_string(child, section_index=f"{section_index}.{child_index + 1}"))

        return result