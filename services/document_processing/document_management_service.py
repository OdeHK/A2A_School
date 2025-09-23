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
from .toc_extractor import TOCExtractor
from .document_loader import DocumentLoader, DocumentType
from .document_chunker import DocumentChunker, ChunkingStrategyType
from .document_library import generate_document_id


logger = logging.getLogger(__name__)


class DocumentManagementService:
    """
    Service that orchestrates the complete document processing workflow.
    Handles upload, metadata extraction, ToC extraction, chunking, and RAG integration.
    """
    
    def __init__(
        self,
        repository: Optional[DocumentRepository] = None,
        toc_extractor: Optional[TOCExtractor] = None,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[DocumentChunker] = None
    ):
        """
        Initialize document processing service.
        
        Args:
            repository: Document repository for storage
            toc_extractor: Enhanced table of contents extractor
            loader: Document loader
            chunker: Document chunker
        """
        self.repository = repository or DocumentRepository()
        self.toc_extractor = toc_extractor or TOCExtractor()
        
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
            if extract_toc:
                logger.info("Extracting table of contents...")
                extraction_result = self.toc_extractor.extract_toc_and_content(
                    str(stored_file_path)
                )
                
                # Save TOC structure data và content data vào session
                toc_structure_data = extraction_result.toc_structure.to_dict()
                content_data = extraction_result.content_data.to_dict()
                
                self.repository.save_toc_structure_data(stored_document_id, toc_structure_data)
                self.repository.save_content_data(stored_document_id, content_data)
                
                logger.info(f"Extracted ToC with {len(extraction_result.toc_structure.sections)} sections")
                logger.info(f"Generated content for {len(extraction_result.content_data.content)} items")
                logger.info(f"Saved TOC structure and content data to session")
            
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
            
            # Add document to library
            document_titles = []
            if extract_toc and 'extraction_result' in locals():
                try:
                    # Extract titles from TOC structure - now it's nested format
                    document_titles = self._extract_titles_from_toc_structure(extraction_result.toc_structure.sections)
                except Exception as e:
                    logger.warning(f"Could not extract titles for document library: {e}")
            
            self.repository.add_document_to_library(
                document_id=stored_document_id,
                name=file_path_obj.stem,  # File name without extension
                path=str(stored_file_path),
                title=document_titles
            )
            
            logger.info(f"Added document {stored_document_id} to document library")
            
            # Create success result
            result = ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                document_id=stored_document_id,
                file_name=file_path_obj.name,
                message=f"Successfully processed {file_path_obj.name}",
                metadata=metadata,
                table_of_contents=None  
            )
            
            logger.info(f"Successfully processed document {stored_document_id}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            
            # Update metadata with error status if stored_document_id exists
            if 'stored_document_id' in locals():
                try:
                    metadata = self.repository.get_document_metadata(stored_document_id)
                    if metadata:
                        metadata.processing_status = ProcessingStatus.FAILED
                        metadata.error_message = str(e)
                        self.repository.save_document_metadata(metadata)
                except Exception as update_error:
                    logger.error(f"Failed to update error status: {update_error}")
            
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                document_id=locals().get('stored_document_id', locals().get('document_id', "unknown")),
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
        Get table of contents for document (created from TOC structure data).
        
        Args:
            document_id: Document identifier
            
        Returns:
            Table of contents or None if not found
        """
        # Lấy TOC structure data thay vì legacy TOC
        toc_structure_data = self.repository.get_toc_structure_data(document_id)
        if not toc_structure_data:
            return None
        # Tạo TableOfContents từ TOC structure data
        return self._create_table_of_contents_from_structure_data(document_id, toc_structure_data)
    
    def get_table_of_contents_as_string(self, document_id: str) -> Optional[str]:
        """
        Get table of contents for document formatted as string.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Table of contents formatted as string or None if not found
        """
        # Lấy TOC structure data trực tiếp
        toc_structure_data = self.repository.get_toc_structure_data(document_id)
        if not toc_structure_data:
            return None
        
        return self._format_toc_structure_as_string(document_id, toc_structure_data)
    
    def list_session_documents(self) -> List[DocumentMetadata]:
        """
        List all documents in current session.
        
        Returns:
            List of document metadata
        """
        return self.repository.list_session_documents()
    
    def get_content_data(self, document_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get content data from TOC extractor for document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of content items or None if not found
        """
        return self.repository.get_content_data(document_id)
    
    def get_toc_structure_data(self, document_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get TOC structure data from TOC extractor for document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of TOC structure items or None if not found
        """
        return self.repository.get_toc_structure_data(document_id)
    
    
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
    
    def _extract_titles_from_toc_structure(self, sections: List[Any]) -> List[str]:
        """
        Extract all titles from nested TOC structure.
        
        Args:
            sections: List of TOC sections with nested children
            
        Returns:
            List of all section titles
        """
        titles = []
        
        def extract_from_section(section):
            """Recursively extract titles from section and children."""
            if hasattr(section, 'section_title'):
                titles.append(section.section_title)
            elif isinstance(section, dict) and 'section_title' in section:
                titles.append(section['section_title'])
            
            # Process children
            children = getattr(section, 'children', section.get('children', []) if isinstance(section, dict) else [])
            for child in children:
                extract_from_section(child)
        
        # Process all sections
        for section in sections:
            extract_from_section(section)
        
        return titles
    
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
    
    def _create_table_of_contents_from_structure_data(self, document_id: str, 
                                                    toc_structure_data: List[Dict[str, Any]]) -> TableOfContents:
        """
        Create TableOfContents object from TOC structure data.
        
        Args:
            document_id: Document identifier
            toc_structure_data: List of TOC structure items
            
        Returns:
            TableOfContents object
        """
        # Convert structure data to TocSection objects
        sections = []
        
        for item_data in toc_structure_data:
            if item_data.get('level') == 1:  # Only add root level sections
                # Skip items without required fields
                if 'id' not in item_data or 'title' not in item_data or 'level' not in item_data:
                    continue
                    
                section = TocSection(
                    section_id=item_data['id'],
                    section_title=item_data['title'],
                    parent_section_id=item_data.get('parent_id'),
                    level=item_data['level'],
                    page_number=item_data.get('page'),
                    children=self._convert_children_from_structure_data(
                        item_data.get('children_ids', []), 
                        toc_structure_data
                    )
                )
                sections.append(section)
        
        return TableOfContents(
            document_id=document_id,
            extraction_method='enhanced_textrank',
            extraction_date=datetime.now(),
            sections=sections,
            raw_text=f"Enhanced extraction with {len(toc_structure_data)} sections"
        )
    
    def _convert_children_from_structure_data(self, children_ids: List[str], 
                                            all_structure_data: List[Dict[str, Any]]) -> List[TocSection]:
        """Convert children IDs to TocSection objects from structure data."""
        children = []
        
        # Create ID mapping - only include items with required fields
        id_to_item = {item['id']: item for item in all_structure_data if 'id' in item}
        
        for child_id in children_ids:
            if child_id in id_to_item:
                child_item = id_to_item[child_id]
                # Skip items without required fields
                if 'title' not in child_item or 'level' not in child_item:
                    continue
                    
                child_section = TocSection(
                    section_id=child_item['id'],
                    section_title=child_item['title'],
                    parent_section_id=child_item.get('parent_id'),
                    level=child_item['level'],
                    page_number=child_item.get('page'),
                    children=self._convert_children_from_structure_data(
                        child_item.get('children_ids', []), 
                        all_structure_data
                    )
                )
                children.append(child_section)
        
        return children
    
    def _format_toc_structure_as_string(self, document_id: str, toc_structure_data: List[Dict[str, Any]]) -> str:
        """
        Format TOC structure data as a readable string.
        
        Args:
            document_id: Document identifier
            toc_structure_data: List of TOC structure items
            
        Returns:
            Formatted string representation
        """
        if not toc_structure_data:
            return "No table of contents available."
        
        result = []
        result.append(f"Table of Contents for Document: {document_id}")
        result.append(f"Extraction Method: enhanced_textrank")
        result.append(f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result.append("-" * 50)
        
        # Group by level and format
        level_1_items = [item for item in toc_structure_data if item.get('level') == 1]
        
        for index, item in enumerate(level_1_items):
            result.extend(self._format_structure_item_as_string(item, toc_structure_data, str(index + 1)))
        
        return "\n".join(result)
    
    def _format_structure_item_as_string(self, item: Dict[str, Any], 
                                       all_structure_data: List[Dict[str, Any]], 
                                       section_index: str) -> List[str]:
        """
        Format a single TOC structure item as string lines.
        
        Args:
            item: TOC structure item to format
            all_structure_data: All structure data for children lookup
            section_index: Current section index
            
        Returns:
            List of formatted string lines
        """
        # Skip items without required fields
        if 'title' not in item:
            return []
            
        page_info = f" (Page {item['page']})" if item.get('page') else ""
        line = f"{section_index} {item['title']}{page_info}"
        
        result = [line]
        
        # Format children
        children_ids = item.get('children_ids', [])
        id_to_item = {item['id']: item for item in all_structure_data if 'id' in item}
        
        for child_index, child_id in enumerate(children_ids):
            if child_id in id_to_item:
                child_item = id_to_item[child_id]
                result.extend(self._format_structure_item_as_string(
                    child_item, 
                    all_structure_data, 
                    f"{section_index}.{child_index + 1}"
                ))

        return result
    
    # === Document Library Management Methods ===
    
    def get_document_library(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current session's document library (all documents).
        
        Returns:
            Dictionary with document_id as key and document info as value
        """
        return self.repository.list_all_documents_in_library()
    
    def get_document_from_library(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get specific document from library by name.
        
        Args:
            name: Document name
            
        Returns:
            Document information or None if not found
        """
        return self.repository.get_document_from_library(name)
    
    def get_document_library_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the document library.
        
        Returns:
            Summary information including document count and list of documents
        """
        library = self.repository.list_all_documents_in_library()
        
        documents_info = []
        for name, doc_info in library.items():
            documents_info.append({
                'name': doc_info['name'],
                'title_count': len(doc_info.get('title', []))
            })
        
        return {
            'total_documents': len(library),
            'documents': documents_info,
            'session_id': self.repository.get_current_session_id()
        }
    
    def add_external_document_to_library(self, file_path: str, extract_bookmarks: bool = True) -> str:
        """
        Add an external document to the library without full processing.
        Useful for referencing documents that don't need RAG processing.
        
        Args:
            file_path: Path to the document file
            extract_bookmarks: Whether to extract PDF bookmarks for title
            
        Returns:
            Generated document_id
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Generate document ID
            doc_name = file_path_obj.stem
            doc_id = generate_document_id(doc_name, str(file_path))
            
            # Extract bookmarks if requested and file is PDF
            titles = []
            if extract_bookmarks and file_path_obj.suffix.lower() == '.pdf':
                try:
                    from PyPDF2 import PdfReader
                    from .document_library import get_all_bookmark_titles
                    
                    reader = PdfReader(str(file_path))
                    titles = get_all_bookmark_titles(reader.outline)
                    
                except Exception as e:
                    logger.warning(f"Could not extract bookmarks from {file_path}: {e}")
            
            # Add to library
            self.repository.add_document_to_library(
                document_id=doc_id,
                name=doc_name,
                path=str(file_path),
                title=titles
            )
            
            logger.info(f"Added external document {doc_id} to library")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding external document to library: {e}")
            raise
    
    def remove_document_from_library(self, document_id: str) -> bool:
        """
        Remove a document from the library.
        
        Args:
            document_id: Document identifier to remove
            
        Returns:
            True if removed, False if not found
        """
        return self.repository.remove_document_from_library(document_id)
    
    def search_documents_in_library(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for documents in the library by name or title.
        
        Args:
            query: Search query
            
        Returns:
            List of matching documents
        """
        library = self.repository.list_all_documents_in_library()
        query_lower = query.lower()
        
        matching_docs = []
        
        for name, doc_info in library.items():
            # Search in name
            if query_lower in doc_info['name'].lower():
                matching_docs.append({
                    'name': doc_info['name'],
                    'match_type': 'name'
                })
                continue
            
            # Search in titles
            for title in doc_info.get('title', []):
                if query_lower in title.lower():
                    matching_docs.append({
                        'name': doc_info['name'],
                        'match_type': 'title',
                        'matched_title': title
                    })
                    break
        
        return matching_docs
    
    def get_document_info_from_library(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a document from the library.
        
        Args:
            name: Document name
            
        Returns:
            Document information or None if not found
        """
        doc_info = self.repository.get_document_from_library(name)
        
        if not doc_info:
            return None
        
        # Add additional information (removed document_id and path fields)
        result = {
            'name': doc_info['name'],
            'title': doc_info.get('title', []),
            'title_count': len(doc_info.get('title', []))
        }
        
        # Note: Metadata checking removed since we no longer use document_id as reference
        result['is_processed'] = False
        
        return result