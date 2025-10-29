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
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any
from services.models import (
    DocumentMetadata, 
    TableOfContents, 
    TableOfContentsSection,
    ProcessingResult, 
    ProcessingStatus
)
from .toc_extractor import TOCExtractor
from .document_loader import DocumentLoader, DocumentType
from .document_chunker import DocumentChunker, ChunkingStrategyType
from services.database_service import DatabaseService

logger = logging.getLogger(__name__)

class DocumentManagementService:
    """
    Service that orchestrates the complete document processing workflow.
    Handles upload, metadata extraction, ToC extraction, chunking, and RAG integration.
    """
    
    def __init__(
        self,
        database_service: DatabaseService,
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
        self.database_service = database_service
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
        username: str,
        rag_service: Optional["RagService"] = None,
        extract_toc: bool = True,
        loader_config: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process an uploaded document through the complete pipeline.
        
        Args:
            file_path: Path to uploaded file
            username: User identifier for document ownership
            rag_service: Optional RAG service for vector storage
            extract_toc: Whether to extract table of contents
            loader_config: Optional loader configuration dictionary
            
        Returns:
            ProcessingResult with status and metadata
        """

        # Generate document ID and store file
        document_id = self._generate_document_id()
        try:
            logger.info(f"Starting document processing for: {file_path}")
            
            # Create loader with specific config if provided
            if loader_config:
                logger.info(f"Using custom loader config: {loader_config}")
                from services.document_processing.document_loader import (
                    DocumentLoader, DocumentType, LoadingStrategyFactory
                )
                
                # Determine document type from file path
                if file_path.startswith("http"):
                    document_type = DocumentType.HTML
                else:
                    document_type = DocumentType.PDF
                
                # Create strategy with config
                strategy = LoadingStrategyFactory.create_strategy(document_type, loader_config)
                temp_loader = DocumentLoader(strategy=strategy)
            else:
                temp_loader = self.loader
            
            # Create initial metadata
            if file_path.startswith("http"):
                parsed_url = urlparse(file_path)
                file_name = parsed_url.path.split("/")[-1] or parsed_url.netloc
                # Extract name without extension for library
                doc_name = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                
                metadata = DocumentMetadata(
                    document_id=document_id,
                    file_name=file_name,
                    file_path=file_path,  
                    file_size=0,
                    upload_date=datetime.now(),
                    processing_status=ProcessingStatus.PROCESSING
                )
                source_for_loader = file_path  # Use original URL string
            else:
                file_path_obj = Path(file_path)
                doc_name = file_path_obj.stem
                file_name = file_path_obj.name
                
                metadata = DocumentMetadata(
                    document_id=document_id,
                    file_name=file_name,
                    file_path=str(file_path_obj),
                    file_size=file_path_obj.stat().st_size,
                    upload_date=datetime.now(),
                    processing_status=ProcessingStatus.PROCESSING
                )
                source_for_loader = str(file_path_obj)

            self.database_service.save_document_metadata(username=username, document_id=document_id, metadata=metadata)

            # Load documents
            logger.info(f"Loading document pages from: {source_for_loader}")
            docs = temp_loader.lazy_load(source_for_loader)
            docs_list = list(docs)
            if not docs_list:
                raise ValueError("No documents were loaded from the file")
            logger.info(f"Loaded {len(docs_list)} document pages")
            
            # Extract table of contents if requested
            if extract_toc:
                logger.info("Extracting table of contents...")
                
                # For websites, pass the document content
                website_content = None
                if file_path.startswith("http"):
                    # Combine all document content for website
                    website_content = "\n\n".join([doc.page_content for doc in docs_list])
                
                extraction_result = self.toc_extractor.extract_toc_and_content(
                    source_for_loader,
                    document_id=document_id,
                    document_content=website_content
                )
                
                # Save TOC structure data và content data vào session
                toc_structure_data = extraction_result.toc_structure.to_dict()
                content_data = extraction_result.content_data.to_dict()
                
                try:
                    idx = next(i for i, c in enumerate(content_data["content"]) if c["title"] == "full_document")
                    content_data["content"] = content_data["content"][:idx+1]
                except StopIteration:
                    pass

                self.database_service.save_toc_structure_data(username=username, document_id=document_id, toc_structure=toc_structure_data)
                self.database_service.save_content_data(username=username, document_id=document_id, content_data=content_data)

                logger.info(f"Extracted ToC with {len(extraction_result.toc_structure.sections)} sections")
                logger.info(f"Generated content for {len(extraction_result.content_data.content)} items")
                logger.info(f"Saved ToC structure and content data to session")

            # Chunk documents
            logger.info("Chunking documents...")
            chunks = self.chunker.chunk(iter(docs_list), document_id=document_id, username=username)
            
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
            self.database_service.save_document_metadata(username=username, document_id=metadata.document_id, metadata=metadata)
            
            # Add document to library
            document_titles = []
            if extract_toc and 'extraction_result' in locals():
                try:
                    # Extract titles from TOC structure - now it's nested format
                    document_titles = self._extract_titles_from_toc_structure(extraction_result.toc_structure.sections)
                    idx = document_titles.index("full_document")
                    document_titles = document_titles[:idx+1]                    
                except Exception as e:
                    logger.warning(f"Could not extract titles for document library: {e}")
            
            self.database_service.add_document_to_library(
                username=username,
                document_id=document_id,
                name=doc_name,
                title=document_titles
            )

            logger.info(f"Added document {file_path} to document library")

            # Create success result
            result = ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                document_id=document_id,
                file_name=file_name,
                message=f"Successfully processed {file_name}",
                metadata=metadata,
                table_of_contents=None  
            )
            
            logger.info(f"Successfully processed document {document_id}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)

            # Update metadata with error status if document_id exists
            if 'document_id' in locals():
                try:
                    metadata = self.database_service.get_document_metadata(username=username, document_id=document_id) #TODO: replace user_id
                    if metadata:
                        metadata.processing_status = ProcessingStatus.FAILED
                        metadata.error_message = str(e)
                        self.database_service.save_document_metadata(username=username, document_id=document_id, metadata=metadata)
                except Exception as update_error:
                    logger.error(f"Failed to update error status: {update_error}")
            
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                document_id=locals().get('stored_document_id', locals().get('document_id', "unknown")),
                file_name=Path(file_path).name,
                message=error_msg,
                error=str(e)
            )
    
    def get_document_metadata(self, username: str, document_id: str) -> Optional[DocumentMetadata]:
        """
        Get document metadata by ID.
        
        Args:
            username: User identifier
            document_id: Document identifier
            
        Returns:
            Document metadata or None if not found
        """
        return self.database_service.get_document_metadata(username=username, document_id=document_id)

    def get_table_of_contents(self, username: str, document_id: str) -> Optional[List[TableOfContentsSection]]:
        """
        Get table of contents for document (created from TOC structure data).
        
        Args:
            document_id: Document identifier
            
        Returns:
            Table of contents or None if not found
        """
        # Lấy TOC structure data thay vì legacy TOC
        toc_structure_data = self.database_service.get_toc_structure_data(username=username, document_id=document_id) 
        if not toc_structure_data:
            return None
        # Tạo TableOfContents từ TOC structure data
        return toc_structure_data

    def get_table_of_contents_as_string(self, username: str, document_id: str) -> Optional[str]:
        """
        Get table of contents for document formatted as string.
        
        Args:
            username: User identifier
            document_id: Document identifier
            
        Returns:
            Table of contents formatted as string or None if not found
        """
        # Lấy TOC structure data trực tiếp
        toc_structure_data = self.database_service.get_toc_structure_data(username=username, document_id=document_id) 
        if not toc_structure_data:
            return None
        logger.info(f"Raw TOC structure data: {toc_structure_data}")

        # Remove full_document entries and convert to dictionary
        return self._format_toc_structure_as_string(document_id, toc_structure_data)


    def get_content_data(self, username: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get content data from TOC extractor for document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of content items or None if not found
        """
        return self.database_service.get_content_data(username=username, document_id=document_id)

    def get_toc_structure_data(self, username: str, document_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get TOC structure data from TOC extractor for document.
        
        Args:
            username: User identifier
            document_id: Document identifier
            
        Returns:
            List of TOC structure items or None if not found
        """
        return self.database_service.get_toc_structure_data(username=username, document_id=document_id)


    def get_document_id_dict(self, username: str) -> Dict[str, str]:
        """
        Get a dictionary mapping document IDs to file names for current session.
        Args:
            username: User identifier
        Returns:
            Dictionary of document_id -> file_name
        """
        document_metadata_list = self.database_service.list_user_documents(username=username)
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
    
    def _format_section_as_string(self, section: TableOfContentsSection, section_index: str) -> List[str]:
        """
        Format a single TableOfContentsSection as string lines.
        
        Args:
            section: TableOfContentsSection to format
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
        # Convert structure data to TableOfContentsSection objects
        sections = []
        
        for item_data in toc_structure_data:
            if item_data.get('level') == 1:  # Only add root level sections
                # Skip items without required fields
                if 'id' not in item_data or 'title' not in item_data or 'level' not in item_data:
                    continue
                    
                section = TableOfContentsSection(
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
                                            all_structure_data: List[Dict[str, Any]]) -> List[TableOfContentsSection]:
        """Convert children IDs to TableOfContentsSection objects from structure data."""
        children = []
        
        # Create ID mapping - only include items with required fields
        id_to_item = {item['id']: item for item in all_structure_data if 'id' in item}
        
        for child_id in children_ids: 
            if child_id in id_to_item:
                child_item = id_to_item[child_id]
                # Skip items without required fields
                if 'title' not in child_item or 'level' not in child_item:
                    continue
                    
                child_section = TableOfContentsSection(
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
    
    def _format_toc_structure_as_string(self, document_id: str, toc_structure_data: List[TableOfContentsSection]) -> str:
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
        result.append("-" * 50)
        
        # Group by level and format
        level_1_items = [item for item in toc_structure_data 
                         if item.level == 1 and item.section_title != "full_document"]
         
        for index, item in enumerate(level_1_items):
            result.extend(self._format_structure_item_as_string(item, toc_structure_data, str(index + 1)))
        
        return "\n".join(result)

    def _format_structure_item_as_string(self, item: TableOfContentsSection,
                                       all_structure_data: List[TableOfContentsSection],
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
        if not item.section_title:
            return []

        page_info = f" (Page {item.page_number})" if item.page_number else ""
        line = f"{section_index} {item.section_title}{page_info}"

        result = [line]
        
        # Format children
        children = item.children

        for child_index, child_item in enumerate(children):
            result.extend(self._format_structure_item_as_string(
                child_item, 
                all_structure_data, 
                f"{section_index}.{child_index + 1}"
            ))

        return result
    