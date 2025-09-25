"""
Enhanced Table of Contents Extractor service.
Generates separate TOC structure and content files with ID mapping.
Uses TextRank summarization for intelligent content extraction.
"""

import logging
import os
import json
import uuid
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict        

# Import from existing TOC system
from services.TOC_generator import TOCGenerator, BookmarkNode, TaskType, TOCStrategyFactory

logger = logging.getLogger(__name__)


# === DATA MODELS ===
@dataclass
class TOCSection:
    """TOC section with nested children structure"""
    section_id: str
    section_title: str
    parent_section_id: Optional[str]
    level: int
    page_number: Optional[int]
    children: List['TOCSection']
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'section_id': self.section_id,
            'section_title': self.section_title,
            'parent_section_id': self.parent_section_id,
            'level': self.level,
            'page_number': self.page_number,
            'children': [child.to_dict() for child in self.children]
        }

@dataclass
class TOCStructure:
    """Complete TOC structure for a document"""
    document_id: str
    extraction_date: str
    sections: List[TOCSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'extraction_date': self.extraction_date,
            'sections': [section.to_dict() for section in self.sections]
        }

@dataclass  
class ContentItem:
    """Content item with corresponding TOC ID"""
    id: str                    # Same ID as TOC item
    title: str                 # Section title (for reference)
    content: str               # Generated content (TextRank summary/extract)
    page_number: Optional[int] # Page number for reference
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ContentData:
    """Complete content data for a document"""
    document_id: str
    extraction_date: str
    content: List[ContentItem]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'extraction_date': self.extraction_date,
            'content': [item.to_dict() for item in self.content]
        }

@dataclass
class TOCExtractionResult:
    """Complete extraction result with both structure and content"""
    pdf_path: str
    toc_structure: TOCStructure
    content_data: ContentData
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "toc_structure": self.toc_structure.to_dict(),
            "content_data": self.content_data.to_dict()
        }


class TOCExtractor:
    """
    TOC Extractor that generates separate TOC and Content files.
    
    Features:
    - Extracts TOC structure with unique IDs
    - Generates content using TextRank summarization
    - Creates separate JSON files for structure and content
    - Maintains ID mapping between TOC and content
    - Supports multiple extraction strategies
    """
    
    def __init__(self, 
                 content_strategy: str = "textrank_extract",
                 embedding_model: str = "Alibaba-NLP/gte-multilingual-base",
                 cache_folder: str = "./model"):
        """
        Initialize TOC Extractor.
        
        Args:
            content_strategy: Strategy for content generation
            embedding_model: Model for TextRank embeddings  
            cache_folder: Cache folder for models
        """
        self.content_strategy = content_strategy
        self.embedding_model = embedding_model
        self.cache_folder = cache_folder
        
        # Configuration for TextRank strategy
        self.toc_config = {
            "embedding_model": embedding_model,
            "cache_folder": cache_folder
        }
        
        logger.info(f"TOC Extractor initialized:")
        logger.info(f"  - Content Strategy: {content_strategy}")
        logger.info(f"  - Embedding Model: {embedding_model}")
        logger.info(f"  - Mode: Separate TOC + Content files with ID mapping")
    
    def extract_toc_and_content(self, pdf_path: str, document_id: str) -> TOCExtractionResult:
        """
        Main method to extract both TOC structure and content.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Optional document ID (auto-generated if not provided)
            
        Returns:
            TOCExtractionResult containing both structure and content
        """
        logger.info(f"Starting TOC extraction for: {pdf_path}")
        
        # Validate PDF file
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # STEP 1: Extract TOC structure using TOCGenerator
        toc_generator = self._create_toc_generator(pdf_path)
        bookmark_tree = toc_generator.generate_toc()

        logger.info(f"Extracted: bookmark_tree {bookmark_tree}")
        
        # STEP 2: Convert to structured format with unique IDs
        toc_sections = self._convert_to_toc_sections(bookmark_tree)
        toc_structure = TOCStructure(
            document_id=document_id,
            extraction_date=datetime.now().isoformat(),
            sections=toc_sections
        )
        
        # STEP 3: Generate content using TextRank for each section
        content_items = self._generate_content_items(toc_sections, bookmark_tree)
        content_data = ContentData(
            document_id=document_id,
            extraction_date=datetime.now().isoformat(),
            content=content_items
        )
        
        # STEP 4: Create final result
        result = TOCExtractionResult(
            pdf_path=pdf_path,
            toc_structure=toc_structure,
            content_data=content_data
        )
        
        # Store document_id for export usage
        result.document_id = document_id
        
        logger.info(f"Extraction completed: {len(toc_structure.sections)} sections, {len(content_data.content)} content items")
        return result
    
    def _create_toc_generator(self, pdf_path: str) -> TOCGenerator:
        """Create TOC generator with appropriate strategy."""
        task_type = TaskType.HYBRID_SUMMARIZE if self.content_strategy == "textrank_extract" else TaskType.SUMMARIZE
        
        return TOCGenerator.create_with_config(
            pdf_path=pdf_path,
            task_type=task_type,
            toc_config=self.toc_config
        )
    
    
    def _convert_to_toc_sections(self, bookmark_tree: List[BookmarkNode]) -> List[TOCSection]:
        """
        Convert BookmarkNode tree to nested TOC sections structure.
        
        Args:
            bookmark_tree: BookmarkNode tree from TOCGenerator
            
        Returns:
            List of TOCSection with nested children
        """
        def process_node(node: BookmarkNode, level: int = 1, parent_id: Optional[str] = None) -> TOCSection:
            """Recursively process nodes and create nested structure."""
            # Generate unique ID
            node_id = f"toc_{uuid.uuid4().hex[:8]}"
            
            # Process children recursively
            children = []
            for child in node.children:
                child_section = process_node(child, level + 1, node_id)
                children.append(child_section)
            
            # Create TOC section
            section = TOCSection(
                section_id=node_id,
                section_title=node.title,
                parent_section_id=parent_id,
                level=level,
                page_number=node.page_number if hasattr(node, 'page_number') else None,
                children=children
            )
            
            return section
        
        # Process root nodes
        sections = []
        for root_node in bookmark_tree:
            section = process_node(root_node, level=1, parent_id=None)
            sections.append(section)
        
        return sections
    
    def _generate_content_items(self, toc_sections: List[TOCSection], 
                               bookmark_tree: List[BookmarkNode]) -> List[ContentItem]:
        """
        Generate content for each TOC section using TextRank.
        
        Args:
            toc_sections: TOC sections with nested structure
            bookmark_tree: Complete bookmark tree including full_document node
            
        Returns:
            List of ContentItem with generated content
        """
        content_items = []
        
        # Create a mapping from title to BookmarkNode for content lookup
        title_to_node = self._create_title_mapping(bookmark_tree)
        
        def process_section(section: TOCSection):
            """Recursively process sections and generate content."""
            try:
                # Find corresponding bookmark node
                bookmark_node = title_to_node.get(section.section_title)
                
                if bookmark_node and bookmark_node.content:
                    # Extract content information
                    content = bookmark_node.content
                    
                    content_item = ContentItem(
                        id=section.section_id,  # Same ID as TOC item for mapping
                        title=section.section_title,
                        content=content,
                        page_number=section.page_number
                    )
                    
                    content_items.append(content_item)
                    logger.debug(f"Generated content for: {section.section_title}")
                    
                else:
                    # Create placeholder for sections without content
                    content_item = ContentItem(
                        id=section.section_id,
                        title=section.section_title,
                        content=f"Nội dung cho '{section.section_title}' chưa được tạo.",
                        page_number=section.page_number
                    )
                    content_items.append(content_item)
                    logger.debug(f"Created placeholder for: {section.section_title}")
                    
            except Exception as e:
                logger.error(f"Error generating content for {section.section_title}: {e}")
                # Create error placeholder
                content_item = ContentItem(
                    id=section.section_id,
                    title=section.section_title, 
                    content=f"Lỗi khi tạo nội dung: {str(e)}",
                    page_number=section.page_number
                )
                content_items.append(content_item)
            
            # Process children
            for child in section.children:
                process_section(child)
        
        # Process all root sections
        for section in toc_sections:
            process_section(section)
        
        logger.info(f"Generated content for {len(content_items)} sections")
        return content_items
    
    def _generate_document_content(self, content_data: List[ContentItem]) -> str:
        """
        Generate combined content from all sections.
        
        Args:
            content_data: List of individual content items
            
        Returns:
            Combined content string
        """
        combined_content = []
        
        for item in content_data:
            # Skip placeholder and error content
            if (not item.content.startswith("Nội dung cho") and 
                not item.content.startswith("Lỗi khi")):
                
                # Add section header and content
                section_header = f"\n=== {item.title} ===\n"
                combined_content.append(section_header)
                combined_content.append(item.content)
                combined_content.append("\n")
        
        if combined_content:
            result = "".join(combined_content).strip()
            logger.debug(f"Generated combined document content: {len(result)} characters")
            return result
        else:
            return "Không có nội dung hợp lệ để tổng hợp."
    
    def _create_title_mapping(self, bookmark_tree: List[BookmarkNode]) -> Dict[str, BookmarkNode]:
        """Create mapping from title to BookmarkNode for content lookup."""
        mapping = {}
        
        def add_to_mapping(node: BookmarkNode):
            mapping[node.title] = node
            for child in node.children:
                add_to_mapping(child)
        
        for root in bookmark_tree:
            add_to_mapping(root)
        
        return mapping
    