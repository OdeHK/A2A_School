"""
Table of Contents Extractor service.
Handles extraction of ToC from documents using both library methods and LLM fallback.
"""

import logging
import pymupdf
from typing import List, Optional, Dict, Any
from pathlib import Path

from datetime import datetime
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

from services.models import TableOfContents, TocSection
from services.rag.llm_service import LLMService

logger = logging.getLogger(__name__)


class TocExtractionResult(BaseModel):
    """Result of ToC extraction attempt"""
    success: bool
    method: str  # "library" or "llm"
    sections: List[TocSection] = []
    raw_text: Optional[str] = None
    error: Optional[str] = None


class TableOfContentsExtractor:
    """
    Service for extracting Table of Contents from documents.
    Uses library-based extraction first, falls back to LLM if needed.
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize ToC extractor.
        
        Args:
            llm_service: LLM service for fallback extraction
        """
        #self.llm_service = llm_service or LLMService(llm_type="google_gen_ai")
    
    def extract_table_of_contents(self, file_path: str, document_id: str) -> TableOfContents:
        """
        Extract table of contents from document.
        Tries library method first, falls back to LLM.
        
        Args:
            file_path: Path to document file
            document_id: Document identifier
            
        Returns:
            TableOfContents object
        """
        logger.info(f"Extracting ToC from {file_path}")
        
        # Try library-based extraction first
        library_result = self.extract_toc_with_library(file_path)
        
        if library_result.success and library_result.sections:
            logger.info(f"Successfully extracted ToC using library method: {len(library_result.sections)} sections")
            return TableOfContents(
                document_id=document_id,
                extraction_method="library",
                extraction_date=datetime.now(),
                sections=library_result.sections,
                raw_text=library_result.raw_text
            )
        
        # Fall back to LLM extraction
        logger.info("Library extraction failed, trying LLM method")

        # TODO: Implement LLM table of contents extraction here
        # llm_result = self._extract_toc_with_llm(file_path)
        
        # if llm_result.success:
        #     logger.info(f"Successfully extracted ToC using LLM method: {len(llm_result.sections)} sections")
        #     return TableOfContents(
        #         document_id=document_id,
        #         extraction_method="llm",
        #         extraction_date=datetime.now(),
        #         sections=llm_result.sections,
        #         raw_text=llm_result.raw_text
        #     )
        
        # # Return empty ToC if both methods fail
        # logger.warning(f"Failed to extract ToC from {file_path}")
        return TableOfContents(
            document_id=document_id,
            extraction_method="failed",
            extraction_date=datetime.now(),
            sections=[],
            raw_text=None
        )
    
    def extract_toc_with_library(self, file_path: str) -> TocExtractionResult:
        """
        Extract ToC using PyMuPDF library.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            TocExtractionResult
        """
        try:
            doc = pymupdf.open(file_path)
            toc = doc.get_toc()
            doc.close()
            
            if not toc:
                return TocExtractionResult(
                    success=False,
                    method="library",
                    error="No table of contents found in document"
                )

            # Convert PyMuPDF ToC format to our format
            sections = self._convert_pymupdf_toc(toc)
            
            return TocExtractionResult(
                success=True,
                method="library",
                sections=sections,
                raw_text=str(toc)
            )
            
        except Exception as e:
            logger.error(f"Library ToC extraction failed: {str(e)}")
            return TocExtractionResult(
                success=False,
                method="library",
                error=str(e)
            )
    
    def _convert_pymupdf_toc(self, toc_data: List) -> List[TocSection]:
        """
        Convert PyMuPDF ToC format to TocSection objects.
        
        Args:
            toc_data: ToC data from PyMuPDF ([level, title, page_num])
            
        Returns:
            List of TocSection objects
        """
        sections = []
        section_stack = {}  # Track parent sections by level

        for section_id, (level, title, page_num) in enumerate(toc_data):

            # Determine parent section
            parent_section_id = None
            if level > 1:
                # Find parent at level-1
                for parent_level in range(level - 1, 0, -1):
                    if parent_level in section_stack:
                        parent_section_id = section_stack[parent_level]
                        break
            
            section = TocSection(
                section_id=f"{section_id + 1}",
                section_title=title.strip(),
                parent_section_id=parent_section_id,
                level=level,
                page_number=page_num if page_num > 0 else None
            )


            # If the last section is a parent (its level is less than current level), add as its child.
            if sections and sections[-1].level < level:
                current_section = sections[-1]
                # Traverse down to the correct parent at level-1
                while current_section.children and current_section.level < level - 1:
                    current_section = current_section.children[-1]
                current_section.children.append(section)
            else:
                # Otherwise, add as a top-level section
                sections.append(section)

            section_stack[level] = section.section_id

            # Clear deeper levels
            keys_to_remove = [k for k in section_stack.keys() if k > level]
            for k in keys_to_remove:
                del section_stack[k]
        
        return sections
    
    def _extract_toc_with_llm(self, file_path: str) -> TocExtractionResult:
        """
        Extract ToC using LLM by analyzing document content.
        
        Args:
            file_path: Path to document file
            
        Returns:
            TocExtractionResult
        """
        try:
            # # Load first few pages to analyze for ToC
            # doc = pymupdf.open(file_path)
            
            # # Extract text from first 10 pages (typical ToC location)
            # doc_text = doc.get_text("text")
            
            # doc.close()
            
            # # Use LLM to extract ToC structure
            # sections = self._analyze_toc_with_llm(toc_text)
            
            return TocExtractionResult(
                success=len(sections) > 0,
                method="llm",
                sections=sections,
                raw_text=toc_text[:2000]  # Store first part of text
            )
            
        except Exception as e:
            logger.error(f"LLM ToC extraction failed: {str(e)}")
            return TocExtractionResult(
                success=False,
                method="llm",
                error=str(e)
            )
    
    def _analyze_toc_with_llm(self, document_text: str) -> List[TocSection]:
        """
        Use LLM to analyze document text and extract ToC structure.
        
        Args:
            document_text: Text content from document pages
            
        Returns:
            List of TocSection objects
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_toc_extraction_system_prompt()),
            ("human", self._get_toc_extraction_user_prompt())
        ])
        
        try:
            # prompt = prompt_template.invoke({
            #     "document_text": document_text[:8000]  # Limit text size
            # })
            
            # response = self.llm_service.invoke(prompt)
            # response_text = getattr(response, 'content', str(response))
            
            # Parse LLM response to extract ToC structure
            return self._parse_llm_toc_response(response_text)
            
        except Exception as e:
            logger.error(f"Error in LLM ToC analysis: {str(e)}")
            return []
    
    def _parse_llm_toc_response(self, response_text: str) -> List[TocSection]:
        """
        Parse LLM response to extract ToC sections.
        
        Args:
            response_text: LLM response text
            
        Returns:
            List of TocSection objects
        """
        # sections = []
        # lines = response_text.split('\\n')
        
        # for line in lines:
        #     line = line.strip()
        #     if not line:
        #         continue
        #     # Example expected formats:
        #     # "1,Chapter Title,5"
        #     # "1.1,Section Title,10"

            
        #     parts = [part.strip() for part in line.split(',')]  
        #     # trichs xuat cac phan tu string
        #     if len(parts) == 3:
        #         section_id = parts[0]
        #         title = parts[1]
        #         page_num = int(parts[2]) if parts[2].isdigit() else None
        #         level = section_id.count('.') + 1 

        #         if title:
        #             section = TocSection(
        #                 section_id=section_id,
        #                 section_title=title,
        #                 level=level,
        #                 page_number=page_num
        #             )
        #             sections.append(section)

        return sections
    

    @staticmethod
    def _get_toc_extraction_system_prompt() -> str:
        """System prompt for ToC extraction."""
        return """You are an expert at analyzing document structure and extracting table of contents information.
        Your task is to identify and extract the table of contents structure from the provided document text.
        
        Focus on:
        1. Identifying chapter/section headings
        2. Determining hierarchical structure
        3. Finding page numbers when available
        4. Maintaining proper formatting
        
        Respond with a clean, structured list of sections in the format:
        Chapter/Section Number. Title ... Page Number
        
        If no clear table of contents is found, extract the main headings and structure from the document."""
    
    @staticmethod
    def _get_toc_extraction_user_prompt() -> str:
        """User prompt for ToC extraction."""
        return """Please analyze the following document text and extract the table of contents structure.
        Look for chapter headings, section titles, and their hierarchical organization.
        
        Document text:
        {document_text}
        
        Please extract and format the table of contents structure."""




if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)
    extractor = TableOfContentsExtractor()
    example_file = "C:\\Users\\likgn\\Repository\\RAG\\example_data\\operation_management.pdf"  
    toc = extractor.extract_table_of_contents(example_file, "doc-001")
    print("\n\n")
    print(toc.sections)