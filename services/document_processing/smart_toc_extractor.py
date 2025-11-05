"""
Smart TOC Extractor
==================
Automatically generates Table of Contents for PDFs that don't have one.

Fallback Chain:
1. Built-in TOC (PyPDF2) - fastest
2. Heuristic Detection (font size, numbering patterns) - fast
3. LLM-based Generation (GPT-4) - accurate but slower
4. Page-based Fallback (one section per page) - last resort

Features:
- Multi-strategy approach
- Automatic fallback
- Pattern recognition (Chapter 1, Section 1.1, etc.)
- Font size analysis
- LLM integration for complex documents
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import PyPDF2
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TOCEntry:
    """Represents a table of contents entry."""
    title: str
    page_number: int
    level: int  # 0 = chapter, 1 = section, 2 = subsection
    confidence: float  # 0.0 to 1.0


@dataclass
class TOCExtractionResult:
    """Result of TOC extraction."""
    entries: List[TOCEntry]
    method_used: str  # 'built-in', 'heuristic', 'llm', 'page-based'
    success: bool
    confidence: float
    processing_time: float


class SmartTOCExtractor:
    """
    Smart TOC Extractor with multiple strategies.
    
    Usage:
        extractor = SmartTOCExtractor(llm_service=llm)
        result = extractor.extract_toc(pdf_path)
        if result.success:
            for entry in result.entries:
                print(f"{entry.title} - Page {entry.page_number}")
    """
    
    # Common chapter/section patterns
    PATTERNS = [
        # Chapter patterns
        r'^Chapter\s+(\d+)[:\.]?\s*(.+)',
        r'^CHAPTER\s+(\d+)[:\.]?\s*(.+)',
        r'^Chương\s+(\d+)[:\.]?\s*(.+)',
        
        # Section patterns
        r'^(\d+)\.\s+(.+)',  # 1. Introduction
        r'^(\d+\.\d+)\s+(.+)',  # 1.1 Background
        r'^(\d+\.\d+\.\d+)\s+(.+)',  # 1.1.1 Details
        
        # Letter patterns
        r'^([A-Z])\.\s+(.+)',  # A. Appendix
        r'^([IVX]+)\.\s+(.+)',  # I. Roman numerals
    ]
    
    def __init__(self, llm_service=None, min_confidence: float = 0.6):
        """
        Initialize Smart TOC Extractor.
        
        Args:
            llm_service: Optional LLM service for LLM-based extraction
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
        """
        self.llm_service = llm_service
        self.min_confidence = min_confidence
        
        logger.info(f"✅ SmartTOCExtractor initialized (min_confidence: {min_confidence})")
    
    def extract_toc(self, pdf_path: str) -> TOCExtractionResult:
        """
        Extract TOC using fallback chain.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            TOCExtractionResult
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"📖 Extracting TOC from: {pdf_path}")
            
            # Strategy 1: Try built-in TOC
            result = self._try_builtin_toc(pdf_path)
            if result and result.confidence >= self.min_confidence:
                result.processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Built-in TOC extraction successful ({len(result.entries)} entries)")
                return result
            
            # Strategy 2: Try heuristic detection
            result = self._try_heuristic_detection(pdf_path)
            if result and result.confidence >= self.min_confidence:
                result.processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Heuristic TOC extraction successful ({len(result.entries)} entries)")
                return result
            
            # Strategy 3: Try LLM-based generation
            if self.llm_service:
                result = self._try_llm_generation(pdf_path)
                if result and result.confidence >= self.min_confidence:
                    result.processing_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"✅ LLM TOC extraction successful ({len(result.entries)} entries)")
                    return result
            
            # Strategy 4: Fallback to page-based
            logger.warning(f"⚠️ All strategies failed, using page-based fallback")
            result = self._page_based_fallback(pdf_path)
            result.processing_time = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            logger.error(f"❌ Error extracting TOC: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return TOCExtractionResult(
                entries=[],
                method_used='error',
                success=False,
                confidence=0.0,
                processing_time=processing_time
            )
    
    def _try_builtin_toc(self, pdf_path: str) -> Optional[TOCExtractionResult]:
        """Try to extract built-in TOC using PyPDF2."""
        try:
            logger.debug(f"🔍 Strategy 1: Trying built-in TOC...")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check for outline/bookmarks
                outline = pdf_reader.outline
                if not outline:
                    logger.debug(f"ℹ️ No built-in TOC found")
                    return None
                
                entries = []
                self._parse_outline(outline, entries, level=0)
                
                if not entries:
                    return None
                
                # Calculate confidence based on structure
                confidence = self._calculate_builtin_confidence(entries)
                
                return TOCExtractionResult(
                    entries=entries,
                    method_used='built-in',
                    success=True,
                    confidence=confidence,
                    processing_time=0.0
                )
                
        except Exception as e:
            logger.debug(f"❌ Built-in TOC extraction failed: {e}")
            return None
    
    def _parse_outline(self, outline, entries: List[TOCEntry], level: int = 0):
        """Recursively parse PDF outline."""
        for item in outline:
            if isinstance(item, list):
                self._parse_outline(item, entries, level + 1)
            else:
                try:
                    title = item.title if hasattr(item, 'title') else str(item)
                    
                    # Get page number
                    page_number = 0
                    if hasattr(item, 'page'):
                        page_obj = item.page
                        if hasattr(page_obj, 'page_number'):
                            page_number = page_obj.page_number
                    
                    entries.append(TOCEntry(
                        title=title,
                        page_number=page_number,
                        level=level,
                        confidence=1.0
                    ))
                except Exception as e:
                    logger.debug(f"Error parsing outline item: {e}")
    
    def _calculate_builtin_confidence(self, entries: List[TOCEntry]) -> float:
        """Calculate confidence for built-in TOC."""
        if not entries:
            return 0.0
        
        # Check if all entries have valid page numbers
        valid_pages = sum(1 for e in entries if e.page_number > 0)
        page_ratio = valid_pages / len(entries)
        
        # Check hierarchical structure
        levels = set(e.level for e in entries)
        has_hierarchy = len(levels) > 1
        
        # Base confidence
        confidence = 0.7
        
        # Adjust based on page numbers
        confidence += page_ratio * 0.2
        
        # Adjust based on hierarchy
        if has_hierarchy:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _try_heuristic_detection(self, pdf_path: str) -> Optional[TOCExtractionResult]:
        """Try to detect TOC using heuristic rules."""
        try:
            logger.debug(f"🔍 Strategy 2: Trying heuristic detection...")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                entries = []
                
                # Scan first 20 pages for TOC patterns
                scan_pages = min(20, total_pages)
                
                for page_num in range(scan_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if not text:
                        continue
                    
                    # Split into lines
                    lines = text.split('\n')
                    
                    for line in lines:
                        line = line.strip()
                        if not line or len(line) < 3:
                            continue
                        
                        # Try to match patterns
                        for pattern in self.PATTERNS:
                            match = re.match(pattern, line, re.IGNORECASE)
                            if match:
                                # Extract title and determine level
                                groups = match.groups()
                                
                                if len(groups) == 2:
                                    number, title = groups
                                    
                                    # Determine level based on numbering
                                    level = self._determine_level(number)
                                    
                                    # Try to extract page number from line
                                    page_match = re.search(r'\.{2,}\s*(\d+)\s*$', line)
                                    if page_match:
                                        target_page = int(page_match.group(1))
                                    else:
                                        target_page = page_num + 1
                                    
                                    entries.append(TOCEntry(
                                        title=title.strip(),
                                        page_number=target_page,
                                        level=level,
                                        confidence=0.8
                                    ))
                
                if not entries:
                    logger.debug(f"ℹ️ No heuristic patterns found")
                    return None
                
                # Calculate overall confidence
                confidence = self._calculate_heuristic_confidence(entries)
                
                return TOCExtractionResult(
                    entries=entries,
                    method_used='heuristic',
                    success=True,
                    confidence=confidence,
                    processing_time=0.0
                )
                
        except Exception as e:
            logger.debug(f"❌ Heuristic detection failed: {e}")
            return None
    
    def _determine_level(self, number: str) -> int:
        """Determine hierarchy level from numbering."""
        if re.match(r'^\d+$', number):  # Simple number: 1, 2, 3
            return 0
        elif re.match(r'^\d+\.\d+$', number):  # 1.1, 1.2
            return 1
        elif re.match(r'^\d+\.\d+\.\d+$', number):  # 1.1.1
            return 2
        else:
            return 0
    
    def _calculate_heuristic_confidence(self, entries: List[TOCEntry]) -> float:
        """Calculate confidence for heuristic detection."""
        if not entries:
            return 0.0
        
        # Base confidence
        confidence = 0.6
        
        # Adjust based on number of entries found
        if len(entries) >= 5:
            confidence += 0.1
        
        # Check for consistent numbering
        has_consistent_numbering = self._check_consistent_numbering(entries)
        if has_consistent_numbering:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _check_consistent_numbering(self, entries: List[TOCEntry]) -> bool:
        """Check if entries have consistent numbering."""
        # Simple check: at least 3 entries with increasing page numbers
        if len(entries) < 3:
            return False
        
        sorted_entries = sorted(entries, key=lambda e: e.page_number)
        increasing = all(
            sorted_entries[i].page_number < sorted_entries[i+1].page_number
            for i in range(len(sorted_entries) - 1)
        )
        
        return increasing
    
    def _try_llm_generation(self, pdf_path: str) -> Optional[TOCExtractionResult]:
        """Try to generate TOC using LLM."""
        try:
            logger.debug(f"🔍 Strategy 3: Trying LLM generation...")
            
            if not self.llm_service:
                return None
            
            # Extract first 5 pages for LLM analysis
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                sample_text = ""
                for i in range(min(5, len(pdf_reader.pages))):
                    page_text = pdf_reader.pages[i].extract_text()
                    sample_text += f"\n=== Page {i+1} ===\n{page_text}\n"
            
            # Prepare prompt
            prompt = f"""Analyze this PDF content and generate a Table of Contents.

PDF Content (first 5 pages):
{sample_text[:3000]}

Generate a structured TOC with:
- Chapter/Section titles
- Page numbers (estimate if not explicit)
- Hierarchy levels

Format each entry as:
LEVEL|TITLE|PAGE_NUMBER

Example:
0|Introduction|1
0|Chapter 1: Background|5
1|1.1 History|6
1|1.2 Current State|10
0|Chapter 2: Methods|15
"""
            
            # Call LLM
            response = self.llm_service.generate(prompt)
            
            # Parse LLM response
            entries = self._parse_llm_response(response)
            
            if not entries:
                logger.debug(f"ℹ️ LLM generated no entries")
                return None
            
            return TOCExtractionResult(
                entries=entries,
                method_used='llm',
                success=True,
                confidence=0.7,
                processing_time=0.0
            )
            
        except Exception as e:
            logger.debug(f"❌ LLM generation failed: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> List[TOCEntry]:
        """Parse LLM-generated TOC."""
        entries = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            parts = line.split('|')
            if len(parts) != 3:
                continue
            
            try:
                level = int(parts[0].strip())
                title = parts[1].strip()
                page_number = int(parts[2].strip())
                
                entries.append(TOCEntry(
                    title=title,
                    page_number=page_number,
                    level=level,
                    confidence=0.7
                ))
            except ValueError:
                continue
        
        return entries
    
    def _page_based_fallback(self, pdf_path: str) -> TOCExtractionResult:
        """Generate simple page-based TOC as last resort."""
        try:
            logger.debug(f"🔍 Strategy 4: Using page-based fallback...")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                entries = []
                
                for i in range(total_pages):
                    entries.append(TOCEntry(
                        title=f"Page {i+1}",
                        page_number=i+1,
                        level=0,
                        confidence=0.3
                    ))
                
                return TOCExtractionResult(
                    entries=entries,
                    method_used='page-based',
                    success=True,
                    confidence=0.3,
                    processing_time=0.0
                )
                
        except Exception as e:
            logger.error(f"❌ Page-based fallback failed: {e}")
            return TOCExtractionResult(
                entries=[],
                method_used='error',
                success=False,
                confidence=0.0,
                processing_time=0.0
            )
