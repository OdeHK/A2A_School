# src/core/professional_pdf_processor.py
# Professional PDF processor using PyMuPDF for accurate bookmark and TOC extraction

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logging.warning("PyMuPDF not installed. Install with: pip install PyMuPDF")

from ..core.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

@dataclass
class BookmarkNode:
    """Cấu trúc dữ liệu cho bookmark node."""
    title: str
    page: int
    level: int
    children: List['BookmarkNode']
    uri: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'page': self.page,
            'level': self.level,
            'children': [child.to_dict() for child in self.children],
            'uri': self.uri
        }

class ProfessionalPDFProcessor:
    """
    Professional PDF processor sử dụng PyMuPDF để extract bookmark và metadata chính xác.
    Kết hợp với LLM để tạo Table of Contents chất lượng cao.
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider
        if not fitz:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")
        logger.info("✅ ProfessionalPDFProcessor initialized with PyMuPDF support")
    
    def extract_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive structure from PDF including bookmarks, metadata, and text.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            Dict: Complete PDF structure analysis
        """
        logger.info(f"🔍 Analyzing PDF structure: {pdf_path}")
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            result = {
                'metadata': self._extract_metadata(doc),
                'bookmarks': self._extract_bookmarks(doc),
                'pages_info': self._extract_pages_info(doc),
                'page_mapped_content': self._extract_page_mapped_content(doc),
                'full_text': self._extract_text_content(doc),
                'table_of_contents': None,  # Will be generated
                'structure_analysis': None  # Will be generated
            }
            
            # Generate enhanced TOC using LLM if available
            if self.llm_provider and result['bookmarks']:
                result['table_of_contents'] = self._generate_enhanced_toc(
                    result['bookmarks'], 
                    result['text_content'][:8000]  # Sample text for context
                )
                
            # Analyze document structure
            result['structure_analysis'] = self._analyze_structure(result)
            
            doc.close()
            logger.info("✅ PDF structure extraction completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF: {e}")
            return {'error': str(e)}
    
    def _extract_metadata(self, doc) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = doc.metadata
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'modification_date': metadata.get('modDate', ''),
            'page_count': doc.page_count,
            'is_encrypted': doc.is_encrypted,
            'is_pdf': doc.is_pdf
        }
    
    def _extract_bookmarks(self, doc) -> List[Dict[str, Any]]:
        """Extract bookmarks with hierarchical structure."""
        try:
            toc = doc.get_toc(simple=False)  # Get detailed TOC
            if not toc:
                logger.warning("No bookmarks found in PDF")
                return []
            
            bookmarks = []
            for item in toc:
                level, title, page, dest = item
                
                # Clean title
                clean_title = self._clean_bookmark_title(title)
                
                bookmark = {
                    'level': level,
                    'title': clean_title,
                    'page': page,
                    'destination': dest if isinstance(dest, dict) else None,
                    'number': self._extract_chapter_number(clean_title)
                }
                bookmarks.append(bookmark)
            
            logger.info(f"📖 Extracted {len(bookmarks)} bookmarks")
            return bookmarks
            
        except Exception as e:
            logger.error(f"Error extracting bookmarks: {e}")
            return []
    
    def _clean_bookmark_title(self, title: str) -> str:
        """Clean and normalize bookmark title."""
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title.strip())
        
        # Remove page numbers at the end
        title = re.sub(r'\s+\.\.\.\s*\d+$', '', title)
        title = re.sub(r'\s+\d+$', '', title)
        
        # Remove leading dots or dashes
        title = re.sub(r'^[.\-\s]+', '', title)
        
        return title.strip()
    
    def _extract_chapter_number(self, title: str) -> Optional[int]:
        """Extract chapter number from title."""
        # Look for patterns like "Chapter 1", "Chương 1", "1.", "1.1", etc.
        patterns = [
            r'(?:Chapter|Chương|Bài)\s+(\d+)',
            r'^(\d+)\.(?:\d+\.)*',
            r'^(\d+)\s',
            r'(\d+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_pages_info(self, doc) -> Dict[str, Any]:
        """Extract information about pages."""
        return {
            'total_pages': doc.page_count,
            'page_sizes': [doc[i].rect for i in range(min(5, doc.page_count))],  # Sample first 5 pages
            'has_images': any(doc[i].get_images() for i in range(min(10, doc.page_count))),
            'estimated_reading_time': doc.page_count * 2  # 2 minutes per page
        }
    
    def _extract_text_content(self, doc, max_pages: int = 50) -> str:
        """Extract text content from PDF (limited pages for performance)."""
        text_content = []
        pages_to_process = min(max_pages, doc.page_count)
        
        for page_num in range(pages_to_process):
            try:
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_content.append(text)
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num}: {e}")
                continue
        
        full_text = '\n'.join(text_content)
        logger.info(f"📄 Extracted text from {pages_to_process} pages ({len(full_text)} characters)")
        return full_text
    
    def _extract_page_mapped_content(self, doc, max_pages: int = None) -> Dict[int, Dict[str, Any]]:
        """
        Extract content mapped to specific pages for accurate chapter association.
        
        Args:
            doc: PyMuPDF document object
            max_pages: Maximum pages to process (None for all pages)
            
        Returns:
            Dict mapping page numbers to page content and metadata
        """
        page_content = {}
        total_pages = doc.page_count
        
        if max_pages is None:
            max_pages = total_pages
        else:
            max_pages = min(max_pages, total_pages)
        
        for page_num in range(max_pages):
            try:
                page = doc[page_num]
                text = page.get_text().strip()
                
                # Get page dimensions and layout info
                rect = page.rect
                
                page_content[page_num + 1] = {  # Use 1-based page numbering
                    'text': text,
                    'char_count': len(text),
                    'word_count': len(text.split()) if text else 0,
                    'dimensions': {
                        'width': rect.width,
                        'height': rect.height
                    },
                    'has_content': bool(text)
                }
                
            except Exception as e:
                logger.warning(f"Error extracting content from page {page_num + 1}: {e}")
                page_content[page_num + 1] = {
                    'text': '',
                    'char_count': 0,
                    'word_count': 0,
                    'dimensions': {'width': 0, 'height': 0},
                    'has_content': False
                }
        
        logger.info(f"📑 Extracted page-mapped content for {max_pages} pages")
        return page_content
    
    def _generate_enhanced_toc(self, bookmarks: List[Dict], context_text: str) -> str:
        """Generate enhanced Table of Contents using LLM."""
        if not self.llm_provider:
            return self._generate_basic_toc(bookmarks)
        
        logger.info("🤖 Generating enhanced TOC using LLM...")
        
        # Prepare bookmark context
        bookmark_context = "\n".join([
            f"Level {bm['level']}: {bm['title']} (Page {bm['page']})"
            for bm in bookmarks[:20]  # Limit to avoid token overflow
        ])
        
        system_prompt = """
        Bạn là chuyên gia phân tích cấu trúc tài liệu chuyên nghiệp. 
        Nhiệm vụ của bạn là tạo mục lục có cấu trúc phân tầng rõ ràng và logic.

        QUY TẮC:
        - LUÔN trả lời bằng tiếng Việt
        - Tạo cấu trúc phân tầng với đầy đủ số thứ tự
        - Giữ nguyên thông tin trang từ bookmark gốc
        - Cải thiện tiêu đề để rõ ràng và nhất quán
        """
        
        user_prompt = f"""
        ## NHIỆM VỤ: Tạo mục lục chuyên nghiệp từ bookmark PDF

        ### BOOKMARK TRÍCH XUẤT TỪ PDF:
        {bookmark_context}

        ### NGỮ CẢNH NỘI DUNG:
        {context_text[:4000]}...

        ### YÊU CẦU:
        1. Tạo cấu trúc phân tầng rõ ràng (1, 1.1, 1.1.1)
        2. Cải thiện tiêu đề để nhất quán và dễ hiểu
        3. Giữ nguyên thông tin trang từ bookmark gốc
        4. Loại bỏ các mục trùng lặp hoặc không quan trọng

        ### ĐỊNH DẠNG ĐẦU RA:
        ```
        1. Tiêu đề chương chính (Page X)
            1.1 Tiêu đề mục con (Page Y)
                1.1.1 Tiêu đề mục con nhỏ (Page Z)
        ```

        Hãy tạo mục lục chuyên nghiệp:
        """
        
        try:
            enhanced_toc = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Clean up the response
            enhanced_toc = re.sub(r'```.*?```', '', enhanced_toc, flags=re.DOTALL)
            enhanced_toc = enhanced_toc.strip()
            
            logger.info("✅ Enhanced TOC generated successfully")
            return enhanced_toc
            
        except Exception as e:
            logger.error(f"Error generating enhanced TOC: {e}")
            return self._generate_basic_toc(bookmarks)
    
    def _generate_basic_toc(self, bookmarks: List[Dict]) -> str:
        """Generate basic TOC from bookmarks without LLM."""
        toc_lines = []
        
        for bookmark in bookmarks:
            level = bookmark['level']
            title = bookmark['title']
            page = bookmark['page']
            
            # Create indentation based on level
            indent = "    " * (level - 1)
            
            # Add numbering if not present
            if not re.match(r'^\d+\.', title):
                # Simple numbering logic
                if level == 1:
                    number = len([b for b in bookmarks[:bookmarks.index(bookmark)+1] if b['level'] == 1])
                    title = f"{number}. {title}"
                elif level == 2:
                    title = f"{title}"  # Keep as is for sub-levels
            
            toc_line = f"{indent}{title} (Page {page})"
            toc_lines.append(toc_line)
        
        return "\n".join(toc_lines)
    
    def _analyze_structure(self, pdf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall document structure."""
        bookmarks = pdf_data.get('bookmarks', [])
        metadata = pdf_data.get('metadata', {})
        
        analysis = {
            'document_type': self._determine_document_type(metadata, bookmarks),
            'complexity_level': self._assess_complexity(bookmarks, metadata),
            'structure_quality': self._assess_structure_quality(bookmarks),
            'learning_estimation': {
                'reading_time_minutes': metadata.get('page_count', 0) * 2,
                'difficulty_level': 'Medium',  # Can be enhanced with content analysis
                'recommended_study_sessions': max(1, metadata.get('page_count', 0) // 20)
            },
            'navigation_quality': {
                'has_bookmarks': len(bookmarks) > 0,
                'bookmark_count': len(bookmarks),
                'max_depth': max([bm['level'] for bm in bookmarks], default=0),
                'coverage_score': len(bookmarks) / max(1, metadata.get('page_count', 1)) * 100
            }
        }
        
        return analysis
    
    def _determine_document_type(self, metadata: Dict, bookmarks: List[Dict]) -> str:
        """Determine type of document based on structure."""
        title = metadata.get('title', '').lower()
        bookmark_titles = ' '.join([bm['title'].lower() for bm in bookmarks[:10]])
        
        if any(keyword in title + bookmark_titles for keyword in ['tutorial', 'guide', 'học', 'giáo trình']):
            return 'Educational Material'
        elif any(keyword in title + bookmark_titles for keyword in ['manual', 'documentation', 'api']):
            return 'Technical Documentation'
        elif any(keyword in title + bookmark_titles for keyword in ['report', 'analysis', 'research']):
            return 'Research/Report'
        else:
            return 'General Document'
    
    def _assess_complexity(self, bookmarks: List[Dict], metadata: Dict) -> str:
        """Assess document complexity."""
        page_count = metadata.get('page_count', 0)
        bookmark_count = len(bookmarks)
        max_depth = max([bm['level'] for bm in bookmarks], default=0)
        
        complexity_score = 0
        
        # Page count factor
        if page_count > 100:
            complexity_score += 3
        elif page_count > 50:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Structure depth factor
        if max_depth > 3:
            complexity_score += 2
        elif max_depth > 2:
            complexity_score += 1
        
        # Bookmark density factor
        if bookmark_count > page_count * 0.1:
            complexity_score += 1
        
        if complexity_score >= 5:
            return 'High'
        elif complexity_score >= 3:
            return 'Medium'
        else:
            return 'Low'
    
    def _assess_structure_quality(self, bookmarks: List[Dict]) -> str:
        """Assess quality of document structure."""
        if not bookmarks:
            return 'Poor - No bookmarks'
        
        # Check for consistent numbering
        numbered_bookmarks = sum(1 for bm in bookmarks if re.match(r'^\d+\.', bm['title']))
        numbering_ratio = numbered_bookmarks / len(bookmarks)
        
        # Check for hierarchical consistency
        levels = [bm['level'] for bm in bookmarks]
        max_level = max(levels)
        has_hierarchy = max_level > 1
        
        quality_score = 0
        
        if numbering_ratio > 0.7:
            quality_score += 2
        elif numbering_ratio > 0.3:
            quality_score += 1
        
        if has_hierarchy:
            quality_score += 2
        
        if len(bookmarks) > 5:
            quality_score += 1
        
        if quality_score >= 4:
            return 'Excellent'
        elif quality_score >= 2:
            return 'Good'
        else:
            return 'Fair'

    def generate_quiz_context(self, pdf_structure: Dict[str, Any], target_section: str = None) -> str:
        """
        Generate appropriate context for quiz generation based on PDF structure.
        
        Args:
            pdf_structure: PDF structure from extract_pdf_structure
            target_section: Specific section to focus on (optional)
            
        Returns:
            str: Contextual content for quiz generation
        """
        logger.info("📝 Generating quiz context from PDF structure...")
        
        text_content = pdf_structure.get('text_content', '')
        bookmarks = pdf_structure.get('bookmarks', [])
        
        if target_section:
            # Find relevant section in bookmarks
            relevant_bookmark = None
            for bookmark in bookmarks:
                if target_section.lower() in bookmark['title'].lower():
                    relevant_bookmark = bookmark
                    break
            
            if relevant_bookmark:
                # Extract text around the target section
                # This is a simplified approach - in reality, you'd need to map
                # bookmark page numbers to text positions
                section_context = self._extract_section_content(
                    text_content, relevant_bookmark['title']
                )
                return section_context
        
        # Return general content for quiz generation
        # Limit to avoid token overflow
        return text_content[:8000] if len(text_content) > 8000 else text_content
    
    def _extract_section_content(self, full_text: str, section_title: str) -> str:
        """Extract content for a specific section."""
        lines = full_text.split('\n')
        section_lines = []
        capturing = False
        
        for line in lines:
            if section_title.lower() in line.lower():
                capturing = True
                section_lines.append(line)
            elif capturing:
                # Check if we hit another section title
                if any(keyword in line.lower() for keyword in ['chapter', 'section', 'chương', 'bài']):
                    if len(section_lines) > 50:  # If we have enough content, stop
                        break
                section_lines.append(line)
                
                # Limit section size
                if len(section_lines) > 200:
                    break
        
    def _analyze_structure_with_pages(self, bookmarks: List[Dict], total_pages: int) -> Dict[str, Any]:
        """
        Analyze document structure with page-level mapping for accurate chapter boundaries.
        
        Args:
            bookmarks: List of bookmark dictionaries
            total_pages: Total number of pages in document
            
        Returns:
            Dict containing structure analysis with page mapping
        """
        if not bookmarks:
            return {
                'has_structure': False,
                'chapter_count': 0,
                'chapters': [],
                'page_ranges': {},
                'structure_quality': 'poor'
            }
        
        # Create chapter mapping with page ranges
        chapters = []
        page_ranges = {}
        
        # Sort bookmarks by page for accurate range calculation
        sorted_bookmarks = sorted(bookmarks, key=lambda x: x['page'])
        
        for i, bookmark in enumerate(sorted_bookmarks):
            chapter_info = {
                'title': bookmark['title'],
                'level': bookmark['level'],
                'start_page': bookmark['page'],
                'end_page': None,  # Will be calculated
                'page_count': 0
            }
            
            # Calculate end page (start of next chapter or end of document)
            if i + 1 < len(sorted_bookmarks):
                next_chapter_page = sorted_bookmarks[i + 1]['page']
                chapter_info['end_page'] = next_chapter_page - 1
            else:
                chapter_info['end_page'] = total_pages
            
            chapter_info['page_count'] = chapter_info['end_page'] - chapter_info['start_page'] + 1
            
            chapters.append(chapter_info)
            
            # Create page range mapping
            for page_num in range(chapter_info['start_page'], chapter_info['end_page'] + 1):
                page_ranges[page_num] = {
                    'chapter_title': bookmark['title'],
                    'chapter_level': bookmark['level'],
                    'chapter_index': i
                }
        
        # Assess structure quality
        avg_chapter_length = sum(ch['page_count'] for ch in chapters) / len(chapters) if chapters else 0
        structure_quality = 'excellent' if avg_chapter_length > 3 else 'good' if avg_chapter_length > 1 else 'basic'
        
        return {
            'has_structure': True,
            'chapter_count': len(chapters),
            'chapters': chapters,
            'page_ranges': page_ranges,
            'average_chapter_length': avg_chapter_length,
            'structure_quality': structure_quality,
            'total_pages_covered': len(page_ranges)
        }
    
    def _get_sample_text_from_pages(self, page_mapped_content: Dict[int, Dict]) -> str:
        """Get sample text from first few pages for context."""
        sample_text = []
        max_chars = 5000
        current_chars = 0
        
        for page_num in sorted(page_mapped_content.keys())[:10]:  # First 10 pages
            page_text = page_mapped_content[page_num].get('text', '')
            if page_text:
                remaining_chars = max_chars - current_chars
                if remaining_chars <= 0:
                    break
                sample_text.append(page_text[:remaining_chars])
                current_chars += len(page_text)
        
        return '\n'.join(sample_text)
    
    def detect_chapters(self, text_content: str) -> List[Dict[str, Any]]:
        """
        Detect chapters from text content for compatibility with DocumentReader.
        
        Args:
            text_content: Full text content of document
            
        Returns:
            List of chapter dictionaries with title, level, start_position
        """
        try:
            # Use basic pattern matching to detect chapters
            chapters = []
            chapter_patterns = [
                r'^(CHƯƠNG|CHAPTER|BÀI|PHẦN)\s+(\d+|[IVXLCDM]+)[\s\.\-:]*(.*?)$',
                r'^\d+\.\s*([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][^\.]*?)$',
                r'^([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s]{5,50})$'
            ]
            
            lines = text_content.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                for pattern in chapter_patterns:
                    match = re.match(pattern, line, re.IGNORECASE | re.MULTILINE)
                    if match:
                        # Extract chapter title
                        if len(match.groups()) >= 2:
                            chapter_title = match.group(3) if match.group(3) else match.group(1)
                        else:
                            chapter_title = match.group(1)
                        
                        # Calculate position in text
                        position = sum(len(lines[j]) + 1 for j in range(i))
                        
                        chapters.append({
                            'number': len(chapters) + 1,  # Sequential chapter numbering
                            'title': chapter_title.strip(),
                            'level': 1,  # Basic level detection
                            'start_position': position,
                            'line_number': i + 1
                        })
                        break
            
            logger.info(f"📚 Detected {len(chapters)} chapters from text content")
            return chapters
            
        except Exception as e:
            logger.error(f"Error detecting chapters: {e}")
            return []


# Utility functions for backward compatibility
def extract_pdf_bookmarks(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract bookmarks from PDF using professional processor."""
    processor = ProfessionalPDFProcessor()
    structure = processor.extract_pdf_structure(pdf_path)
    return structure.get('bookmarks', [])

def generate_professional_toc(pdf_path: str, llm_provider: LLMProvider = None) -> str:
    """Generate professional TOC from PDF."""
    processor = ProfessionalPDFProcessor(llm_provider)
    structure = processor.extract_pdf_structure(pdf_path)
    return structure.get('table_of_contents', 'No TOC available')