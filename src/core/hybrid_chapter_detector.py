# src/core/hybrid_chapter_detector.py
# Hybrid Chapter Detector với OCR, Computer Vision và LLM

import logging
import re
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

# OCR Libraries
try:
    import pytesseract
    import easyocr
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    logging.warning("OCR libraries not installed. Install with: pip install pytesseract easyocr pillow opencv-python")

# PDF handling
try:
    import fitz  # PyMuPDF
    import pdf2image
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    logging.warning("PDF libraries not installed. Install with: pip install PyMuPDF pdf2image")

logger = logging.getLogger(__name__)

@dataclass
class TOCEntry:
    """Cấu trúc dữ liệu cho mục lục entry."""
    title: str
    page: Optional[int]
    level: int
    confidence: float
    source: str  # 'text', 'ocr', 'cv', 'llm'
    bbox: Optional[Tuple[int, int, int, int]] = None  # Bounding box từ CV

class HybridChapterDetector:
    """
    Chapter Detector hybrid kết hợp nhiều phương pháp:
    1. Traditional Pattern Matching (cho text có cấu trúc)
    2. OCR (cho PDF scan và hình ảnh)
    3. Computer Vision (để detect visual structure)
    4. LLM (để intelligent parsing và validation)
    """
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.ocr_readers = self._initialize_ocr()
        self.pattern_detector = self._initialize_patterns()
        logger.info("✅ HybridChapterDetector đã được khởi tạo.")

    def _initialize_ocr(self):
        """Khởi tạo các OCR engines."""
        readers = {}
        if HAS_OCR:
            try:
                readers['easyocr'] = easyocr.Reader(['en', 'vi'])
                readers['tesseract'] = True
                logger.info("✅ OCR engines đã sẵn sàng (EasyOCR + Tesseract)")
            except Exception as e:
                logger.warning(f"Không thể khởi tạo OCR: {e}")
        return readers

    def _initialize_patterns(self):
        """Khởi tạo patterns cho traditional detection."""
        return {
            'main_chapter': [
                r'^(Chapter|Chương|CHAPTER|CHƯƠNG)\s+([IVX\d]+)[:\.\s]+(.+?)(?:\s*\.+\s*(\d+))?$',
                r'^([IVX\d]+)\.\s+(.+?)(?:\s*\.+\s*(\d+))?$',
                r'^(\d+)\s+(.+?)(?:\s*\.+\s*(\d+))?$'
            ],
            'sub_section': [
                r'^(\d+\.\d+)\s+(.+?)(?:\s*\.+\s*(\d+))?$',
                r'^([A-Z])\.\s+(.+?)(?:\s*\.+\s*(\d+))?$'
            ],
            'sub_sub_section': [
                r'^(\d+\.\d+\.\d+)\s+(.+?)(?:\s*\.+\s*(\d+))?$'
            ]
        }

    def detect_table_of_contents(
        self, 
        source: Union[str, bytes, Path], 
        source_type: str = 'auto'
    ) -> List[TOCEntry]:
        """
        Phát hiện table of contents từ nhiều nguồn khác nhau.
        
        Args:
            source: Text, file path, hoặc binary data
            source_type: 'text', 'image', 'pdf', 'auto'
            
        Returns:
            List[TOCEntry]: Danh sách entries được detect
        """
        logger.info(f"🔍 Bắt đầu detect TOC với source_type: {source_type}")
        
        # Auto-detect source type
        if source_type == 'auto':
            source_type = self._detect_source_type(source)
        
        # Dispatch theo source type
        if source_type == 'text':
            return self._detect_from_text(source)
        elif source_type == 'image':
            return self._detect_from_image(source)
        elif source_type == 'pdf':
            return self._detect_from_pdf(source)
        else:
            logger.error(f"Unsupported source_type: {source_type}")
            return []

    def _detect_source_type(self, source) -> str:
        """Auto-detect loại source."""
        if isinstance(source, str):
            if source.lower().endswith(('.pdf',)):
                return 'pdf'
            elif source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                return 'image'
            else:
                return 'text'
        elif isinstance(source, (bytes, Path)):
            return 'pdf'  # Assume binary data is PDF
        else:
            return 'text'

    def _detect_from_text(self, text: str) -> List[TOCEntry]:
        """Detect TOC từ plain text sử dụng pattern matching + LLM."""
        logger.info("📝 Detecting từ text với pattern matching + LLM")
        
        entries = []
        lines = text.split('\n')
        
        # Phase 1: Pattern-based detection
        pattern_entries = self._pattern_based_detection(lines)
        entries.extend(pattern_entries)
        
        # Phase 2: LLM-based enhancement nếu có LLM provider
        if self.llm_provider and len(entries) < 5:
            llm_entries = self._llm_based_detection(text)
            entries.extend(llm_entries)
        
        # Phase 3: Merge và deduplicate
        return self._merge_and_deduplicate(entries)

    def _detect_from_image(self, image_source: Union[str, Path, np.ndarray]) -> List[TOCEntry]:
        """Detect TOC từ hình ảnh sử dụng OCR + Computer Vision."""
        if not HAS_OCR:
            logger.error("OCR libraries not available for image processing")
            return []
            
        logger.info("🖼️ Detecting từ image với OCR + Computer Vision")
        
        # Load image
        if isinstance(image_source, (str, Path)):
            image = cv2.imread(str(image_source))
        else:
            image = image_source
            
        if image is None:
            logger.error("Không thể load image")
            return []
        
        entries = []
        
        # Phase 1: OCR-based detection
        ocr_entries = self._ocr_based_detection(image)
        entries.extend(ocr_entries)
        
        # Phase 2: Computer Vision-based structure detection
        cv_entries = self._computer_vision_detection(image)
        entries.extend(cv_entries)
        
        # Phase 3: LLM validation
        if self.llm_provider:
            validated_entries = self._llm_validate_entries(entries, image)
            return validated_entries
        
        return self._merge_and_deduplicate(entries)

    def _detect_from_pdf(self, pdf_source: Union[str, Path, bytes]) -> List[TOCEntry]:
        """Detect TOC từ PDF sử dụng tất cả phương pháp."""
        if not HAS_PDF:
            logger.error("PDF libraries not available")
            return []
            
        logger.info("📄 Detecting từ PDF với multiple approaches")
        
        entries = []
        
        # Phase 1: Extract embedded TOC từ PDF metadata
        metadata_entries = self._extract_pdf_toc_metadata(pdf_source)
        entries.extend(metadata_entries)
        
        # Phase 2: OCR trên các trang đầu (thường chứa TOC)
        if len(entries) < 5:  # Nếu metadata không đủ
            ocr_entries = self._extract_pdf_toc_ocr(pdf_source)
            entries.extend(ocr_entries)
        
        # Phase 3: Full text extraction + pattern matching
        text_entries = self._extract_pdf_toc_text(pdf_source)
        entries.extend(text_entries)
        
        return self._merge_and_deduplicate(entries)

    def _pattern_based_detection(self, lines: List[str]) -> List[TOCEntry]:
        """Traditional pattern matching approach được cải tiến."""
        entries = []
        
        for line_idx, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean or len(line_clean) < 3:
                continue
                
            # Thử match với các patterns
            for level, (level_name, patterns) in enumerate(self.pattern_detector.items(), 1):
                for pattern in patterns:
                    match = re.search(pattern, line_clean, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        
                        # Extract thông tin
                        if len(groups) >= 2:
                            number = groups[0]
                            title = groups[1]
                            page = int(groups[2]) if len(groups) > 2 and groups[2] and groups[2].isdigit() else None
                        else:
                            number = None
                            title = groups[0] if groups else line_clean
                            page = None
                        
                        entry = TOCEntry(
                            title=title.strip(),
                            page=page,
                            level=level,
                            confidence=0.8,
                            source='text'
                        )
                        entries.append(entry)
                        break
                else:
                    continue
                break
        
        return entries

    def _llm_based_detection(self, text: str) -> List[TOCEntry]:
        """Sử dụng LLM để detect và parse table of contents."""
        if not self.llm_provider:
            return []
            
        logger.info("🤖 Sử dụng LLM để detect table of contents")
        
        system_prompt = """
        Bạn là chuyên gia phân tích cấu trúc tài liệu. Nhiệm vụ của bạn là tìm và trích xuất 
        table of contents (mục lục) từ văn bản, kể cả khi nó không có format chuẩn.

        QUY TẮC:
        - LUÔN trả lời bằng tiếng Việt
        - Chỉ trả về JSON array, không giải thích thêm
        """
        
        user_prompt = f"""
        Phân tích văn bản sau và tìm table of contents. Trả về JSON array với format:
        [
            {{"title": "Tiêu đề chương/mục", "page": số_trang_hoặc_null, "level": cấp_độ_1_2_3}}
        ]

        Văn bản:
        {text[:8000]}...
        """
        
        try:
            response = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Parse JSON response
            import json
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                entries_data = json.loads(json_match.group())
                return [
                    TOCEntry(
                        title=entry['title'],
                        page=entry.get('page'),
                        level=entry.get('level', 1),
                        confidence=0.7,
                        source='llm'
                    )
                    for entry in entries_data
                ]
        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
        
        return []

    def _ocr_based_detection(self, image: np.ndarray) -> List[TOCEntry]:
        """OCR-based detection với EasyOCR và Tesseract."""
        entries = []
        
        if 'easyocr' in self.ocr_readers:
            try:
                # EasyOCR detection
                results = self.ocr_readers['easyocr'].readtext(image)
                
                for bbox, text, confidence in results:
                    if confidence > 0.5 and len(text.strip()) > 3:
                        # Phân tích text để xác định level và page
                        level = self._analyze_text_level(text)
                        page = self._extract_page_number(text)
                        
                        entry = TOCEntry(
                            title=text.strip(),
                            page=page,
                            level=level,
                            confidence=confidence,
                            source='ocr',
                            bbox=tuple(map(int, bbox[0] + bbox[2]))  # Convert to (x1,y1,x2,y2)
                        )
                        entries.append(entry)
            except Exception as e:
                logger.error(f"EasyOCR failed: {e}")
        
        return entries

    def _computer_vision_detection(self, image: np.ndarray) -> List[TOCEntry]:
        """Computer Vision để detect visual structure của TOC."""
        entries = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect lines (cho dotted lines trong TOC)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            # Detect text regions
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze layout để determine TOC structure
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 15:  # Reasonable text size
                    # Estimate level based on x position (indentation)
                    level = max(1, (x // 30) + 1)
                    
                    # Extract text trong region này (cần OCR)
                    roi = image[y:y+h, x:x+w]
                    # ... OCR processing cho roi
                    
            logger.info(f"CV detected {len(entries)} potential TOC entries")
            
        except Exception as e:
            logger.error(f"Computer vision detection failed: {e}")
        
        return entries

    def _extract_pdf_toc_metadata(self, pdf_source) -> List[TOCEntry]:
        """Extract TOC từ PDF metadata/bookmarks."""
        if not HAS_PDF:
            return []
            
        entries = []
        try:
            doc = fitz.open(pdf_source)
            toc = doc.get_toc()
            
            for level, title, page in toc:
                entry = TOCEntry(
                    title=title,
                    page=page,
                    level=level,
                    confidence=0.95,
                    source='pdf_metadata'
                )
                entries.append(entry)
                
            doc.close()
            logger.info(f"Extracted {len(entries)} từ PDF metadata")
            
        except Exception as e:
            logger.error(f"PDF metadata extraction failed: {e}")
            
        return entries

    def _extract_pdf_toc_ocr(self, pdf_source) -> List[TOCEntry]:
        """OCR trên các trang đầu của PDF để tìm TOC."""
        if not HAS_PDF or not HAS_OCR:
            return []
            
        entries = []
        try:
            # Convert các trang đầu thành images
            images = pdf2image.convert_from_path(pdf_source, first_page=1, last_page=5)
            
            for page_num, image in enumerate(images, 1):
                # Convert PIL to OpenCV
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # OCR detection
                page_entries = self._ocr_based_detection(cv_image)
                
                # Filter để chỉ giữ entries có vẻ như TOC
                toc_entries = [e for e in page_entries if self._looks_like_toc_entry(e.title)]
                entries.extend(toc_entries)
                
            logger.info(f"OCR trên PDF tìm được {len(entries)} entries")
            
        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            
        return entries

    def _extract_pdf_toc_text(self, pdf_source) -> List[TOCEntry]:
        """Extract text từ PDF và pattern matching."""
        if not HAS_PDF:
            return []
            
        try:
            doc = fitz.open(pdf_source)
            full_text = ""
            
            # Extract text từ 10 trang đầu
            for page_num in range(min(10, doc.page_count)):
                page = doc[page_num]
                full_text += page.get_text()
                
            doc.close()
            
            # Pattern matching trên full text
            return self._detect_from_text(full_text)
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return []

    def _analyze_text_level(self, text: str) -> int:
        """Phân tích text để xác định level (1, 2, 3...)."""
        # Check for numbering patterns
        if re.match(r'^\d+\.\d+\.\d+', text):
            return 3
        elif re.match(r'^\d+\.\d+', text):
            return 2
        elif re.match(r'^\d+\.', text) or re.match(r'^(Chapter|Chương)', text, re.I):
            return 1
        else:
            return 2  # Default

    def _extract_page_number(self, text: str) -> Optional[int]:
        """Extract page number từ text."""
        # Look for patterns như "... 25" hoặc "... page 25"
        matches = re.findall(r'\.+\s*(\d+)$', text)
        if matches:
            return int(matches[-1])
        
        matches = re.findall(r'\b(\d+)\s*$', text)
        if matches:
            return int(matches[-1])
            
        return None

    def _looks_like_toc_entry(self, text: str) -> bool:
        """Kiểm tra xem text có giống TOC entry không."""
        # Heuristics để identify TOC entries
        if len(text.strip()) < 5:
            return False
            
        # Has page number at end
        if re.search(r'\.+\s*\d+\s*$', text):
            return True
            
        # Has chapter/section indicators
        if re.search(r'^(Chapter|Chương|Section|Phần|\d+\.)', text, re.I):
            return True
            
        # Reasonable length
        if 10 <= len(text) <= 150:
            return True
            
        return False

    def _merge_and_deduplicate(self, entries: List[TOCEntry]) -> List[TOCEntry]:
        """Merge entries từ các sources và loại bỏ duplicates."""
        if not entries:
            return []
            
        # Sort theo confidence và source priority
        source_priority = {'pdf_metadata': 4, 'text': 3, 'ocr': 2, 'llm': 1, 'cv': 0}
        entries.sort(key=lambda x: (-x.confidence, -source_priority.get(x.source, 0)))
        
        # Deduplicate based on title similarity
        unique_entries = []
        seen_titles = set()
        
        for entry in entries:
            title_normalized = re.sub(r'\s+', ' ', entry.title.lower().strip())
            
            # Check similarity với existing titles
            is_duplicate = False
            for seen_title in seen_titles:
                if self._text_similarity(title_normalized, seen_title) > 0.8:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_entries.append(entry)
                seen_titles.add(title_normalized)
        
        # Sort theo level và page
        unique_entries.sort(key=lambda x: (x.level, x.page or 0))
        
        logger.info(f"Merged {len(entries)} entries into {len(unique_entries)} unique entries")
        return unique_entries

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity score."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def _llm_validate_entries(self, entries: List[TOCEntry], image: np.ndarray) -> List[TOCEntry]:
        """Sử dụng LLM để validate và improve entries."""
        if not self.llm_provider or not entries:
            return entries
            
        # Convert entries to text format for LLM
        entries_text = "\n".join([
            f"Level {e.level}: {e.title} (Page {e.page or '?'}) [Confidence: {e.confidence:.2f}]"
            for e in entries
        ])
        
        system_prompt = """
        Bạn là chuyên gia phân tích table of contents. Nhiệm vụ của bạn là validate và cải thiện 
        kết quả detect từ OCR/CV, loại bỏ noise và sửa lỗi.

        QUY TẮC: LUÔN trả lời bằng tiếng Việt
        """
        
        user_prompt = f"""
        Đây là kết quả detect table of contents từ hình ảnh:

        {entries_text}

        Hãy validate và improve kết quả này:
        1. Loại bỏ entries không phải TOC (noise)
        2. Sửa lỗi OCR trong title
        3. Xác định đúng level hierarchy
        4. Trả về JSON array cải thiện

        Format: [{{"title": "...", "page": ..., "level": ...}}]
        """
        
        try:
            response = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Parse improved entries
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                improved_data = json.loads(json_match.group())
                return [
                    TOCEntry(
                        title=entry['title'],
                        page=entry.get('page'),
                        level=entry.get('level', 1),
                        confidence=0.9,
                        source='llm_validated'
                    )
                    for entry in improved_data
                ]
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
        
        return entries

    # Method để tương thích với interface cũ
    def detect_chapters(self, text: str) -> List[Dict[str, Any]]:
        """Compatibility method cho interface cũ."""
        entries = self.detect_table_of_contents(text, 'text')
        
        # Convert TOCEntry to old format
        return [
            {
                'number': i + 1,
                'title': entry.title,
                'line_number': 0,  # Không có line number trong new format
                'level': entry.level,
                'page': entry.page
            }
            for i, entry in enumerate(entries)
        ]