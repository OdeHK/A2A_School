# src/core/unified_document_processor.py
# Unified pipeline tích hợp tất cả approaches

import logging
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Import các detector khác nhau
from .document_reader_optimized import OptimizedChapterDetector
from .hybrid_chapter_detector import HybridChapterDetector
from ..agents.summarization_agent_optimized import SummarizationAgentOptimized
from ..agents.quiz_generation_agent_optimized import QuizGenerationAgentOptimized

logger = logging.getLogger(__name__)

class UnifiedDocumentProcessor:
    """
    Unified Document Processor kết hợp tất cả approaches:
    1. Traditional Pattern Matching
    2. OCR + Computer Vision  
    3. LLM-based intelligent parsing
    4. Hybrid approach với fallback mechanisms
    """
    
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        
        # Initialize các detectors
        self.traditional_detector = OptimizedChapterDetector()
        self.hybrid_detector = HybridChapterDetector(llm_provider)
        
        # Initialize agents
        self.summarizer = SummarizationAgentOptimized(llm_provider)
        self.quiz_generator = QuizGenerationAgentOptimized(llm_provider)
        
        logger.info("✅ UnifiedDocumentProcessor đã được khởi tạo.")

    def process_document(
        self, 
        content: Union[str, Path], 
        content_type: str = 'auto',
        processing_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Xử lý tài liệu với unified approach.
        
        Args:
            content: Nội dung tài liệu (text, file path, etc.)
            content_type: 'text', 'pdf', 'image', 'auto'
            processing_options: Các tùy chọn xử lý
            
        Returns:
            Dict chứa tất cả kết quả phân tích
        """
        if processing_options is None:
            processing_options = {
                'enable_ocr': True,
                'enable_cv': True, 
                'enable_llm': True,
                'fallback_traditional': True,
                'quality_threshold': 0.7
            }
        
        logger.info(f"🚀 Bắt đầu xử lý tài liệu với unified approach")
        
        result = {
            'content_type': content_type,
            'processing_options': processing_options,
            'chapters': [],
            'table_of_contents': '',
            'summaries': {},
            'quiz_ready': False,
            'confidence_score': 0.0,
            'processing_methods_used': [],
            'errors': []
        }
        
        try:
            # Phase 1: Chapter Detection với multiple approaches
            chapters = self._detect_chapters_unified(content, content_type, processing_options)
            result['chapters'] = chapters
            
            # Phase 2: Generate Table of Contents
            if chapters:
                toc = self._generate_toc_from_chapters(chapters, content)
                result['table_of_contents'] = toc
                
            # Phase 3: Document Analysis
            if isinstance(content, str):  # Text content
                analysis = self.summarizer.generate_comprehensive_analysis(content, chapters)
                result.update(analysis)
                
            # Phase 4: Quiz Readiness Assessment
            result['quiz_ready'] = len(chapters) >= 3 and len(str(content)) > 1000
            
            # Phase 5: Quality Assessment
            result['confidence_score'] = self._calculate_overall_confidence(result)
            
            logger.info(f"✅ Hoàn thành xử lý tài liệu. Confidence: {result['confidence_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình xử lý: {e}")
            result['errors'].append(str(e))
            
        return result

    def _detect_chapters_unified(
        self, 
        content: Union[str, Path], 
        content_type: str,
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect chapters bằng unified approach với fallback."""
        chapters = []
        methods_used = []
        
        # Method 1: Hybrid Detector (if content supports it)
        if content_type in ['pdf', 'image'] or options.get('enable_ocr', True):
            try:
                logger.info("🔄 Thử Hybrid Detection (OCR + CV + LLM)")
                hybrid_chapters = self.hybrid_detector.detect_table_of_contents(content, content_type)
                
                if hybrid_chapters:
                    # Convert TOCEntry to standard format
                    chapters = [
                        {
                            'number': i + 1,
                            'title': entry.title,
                            'line_number': 0,
                            'level': entry.level,
                            'page': entry.page,
                            'confidence': entry.confidence,
                            'source': entry.source
                        }
                        for i, entry in enumerate(hybrid_chapters)
                    ]
                    methods_used.append('hybrid')
                    logger.info(f"✅ Hybrid detection tìm được {len(chapters)} chapters")
                    
            except Exception as e:
                logger.warning(f"Hybrid detection failed: {e}")
        
        # Method 2: Traditional Pattern Matching (fallback or text-only)
        if not chapters or options.get('fallback_traditional', True):
            try:
                logger.info("🔄 Thử Traditional Pattern Matching")
                
                # Convert content to text if needed
                text_content = self._extract_text_content(content, content_type)
                
                if text_content:
                    traditional_chapters = self.traditional_detector.detect_hierarchical_structure(text_content)
                    
                    if traditional_chapters and (not chapters or len(traditional_chapters) > len(chapters)):
                        chapters = traditional_chapters
                        methods_used.append('traditional')
                        logger.info(f"✅ Traditional detection tìm được {len(chapters)} chapters")
                        
            except Exception as e:
                logger.warning(f"Traditional detection failed: {e}")
        
        # Method 3: LLM-only approach (last resort)
        if not chapters and options.get('enable_llm', True):
            try:
                logger.info("🔄 Thử LLM-only detection")
                text_content = self._extract_text_content(content, content_type)
                
                if text_content:
                    llm_chapters = self._llm_only_chapter_detection(text_content)
                    if llm_chapters:
                        chapters = llm_chapters
                        methods_used.append('llm_only')
                        logger.info(f"✅ LLM-only detection tìm được {len(chapters)} chapters")
                        
            except Exception as e:
                logger.warning(f"LLM-only detection failed: {e}")
        
        # Add metadata về methods used
        for chapter in chapters:
            chapter['detection_methods'] = methods_used
            
        return chapters

    def _extract_text_content(self, content: Union[str, Path], content_type: str) -> str:
        """Extract text content từ các loại input khác nhau."""
        if isinstance(content, str):
            return content
            
        # Handle file paths
        if isinstance(content, Path) or (isinstance(content, str) and Path(content).exists()):
            file_path = Path(content)
            
            if content_type == 'pdf' or file_path.suffix.lower() == '.pdf':
                # Extract text from PDF
                try:
                    import fitz
                    doc = fitz.open(file_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                    return text
                except:
                    logger.warning("Could not extract text from PDF")
                    
            elif file_path.suffix.lower() in ['.txt', '.md']:
                return file_path.read_text(encoding='utf-8')
                
        return ""

    def _llm_only_chapter_detection(self, text: str) -> List[Dict[str, Any]]:
        """LLM-only approach cho chapter detection."""
        system_prompt = """
        Bạn là chuyên gia phân tích cấu trúc tài liệu. Nhiệm vụ của bạn là phân tích văn bản 
        và xác định cấu trúc chương/mục một cách thông minh.

        QUY TẮC: LUÔN trả lời bằng tiếng Việt và chỉ trả về JSON.
        """
        
        user_prompt = f"""
        Phân tích văn bản sau và xác định cấu trúc chương/mục. Trả về JSON array:
        [{{"number": số_thứ_tự, "title": "tiêu_đề", "level": cấp_độ_1_2_3, "confidence": độ_tin_cậy}}]

        Văn bản (10000 ký tự đầu):
        {text[:10000]}...
        """
        
        try:
            response = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Parse JSON
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                chapters_data = json.loads(json_match.group())
                return [
                    {
                        'number': ch.get('number', i+1),
                        'title': ch['title'],
                        'line_number': 0,
                        'level': ch.get('level', 1),
                        'confidence': ch.get('confidence', 0.6),
                        'source': 'llm_only'
                    }
                    for i, ch in enumerate(chapters_data)
                ]
        except Exception as e:
            logger.error(f"LLM-only detection failed: {e}")
            
        return []

    def _generate_toc_from_chapters(self, chapters: List[Dict], content: str) -> str:
        """Generate table of contents từ detected chapters."""
        try:
            # Prepare text content for TOC generation
            text_content = str(content) if isinstance(content, str) else ""
            
            # Use optimized summarizer với chapter info
            bookmarks = [
                {
                    'title': ch['title'],
                    'number': ch['number'],
                    'level': ch.get('level', 1)
                }
                for ch in chapters
            ]
            
            toc = self.summarizer.generate_hierarchical_toc(text_content, bookmarks)
            return toc
            
        except Exception as e:
            logger.error(f"TOC generation failed: {e}")
            return "Không thể tạo mục lục."

    def _calculate_overall_confidence(self, result: Dict[str, Any]) -> float:
        """Tính confidence score tổng thể."""
        scores = []
        
        # Chapter detection confidence
        if result['chapters']:
            chapter_confidences = [ch.get('confidence', 0.5) for ch in result['chapters']]
            scores.append(sum(chapter_confidences) / len(chapter_confidences))
        else:
            scores.append(0.1)
            
        # TOC quality
        if result.get('table_of_contents') and len(result['table_of_contents']) > 100:
            scores.append(0.8)
        else:
            scores.append(0.3)
            
        # Processing success
        if not result.get('errors'):
            scores.append(0.9)
        else:
            scores.append(0.4)
            
        return sum(scores) / len(scores) if scores else 0.0

    def generate_quiz_from_document(
        self, 
        content: str, 
        chapters: List[Dict], 
        quiz_requirements: str,
        num_questions: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate quiz từ document đã được phân tích.
        Đảm bảo câu hỏi chỉ dựa trên nội dung tài liệu.
        """
        logger.info(f"📝 Tạo quiz với {num_questions} câu hỏi từ tài liệu")
        
        # Create question plan
        plan = self.summarizer.create_advanced_question_plan(
            user_request=f"{quiz_requirements}. Tạo {num_questions} câu hỏi dựa trên nội dung tài liệu.",
            target_difficulty="medium"
        )
        
        if not plan:
            logger.warning("Không thể tạo question plan")
            return []
        
        all_questions = []
        
        # Generate questions theo plan
        for task in plan[:num_questions]:  # Limit số lượng tasks
            try:
                # Extract relevant content cho task này
                task_content = self._extract_relevant_content(content, task, chapters)
                
                if task_content:
                    questions = self.quiz_generator.generate_professional_questions(
                        content=task_content,
                        task_details=task
                    )
                    all_questions.extend(questions)
                    
            except Exception as e:
                logger.error(f"Lỗi khi tạo câu hỏi cho task {task.get('section_title', 'unknown')}: {e}")
        
        # Limit total questions
        final_questions = all_questions[:num_questions]
        
        logger.info(f"✅ Đã tạo {len(final_questions)} câu hỏi từ tài liệu")
        return final_questions

    def _extract_relevant_content(
        self, 
        full_content: str, 
        task: Dict, 
        chapters: List[Dict]
    ) -> str:
        """Extract nội dung relevant cho một task cụ thể."""
        section_title = task.get('section_title', '')
        
        # Try to find matching chapter
        matching_chapter = None
        for chapter in chapters:
            if section_title.lower() in chapter['title'].lower():
                matching_chapter = chapter
                break
        
        if matching_chapter:
            # Extract content around this chapter
            # (This is simplified - in practice you'd use line numbers or page info)
            lines = full_content.split('\n')
            chapter_line = matching_chapter.get('line_number', 0)
            
            # Extract context around chapter
            start_line = max(0, chapter_line - 10)
            end_line = min(len(lines), chapter_line + 100)
            
            relevant_content = '\n'.join(lines[start_line:end_line])
            return relevant_content
        
        # Fallback: return subset of full content
        return full_content[:5000]  # First 5000 chars

    # Compatibility methods
    def detect_chapters(self, text: str) -> List[Dict[str, Any]]:
        """Compatibility method."""
        return self._detect_chapters_unified(text, 'text', {})
        
    def generate_table_of_contents(self, text: str) -> str:
        """Compatibility method."""
        chapters = self.detect_chapters(text)
        return self._generate_toc_from_chapters(chapters, text)