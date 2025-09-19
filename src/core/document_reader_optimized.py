# src/core/document_reader_optimized.py
# Module tối ưu cho việc đọc và phân tích cấu trúc tài liệu

import logging
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChapterCandidate:
    """Cấu trúc dữ liệu cho ứng viên chapter."""
    number: Optional[int]
    title: str
    line_number: int
    level: int
    quality_score: float
    pattern_type: str
    context_score: float
    importance: str

class OptimizedChapterDetector:
    """
    Chapter Detector được tối ưu để tạo cấu trúc phân tầng chính xác.
    Sử dụng AI-driven approach để phân tích cấu trúc tài liệu.
    """
    
    def __init__(self):
        self.profile = {
            'patterns': [
                {
                    'pattern': r'^(Chương|Chapter|CHƯƠNG)\s+([IVX]+|\d+)[:\.\s]+(.+)$',
                    'type': 'main_chapter',
                    'level': 1,
                    'weight': 10
                },
                {
                    'pattern': r'^(Bài|Lesson|BÀI)\s+(\d+)[:\.\s]+(.+)$',
                    'type': 'main_chapter', 
                    'level': 1,
                    'weight': 9
                },
                {
                    'pattern': r'^(\d+)\.\s+(.+)$',
                    'type': 'numbered_section',
                    'level': 1,
                    'weight': 8
                },
                {
                    'pattern': r'^(\d+)\.(\d+)\.\s+(.+)$',
                    'type': 'sub_section',
                    'level': 2, 
                    'weight': 6
                },
                {
                    'pattern': r'^(\d+)\.(\d+)\.(\d+)\.\s+(.+)$',
                    'type': 'sub_sub_section',
                    'level': 3,
                    'weight': 4
                },
                {
                    'pattern': r'^([A-Z])\.\s+(.+)$',
                    'type': 'alphabetic_section',
                    'level': 2,
                    'weight': 5
                },
                {
                    'pattern': r'^(I+|V+|X+|L+|C+|D+|M+)\.\s+(.+)$',
                    'type': 'roman_section',
                    'level': 1,
                    'weight': 7
                }
            ],
            'important_keywords': [
                'giới thiệu', 'tổng quan', 'khái niệm', 'cơ bản', 'nâng cao',
                'ứng dụng', 'thực hành', 'bài tập', 'ví dụ', 'kết luận',
                'tóm tắt', 'đánh giá', 'kiểm tra', 'phương pháp', 'nguyên lý'
            ],
            'exclude_keywords': [
                'trang', 'page', 'tác giả', 'author', 'nguồn', 'source',
                'tham khảo', 'reference', 'mục lục', 'index', 'danh sách'
            ],
            'quality_thresholds': {
                'minimum': 5.0,
                'good': 7.0, 
                'excellent': 9.0
            }
        }
        logger.info("✅ OptimizedChapterDetector đã được khởi tạo.")

    def detect_hierarchical_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Phát hiện cấu trúc phân tầng thông minh của tài liệu.
        
        Args:
            text (str): Nội dung tài liệu đầy đủ
            
        Returns:
            List[Dict]: Cấu trúc phân tầng được tối ưu
        """
        logger.info("🔍 Bắt đầu phân tích cấu trúc phân tầng thông minh...")
        
        # Bước 1: Tiền xử lý và chuẩn hóa văn bản
        processed_lines = self._preprocess_text(text)
        
        # Bước 2: Tìm tất cả ứng viên chapters
        candidates = self._extract_chapter_candidates(processed_lines)
        
        # Bước 3: Đánh giá chất lượng và ngữ cảnh
        scored_candidates = self._score_candidates(candidates, processed_lines)
        
        # Bước 4: Xây dựng cấu trúc phân tầng
        hierarchical_structure = self._build_hierarchy(scored_candidates)
        
        # Bước 5: Tối ưu và làm sạch kết quả
        final_structure = self._optimize_final_structure(hierarchical_structure)
        
        logger.info(f"✅ Hoàn thành phân tích: {len(final_structure)} mục với cấu trúc phân tầng.")
        return final_structure

    def _preprocess_text(self, text: str) -> List[str]:
        """Tiền xử lý văn bản để cải thiện chất lượng phân tích."""
        lines = text.split('\n')
        processed = []
        
        for i, line in enumerate(lines):
            # Chuẩn hóa spacing và encoding
            cleaned = re.sub(r'\s+', ' ', line.strip())
            
            # Bỏ qua dòng trống, quá ngắn, hoặc quá dài
            if not cleaned or len(cleaned) < 3 or len(cleaned) > 300:
                continue
                
            # Bỏ qua dòng chỉ chứa số trang
            if re.match(r'^\d+$', cleaned):
                continue
                
            # Bỏ qua dòng chứa từ khóa loại trừ
            if any(kw in cleaned.lower() for kw in self.profile['exclude_keywords']):
                continue
                
            processed.append(cleaned)
            
        logger.info(f"Đã xử lý {len(processed)} dòng hợp lệ từ {len(lines)} dòng gốc.")
        return processed

    def _extract_chapter_candidates(self, lines: List[str]) -> List[ChapterCandidate]:
        """Trích xuất tất cả ứng viên chapters với thông tin chi tiết."""
        candidates = []
        
        for line_idx, line in enumerate(lines[:10000]):  # Tăng phạm vi quét
            for pattern_info in self.profile['patterns']:
                match = re.match(pattern_info['pattern'], line, re.IGNORECASE)
                if match:
                    # Trích xuất thông tin từ pattern
                    groups = match.groups()
                    
                    if pattern_info['type'] in ['sub_section', 'sub_sub_section']:
                        # Patterns có nhiều nhóm số
                        number_str = groups[0]  # Lấy số đầu tiên
                        title = groups[-1]      # Lấy title cuối cùng
                    else:
                        # Patterns đơn giản
                        number_str = groups[0] if len(groups) > 1 else None
                        title = groups[-1]
                    
                    # Tạo candidate
                    candidate = ChapterCandidate(
                        number=self._extract_number(number_str),
                        title=title.strip(),
                        line_number=line_idx,
                        level=pattern_info['level'],
                        quality_score=0.0,  # Sẽ tính sau
                        pattern_type=pattern_info['type'],
                        context_score=0.0,  # Sẽ tính sau
                        importance='medium'
                    )
                    
                    candidates.append(candidate)
                    break  # Tìm thấy pattern phù hợp, chuyển dòng tiếp theo
        
        logger.info(f"Tìm được {len(candidates)} ứng viên chapters.")
        return candidates

    def _extract_number(self, number_str: Optional[str]) -> Optional[int]:
        """Trích xuất số từ chuỗi (hỗ trợ số La Mã và Arabic)."""
        if not number_str:
            return None
            
        # Số Arabic
        arabic_match = re.search(r'\d+', number_str)
        if arabic_match:
            return int(arabic_match.group())
            
        # Số La Mã đơn giản
        roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 
                    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
        if number_str.upper() in roman_map:
            return roman_map[number_str.upper()]
            
        return None

    def _score_candidates(self, candidates: List[ChapterCandidate], 
                         lines: List[str]) -> List[ChapterCandidate]:
        """Đánh giá chất lượng từng candidate với thuật toán nâng cao."""
        for candidate in candidates:
            # Điểm cơ bản từ pattern type
            base_score = self._get_pattern_base_score(candidate.pattern_type)
            
            # Điểm chất lượng title
            title_score = self._evaluate_title_quality(candidate.title)
            
            # Điểm ngữ cảnh xung quanh
            context_score = self._evaluate_context(candidate, lines)
            
            # Điểm vị trí trong tài liệu
            position_score = self._evaluate_position(candidate, len(lines))
            
            # Tính tổng điểm có trọng số
            candidate.quality_score = (
                base_score * 0.3 +
                title_score * 0.4 + 
                context_score * 0.2 +
                position_score * 0.1
            )
            
            candidate.context_score = context_score
            
            # Đánh giá mức độ quan trọng
            if candidate.quality_score >= self.profile['quality_thresholds']['excellent']:
                candidate.importance = 'high'
            elif candidate.quality_score >= self.profile['quality_thresholds']['good']:
                candidate.importance = 'medium'
            else:
                candidate.importance = 'low'
        
        return candidates

    def _get_pattern_base_score(self, pattern_type: str) -> float:
        """Điểm cơ bản cho từng loại pattern."""
        scores = {
            'main_chapter': 10.0,
            'numbered_section': 8.0,
            'sub_section': 6.0,
            'sub_sub_section': 4.0,
            'alphabetic_section': 5.0,
            'roman_section': 7.0
        }
        return scores.get(pattern_type, 3.0)

    def _evaluate_title_quality(self, title: str) -> float:
        """Đánh giá chất lượng tiêu đề."""
        score = 0.0
        
        # Độ dài phù hợp
        if 10 <= len(title) <= 100:
            score += 4.0
        elif 5 <= len(title) <= 150:
            score += 2.0
        else:
            score += 0.5
            
        # Chứa từ khóa quan trọng
        keyword_bonus = sum(2.0 for kw in self.profile['important_keywords'] 
                           if kw in title.lower())
        score += min(keyword_bonus, 6.0)  # Tối đa 6 điểm từ keywords
        
        # Cấu trúc ngữ pháp tốt (có động từ, danh từ)
        if re.search(r'[a-zA-ZÀ-ỹ]+', title):  # Chứa chữ cái
            score += 2.0
            
        # Không chứa ký tự đặc biệt lạ
        if not re.search(r'[^\w\s\-\.\(\)àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]', title):
            score += 1.0
            
        return min(score, 10.0)  # Tối đa 10 điểm

    def _evaluate_context(self, candidate: ChapterCandidate, lines: List[str]) -> float:
        """Đánh giá ngữ cảnh xung quanh candidate."""
        score = 0.0
        line_idx = candidate.line_number
        
        # Lấy ngữ cảnh xung quanh (5 dòng trước và sau)
        start_idx = max(0, line_idx - 5)
        end_idx = min(len(lines), line_idx + 5)
        context_lines = lines[start_idx:end_idx]
        context_text = ' '.join(context_lines).lower()
        
        # Bonus nếu có keywords liên quan trong ngữ cảnh
        context_keywords = [
            'mục tiêu', 'nội dung', 'kiến thức', 'kỹ năng', 'học', 'hiểu',
            'phương pháp', 'cách thức', 'quy trình', 'bước', 'giai đoạn'
        ]
        
        keyword_count = sum(1 for kw in context_keywords if kw in context_text)
        score += keyword_count * 0.5
        
        # Bonus nếu dòng trước/sau có format tương tự (cấu trúc nhất quán)
        if line_idx > 0 and line_idx < len(lines) - 1:
            prev_line = lines[line_idx - 1].strip()
            next_line = lines[line_idx + 1].strip()
            
            # Kiểm tra tính nhất quán format
            if any(re.match(pattern['pattern'], prev_line) or re.match(pattern['pattern'], next_line) 
                  for pattern in self.profile['patterns']):
                score += 2.0
        
        # Penalty nếu quá gần với candidate khác
        nearby_candidates_count = sum(1 for other_line in range(max(0, line_idx-3), min(len(lines), line_idx+3))
                                    if other_line != line_idx and 
                                    any(re.match(p['pattern'], lines[other_line]) for p in self.profile['patterns']))
        if nearby_candidates_count > 2:
            score -= 1.0
            
        return max(0.0, min(score, 10.0))

    def _evaluate_position(self, candidate: ChapterCandidate, total_lines: int) -> float:
        """Đánh giá vị trí của candidate trong tài liệu."""
        position_ratio = candidate.line_number / total_lines
        
        # Chapters thường xuất hiện đầu document hoặc phân bố đều
        if position_ratio < 0.1:  # 10% đầu
            return 8.0
        elif position_ratio > 0.9:  # 10% cuối
            return 3.0
        else:
            return 6.0

    def _build_hierarchy(self, candidates: List[ChapterCandidate]) -> List[Dict[str, Any]]:
        """Xây dựng cấu trúc phân tầng từ candidates."""
        # Lọc candidates chất lượng cao
        quality_candidates = [c for c in candidates 
                            if c.quality_score >= self.profile['quality_thresholds']['minimum']]
        
        # Sắp xếp theo chất lượng và vị trí
        quality_candidates.sort(key=lambda x: (-x.quality_score, x.line_number))
        
        # Phân loại theo level
        level_groups = {1: [], 2: [], 3: []}
        for candidate in quality_candidates:
            if candidate.level in level_groups:
                level_groups[candidate.level].append(candidate)
        
        # Xây dựng cấu trúc phân tầng
        hierarchy = []
        
        # Level 1: Main chapters (tối đa 12)
        main_chapters = level_groups[1][:12]
        
        for i, main_chapter in enumerate(main_chapters):
            main_dict = {
                'number': main_chapter.number or (i + 1),
                'title': main_chapter.title,
                'line_number': main_chapter.line_number,
                'level': 1,
                'display_number': str(i + 1),
                'quality_score': main_chapter.quality_score,
                'children': []
            }
            
            # Tìm sub-chapters cho main chapter này
            main_line = main_chapter.line_number
            next_main_line = main_chapters[i+1].line_number if i+1 < len(main_chapters) else float('inf')
            
            # Level 2: Sub-chapters
            relevant_subs = [c for c in level_groups[2] 
                           if main_line < c.line_number < next_main_line][:5]  # Tối đa 5 sub
            
            for j, sub_chapter in enumerate(relevant_subs):
                sub_dict = {
                    'number': sub_chapter.number or (j + 1),
                    'title': sub_chapter.title,
                    'line_number': sub_chapter.line_number,
                    'level': 2,
                    'display_number': f"{i + 1}.{j + 1}",
                    'parent_number': str(i + 1),
                    'quality_score': sub_chapter.quality_score,
                    'children': []
                }
                
                # Level 3: Sub-sub-chapters
                sub_line = sub_chapter.line_number
                next_sub_line = relevant_subs[j+1].line_number if j+1 < len(relevant_subs) else next_main_line
                
                relevant_subsubs = [c for c in level_groups[3] 
                                  if sub_line < c.line_number < next_sub_line][:3]  # Tối đa 3 sub-sub
                
                for k, subsub_chapter in enumerate(relevant_subsubs):
                    subsub_dict = {
                        'number': subsub_chapter.number or (k + 1),
                        'title': subsub_chapter.title,
                        'line_number': subsub_chapter.line_number,
                        'level': 3,
                        'display_number': f"{i + 1}.{j + 1}.{k + 1}",
                        'parent_number': f"{i + 1}.{j + 1}",
                        'quality_score': subsub_chapter.quality_score
                    }
                    sub_dict['children'].append(subsub_dict)
                
                main_dict['children'].append(sub_dict)
            
            hierarchy.append(main_dict)
        
        return hierarchy

    def _optimize_final_structure(self, hierarchy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tối ưu cấu trúc cuối cùng để phù hợp với mục lục."""
        optimized = []
        
        for main_chapter in hierarchy:
            # Làm sạch main chapter
            clean_main = {
                'number': main_chapter['display_number'],
                'title': main_chapter['title'],
                'line_number': main_chapter['line_number'],
                'level': 1
            }
            optimized.append(clean_main)
            
            # Thêm sub-chapters
            for sub_chapter in main_chapter.get('children', []):
                clean_sub = {
                    'number': sub_chapter['display_number'], 
                    'title': sub_chapter['title'],
                    'line_number': sub_chapter['line_number'],
                    'level': 2
                }
                optimized.append(clean_sub)
                
                # Thêm sub-sub-chapters
                for subsub_chapter in sub_chapter.get('children', []):
                    clean_subsub = {
                        'number': subsub_chapter['display_number'],
                        'title': subsub_chapter['title'], 
                        'line_number': subsub_chapter['line_number'],
                        'level': 3
                    }
                    optimized.append(clean_subsub)
        
        # Sắp xếp lại theo line_number để đảm bảo thứ tự
        optimized.sort(key=lambda x: x['line_number'])
        
        logger.info(f"Tối ưu hoàn thành: {len(optimized)} mục trong cấu trúc phân tầng.")
        return optimized

    # Compatibility method để tương thích với DocumentReader
    def detect_chapters(self, text: str) -> List[Dict[str, Any]]:
        """
        Method tương thích với interface cũ để không phá vỡ existing code.
        """
        return self.detect_hierarchical_structure(text)

# Hàm tiện ích để tương thích với code cũ
def create_optimized_detector() -> OptimizedChapterDetector:
    """Factory function để tạo OptimizedChapterDetector."""
    return OptimizedChapterDetector()