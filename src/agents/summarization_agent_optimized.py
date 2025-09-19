# src/agents/summarization_agent_optimized.py
# Agent chuyên biệt cho tóm tắt và tạo mục lục thông minh với cấu trúc phân tầng

import logging
import re
import json
from typing import Dict, Any, Optional, List, Tuple

# Sử dụng relative import để giữ cấu trúc module
from ..core.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

class SummarizationAgentOptimized:
    """
    Agent tối ưu cho việc tóm tắt văn bản và tạo mục lục có cấu trúc phân tầng.
    Được thiết kế để xử lý bookmark và tạo output chuyên nghiệp.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.table_of_contents = None
        self.document_structure = None
        logger.info("✅ SummarizationAgentOptimized đã được khởi tạo.")

    def generate_hierarchical_toc(self, document_text: str, existing_bookmarks: List[Dict] = None) -> str:
        """
        Tạo mục lục phân tầng thông minh dựa trên nội dung và bookmark có sẵn.
        
        Args:
            document_text (str): Nội dung đầy đủ của tài liệu
            existing_bookmarks (List[Dict], optional): Bookmark có sẵn từ PDF
            
        Returns:
            str: Mục lục được định dạng theo cấu trúc phân cấp
        """
        logger.info("🗺️ Đang tạo Mục lục phân tầng chuyên nghiệp...")
        
        # Phân tích bookmark hiện có
        bookmark_context = ""
        if existing_bookmarks:
            bookmark_context = f"\n\n## BOOKMARK CÓ SẴN:\n"
            for i, bookmark in enumerate(existing_bookmarks[:10]):  # Chỉ lấy 10 bookmark đầu
                title = bookmark.get('title', f'Chương {i+1}')
                number = bookmark.get('number', i+1)
                bookmark_context += f"- {number}. {title}\n"
        
        system_prompt = """
        Bạn là chuyên gia phân tích cấu trúc tài liệu học thuật chuyên nghiệp. 
        Nhiệm vụ của bạn là tạo một mục lục có cấu trúc phân tầng chi tiết và khoa học.

        QUY TẮC NGHIÊM NGẶT:
        - LUÔN LUÔN trả lời bằng tiếng Việt
        - KHÔNG sử dụng tiếng Anh trong phản hồi trừ thuật ngữ kỹ thuật
        - Tạo cấu trúc phân tầng rõ ràng với ít nhất 3-4 cấp độ
        """
        
        user_prompt = f"""
        ## NHIỆM VỤ: Tạo mục lục phân tầng chuyên nghiệp

        ### YÊU CẦU CHẤT LƯỢNG CAO:
        1. **Cấu trúc phân tầng:** Tối thiểu 3 cấp độ (1, 1.1, 1.1.1)
        2. **Định dạng chuẩn:** Số thứ tự + Tiêu đề + (Page X)
        3. **Phân tích sâu:** Chia nhỏ các chủ đề lớn thành mục con cụ thể
        4. **Tính logic:** Các mục con phải liên quan trực tiếp đến mục cha

        ### VÍ DỤ ĐỊNH DẠNG MONG MUỐN:
        ```
        1. Giới thiệu về Lập trình PHP (Page 1)
            1.1 Khái niệm cơ bản và lịch sử (Page 1)
                1.1.1 PHP là gì và ưu điểm (Page 1)
                1.1.2 Lịch sử phát triển của PHP (Page 2)
            1.2 Cài đặt môi trường phát triển (Page 3)
                1.2.1 Cài đặt XAMPP (Page 3)
                1.2.2 Cấu hình PHP và MySQL (Page 4)

        2. Cú pháp cơ bản trong PHP (Page 5)
            2.1 Biến và kiểu dữ liệu (Page 5)
                2.1.1 Khai báo biến (Page 5)
                2.1.2 Các kiểu dữ liệu cơ bản (Page 6)
            2.2 Cấu trúc điều khiển (Page 7)
                2.2.1 Câu lệnh if-else (Page 7)
                2.2.2 Vòng lặp for và while (Page 8)
        ```

        ### HƯỚNG DẪN PHÂN TÍCH:
        - Xác định các chủ đề chính (Level 1)
        - Chia nhỏ thành các khái niệm con (Level 2)  
        - Phân tích chi tiết các ví dụ và kỹ thuật (Level 3)
        - Ước tính số trang dựa trên độ dài nội dung{bookmark_context}

        ### NỘI DUNG TÀI LIỆU CẦN PHÂN TÍCH:
        {document_text[:12000]}...

        ### YÊU CẦU ĐẦU RA:
        Tạo mục lục hoàn chỉnh với cấu trúc phân tầng, mỗi mục có đầy đủ số thứ tự, tiêu đề mô tả và ước tính trang.
        """
        
        try:
            self.table_of_contents = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Xử lý và làm sạch output
            self.table_of_contents = self._clean_and_format_toc(self.table_of_contents)
            
            logger.info("✅ Đã tạo Mục lục phân tầng thành công.")
            return self.table_of_contents
        except Exception as e:
            logger.error(f"Lỗi khi tạo mục lục: {e}")
            return "Không thể tạo mục lục do lỗi hệ thống."

    def _clean_and_format_toc(self, raw_toc: str) -> str:
        """Làm sạch và định dạng lại mục lục."""
        # Loại bỏ các markdown formatting
        cleaned = re.sub(r'```.*?```', '', raw_toc, flags=re.DOTALL)
        cleaned = re.sub(r'#+\s*', '', cleaned)
        
        # Đảm bảo format đúng cho số trang
        cleaned = re.sub(r'\(Page\s*(\d+)\)', r'(Page \1)', cleaned)
        cleaned = re.sub(r'\(Trang\s*(\d+)\)', r'(Page \1)', cleaned)
        
        return cleaned.strip()

    def create_advanced_question_plan(self, user_request: str, target_difficulty: str = "medium") -> Optional[List[Dict]]:
        """
        Tạo kế hoạch sinh câu hỏi nâng cao với phân tích sâu.
        
        Args:
            user_request (str): Yêu cầu chi tiết về bộ đề
            target_difficulty (str): Mức độ khó mong muốn
            
        Returns:
            List[Dict]: Kế hoạch chi tiết để sinh câu hỏi
        """
        if not self.table_of_contents:
            logger.warning("Cần tạo mục lục trước khi lập kế hoạch câu hỏi.")
            return None

        logger.info("📝 Đang tạo Kế hoạch sinh câu hỏi nâng cao...")
        
        difficulty_map = {
            "easy": "Câu hỏi cơ bản, kiểm tra hiểu biết định nghĩa và khái niệm",
            "medium": "Câu hỏi ứng dụng, yêu cầu phân tích và so sánh",
            "hard": "Câu hỏi tổng hợp, đánh giá khả năng sáng tạo và giải quyết vấn đề"
        }
        
        difficulty_desc = difficulty_map.get(target_difficulty, difficulty_map["medium"])
        
        system_prompt = """
        Bạn là chuyên gia thiết kế đề thi và đánh giá giáo dục chuyên nghiệp. 
        Nhiệm vụ của bạn là tạo kế hoạch phân bổ câu hỏi khoa học và cân bằng.

        QUY TẮC NGHIÊM NGẶT:
        - LUÔN LUÔN trả lời bằng tiếng Việt
        - KHÔNG sử dụng tiếng Anh trong phản hồi
        - Tạo kế hoạch cân bằng giữa các cấp độ kiến thức
        """
        
        user_prompt = f"""
        ## NHIỆM VỤ: Thiết kế kế hoạch sinh câu hỏi chuyên nghiệp

        ### THÔNG TIN ĐẦU VÀO:
        **Mục lục tài liệu:**
        {self.table_of_contents}

        **Yêu cầu giáo viên:**
        {user_request}

        **Mức độ khó:**
        {difficulty_desc}

        ### YÊU CẦU THIẾT KẾ:
        1. **Phân bổ cân bằng:** Đảm bảo tất cả chương quan trọng đều có câu hỏi
        2. **Đa dạng loại câu hỏi:** Kết hợp nhiều hình thức đánh giá
        3. **Phù hợp đối tượng:** Phù hợp với trình độ và mục tiêu học tập

        ### ĐỊNH DẠNG ĐẦU RA (JSON):
        Trả về danh sách các task, mỗi task có:
        - "section_title": Tiêu đề mục (bằng tiếng Việt)
        - "section_level": Cấp độ của mục (1, 2, 3...)
        - "num_questions": Số câu hỏi cho mục này
        - "question_types": Loại câu hỏi (trắc nghiệm, tự luận, thực hành...)
        - "difficulty_level": Mức độ khó cụ thể
        - "learning_objectives": Mục tiêu học tập cần đánh giá
        - "query_string": Câu mô tả ngữ cảnh để tìm kiếm nội dung
        - "estimation_time": Thời gian dự kiến làm bài (phút)

        ### LƯU Ý QUAN TRỌNG:
        - Tổng số câu hỏi phải khớp chính xác với yêu cầu
        - Ưu tiên các mục có nhiều nội dung quan trọng
        - Cân bằng giữa lý thuyết và thực hành
        - Chỉ trả về JSON, không giải thích thêm
        """
        
        try:
            plan_str = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Xử lý JSON response
            plan_str = self._extract_json_from_response(plan_str)
            question_plan = json.loads(plan_str)
            
            logger.info(f"✅ Đã tạo kế hoạch với {len(question_plan)} task.")
            return question_plan
            
        except json.JSONDecodeError as e:
            logger.error(f"Lỗi parse JSON: {e}\nResponse: {plan_str}")
            return None
        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch: {e}")
            return None

    def _extract_json_from_response(self, response: str) -> str:
        """Trích xuất JSON từ response của LLM."""
        # Tìm JSON trong markdown blocks
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Tìm JSON không có markdown
        json_match = re.search(r'(\[.*?\])', response, re.DOTALL)
        if json_match:
            return json_match.group(1)
            
        return response.strip()

    def summarize_with_context(
        self, 
        text_to_summarize: str, 
        context_info: str = "",
        style: str = "paragraph", 
        length: str = "medium",
        focus_areas: List[str] = None
    ) -> str:
        """
        Tóm tắt nâng cao với ngữ cảnh và trọng tâm cụ thể.
        """
        if not text_to_summarize or not text_to_summarize.strip():
            logger.warning("Văn bản đầu vào để tóm tắt bị rỗng.")
            return "Không có nội dung để tóm tắt."

        logger.info(f"Bắt đầu tóm tắt nâng cao: {style} - {length}")

        # Mapping độ dài và style
        length_map = {
            "short": "1-2 câu ngắn gọn, chỉ ý chính",
            "medium": "3-5 câu cân bằng, bao quát các khía cạnh quan trọng",
            "long": "một đoạn văn chi tiết 150-200 từ, phân tích sâu"
        }
        
        style_map = {
            "paragraph": "đoạn văn xuôi mạch lạc và có logic",
            "bullet_points": "các gạch đầu dòng rõ ràng, mỗi điểm một ý chính",
            "structured": "cấu trúc có tiêu đề phụ và phân loại theo chủ đề"
        }

        length_instruction = length_map.get(length, length_map["medium"])
        style_instruction = style_map.get(style, style_map["paragraph"])
        
        # Xử lý focus areas
        focus_instruction = ""
        if focus_areas:
            focus_instruction = f"\n**Trọng tâm đặc biệt:** {', '.join(focus_areas)}"

        system_prompt = """
        Bạn là chuyên gia phân tích và tóm tắt văn bản học thuật với 15 năm kinh nghiệm. 
        Bạn có khả năng chắt lọc thông tin cốt lõi và trình bày một cách logic, khoa học.

        QUY TẮC NGHIÊM NGẶT:
        - LUÔN LUÔN trả lời bằng tiếng Việt
        - KHÔNG sử dụng tiếng Anh trừ thuật ngữ chuyên môn cần thiết
        - Giữ nguyên tính chính xác của thông tin gốc
        - Đảm bảo tính mạch lạc và logic trong trình bày
        """

        user_prompt = f"""
        ## NHIỆM VỤ TÓM TẮT CHUYÊN NGHIỆP

        ### THÔNG TIN NGỮ CẢNH:
        {context_info if context_info else "Tóm tắt nội dung chung"}

        ### YÊU CẦU CHI TIẾT:
        - **Độ dài:** {length_instruction}
        - **Định dạng:** {style_instruction}
        - **Trọng tâm:** Tập trung vào khái niệm chính, định nghĩa quan trọng và ứng dụng thực tế
        {focus_instruction}

        ### HƯỚNG DẪN THỰC HIỆN:
        1. Đọc kỹ và hiểu nội dung toàn diện
        2. Xác định các ý chính và mối liên hệ
        3. Chắt lọc thông tin quan trọng nhất
        4. Trình bày theo đúng yêu cầu về độ dài và định dạng
        5. Đảm bảo tính chính xác và khách quan

        ### NỘI DUNG CẦN TÓM TẮT:
        ```
        {text_to_summarize[:6000]}{"..." if len(text_to_summarize) > 6000 else ""}
        ```

        ### KẾT QUA TÓM TẮT:
        """

        try:
            summary = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Làm sạch và format output
            summary = self._clean_summary_output(summary)
            
            logger.info("✅ Tóm tắt nâng cao hoàn thành.")
            return summary
        except Exception as e:
            logger.error(f"Lỗi khi tóm tắt: {e}")
            return "Rất tiếc, đã có lỗi xảy ra trong quá trình tóm tắt."

    def _clean_summary_output(self, summary: str) -> str:
        """Làm sạch output tóm tắt."""
        # Loại bỏ các marker không cần thiết
        summary = re.sub(r'### KẾT QUA TÓM TẮT:?\s*', '', summary)
        summary = re.sub(r'```.*?```', '', summary, flags=re.DOTALL)
        
        return summary.strip()

    def generate_comprehensive_analysis(
        self, 
        document_text: str, 
        bookmarks: List[Dict] = None,
        analysis_depth: str = "deep"
    ) -> Dict[str, Any]:
        """
        Tạo phân tích toàn diện về tài liệu với chất lượng chuyên nghiệp.
        
        Args:
            document_text (str): Nội dung đầy đủ
            bookmarks (List[Dict]): Bookmark có sẵn
            analysis_depth (str): Mức độ phân tích (quick, standard, deep)
            
        Returns:
            Dict[str, Any]: Báo cáo phân tích toàn diện
        """
        logger.info("📋 Bắt đầu phân tích toàn diện chuyên nghiệp...")
        
        result = {
            "document_metadata": {
                "total_length": len(document_text),
                "estimated_pages": max(1, len(document_text) // 2000),
                "estimated_reading_time": max(5, len(document_text) // 1000),  # phút
                "complexity_level": self._assess_complexity(document_text),
                "language": "Vietnamese",
                "num_bookmarks": len(bookmarks) if bookmarks else 0
            },
            "hierarchical_toc": "",
            "chapter_summaries": {},
            "overall_summary": "",
            "key_concepts": [],
            "learning_outcomes": [],
            "difficulty_assessment": "",
            "recommended_study_time": "",
            "question_generation_readiness": False
        }
        
        try:
            # 1. Tạo mục lục phân tầng
            result["hierarchical_toc"] = self.generate_hierarchical_toc(document_text, bookmarks)
            
            # 2. Tóm tắt tổng quan
            result["overall_summary"] = self.summarize_with_context(
                text_to_summarize=document_text,
                context_info="Tóm tắt tổng quan toàn bộ tài liệu",
                style="structured",
                length="long"
            )
            
            # 3. Phân tích độ phức tạp và đề xuất
            result["difficulty_assessment"] = self._assess_difficulty_and_recommendations(document_text)
            
            # 4. Đánh giá khả năng sinh câu hỏi
            result["question_generation_readiness"] = bool(result["hierarchical_toc"] and len(document_text) > 1000)
            
            logger.info("✅ Hoàn thành phân tích toàn diện.")
            return result
            
        except Exception as e:
            logger.error(f"Lỗi trong phân tích toàn diện: {e}")
            result["error"] = str(e)
            return result

    def _assess_complexity(self, text: str) -> str:
        """Đánh giá độ phức tạp của văn bản."""
        # Phân tích đơn giản dựa trên các chỉ số
        avg_sentence_length = len(text.split()) / max(1, len(text.split('.')))
        
        if avg_sentence_length > 25:
            return "High"
        elif avg_sentence_length > 15:
            return "Medium"
        else:
            return "Low"

    def _assess_difficulty_and_recommendations(self, text: str) -> str:
        """Đánh giá độ khó và đưa ra khuyến nghị."""
        system_prompt = """
        Bạn là chuyên gia đánh giá độ khó tài liệu học thuật. 
        Hãy phân tích và đưa ra đánh giá khách quan về mức độ khó của tài liệu.

        QUY TẮC: LUÔN trả lời bằng tiếng Việt.
        """
        
        user_prompt = f"""
        Đánh giá độ khó của tài liệu này và đưa ra khuyến nghị học tập:

        ### YÊU CẦU:
        1. Mức độ khó: Cơ bản/Trung bình/Nâng cao
        2. Đối tượng phù hợp
        3. Thời gian học tập khuyến nghị
        4. Kiến thức tiên quyết cần có
        5. Phương pháp học tập hiệu quả

        ### NỘI DUNG MẪU:
        {text[:3000]}...

        Trả lời ngắn gọn, cụ thể và hữu ích.
        """
        
        try:
            return self.llm_provider.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except:
            return "Không thể đánh giá độ khó do lỗi hệ thống."

    # Compatibility methods để giữ tương thích với code cũ
    def summarize(self, text_to_summarize: str, context_info: str = "", style: str = "paragraph", length: str = "medium") -> str:
        """Method tương thích với interface cũ."""
        return self.summarize_with_context(text_to_summarize, context_info, style, length)
    
    def generate_table_of_contents(self, document_text: str) -> str:
        """Method tương thích với interface cũ."""
        return self.generate_hierarchical_toc(document_text)
    
    def create_question_plan(self, user_request: str) -> Optional[List[Dict]]:
        """Method tương thích với interface cũ."""
        return self.create_advanced_question_plan(user_request)