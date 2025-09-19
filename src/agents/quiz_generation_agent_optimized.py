# src/agents/quiz_generation_agent_optimized.py
# Agent tối ưu cho việc sinh câu hỏi với chất lượng chuyên nghiệp

import logging
import json
import re
from typing import Dict, Any, List, Optional

from ..core.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

class QuizGenerationAgentOptimized:
    """
    Agent tối ưu cho việc sinh câu hỏi kiểm tra với chất lượng chuyên nghiệp.
    Được thiết kế để tạo câu hỏi chi tiết với giải thích đầy đủ.
    """
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.generated_questions = []
        logger.info("✅ QuizGenerationAgentOptimized đã được khởi tạo.")

    def generate_professional_questions(
        self, 
        content: str, 
        task_details: Dict[str, Any], 
        question_format: str = "multiple_choice"
    ) -> List[Dict[str, Any]]:
        """
        Sinh câu hỏi chuyên nghiệp với format chi tiết và giải thích đầy đủ.
        
        Args:
            content (str): Nội dung tài liệu để tạo câu hỏi
            task_details (Dict): Chi tiết về task (từ kế hoạch)
            question_format (str): Loại câu hỏi (multiple_choice, essay, practical)
            
        Returns:
            List[Dict]: Danh sách câu hỏi với format chuyên nghiệp
        """
        logger.info(f"🎯 Sinh câu hỏi chuyên nghiệp cho: {task_details.get('section_title', 'Unknown')}")
        
        section_title = task_details.get('section_title', 'Chương học')
        num_questions = task_details.get('num_questions', 1)
        difficulty_level = task_details.get('difficulty_level', 'medium')
        learning_objectives = task_details.get('learning_objectives', [])
        question_types = task_details.get('question_types', ['trắc nghiệm'])
        
        # Map difficulty sang tiếng Việt
        difficulty_map = {
            "easy": "Cơ bản - Nhận biết và hiểu",
            "medium": "Trung bình - Vận dụng và phân tích", 
            "hard": "Nâng cao - Đánh giá và sáng tạo"
        }
        difficulty_desc = difficulty_map.get(difficulty_level, difficulty_map["medium"])
        
        system_prompt = """
        Bạn là giáo viên đại học chuyên nghiệp với 20 năm kinh nghiệm trong việc thiết kế đề thi chất lượng cao.
        Bạn có khả năng tạo ra những câu hỏi sâu sắc, khoa học và phù hợp với mục tiêu đánh giá.

        NGUYÊN TẮC QUAN TRỌNG NHẤT:
        - CHỈ tạo câu hỏi dựa trên NỘI DUNG HỌC LIỆU được cung cấp
        - KHÔNG được sử dụng kiến thức bên ngoài tài liệu
        - KHÔNG được tạo câu hỏi về chủ đề không xuất hiện trong nội dung
        - Phải trích dẫn chính xác từ tài liệu gốc

        QUY TẮC NGHIÊM NGẶT:
        - LUÔN LUÔN trả lời bằng tiếng Việt
        - KHÔNG sử dụng tiếng Anh trong câu hỏi trừ thuật ngữ chuyên môn có trong tài liệu
        - Mỗi câu hỏi phải có giải thích chi tiết dựa trên nội dung tài liệu
        - Đảm bảo tính chính xác về mặt học thuật theo đúng tài liệu
        """
        
        user_prompt = f"""
        ## NHIỆM VỤ: Thiết kế câu hỏi kiểm tra chuyên nghiệp

        ### THÔNG TIN CHƯƠNG MỤC:
        **Tiêu đề:** {section_title}
        **Số lượng câu hỏi:** {num_questions}
        **Mức độ khó:** {difficulty_desc}
        **Loại câu hỏi:** {', '.join(question_types)}
        **Mục tiêu đánh giá:** {', '.join(learning_objectives) if learning_objectives else 'Kiến thức tổng quát'}

        ### YÊU CẦU CHẤT LƯỢNG CAO:
        1. **Câu hỏi rõ ràng:** Không gây nhầm lẫn, đi thẳng vào vấn đề
        2. **Đáp án chính xác:** Đáp án đúng phải hoàn toàn chính xác về mặt khoa học
        3. **Phương án nhiễu hợp lý:** Các lựa chọn sai phải hợp lý, không quá dễ loại trừ
        4. **Giải thích chi tiết:** Mỗi câu phải có giải thích tại sao đáp án này đúng và các đáp án khác sai

        ### ĐỊNH DẠNG ĐẦU RA CHUẨN:
        ```json
        [
            {{
                "question_id": 1,
                "question_type": "multiple_choice",
                "question_text": "Câu hỏi chi tiết và rõ ràng?",
                "options": {{
                    "A": "Phương án A - mô tả cụ thể",
                    "B": "Phương án B - mô tả cụ thể", 
                    "C": "Phương án C - mô tả cụ thể",
                    "D": "Phương án D - mô tả cụ thể"
                }},
                "correct_answer": "A",
                "explanation": {{
                    "correct_reason": "Giải thích chi tiết tại sao đáp án A đúng, bao gồm lý thuyết nền tảng và ví dụ minh họa.",
                    "incorrect_reasons": {{
                        "B": "Lý do tại sao phương án B không chính xác.",
                        "C": "Lý do tại sao phương án C không chính xác.",
                        "D": "Lý do tại sao phương án D không chính xác."
                    }}
                }},
                "difficulty": "{difficulty_level}",
                "topic": "{section_title}",
                "cognitive_level": "Nhận biết/Hiểu/Vận dụng/Phân tích/Đánh giá/Sáng tạo",
                "estimated_time": 2,
                "learning_objective": "Mục tiêu học tập cụ thể được đánh giá"
            }}
        ]
        ```

        ### HƯỚNG DẪN TẠO CÂU HỎI:
        - Câu hỏi PHẢI dựa HOÀN TOÀN trên nội dung học liệu được cung cấp
        - KHÔNG được sử dụng kiến thức bên ngoài tài liệu
        - KHÔNG được tạo câu hỏi về chủ đề không có trong nội dung
        - Tập trung vào các khái niệm, định nghĩa và thông tin cụ thể trong tài liệu
        - Đảm bảo tính chính xác 100% theo nội dung gốc

        ### NỘI DUNG HỌC LIỆU:
        ```
        {content[:8000]}{"..." if len(content) > 8000 else ""}
        ```

        ### YÊU CẦU THỰC HIỆN:
        1. ĐỌC KỸ toàn bộ nội dung học liệu được cung cấp
        2. CHỈ tạo câu hỏi về những gì có trong nội dung này
        3. Trích xuất các khái niệm, định nghĩa, nguyên lý từ tài liệu
        4. Tạo câu hỏi kiểm tra hiểu biết về nội dung đã đọc
        5. Giải thích dựa trên chính xác những gì có trong tài liệu
        6. TUYỆT ĐỐI KHÔNG tạo câu hỏi về kiến thức ngoài tài liệu

        ### NHẮC NHỞ QUAN TRỌNG:
        - Nếu tài liệu nói về "Prompt Engineering", hãy tạo câu hỏi về Prompt Engineering
        - Nếu tài liệu nói về "Machine Learning", hãy tạo câu hỏi về Machine Learning  
        - KHÔNG tạo câu hỏi về chủ đề khác ngoài những gì có trong tài liệu

        ### KẾT QUẢ (CHỈ TRẢ VỀ JSON):
        """
        
        try:
            response = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Trích xuất và parse JSON
            questions_json = self._extract_json_from_response(response)
            questions = json.loads(questions_json)
            
            # Validate và clean up questions
            validated_questions = self._validate_and_clean_questions(questions, section_title)
            
            # Lưu vào cache
            self.generated_questions.extend(validated_questions)
            
            logger.info(f"✅ Đã sinh {len(validated_questions)} câu hỏi chuyên nghiệp.")
            return validated_questions
            
        except json.JSONDecodeError as e:
            logger.error(f"Lỗi parse JSON câu hỏi: {e}")
            return self._create_fallback_question(section_title, content)
        except Exception as e:
            logger.error(f"Lỗi khi sinh câu hỏi: {e}")
            return self._create_fallback_question(section_title, content)

    def _extract_json_from_response(self, response: str) -> str:
        """Trích xuất JSON từ response của LLM."""
        # Tìm JSON trong markdown code blocks
        json_patterns = [
            r'```json\s*(\[.*?\])\s*```',
            r'```\s*(\[.*?\])\s*```', 
            r'(\[.*?\])'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1)
        
        # Nếu không tìm thấy, thử toàn bộ response
        return response.strip()

    def _validate_and_clean_questions(self, questions: List[Dict], section_title: str) -> List[Dict]:
        """Validate và làm sạch danh sách câu hỏi."""
        validated = []
        
        for i, q in enumerate(questions):
            try:
                # Đảm bảo có đủ các field bắt buộc
                required_fields = ['question_text', 'options', 'correct_answer', 'explanation']
                if not all(field in q for field in required_fields):
                    logger.warning(f"Câu hỏi {i+1} thiếu field bắt buộc, bỏ qua.")
                    continue
                
                # Validate options
                if not isinstance(q['options'], dict) or len(q['options']) < 2:
                    logger.warning(f"Câu hỏi {i+1} có options không hợp lệ, bỏ qua.")
                    continue
                
                # Đảm bảo correct_answer hợp lệ
                if q['correct_answer'] not in q['options']:
                    logger.warning(f"Câu hỏi {i+1} có correct_answer không khớp với options, bỏ qua.")
                    continue
                
                # Thêm các field mặc định nếu thiếu
                q.setdefault('question_id', i + 1)
                q.setdefault('question_type', 'multiple_choice')
                q.setdefault('difficulty', 'medium')
                q.setdefault('topic', section_title)
                q.setdefault('cognitive_level', 'Hiểu')
                q.setdefault('estimated_time', 2)
                q.setdefault('learning_objective', f'Kiến thức về {section_title}')
                
                validated.append(q)
                
            except Exception as e:
                logger.error(f"Lỗi khi validate câu hỏi {i+1}: {e}")
                continue
        
        return validated

    def _create_fallback_question(self, section_title: str, content: str) -> List[Dict]:
        """Tạo câu hỏi dự phòng khi có lỗi."""
        return [
            {
                "question_id": 1,
                "question_type": "multiple_choice",
                "question_text": f"Theo nội dung đã học về {section_title}, khái niệm nào sau đây là quan trọng nhất?",
                "options": {
                    "A": "Khái niệm cơ bản trong lý thuyết",
                    "B": "Ứng dụng thực tế trong công việc", 
                    "C": "Phương pháp tiếp cận vấn đề",
                    "D": "Tất cả các khái niệm trên"
                },
                "correct_answer": "D",
                "explanation": {
                    "correct_reason": f"Trong {section_title}, tất cả các khía cạnh đều quan trọng và bổ sung cho nhau.",
                    "incorrect_reasons": {
                        "A": "Chỉ tập trung vào lý thuyết là chưa đủ.",
                        "B": "Chỉ tập trung vào ứng dụng mà thiếu nền tảng lý thuyết là không đầy đủ.",
                        "C": "Phương pháp quan trọng nhưng cần kết hợp với kiến thức cơ bản."
                    }
                },
                "difficulty": "medium",
                "topic": section_title,
                "cognitive_level": "Hiểu",
                "estimated_time": 2,
                "learning_objective": f"Hiểu tổng quan về {section_title}"
            }
        ]

    def generate_essay_questions(
        self, 
        content: str, 
        task_details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Sinh câu hỏi tự luận chuyên nghiệp với tiêu chí chấm điểm chi tiết.
        """
        logger.info("📝 Sinh câu hỏi tự luận chuyên nghiệp...")
        
        section_title = task_details.get('section_title', 'Chương học')
        num_questions = task_details.get('num_questions', 1)
        difficulty_level = task_details.get('difficulty_level', 'medium')
        
        system_prompt = """
        Bạn là giáo viên đại học chuyên nghiệp thiết kế câu hỏi tự luận chất lượng cao.
        Bạn có khả năng tạo ra những câu hỏi sâu sắc và tiêu chí đánh giá khoa học.

        NGUYÊN TẮC QUAN TRỌNG NHẤT:
        - CHỈ tạo câu hỏi dựa trên NỘI DUNG HỌC LIỆU được cung cấp
        - KHÔNG được sử dụng kiến thức bên ngoài tài liệu
        - Câu hỏi phải kiểm tra hiểu biết về nội dung đã học

        QUY TẮC: LUÔN trả lời bằng tiếng Việt.
        """
        
        user_prompt = f"""
        ## NHIỆM VỤ: Thiết kế câu hỏi tự luận chuyên nghiệp

        ### THÔNG TIN:
        - Chương mục: {section_title}
        - Số câu hỏi: {num_questions}
        - Mức độ: {difficulty_level}

        ### ĐỊNH DẠNG ĐẦU RA:
        ```json
        [
            {{
                "question_id": 1,
                "question_type": "essay",
                "question_text": "Câu hỏi tự luận chi tiết...",
                "expected_length": "200-300 từ",
                "scoring_criteria": {{
                    "content_accuracy": "Tiêu chí về độ chính xác nội dung (30%)",
                    "logical_structure": "Tiêu chí về cấu trúc logic (25%)",
                    "examples_application": "Tiêu chí về ví dụ và ứng dụng (25%)",
                    "language_presentation": "Tiêu chí về ngôn ngữ và trình bày (20%)"
                }},
                "sample_answer": "Câu trả lời mẫu chi tiết...",
                "common_mistakes": ["Lỗi thường gặp 1", "Lỗi thường gặp 2"],
                "difficulty": "{difficulty_level}",
                "topic": "{section_title}",
                "estimated_time": 15
            }}
        ]
        ```

        ### NỘI DUNG:
        {content[:6000]}...

        Tạo câu hỏi tự luận sâu sắc với tiêu chí chấm điểm chi tiết.
        """
        
        try:
            response = self.llm_provider.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            questions_json = self._extract_json_from_response(response)
            questions = json.loads(questions_json)
            
            # Validate essay questions
            for q in questions:
                q.setdefault('question_type', 'essay')
                q.setdefault('estimated_time', 15)
            
            logger.info(f"✅ Đã sinh {len(questions)} câu hỏi tự luận.")
            return questions
            
        except Exception as e:
            logger.error(f"Lỗi khi sinh câu hỏi tự luận: {e}")
            return []

    def format_quiz_for_export(self, questions: List[Dict], quiz_metadata: Dict = None) -> Dict[str, Any]:
        """
        Format câu hỏi để xuất file hoặc hiển thị.
        
        Args:
            questions (List[Dict]): Danh sách câu hỏi
            quiz_metadata (Dict): Thông tin meta về bộ đề
            
        Returns:
            Dict: Bộ đề được format hoàn chỉnh
        """
        logger.info("📄 Format bộ đề để xuất file...")
        
        if not quiz_metadata:
            quiz_metadata = {
                "title": "Bộ Câu Hỏi Kiểm Tra",
                "description": "Được tạo bởi AI Assistant",
                "total_questions": len(questions),
                "estimated_time": sum(q.get('estimated_time', 2) for q in questions),
                "difficulty_distribution": self._calculate_difficulty_distribution(questions)
            }
        
        formatted_quiz = {
            "quiz_metadata": quiz_metadata,
            "instructions": {
                "general": "Đọc kỹ câu hỏi trước khi trả lời. Chọn đáp án đúng nhất.",
                "time_limit": f"{quiz_metadata.get('estimated_time', 30)} phút",
                "scoring": "Mỗi câu đúng được 1 điểm, câu sai không bị trừ điểm."
            },
            "questions": questions,
            "answer_key": self._generate_answer_key(questions),
            "statistics": {
                "total_questions": len(questions),
                "question_types": self._count_question_types(questions),
                "topics_covered": list(set(q.get('topic', 'Unknown') for q in questions)),
                "cognitive_levels": self._count_cognitive_levels(questions)
            }
        }
        
        return formatted_quiz

    def _calculate_difficulty_distribution(self, questions: List[Dict]) -> Dict[str, int]:
        """Tính phân bố độ khó của câu hỏi."""
        distribution = {"easy": 0, "medium": 0, "hard": 0}
        for q in questions:
            difficulty = q.get('difficulty', 'medium')
            if difficulty in distribution:
                distribution[difficulty] += 1
        return distribution

    def _count_question_types(self, questions: List[Dict]) -> Dict[str, int]:
        """Đếm số lượng từng loại câu hỏi."""
        types = {}
        for q in questions:
            q_type = q.get('question_type', 'multiple_choice')
            types[q_type] = types.get(q_type, 0) + 1
        return types

    def _count_cognitive_levels(self, questions: List[Dict]) -> Dict[str, int]:
        """Đếm số lượng câu hỏi theo cấp độ nhận thức."""
        levels = {}
        for q in questions:
            level = q.get('cognitive_level', 'Hiểu')
            levels[level] = levels.get(level, 0) + 1
        return levels

    def _generate_answer_key(self, questions: List[Dict]) -> List[Dict]:
        """Tạo đáp án cho bộ đề."""
        answer_key = []
        for q in questions:
            if q.get('question_type') == 'multiple_choice':
                answer_key.append({
                    "question_id": q.get('question_id'),
                    "correct_answer": q.get('correct_answer'),
                    "explanation": q.get('explanation', {}).get('correct_reason', 'Không có giải thích')
                })
            elif q.get('question_type') == 'essay':
                answer_key.append({
                    "question_id": q.get('question_id'),
                    "sample_answer": q.get('sample_answer', 'Không có câu trả lời mẫu'),
                    "scoring_criteria": q.get('scoring_criteria', {})
                })
        return answer_key

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê về quá trình sinh câu hỏi."""
        if not self.generated_questions:
            return {"message": "Chưa có câu hỏi nào được sinh"}
        
        return {
            "total_generated": len(self.generated_questions),
            "types": self._count_question_types(self.generated_questions),
            "difficulties": self._calculate_difficulty_distribution(self.generated_questions),
            "topics": list(set(q.get('topic', 'Unknown') for q in self.generated_questions)),
            "average_time": sum(q.get('estimated_time', 2) for q in self.generated_questions) / len(self.generated_questions)
        }

    # Compatibility methods để tương thích với code cũ
    def generate_quiz(self, document_id: int, user_request: str) -> str:
        """
        Method tương thích với interface cũ để không phá vỡ existing code.
        """
        logger.info(f"🔄 Compatibility mode: generate_quiz for document {document_id}")
        
        try:
            # Parse user request để lấy thông tin
            num_questions = self._extract_number_from_request(user_request)
            difficulty = self._extract_difficulty_from_request(user_request)
            
            # Tạo sample content để demo
            sample_content = """
            Nội dung tài liệu học thuật với các khái niệm quan trọng.
            Bao gồm lý thuyết cơ bản, ứng dụng thực tế và phương pháp tiếp cận.
            """
            
            # Tạo task details
            task_details = {
                'section_title': 'Kiến thức tổng quát',
                'num_questions': min(num_questions, 5),  # Giới hạn 5 câu để demo
                'question_types': ['trắc nghiệm'],
                'difficulty_level': difficulty,
                'learning_objectives': ['Kiểm tra hiểu biết cơ bản'],
                'estimation_time': num_questions * 2
            }
            
            # Sinh câu hỏi
            questions = self.generate_professional_questions(sample_content, task_details)
            
            if not questions:
                return "❌ Không thể tạo câu hỏi. Vui lòng thử lại."
            
            # Format output
            return self._format_quiz_output(questions, user_request)
            
        except Exception as e:
            logger.error(f"Lỗi trong compatibility mode: {e}")
            return f"❌ Lỗi khi tạo quiz: {str(e)}"

    def _extract_number_from_request(self, request: str) -> int:
        """Trích xuất số câu hỏi từ user request."""
        import re
        match = re.search(r'(\d+)\s*câu', request.lower())
        return int(match.group(1)) if match else 3

    def _extract_difficulty_from_request(self, request: str) -> str:
        """Trích xuất độ khó từ user request."""
        request_lower = request.lower()
        if 'dễ' in request_lower or 'easy' in request_lower:
            return 'easy'
        elif 'khó' in request_lower or 'hard' in request_lower:
            return 'hard'
        else:
            return 'medium'

    def _format_quiz_output(self, questions: List[Dict], user_request: str) -> str:
        """Format output cho compatibility mode."""
        output = f"🎉 **BỘ ĐỀ KIỂM TRA ĐÃ TẠO XONG** 🎉\n"
        output += f"📋 **Yêu cầu:** {user_request}\n"
        output += f"📊 **Số câu hỏi:** {len(questions)}\n\n"
        output += "="*60 + "\n\n"
        
        for i, question in enumerate(questions):
            output += f"**CÂU HỎI {i+1}:** {question.get('question_text', 'Không có câu hỏi')}\n\n"
            
            # Hiển thị options
            for key, value in question.get('options', {}).items():
                marker = "✓" if key == question.get('correct_answer') else " "
                output += f"   {key}) {value} {marker}\n"
            
            output += f"\n**💡 Giải thích:** {question.get('explanation', {}).get('correct_reason', 'Không có giải thích')}\n"
            output += f"**⏱️ Thời gian:** {question.get('estimated_time', 2)} phút\n"
            output += f"**📈 Mức độ:** {question.get('difficulty', 'medium')}\n"
            output += "\n" + "-"*50 + "\n\n"
        
        return output