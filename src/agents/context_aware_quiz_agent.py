# src/agents/context_aware_quiz_agent.py
# Context-aware quiz generation agent - NO EXAMPLE LEAK

import logging
import json
import re
from typing import Dict, Any, List, Optional

from ..core.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

class ContextAwareQuizAgent:
    """
    Quiz generation agent that STRICTLY uses only the provided document content.
    Designed to prevent context leak and generate questions only from actual content.
    """
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.generated_questions = []
        logger.info("✅ ContextAwareQuizAgent initialized with strict content-only mode")

    def generate_content_based_questions(
        self, 
        document_content: str, 
        section_title: str,
        num_questions: int = 5,
        question_type: str = "multiple_choice",
        difficulty: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        Generate questions STRICTLY from document content only.
        
        Args:
            document_content (str): The actual document text to create questions from
            section_title (str): Title of the section for context
            num_questions (int): Number of questions to generate
            question_type (str): Type of questions (multiple_choice, true_false, essay)
            difficulty (str): Difficulty level (easy, medium, hard)
            
        Returns:
            List[Dict]: Questions generated from document content only
        """
        logger.info(f"🎯 Generating {num_questions} {question_type} questions from content: {section_title}")
        
        # Analyze content first to ensure we have enough material
        content_analysis = self._analyze_content_for_questions(document_content)
        
        if not content_analysis['suitable_for_questions']:
            logger.warning("⚠️ Document content may not be suitable for question generation")
            return []
        
        if question_type == "multiple_choice":
            return self._generate_multiple_choice_from_content(
                document_content, section_title, num_questions, difficulty
            )
        elif question_type == "true_false":
            return self._generate_true_false_from_content(
                document_content, section_title, num_questions, difficulty
            )
        elif question_type == "essay":
            return self._generate_essay_from_content(
                document_content, section_title, num_questions, difficulty
            )
        else:
            logger.error(f"Unsupported question type: {question_type}")
            return []

    def _analyze_content_for_questions(self, content: str) -> Dict[str, Any]:
        """Analyze if content is suitable for question generation."""
        analysis = {
            'suitable_for_questions': False,
            'content_length': len(content),
            'has_concepts': False,
            'has_facts': False,
            'has_examples': False,
            'key_topics': []
        }
        
        # Check content length
        if len(content) < 200:
            return analysis
        
        # Look for conceptual content
        concept_indicators = [
            'định nghĩa', 'khái niệm', 'là gì', 'tức là', 'có nghĩa', 'được hiểu',
            'definition', 'concept', 'means', 'refers to', 'defined as'
        ]
        analysis['has_concepts'] = any(indicator in content.lower() for indicator in concept_indicators)
        
        # Look for factual content
        fact_indicators = [
            'năm', 'ngày', 'tháng', 'được tạo', 'phát minh', 'phát triển',
            'số lượng', 'phần trăm', 'bao gồm', 'consists of', 'includes'
        ]
        analysis['has_facts'] = any(indicator in content.lower() for indicator in fact_indicators)
        
        # Look for examples
        example_indicators = [
            'ví dụ', 'chẳng hạn', 'như là', 'for example', 'such as', 'instance'
        ]
        analysis['has_examples'] = any(indicator in content.lower() for indicator in example_indicators)
        
        # Extract key topics (simplified)
        sentences = content.split('.')
        key_topics = []
        for sentence in sentences[:10]:  # Analyze first 10 sentences
            if len(sentence.strip()) > 50:  # Meaningful sentences
                key_topics.append(sentence.strip()[:100])  # First 100 chars
        
        analysis['key_topics'] = key_topics
        analysis['suitable_for_questions'] = (
            analysis['has_concepts'] or 
            analysis['has_facts'] or 
            len(analysis['key_topics']) >= 3
        )
        
        return analysis

    def _generate_multiple_choice_from_content(
        self, 
        content: str, 
        section_title: str, 
        num_questions: int, 
        difficulty: str
    ) -> List[Dict[str, Any]]:
        """Generate multiple choice questions from content only."""
        
        # Extract key information from content
        content_summary = self._extract_key_information(content)
        
        system_prompt = """
        Bạn là giáo viên chuyên nghiệp tạo câu hỏi kiểm tra từ tài liệu học tập.
        
        QUY TẮC NGHIÊM NGẶT:
        - CHỈ tạo câu hỏi dựa trên nội dung tài liệu được cung cấp
        - KHÔNG sử dụng kiến thức bên ngoài tài liệu
        - KHÔNG tạo ví dụ từ kiến thức riêng của bạn
        - LUÔN trả lời bằng tiếng Việt
        - Đảm bảo câu hỏi có thể trả lời được từ nội dung đã cho
        """
        
        user_prompt = f"""
        ## NHIỆM VỤ: Tạo câu hỏi trắc nghiệm từ nội dung tài liệu

        ### THÔNG TIN MẠCH:
        - Chủ đề: {section_title}
        - Số câu hỏi: {num_questions}
        - Mức độ: {difficulty}

        ### NỘI DUNG TÀI LIỆU (CHỈ SỬ DỤNG THÔNG TIN NÀY):
        ```
        {content[:6000]}
        ```

        ### YÊU CẦU:
        1. CHỈ tạo câu hỏi từ nội dung trên
        2. Đảm bảo đáp án có trong nội dung
        3. Tạo các phương án nhiễu hợp lý nhưng sai theo nội dung
        4. Giải thích dựa trên đoạn văn cụ thể trong tài liệu

        ### ĐỊNH DẠNG JSON:
        ```json
        [
            {{
                "question_id": 1,
                "question_text": "Câu hỏi dựa trên nội dung tài liệu?",
                "options": {{
                    "A": "Đáp án A từ tài liệu",
                    "B": "Đáp án B từ tài liệu", 
                    "C": "Đáp án C từ tài liệu",
                    "D": "Đáp án D từ tài liệu"
                }},
                "correct_answer": "A",
                "explanation": {{
                    "correct_reason": "Đúng vì trong tài liệu có đề cập: [trích dẫn cụ thể]",
                    "content_reference": "Đoạn văn hoặc câu cụ thể trong tài liệu làm căn cứ"
                }},
                "source_content": "Trích dẫn đoạn văn từ tài liệu làm cơ sở cho câu hỏi"
            }}
        ]
        ```

        ### LƯU Ý QUAN TRỌNG:
        - Mỗi câu hỏi phải có trường "source_content" trích dẫn từ tài liệu
        - Giải thích phải tham chiếu cụ thể đến nội dung
        - KHÔNG được sử dụng thông tin ngoài tài liệu

        Tạo {num_questions} câu hỏi chất lượng cao:
        """
        
        try:
            response = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Extract and parse JSON
            questions_json = self._extract_json_from_response(response)
            questions = json.loads(questions_json)
            
            # Validate that questions are content-based
            validated_questions = self._validate_content_based_questions(questions, content)
            
            logger.info(f"✅ Generated {len(validated_questions)} content-based questions")
            return validated_questions
            
        except Exception as e:
            logger.error(f"❌ Error generating questions: {e}")
            return []

    def _generate_true_false_from_content(
        self, 
        content: str, 
        section_title: str, 
        num_questions: int, 
        difficulty: str
    ) -> List[Dict[str, Any]]:
        """Generate true/false questions from content only."""
        
        system_prompt = """
        Bạn là giáo viên tạo câu hỏi đúng/sai từ tài liệu học tập.
        
        QUY TẮC:
        - CHỈ tạo câu hỏi từ nội dung tài liệu
        - Câu hỏi phải rõ ràng là đúng hoặc sai theo tài liệu
        - Trả lời bằng tiếng Việt
        """
        
        user_prompt = f"""
        ## NHIỆM VỤ: Tạo câu hỏi đúng/sai từ nội dung

        ### NỘI DUNG TÀI LIỆU:
        ```
        {content[:5000]}
        ```

        ### YÊU CẦU:
        - Tạo {num_questions} câu hỏi đúng/sai
        - Dựa trên thông tin chính xác trong tài liệu
        - Mix câu đúng và câu sai

        ### ĐỊNH DẠNG JSON:
        ```json
        [
            {{
                "question_id": 1,
                "question_text": "Khẳng định về nội dung trong tài liệu",
                "correct_answer": true,
                "explanation": "Giải thích dựa trên nội dung tài liệu",
                "source_content": "Trích dẫn từ tài liệu"
            }}
        ]
        ```

        Tạo câu hỏi:
        """
        
        try:
            response = self.llm_provider.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            questions_json = self._extract_json_from_response(response)
            questions = json.loads(questions_json)
            
            # Add question type
            for q in questions:
                q['question_type'] = 'true_false'
                q['topic'] = section_title
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating true/false questions: {e}")
            return []

    def _generate_essay_from_content(
        self, 
        content: str, 
        section_title: str, 
        num_questions: int, 
        difficulty: str
    ) -> List[Dict[str, Any]]:
        """Generate essay questions from content only."""
        
        system_prompt = """
        Bạn là giáo viên tạo câu hỏi tự luận từ tài liệu học tập.
        
        QUY TẮC:
        - Câu hỏi phải có thể trả lời được từ nội dung tài liệu
        - Không yêu cầu kiến thức bên ngoài tài liệu
        - Trả lời bằng tiếng Việt
        """
        
        user_prompt = f"""
        ## NHIỆM VỤ: Tạo câu hỏi tự luận từ nội dung

        ### NỘI DUNG TÀI LIỆU:
        ```
        {content[:5000]}
        ```

        ### YÊU CẦU:
        - Tạo {num_questions} câu hỏi tự luận
        - Học sinh có thể trả lời từ nội dung đã cho
        - Bao gồm hướng dẫn chấm điểm

        ### ĐỊNH DẠNG JSON:
        ```json
        [
            {{
                "question_id": 1,
                "question_text": "Câu hỏi tự luận về nội dung",
                "expected_length": "150-200 từ",
                "scoring_criteria": {{
                    "key_points": ["Điểm 1 từ tài liệu", "Điểm 2 từ tài liệu"],
                    "examples_required": true
                }},
                "sample_answer": "Câu trả lời mẫu dựa trên tài liệu",
                "source_content": "Phần tài liệu liên quan"
            }}
        ]
        ```

        Tạo câu hỏi:
        """
        
        try:
            response = self.llm_provider.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            questions_json = self._extract_json_from_response(response)
            questions = json.loads(questions_json)
            
            # Add question type
            for q in questions:
                q['question_type'] = 'essay'
                q['topic'] = section_title
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating essay questions: {e}")
            return []

    def _extract_key_information(self, content: str) -> Dict[str, List[str]]:
        """Extract key information categories from content."""
        info = {
            'definitions': [],
            'facts': [],
            'processes': [],
            'examples': []
        }
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
                
            # Look for definitions
            if any(word in sentence.lower() for word in ['định nghĩa', 'là', 'tức là', 'có nghĩa']):
                info['definitions'].append(sentence)
            
            # Look for facts
            elif any(word in sentence.lower() for word in ['năm', 'được tạo', 'phát triển', 'bao gồm']):
                info['facts'].append(sentence)
            
            # Look for processes
            elif any(word in sentence.lower() for word in ['bước', 'quy trình', 'cách', 'phương pháp']):
                info['processes'].append(sentence)
            
            # Look for examples
            elif any(word in sentence.lower() for word in ['ví dụ', 'chẳng hạn', 'như']):
                info['examples'].append(sentence)
        
        # Limit each category
        for key in info:
            info[key] = info[key][:5]  # Max 5 items per category
        
        return info

    def _validate_content_based_questions(self, questions: List[Dict], content: str) -> List[Dict]:
        """Validate that questions are truly based on content."""
        validated = []
        
        for question in questions:
            # Check if question has source_content field
            if 'source_content' not in question:
                logger.warning(f"Question {question.get('question_id')} missing source_content")
                continue
            
            # Check if source content exists in the document
            source_content = question['source_content']
            if source_content and any(part.strip() in content for part in source_content.split() if len(part) > 5):
                validated.append(question)
            else:
                logger.warning(f"Question {question.get('question_id')} source not found in content")
        
        return validated

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response."""
        # Try to find JSON in code blocks
        json_patterns = [
            r'```json\s*(\[.*?\])\s*```',
            r'```\s*(\[.*?\])\s*```', 
            r'(\[.*?\])'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1)
        
        return response.strip()

    def generate_quiz_from_pdf_structure(
        self, 
        pdf_structure: Dict[str, Any], 
        quiz_requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate quiz from PDF structure analysis.
        
        Args:
            pdf_structure: Output from ProfessionalPDFProcessor
            quiz_requirements: Requirements for quiz generation
            
        Returns:
            List of questions based on PDF content
        """
        logger.info("📝 Generating quiz from PDF structure...")
        
        all_questions = []
        content = pdf_structure.get('text_content', '')
        bookmarks = pdf_structure.get('bookmarks', [])
        
        # Generate questions for each major section
        for bookmark in bookmarks[:5]:  # Limit to first 5 major sections
            if bookmark['level'] <= 2:  # Only major sections
                section_content = self._extract_section_content_from_full_text(
                    content, bookmark['title']
                )
                
                if len(section_content) > 200:  # Ensure sufficient content
                    section_questions = self.generate_content_based_questions(
                        document_content=section_content,
                        section_title=bookmark['title'],
                        num_questions=quiz_requirements.get('questions_per_section', 2),
                        question_type=quiz_requirements.get('question_type', 'multiple_choice'),
                        difficulty=quiz_requirements.get('difficulty', 'medium')
                    )
                    all_questions.extend(section_questions)
        
        logger.info(f"✅ Generated {len(all_questions)} questions from PDF structure")
        return all_questions

    def _extract_section_content_from_full_text(self, full_text: str, section_title: str) -> str:
        """Extract content for a specific section from full text."""
        lines = full_text.split('\n')
        section_lines = []
        capturing = False
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                if capturing:
                    section_lines.append('')
                continue
            
            # Check if this line contains the section title
            if section_title.lower() in line_clean.lower():
                capturing = True
                section_lines.append(line_clean)
                continue
            
            if capturing:
                # Check if we hit another major section
                if any(keyword in line_clean.lower() for keyword in ['chapter', 'chương', 'bài']):
                    # If this looks like a new section and we have enough content, stop
                    if len('\n'.join(section_lines)) > 500:
                        break
                
                section_lines.append(line_clean)
                
                # Limit section size to avoid token overflow
                if len(section_lines) > 100:
                    break
        
        return '\n'.join(section_lines)


# Compatibility function for existing code
def create_context_aware_quiz_agent(llm_provider: LLMProvider) -> ContextAwareQuizAgent:
    """Factory function to create ContextAwareQuizAgent."""
    return ContextAwareQuizAgent(llm_provider)