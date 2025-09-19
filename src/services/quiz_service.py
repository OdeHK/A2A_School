# src/services/quiz_service.py
# Quiz service using context-aware quiz generation

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class QuizService:
    """
    Quiz service that uses ContextAwareQuizAgent to generate questions 
    strictly from document content without external knowledge leakage.
    """
    
    def __init__(self, config, db_manager, vector_store, quiz_agent):
        """
        Initialize QuizService with context-aware quiz agent.
        
        Args:
            config: App configuration
            db_manager: Database manager
            vector_store: Vector store for content retrieval
            quiz_agent: ContextAwareQuizAgent instance
        """
        self.config = config
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.quiz_agent = quiz_agent
        logger.info('✅ QuizService (with ContextAwareQuizAgent) initialized.')

    def generate_quiz(self, document_text: str, num_questions: int, difficulty: str, scope: str) -> str:
        """
        Generate quiz using context-aware agent that strictly uses document content.
        
        Args:
            document_text: The actual document content
            num_questions: Number of questions to generate
            difficulty: Difficulty level (easy, medium, hard)
            scope: Scope or topic of the quiz
            
        Returns:
            str: Generated quiz in formatted text
        """
        try:
            logger.info(f"🎯 Generating content-aware quiz: {num_questions} questions, difficulty: {difficulty}, scope: {scope}")
            
            # Use context-aware quiz agent to generate questions from document content only
            questions = self.quiz_agent.generate_content_based_questions(
                document_content=document_text,
                section_title=scope,
                num_questions=min(num_questions, 10),  # Limit to max 10 questions
                question_type="multiple_choice",
                difficulty=difficulty
            )
            
            if not questions:
                logger.warning("No questions generated from document content")
                return "⚠️ Không thể tạo câu hỏi từ nội dung tài liệu. Vui lòng kiểm tra lại nội dung."
            
            # Format questions for display
            formatted_quiz = self._format_quiz_for_display(questions, scope, difficulty)
            
            logger.info(f"✅ Successfully generated {len(questions)} content-based questions")
            return formatted_quiz
            
        except Exception as e:
            logger.error(f"❌ Error generating quiz: {e}")
            return f"Lỗi khi tạo quiz: {str(e)}"

    def _format_quiz_for_display(self, questions: List[Dict[str, Any]], scope: str, difficulty: str) -> str:
        """Format questions for user-friendly display."""
        if not questions:
            return "Không có câu hỏi nào được tạo."
        
        formatted_parts = []
        
        # Header
        formatted_parts.append(f"# 📝 BỘ CÂU HỎI KIỂM TRA")
        formatted_parts.append(f"**Chủ đề:** {scope}")
        formatted_parts.append(f"**Mức độ:** {difficulty.title()}")
        formatted_parts.append(f"**Số câu hỏi:** {len(questions)}")
        formatted_parts.append(f"**Thời gian ước tính:** {len(questions) * 2} phút")
        formatted_parts.append("---\n")
        
        # Questions
        for i, question in enumerate(questions, 1):
            formatted_parts.append(f"## Câu {i}")
            formatted_parts.append(f"**{question.get('question_text', 'N/A')}**\n")
            
            # Options
            options = question.get('options', {})
            for key, value in options.items():
                formatted_parts.append(f"{key}. {value}")
            
            formatted_parts.append("")  # Empty line
            
            # Show correct answer and explanation
            correct = question.get('correct_answer', 'N/A')
            explanation = question.get('explanation', {})
            
            formatted_parts.append(f"**Đáp án đúng:** {correct}")
            
            if isinstance(explanation, dict):
                correct_reason = explanation.get('correct_reason', 'Không có giải thích')
                formatted_parts.append(f"**Giải thích:** {correct_reason}")
                
                # Show source content if available
                source = question.get('source_content', '')
                if source:
                    formatted_parts.append(f"**Căn cứ từ tài liệu:** _{source[:200]}..._")
            else:
                formatted_parts.append(f"**Giải thích:** {explanation}")
            
            formatted_parts.append("---\n")
        
        return "\n".join(formatted_parts)

    def generate_quiz_from_pdf(self, pdf_structure: Dict[str, Any], quiz_requirements: Dict[str, Any]) -> str:
        """
        Generate quiz from PDF structure using professional PDF processor results.
        
        Args:
            pdf_structure: Output from ProfessionalPDFProcessor
            quiz_requirements: Quiz generation requirements
            
        Returns:
            str: Formatted quiz text
        """
        try:
            logger.info("📄 Generating quiz from PDF structure...")
            
            # Use the quiz agent's PDF structure method
            questions = self.quiz_agent.generate_quiz_from_pdf_structure(
                pdf_structure=pdf_structure,
                quiz_requirements=quiz_requirements
            )
            
            if not questions:
                return "⚠️ Không thể tạo câu hỏi từ cấu trúc PDF."
            
            # Format for display
            scope = quiz_requirements.get('scope', 'Tài liệu PDF')
            difficulty = quiz_requirements.get('difficulty', 'medium')
            
            return self._format_quiz_for_display(questions, scope, difficulty)
            
        except Exception as e:
            logger.error(f"Error generating quiz from PDF: {e}")
            return f"Lỗi khi tạo quiz từ PDF: {str(e)}"

    def generate_multiple_choice_quiz(self, content: str, section: str, num_questions: int = 5) -> List[Dict]:
        """Generate multiple choice quiz from content."""
        return self.quiz_agent.generate_content_based_questions(
            document_content=content,
            section_title=section,
            num_questions=num_questions,
            question_type="multiple_choice",
            difficulty="medium"
        )

    def generate_true_false_quiz(self, content: str, section: str, num_questions: int = 5) -> List[Dict]:
        """Generate true/false quiz from content."""
        return self.quiz_agent.generate_content_based_questions(
            document_content=content,
            section_title=section,
            num_questions=num_questions,
            question_type="true_false",
            difficulty="medium"
        )

    def generate_essay_quiz(self, content: str, section: str, num_questions: int = 3) -> List[Dict]:
        """Generate essay quiz from content."""
        return self.quiz_agent.generate_content_based_questions(
            document_content=content,
            section_title=section,
            num_questions=num_questions,
            question_type="essay",
            difficulty="medium"
        )

    def get_quiz_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated quizzes."""
        if hasattr(self.quiz_agent, 'get_generation_statistics'):
            return self.quiz_agent.get_generation_statistics()
        return {"message": "No statistics available"}

    def validate_quiz_quality(self, questions: List[Dict], original_content: str) -> Dict[str, Any]:
        """
        Validate that quiz questions are properly based on content.
        
        Args:
            questions: List of generated questions
            original_content: Original document content
            
        Returns:
            Dict with validation results
        """
        validation_results = {
            'total_questions': len(questions),
            'valid_questions': 0,
            'issues': [],
            'quality_score': 0.0
        }
        
        for i, question in enumerate(questions):
            question_id = question.get('question_id', i + 1)
            
            # Check for source content
            if 'source_content' not in question:
                validation_results['issues'].append(f"Question {question_id}: Missing source content")
                continue
            
            # Check if source content exists in original
            source_content = question.get('source_content', '')
            if source_content:
                # Simple check if key phrases from source exist in original
                source_words = source_content.split()
                found_words = sum(1 for word in source_words if len(word) > 4 and word in original_content)
                
                if found_words / max(1, len(source_words)) > 0.3:  # 30% word overlap threshold
                    validation_results['valid_questions'] += 1
                else:
                    validation_results['issues'].append(f"Question {question_id}: Source content not found in original")
            else:
                validation_results['issues'].append(f"Question {question_id}: Empty source content")
        
        # Calculate quality score
        if validation_results['total_questions'] > 0:
            validation_results['quality_score'] = validation_results['valid_questions'] / validation_results['total_questions']
        
        return validation_results

    def generate_quiz_for_chapter(self, doc_id: str, chapter_title: str, num_questions: int, difficulty: str, document_service) -> str:
        """
        Generate quiz for a specific chapter using accurate chapter mapping.
        
        Args:
            doc_id: Document ID
            chapter_title: Title of the chapter to generate quiz for
            num_questions: Number of questions to generate
            difficulty: Difficulty level (easy, medium, hard)
            document_service: DocumentService instance to get chapter content
            
        Returns:
            str: Generated quiz in formatted text
        """
        try:
            logger.info(f"🎯 Generating quiz for chapter '{chapter_title}' in document {doc_id}")
            
            # Get chapter content using the new accurate mapping
            chapter_content = document_service.get_chapter_content(doc_id, chapter_title)
            
            if not chapter_content:
                return f"⚠️ Không tìm thấy nội dung cho chương '{chapter_title}'. Vui lòng kiểm tra lại tên chương."
            
            # Generate questions using context-aware agent
            questions = self.quiz_agent.generate_content_based_questions(
                document_content=chapter_content,
                section_title=chapter_title,
                num_questions=min(num_questions, 10),  # Limit to max 10 questions
                question_type="multiple_choice",
                difficulty=difficulty
            )
            
            if not questions:
                logger.warning(f"No questions generated for chapter '{chapter_title}'")
                return f"⚠️ Không thể tạo câu hỏi cho chương '{chapter_title}'. Nội dung có thể không phù hợp."
            
            # Format questions for display
            formatted_quiz = self._format_quiz_for_display(questions, chapter_title, difficulty)
            
            # Save quiz to database
            self._save_quiz_to_database(doc_id, formatted_quiz, len(questions), difficulty, chapter_title)
            
            logger.info(f"✅ Successfully generated {len(questions)} questions for chapter '{chapter_title}'")
            return formatted_quiz
            
        except Exception as e:
            logger.error(f"❌ Error generating quiz for chapter: {e}")
            return f"Lỗi khi tạo quiz cho chương: {str(e)}"

    def _save_quiz_to_database(self, doc_id: str, quiz_content: str, num_questions: int, difficulty: str, scope: str):
        """Save generated quiz to database."""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO quizzes (document_id, quiz_content, num_questions, difficulty, scope)
                    VALUES (?, ?, ?, ?, ?)
                """, (doc_id, quiz_content, num_questions, difficulty, scope))
                conn.commit()
                logger.info(f"✅ Quiz saved to database for document {doc_id}")
        except Exception as e:
            logger.error(f"❌ Error saving quiz to database: {e}")

    def get_quiz_history(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get quiz history from database.
        
        Args:
            doc_id: Optional document ID to filter quizzes
            
        Returns:
            List of quiz records
        """
        try:
            with self.db_manager._get_connection() as conn:
                conn.row_factory = self.db_manager._get_connection().row_factory
                cursor = conn.cursor()
                
                if doc_id:
                    cursor.execute("""
                        SELECT q.*, d.filename 
                        FROM quizzes q 
                        JOIN documents d ON q.document_id = d.id 
                        WHERE q.document_id = ? 
                        ORDER BY q.created_at DESC
                    """, (doc_id,))
                else:
                    cursor.execute("""
                        SELECT q.*, d.filename 
                        FROM quizzes q 
                        JOIN documents d ON q.document_id = d.id 
                        ORDER BY q.created_at DESC
                    """)
                
                quizzes = []
                for row in cursor.fetchall():
                    quizzes.append(dict(row))
                
                logger.info(f"✅ Retrieved {len(quizzes)} quiz records")
                return quizzes
                
        except Exception as e:
            logger.error(f"❌ Error retrieving quiz history: {e}")
            return []