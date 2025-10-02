"""
Quiz to Google Form Converter
Chuyển đổi từ JSON quiz format sang Google Form API Schema
"""
import os
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from services.models import QuizQuestion, QuizQuestionOutput


class QuestionType(Enum):
    """Enum định nghĩa các loại câu hỏi được hỗ trợ"""
    MULTIPLE_CHOICE = "multiple_choice"
    ESSAY = "essay"


class GoogleFormItemType(Enum):
    """Enum định nghĩa các loại Item trong Google Form API"""
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    PARAGRAPH_TEXT = "PARAGRAPH_TEXT"


class QuestionMapper:
    """Class chuyển đổi từng loại câu hỏi sang Google Form format"""
    
    def map_multiple_choice(self, question: QuizQuestion, index: int) -> Dict[str, Any]:
        """
        Chuyển đổi câu hỏi multiple choice
        
        Args:
            question: QuizQuestion object
            index: Vị trí của câu hỏi trong form
            
        Returns:
            Dict: Google Form createItem request structure
        """
        options = [{"value": option} for option in question.options or []]
        
        return {
            "createItem": {
                "item": {
                    "title": question.title,
                    "questionItem": {
                        "question": {
                            "required": True,
                            "choiceQuestion": {
                                "type": "RADIO",
                                "options": options,
                                "shuffle": False
                            }
                        }
                    }
                },
                "location": {"index": index}
            }
        }
    
    def map_essay_question(self, question: QuizQuestion, index: int) -> Dict[str, Any]:
        """
        Chuyển đổi câu hỏi essay

        Args:
            question: QuizQuestion object
            index: Vị trí của câu hỏi trong form
            
        Returns:
            Dict: Google Form createItem request structure
        """
        return {
            "createItem": {
                "item": {
                    "title": question.title,
                    "questionItem": {
                        "question": {
                            "required": True,
                            "textQuestion": {
                                "paragraph": True
                            }
                        }
                    }
                },
                "location": {"index": index}
            }
        }
    
    def map_question(self, question: QuizQuestion, index: int) -> Dict[str, Any]:
        """
        Chuyển đổi câu hỏi dựa trên type
        
        Args:
            question: QuizQuestion object
            index: Vị trí của câu hỏi trong form
            
        Returns:
            Dict: Google Form createItem request structure
        """
        mapping_methods = {
            QuestionType.MULTIPLE_CHOICE.value: self.map_multiple_choice,
            QuestionType.ESSAY.value: self.map_essay_question
        }
        
        method = mapping_methods.get(question.type)
        if method:
            return method(question, index)
        else:
            raise ValueError(f"Không hỗ trợ loại câu hỏi: {question.type}")


class GoogleFormSchemaBuilder:
    """Class tạo cấu trúc Google Form Schema hoàn chỉnh"""
    
    def build_form_creation_request(self, title: str) -> Dict[str, Any]:
        """
        Tạo request tạo form ban đầu (chỉ có title theo yêu cầu của Google API)
        
        Args:
            title: Tiêu đề form
            
        Returns:
            Dict: Form creation request
        """
        return {
            "info": {
                "title": title
            }
        }
    
    def build_form_info_update_request(self, description: str) -> Dict[str, Any]:
        """
        Tạo request để update form info (thêm description)
        
        Args:
            description: Mô tả form
            
        Returns:
            Dict: updateFormInfo request
        """
        return {
            "updateFormInfo": {
                "info": {
                    "description": description
                },
                "updateMask": "description"
            }
        }
    
    def build_items_requests(self, requests: List[Dict[str, Any]], description: Optional[str] = None) -> Dict[str, Any]:
        """
        Tạo requests để thêm các item vào form và update description
        
        Args:
            requests: List các createItem requests
            description: Mô tả form (optional)
            
        Returns:
            Dict: Batch requests để thêm items và update info
        """
        all_requests = []
        
        # Thêm updateFormInfo request nếu có description
        if description:
            info_update_request = self.build_form_info_update_request(description)
            all_requests.append(info_update_request)
        
        # Thêm các createItem requests
        all_requests.extend(requests)
        
        return {
            "requests": all_requests
        }
    
    def build_form_schema(self, title: str, description: str, 
                         items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tạo schema hoàn chỉnh cho Google Form (backward compatibility)
        
        Args:
            title: Tiêu đề form
            description: Mô tả form
            items: List các item (câu hỏi)
            
        Returns:
            Dict: Google Form schema hoàn chỉnh
        """
        # Trả về cả form creation và items requests
        form_creation = self.build_form_creation_request(title)
        items_requests = self.build_items_requests(items, description)
        
        return {
            "form_creation": form_creation,
            "items_requests": items_requests
        }


class QuizToGoogleFormConverter:
    """
    Main Converter class
    Chuyển đổi từ Quiz JSON format sang Google Form API Schema
    """
    
    def __init__(self):
        self.question_mapper = QuestionMapper()
        self.schema_builder = GoogleFormSchemaBuilder()
    
    def load_quiz_from_file(self, file_path: str) -> QuizQuestionOutput:
        """
        Load quiz data từ file JSON
        
        Args:
            file_path: Đường dẫn đến file JSON
            
        Returns:
            Dict: Quiz data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return QuizQuestionOutput(**json.load(file))
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Lỗi parse JSON: {e}")
    
    def convert_quiz_to_google_form(self, quiz_data: QuizQuestionOutput) -> Dict[str, Any]:
        """
        Chuyển đổi quiz data sang Google Form schema
        
        Args:
            quiz_data: Dictionary chứa dữ liệu quiz
            
        Returns:
            Dict: Google Form API schema
        """
        
        # TODO: Validate quiz_data
        
        # Chuyển đổi từng câu hỏi
        google_form_requests = []
        for index, question in enumerate(quiz_data.questions):
            try:
                request = self.question_mapper.map_question(question, index)
                google_form_requests.append(request)
            except Exception as e:
                print(f"Lỗi khi chuyển đổi câu hỏi '{question.title}': {e}")
        
        # Tạo schema hoàn chỉnh
        title = "Untitled Quiz"
        description = ""
        return self.schema_builder.build_form_schema(title, description, google_form_requests)
    
    def convert_file_to_google_form(self, input_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Chuyển đổi file JSON thành Google Form schema
        
        Args:
            input_file: Đường dẫn file JSON input
            output_file: Đường dẫn file output (optional)
            
        Returns:
            Dict: Google Form schema
        """
        if os.path.exists(input_file) is False:
            raise FileNotFoundError(f"Không tìm thấy file: {input_file}")
        
        # Load dữ liệu
        quiz_data = self.load_quiz_from_file(input_file)

        # TODO: Sử dụng pydantic để validate toàn bộ quiz_data nếu cần thiết

        # Chuyển đổi
        google_form_schema = self.convert_quiz_to_google_form(quiz_data)
        
        # Lưu file output nếu được chỉ định
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(google_form_schema, file, ensure_ascii=False, indent=2)
            print(f"Đã lưu Google Form schema vào: {output_file}")
        
        return google_form_schema


def main():
    """
    Demo function để test converter
    """
    converter = QuizToGoogleFormConverter()
    
    try:
        # Test với file quiz_data.json hiện có
        input_file = r"demo/InteractWithGGForm/quiz_data.json"
        
        print("🚀 Bắt đầu chuyển đổi...")
        result = converter.convert_file_to_google_form(input_file)
        
        print("\n✅ Chuyển đổi thành công!")
        print(f"📄 Số câu hỏi được chuyển đổi: {len(result.get('items', []))}")
        print(f"📝 Tiêu đề form: {result.get('info', {}).get('title', 'N/A')}")
        
        # In ra một phần kết quả để xem
        print("\n📋 Preview kết quả:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    main()
