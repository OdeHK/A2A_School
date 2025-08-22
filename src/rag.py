
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import time
import json


class RAGManager:
    """
    Quản lý pipeline RAG: chunking, embedding, truy vấn và tạo quiz.
    Phiên bản này có xử lý chuỗi JSON trả về từ LLM tĩnh hơn.
    """
    def __init__(self, chunker, embedder, llm):
        self.chunker = chunker
        self.embedder = embedder
        self.llm = llm
        self.retriever = None

    def setup_with_text(self, text_content: str):
        chunks = self.chunker.split_text(text_content)
        docs = [Document(page_content=chunk) for chunk in chunks]
        vectorstore = Chroma.from_documents(documents=docs, embedding=self.embedder)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        print("✅ Đã xây dựng xong vector store và retriever.")

    def query(self, question: str) -> str:
        """Thực hiện truy vấn RAG cơ bản."""
        if not self.retriever:
            return "Lỗi: Hệ thống chưa được khởi tạo với tài liệu. Vui lòng tải file lên."
        
        try:
            # Lấy tài liệu liên quan sử dụng invoke
            search_kwargs = {"k": 5}
            docs = self.retriever.invoke(question, config={"search_kwargs": search_kwargs})
            
            if not docs:
                return "Không tìm thấy thông tin liên quan trong tài liệu."
            
            # Tạo context từ các tài liệu
            context = "\n".join([doc.page_content for doc in docs])
            
            # Tạo prompt
            prompt = f"""Bạn là một trợ lý AI hữu ích. 
            Hãy trả lời câu hỏi dưới đây dựa trên thông tin được cung cấp.
            Nếu câu trả lời không có trong nội dung, hãy nói thật là bạn không biết.

            Nội dung tham khảo:
            {context}

            Câu hỏi:
            {question}

            Trả lời:"""
            
            # Gọi LLM
            response = self.llm.invoke(prompt)
            return response if isinstance(response, str) else str(response)
            
        except Exception as e:
            print(f"❌ Lỗi trong query: {str(e)}")
            return "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn. Vui lòng thử lại."

    def generate_quiz(self, num_questions: int = 5):
        """Tạo bài quiz từ nội dung tài liệu."""
        if not self.retriever:
            return "Lỗi: Hệ thống chưa được khởi tạo với tài liệu."

        try:
            # Lấy nội dung từ tài liệu sử dụng invoke
            search_kwargs = {"k": 10}  # Lấy nhiều đoạn hơn cho quiz
            all_docs = self.retriever.invoke("", config={"search_kwargs": search_kwargs})
            
            if not all_docs:
                return "Không thể tạo quiz do không tìm thấy đủ nội dung trong tài liệu."
            
            full_context = "\n---\n".join([doc.page_content for doc in all_docs])

            quiz_prompt = f"""
            Bạn là một API tạo quiz. Hãy trả về chính xác một mảng JSON với {num_questions} câu hỏi.
            Mỗi câu hỏi phải có format như sau:
            {{
                "question": "Nội dung câu hỏi?",
                "choices": [
                    "A. Lựa chọn thứ nhất",
                    "B. Lựa chọn thứ hai",
                    "C. Lựa chọn thứ ba",
                    "D. Lựa chọn thứ tư"
                ],
                "correct_answer": 1
            }}
            
            Lưu ý:
            - correct_answer là số từ 1-4 chỉ vị trí đáp án đúng
            - Mỗi lựa chọn phải bắt đầu bằng "A.", "B.", "C.", "D."
            - KHÔNG THÊM GIẢI THÍCH. CHỈ TRẢ VỀ JSON.

            Tạo {num_questions} câu hỏi từ nội dung sau:
            {full_context}
            """

            print("🎯 Bắt đầu tạo quiz...")
            try:
                response = self.llm.invoke(quiz_prompt)
                print("📝 Kết quả từ LLM:", response)

                # Làm sạch và chuẩn hóa response
                response = response.strip()
                
                # Xử lý các trường hợp đặc biệt
                if response.lower().startswith('json'):
                    response = response[4:].strip()
                response = response.replace('```json', '').replace('```', '')
                
                # Tìm và trích xuất JSON array
                start_idx = response.find('[')
                end_idx = response.rfind(']') + 1
                if start_idx != -1 and end_idx > start_idx:
                    response = response[start_idx:end_idx]

                # Parse JSON
                try:
                    quiz_data = json.loads(response)
                    # Validate format
                    for question in quiz_data:
                        if not all(key in question for key in ['question', 'choices', 'correct_answer']):
                            raise ValueError("Thiếu trường bắt buộc trong câu hỏi")
                        if len(question['choices']) != 4:
                            raise ValueError("Mỗi câu hỏi phải có đúng 4 lựa chọn")
                    return quiz_data
                except json.JSONDecodeError as e:
                    print("❌ Lỗi JSON:", str(e))
                    print("📝 Response gốc:", response)
                    return []
            except Exception as e:
                print("❌ Lỗi không xác định:", str(e))
                return []
        except Exception as e:
            print("❌ Lỗi trong generate_quiz:", str(e))
            return []
    def act_as_expert(self, question: str, chat_history: list = []):
        """Trò chuyện như một chuyên gia dựa trên tài liệu có liên quan."""
        if not self.retriever:
            return "Lỗi: Vui lòng tải lên tài liệu trước khi sử dụng chức năng này."

        try:
            # Sử dụng invoke thay vì get_relevant_documents
            search_kwargs = {"k": 5}
            relevant_docs = self.retriever.invoke(question, config={"search_kwargs": search_kwargs})
            if not relevant_docs:
                return "Tôi không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi của bạn."

            # Xây dựng context từ các đoạn liên quan
            context = "\n---\n".join([doc.page_content for doc in relevant_docs])

            # Xây dựng lịch sử chat
            history_str = ""
            if chat_history:
                history_str = "\n".join([
                    f"{'🧑 Người dùng' if msg['role']=='user' else '🤖 Chuyên gia'}: {msg['content']}"
                    for msg in chat_history[-3:]  # Chỉ lấy 3 tin nhắn gần nhất
                ])

            # Tạo prompt chi tiết cho LLM
            prompt = f"""Bạn là một chuyên gia trong lĩnh vực được đề cập trong tài liệu.
            Hãy trả lời câu hỏi dựa trên kiến thức từ tài liệu một cách chuyên nghiệp và học thuật.

            {'--- Lịch sử trò chuyện gần đây ---\n' + history_str if history_str else ''}

            --- Thông tin từ tài liệu ---
            {context}

            --- Câu hỏi hiện tại ---
            {question}

            Hãy trả lời một cách:
            1. Chính xác: Chỉ dựa vào thông tin từ tài liệu
            2. Chuyên nghiệp: Dùng ngôn ngữ học thuật phù hợp
            3. Dễ hiểu: Giải thích rõ ràng cho người học
            4. Đầy đủ: Đề cập đến tất cả các khía cạnh quan trọng

            Câu trả lời của chuyên gia:"""

            # Gọi LLM với retry tự động
            response = self.llm.invoke(prompt)
            return response if response else "Xin lỗi, tôi đang gặp vấn đề trong việc xử lý. Vui lòng thử lại."

        except Exception as e:
            print(f"❌ Lỗi trong act_as_expert: {str(e)}")
            return "Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại sau."

    def suggest_research_topics(self, num_topics: int = 3):
        """Gợi mở các hướng đề tài nghiên cứu dựa trên tài liệu."""
        if not self.retriever:
            return "Lỗi: Vui lòng tải lên tài liệu trước khi sử dụng chức năng này."
        
        try:
            # Lấy các đoạn quan trọng từ tài liệu sử dụng invoke
            search_kwargs = {"k": 5}
            relevant_docs = self.retriever.invoke("", config={"search_kwargs": search_kwargs})
            
            if not relevant_docs:
                return "Không tìm thấy đủ thông tin trong tài liệu để đề xuất hướng nghiên cứu."

            # Tổng hợp nội dung có liên quan
            context = "\n\n".join([doc.page_content for doc in relevant_docs])  # Sử dụng tất cả đoạn liên quan

            prompt = f"""Bạn là một giáo sư hướng dẫn có kinh nghiệm.
            Dựa trên nội dung tài liệu được cung cấp, hãy đề xuất {num_topics} hướng nghiên cứu tiềm năng cho sinh viên.

            --- Nội dung tài liệu ---
            {context}

            Cho mỗi đề xuất, hãy cung cấp:
            1. Tên đề tài
            2. Mục tiêu nghiên cứu
            3. 2-3 câu hỏi nghiên cứu chính
            4. Phương pháp nghiên cứu đề xuất
            5. Kết quả dự kiến và ý nghĩa

            Format phản hồi:
            # Đề tài 1: [Tên đề tài]
            - Mục tiêu: ...
            - Câu hỏi nghiên cứu: ...
            - Phương pháp: ...
            - Kết quả dự kiến: ...

            [Tiếp tục với các đề tài khác]"""

            response = self.llm.invoke(prompt)
            return response if response else "Xin lỗi, tôi đang gặp vấn đề trong việc xử lý. Vui lòng thử lại."

        except Exception as e:
            print(f"❌ Lỗi trong suggest_research_topics: {str(e)}")
            return "Đã xảy ra lỗi khi tạo gợi ý nghiên cứu. Vui lòng thử lại sau."