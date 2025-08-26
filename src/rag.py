from langchain_community.vectorstores import Chroma
from langchain.schema import Document  # ← ĐẢM BẢO CÓ IMPORT NÀY
from .fallback_llm import FallbackLLM, OfflineLLM
import json
import re
import traceback
class RAGManager:
    """
    Quản lý pipeline RAG: chunking, embedding, truy vấn và tạo quiz.
    Phiên bản này có xử lý chuỗi JSON trả về từ LLM tĩnh hơn và hỗ trợ ký tự toán học.
    """
    def __init__(self, chunker, embedder, llm):
        self.chunker = chunker
        self.embedder = embedder
        self.llm = llm
        self.fallback_llm = FallbackLLM()
        self.offline_llm = OfflineLLM()
        self.retriever = None
        self.vector_store = None

    def enhanced_math_aware_chunking(self, text_content: str):
        """
        Chia văn bản thành chunks với việc bảo vệ toàn diện các biểu thức toán học và cấu trúc học thuật.
        """
        lines = text_content.split('\n')
        chunks = []
        current_chunk = ""
        current_topic = ""
        
        # Enhanced math patterns - bao phủ nhiều trường hợp hơn
        math_patterns = [
            # Basic operators và symbols
            r'.*?[=+\-*/^√∫∑∏∈∉⊆⊇∪∩∅→←↔⇒⇔≠≤≥≈≡∼≃∝∞∂∇].*?',
            # Greek letters
            r'.*?[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ].*?',
            # Functions
            r'.*?\b(sin|cos|tan|log|ln|exp|det|tr|rank|dim|deg|gcd|lcm|max|min|sup|inf|lim)\s*\(.*?\).*?',
            # LaTeX expressions
            r'.*?\$.*?\$.*?',
            r'.*?\$\$.*?\$\$.*?',
            r'.*?\\begin\{.*?\}.*?\\end\{.*?\}.*?',
            r'.*?\\frac\{.*?\}\{.*?\}.*?',
            r'.*?\\sqrt\{.*?\}.*?',
            r'.*?\\[a-zA-Z]+\{.*?\}.*?',
            # Matrices và arrays
            r'.*?\[.*?=.*?\].*?',
            r'.*?\|.*?\|.*?',  # Determinants
            r'.*?det\s*\(.*?\).*?',
            # Numbers và equations
            r'.*?\d+\.\d+.*?=.*?',
            r'.*?[a-zA-Z]\s*[=≈≠]\s*\d+.*?',
            # Problem structures
            r'.*?\b(Problem|Bài|Câu|Ví dụ|Example)\s*\d+.*?',
            r'.*?\b(Định lý|Theorem|Lemma|Corollary).*?',
            r'.*?\b(Chứng minh|Proof|Solution|Giải).*?',
        ]
        
        # Academic structure patterns
        header_patterns = [
            r'^#+\s+.*?$',  # Markdown headers
            r'^\d+\.\s+.*?$',  # Numbered sections
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headers
            r'^\*\*.*?\*\*$',  # Bold headers
            r'^Chương\s+\d+.*?$',  # Vietnamese chapters
            r'^Bài\s+\d+.*?$',  # Vietnamese lessons
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect headers and topics
            is_header = any(re.match(pattern, line) for pattern in header_patterns)
            if is_header:
                # Start new chunk for new topic
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'topic': current_topic,
                        'type': 'content'
                    })
                current_topic = line
                current_chunk = f"[TOPIC]{line}[/TOPIC]\n"
                continue
            
            # Detect math content
            is_math_line = any(re.search(pattern, line) for pattern in math_patterns)
            if is_math_line:
                line = f"[MATH]{line}[/MATH]"
            
            # Smart chunking based on content type
            estimated_chunk_size = len(current_chunk) + len(line)
            
            # Dynamic chunk size based on content
            if is_math_line:
                max_chunk_size = 1500  # Larger for math content
            elif current_topic and 'ví dụ' in current_topic.lower():
                max_chunk_size = 2000  # Larger for examples
            else:
                max_chunk_size = 1000  # Standard size
            
            # Check if we should start new chunk
            should_split = False
            
            if estimated_chunk_size > max_chunk_size:
                should_split = True
            elif i < len(lines) - 1:
                # Look ahead for natural break points
                next_line = lines[i + 1].strip()
                if any(re.match(pattern, next_line) for pattern in header_patterns):
                    should_split = True
                elif next_line.startswith(('Ví dụ', 'Example', 'Bài tập', 'Exercise')):
                    should_split = True
            
            if should_split and current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip(),
                    'topic': current_topic,
                    'type': 'math' if '[MATH]' in current_chunk else 'content'
                })
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'topic': current_topic,
                'type': 'math' if '[MATH]' in current_chunk else 'content'
            })
        
        return chunks

    def math_aware_chunking(self, text_content: str):
        """
        Chia văn bản thành chunks với việc bảo vệ các biểu thức toán học.
        Deprecated: Sử dụng enhanced_math_aware_chunking thay thế.
        """
        enhanced_chunks = self.enhanced_math_aware_chunking(text_content)
        # Return simple list for backward compatibility
        return [chunk['content'] for chunk in enhanced_chunks]

    def setup_with_text(self, text_content: str):
        """
        Thiết lập hệ thống RAG với nội dung văn bản sử dụng enhanced chunking.
        """
        # Sử dụng enhanced math-aware chunking
        enhanced_chunks = self.enhanced_math_aware_chunking(text_content)
        
        # Process chunks with metadata preservation
        documents = []
        for i, chunk_info in enumerate(enhanced_chunks):
            content = chunk_info['content']
            
            # Clean markers nhưng giữ nguyên cấu trúc
            clean_content = content.replace("[MATH]", "").replace("[/MATH]", "")
            clean_content = clean_content.replace("[TOPIC]", "").replace("[/TOPIC]", "")
            
            if clean_content.strip():
                # Tạo metadata phong phú
                metadata = {
                    'chunk_id': i,
                    'topic': chunk_info['topic'],
                    'type': chunk_info['type'],
                    'has_math': '[MATH]' in content,
                    'has_topic': '[TOPIC]' in content,
                    'length': len(clean_content),
                    'position': i / len(enhanced_chunks)  # Relative position
                }
                
                # Add context from neighboring chunks
                if i > 0:
                    prev_chunk = enhanced_chunks[i-1]
                    metadata['prev_topic'] = prev_chunk['topic']
                    metadata['prev_type'] = prev_chunk['type']
                
                if i < len(enhanced_chunks) - 1:
                    next_chunk = enhanced_chunks[i+1]
                    metadata['next_topic'] = next_chunk['topic']
                    metadata['next_type'] = next_chunk['type']
                
                documents.append(Document(
                    page_content=clean_content.strip(),
                    metadata=metadata
                ))
        
        # Xây dựng vector store với enhanced documents
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder,
            persist_directory="./chroma_db"
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7  # Balance relevance vs diversity
            }
        )
        
        print(f"✅ Đã setup RAG với {len(documents)} enhanced chunks (có metadata)")
        return True

    def query(self, question: str) -> str:
        """
        Truy vấn hệ thống RAG với câu hỏi.
        """
        try:
            if not self.retriever:
                return "Lỗi: Chưa có tài liệu nào được xử lý. Vui lòng tải lên tài liệu trước."
            
            # Tìm kiếm documents liên quan
            docs = self.retriever.invoke(question, config={"search_kwargs": {"k": 7}})
            
            if not docs:
                return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu."
            
            # Xử lý LaTeX trong context
            from .document_reader import DocumentReader
            doc_reader = DocumentReader()
            
            # Ưu tiên nội dung có toán học
            context_parts = []
            math_indicators = ['=', '+', '-', 'λ', 'α', 'ℝ', 'ℂ', 'det(', '∈', '∉', '⊆']
            
            for doc in docs:
                processed_content = doc_reader.normalize_math_text(doc.page_content)
                
                # Đưa nội dung toán học lên đầu
                if any(indicator in processed_content for indicator in math_indicators):
                    context_parts.insert(0, processed_content)
                else:
                    context_parts.append(processed_content)
            
            context = "\n".join(context_parts[:5])  # Giới hạn context
            
            # Tạo prompt
            prompt = f"""Quy tắc trả lời:
- LUÔN sử dụng ký hiệu Unicode: λ, α, β, ℝ, ℂ, ∈, ∉, ⊆, ⊇
- KHÔNG sử dụng LaTeX commands: \\lambda, \\mathbb{{R}}, \\alpha
- GIỮ NGUYÊN công thức toán học trong dấu ngoặc vuông: [Av = λv]
- Trả lời chính xác, chi tiết và dễ hiểu

Ví dụ đúng: "Eigenvalue λ ∈ ℂ của ma trận A ∈ ℝ^{{n×n}}"
Ví dụ sai: "Eigenvalue \\lambda \\in \\mathbb{{C}} của ma trận A"

Context từ tài liệu:
{context}

Câu hỏi: {question}

Trả lời:"""

            # Gọi LLM với fallback
            response = self._call_llm_with_fallback(prompt)
            
            # Post-process response để đảm bảo Unicode
            response = doc_reader.normalize_math_text(response)
            
            return response
            
        except Exception as e:
            print(f"❌ Lỗi trong query: {str(e)}")
            return "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."

    def generate_quiz(self, num_questions: int = 5):
        """Tạo quiz từ documents với xử lý LaTeX."""
        try:
            print(f"🎯 Bắt đầu tạo quiz với {num_questions} câu hỏi...")
            
            if not self.retriever:
                error_msg = "Lỗi: Chưa có tài liệu nào được xử lý. Vui lòng tải lên tài liệu trước."
                print(f"❌ {error_msg}")
                return error_msg
                
            # Lấy nhiều documents để có đủ nội dung
            print("🔍 Đang tìm kiếm nội dung từ tài liệu...")
            all_docs = self.retriever.invoke("", config={"search_kwargs": {"k": 15}})
            
            if not all_docs:
                error_msg = "Lỗi: Không tìm thấy nội dung phù hợp để tạo quiz."
                print(f"❌ {error_msg}")
                return error_msg
            
            print(f"✅ Tìm thấy {len(all_docs)} đoạn văn bản")
                
            # Xử lý LaTeX trong documents
            from .document_reader import DocumentReader
            doc_reader = DocumentReader()
            
            processed_docs = []
            math_docs = []
            regular_docs = []
            
            print("🔄 Đang xử lý và phân loại nội dung...")
            
            # Phân loại và xử lý documents
            for i, doc in enumerate(all_docs):
                try:
                    processed_content = doc_reader.normalize_math_text(doc.page_content)
                    
                    # Tạo document object đơn giản
                    class SimpleDoc:
                        def __init__(self, content, metadata=None):
                            self.page_content = content
                            self.metadata = metadata or {}
                    
                    processed_doc = SimpleDoc(
                        processed_content, 
                        doc.metadata if hasattr(doc, 'metadata') else {}
                    )
                    
                    # Phân loại theo nội dung
                    math_indicators = ['=', '+', '-', 'λ', 'α', 'ℝ', 'ℂ', 'det(', '∈', '∉', '⊆']
                    
                    if any(indicator in processed_content for indicator in math_indicators):
                        math_docs.append(processed_doc)
                    else:
                        regular_docs.append(processed_doc)
                        
                except Exception as doc_error:
                    print(f"⚠️ Lỗi khi xử lý document {i+1}: {doc_error}")
                    continue
            
            print(f"📊 Phân loại: {len(math_docs)} đoạn toán học, {len(regular_docs)} đoạn thường")
            
            # Cân bằng content: 60% toán học, 40% thông thường
            max_math = max(1, min(len(math_docs), int(num_questions * 0.6 * 2)))  # x2 vì cần nhiều context
            max_regular = max(1, min(len(regular_docs), int(num_questions * 0.4 * 2)))
            
            selected_docs = math_docs[:max_math] + regular_docs[:max_regular]
            
            if not selected_docs:
                error_msg = "Lỗi: Không thể xử lý nội dung tài liệu. Vui lòng thử tài liệu khác."
                print(f"❌ {error_msg}")
                return error_msg
            
            print(f"✅ Chọn {len(selected_docs)} đoạn để tạo quiz")
            
            # Tạo context - GIỚI HẠN CONTEXT ĐỂ TRÁNH QUÁ DÀI
            context_parts = []
            total_length = 0
            
            for doc in selected_docs:
                if total_length + len(doc.page_content) > 4000:  # Giới hạn 4K chars
                    break
                context_parts.append(doc.page_content)
                total_length += len(doc.page_content)
            
            context = "\n\n---\n\n".join(context_parts)
            
            print(f"📝 Context length: {len(context)} characters")
            
            # Tạo prompt ngắn gọn hơn cho quiz generation
            quiz_prompt = f"""Tạo {min(num_questions, 3)} câu hỏi trắc nghiệm từ nội dung sau.

QUAN TRỌNG: CHỈ trả về JSON array, không có text khác.

Format:
[{{"question": "Câu hỏi?", "choices": ["A. ...", "B. ...", "C. ...", "D. ..."], "correct_answer": 1}}]

Nội dung:
{context[:2000]}...

JSON:"""

            print("🤖 Đang gọi LLM để tạo quiz...")
            print(f"📏 Prompt length: {len(quiz_prompt)} characters")
            
            # Gọi LLM với timeout ngắn hơn
            response = self._call_llm_with_fallback(quiz_prompt)
            
            if not response or response.strip() == "":
                error_msg = "Lỗi: LLM không trả về kết quả. Vui lòng thử lại."
                print(f"❌ {error_msg}")
                return error_msg
            
            print(f"📝 Kết quả từ LLM ({len(response)} chars): {response[:300]}...")
            
            # Xử lý và parse JSON
            try:
                # Làm sạch response
                response = response.strip()
                
                # Loại bỏ markdown code blocks nếu có
                if '```json' in response:
                    start = response.find('```json') + 7
                    end = response.find('```', start)
                    if end != -1:
                        response = response[start:end].strip()
                elif '```' in response:
                    start = response.find('```') + 3
                    end = response.find('```', start)
                    if end != -1:
                        response = response[start:end].strip()
                
                # Tìm JSON array
                start_idx = response.find('[')
                end_idx = response.rfind(']') + 1
                
                if start_idx == -1 or end_idx == 0:
                    print(f"❌ Không tìm thấy JSON array trong response: {response}")
                    # Thử tạo JSON từ text nếu có thể
                    if "câu hỏi" in response.lower():
                        return self._extract_quiz_from_text(response, num_questions)
                    raise ValueError("Không tìm thấy JSON array trong response")
                
                json_str = response[start_idx:end_idx]
                print(f"📝 JSON được trích xuất: {json_str[:500]}...")
                
                # Parse JSON
                quiz_data = json.loads(json_str)
                
                if not isinstance(quiz_data, list):
                    raise ValueError(f"JSON không phải là array: {type(quiz_data)}")
                
                if len(quiz_data) == 0:
                    raise ValueError("Array JSON rỗng")
                
                print(f"✅ Parse JSON thành công: {len(quiz_data)} câu hỏi")
                
                # Validate từng câu hỏi
                validated_questions = []
                for i, q in enumerate(quiz_data):
                    try:
                        # Kiểm tra các field bắt buộc
                        if not all(key in q for key in ['question', 'choices', 'correct_answer']):
                            print(f"⚠️ Câu hỏi {i+1} thiếu field bắt buộc")
                            continue
                        
                        # Kiểm tra choices
                        if not isinstance(q['choices'], list) or len(q['choices']) != 4:
                            print(f"⚠️ Câu hỏi {i+1} không có đúng 4 lựa chọn")
                            continue
                        
                        # Kiểm tra correct_answer
                        if not isinstance(q['correct_answer'], int) or q['correct_answer'] < 1 or q['correct_answer'] > 4:
                            print(f"⚠️ Câu hỏi {i+1} có correct_answer không hợp lệ")
                            continue
                        
                        validated_questions.append(q)
                        
                    except Exception as validation_error:
                        print(f"⚠️ Lỗi validate câu hỏi {i+1}: {validation_error}")
                        continue
                
                if len(validated_questions) == 0:
                    error_msg = "Lỗi: Không có câu hỏi nào hợp lệ sau khi validate."
                    print(f"❌ {error_msg}")
                    # Fallback: Tạo quiz template
                    return json.loads(self._generate_template_quiz())
                
                print(f"✅ Tạo quiz thành công: {len(validated_questions)} câu hỏi hợp lệ")
                return validated_questions
                
            except json.JSONDecodeError as e:
                error_msg = f"Lỗi: Không thể phân tích JSON từ AI. Chi tiết: {str(e)}"
                print(f"❌ {error_msg}")
                print(f"📝 Response gốc: {response}")
                # Fallback: Thử extract từ text
                try:
                    return self._extract_quiz_from_text(response, num_questions)
                except:
                    return json.loads(self._generate_template_quiz())
                
            except Exception as parse_error:
                error_msg = f"Lỗi: Định dạng quiz không đúng. Chi tiết: {str(parse_error)}"
                print(f"❌ {error_msg}")
                print(f"📝 Response gốc: {response}")
                return json.loads(self._generate_template_quiz())
            
        except Exception as e:
            error_msg = f"Lỗi: Hệ thống gặp sự cố. Chi tiết: {str(e)}"
            print(f"❌ Lỗi trong generate_quiz: {str(e)}")
            traceback.print_exc()
            return json.loads(self._generate_template_quiz())

    def _extract_quiz_from_text(self, text: str, num_questions: int):
        """
        Thử extract quiz từ text khi JSON parsing fail.
        """
        try:
            print("🔄 Đang thử extract quiz từ plain text...")
            
            # Simple fallback: tạo quiz từ context
            quiz = [
                {
                    "question": "Dựa trên nội dung đã học, khái niệm nào sau đây là quan trọng nhất?",
                    "choices": [
                        "A. Khái niệm cơ bản",
                        "B. Khái niệm nâng cao", 
                        "C. Khái niệm ứng dụng",
                        "D. Tất cả đều quan trọng"
                    ],
                    "correct_answer": 4
                },
                {
                    "question": "Để hiểu sâu hơn về chủ đề này, bạn nên làm gì?",
                    "choices": [
                        "A. Đọc thêm tài liệu",
                        "B. Thực hành bài tập",
                        "C. Thảo luận với bạn bè",
                        "D. Cả ba phương pháp trên"
                    ],
                    "correct_answer": 4
                }
            ]
            
            return quiz[:num_questions]
            
        except Exception as e:
            print(f"❌ Extract từ text cũng lỗi: {e}")
            return json.loads(self._generate_template_quiz())
    
    def act_as_expert(self, question: str, chat_history: list = []) -> str:
        """
        Hoạt động như một chuyên gia trong lĩnh vực tài liệu.
        """
        try:
            if not self.retriever:
                return "Lỗi: Chưa có tài liệu nào được xử lý. Vui lòng tải lên tài liệu trước."
            
            # Tìm kiếm context
            docs = self.retriever.invoke(question, config={"search_kwargs": {"k": 5}})
            
            if not docs:
                return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi này."
            
            # Xử lý LaTeX trong context
            from .document_reader import DocumentReader
            doc_reader = DocumentReader()
            
            context = "\n".join([doc_reader.normalize_math_text(doc.page_content) for doc in docs])
            
            # Xây dựng lịch sử chat
            history_text = ""
            if chat_history:
                recent_history = chat_history[-3:]  # Chỉ lấy 3 tin nhắn gần nhất
                for msg in recent_history:
                    if msg.get('role') == 'user':
                        history_text += f"Học sinh: {msg.get('content', '')}\n"
                    elif msg.get('role') == 'assistant':
                        history_text += f"Chuyên gia: {msg.get('content', '')}\n"
            
            # Tạo prompt chuyên gia
            expert_prompt = f"""Bạn là một chuyên gia giàu kinh nghiệm trong lĩnh vực này. Hãy trả lời câu hỏi một cách chuyên nghiệp, chi tiết và học thuật.

QUY TẮC:
- LUÔN sử dụng ký hiệu Unicode: λ, α, β, ℝ, ℂ, ∈, ∉, ⊆
- KHÔNG sử dụng LaTeX commands: \\lambda, \\mathbb{{R}}
- Giải thích rõ ràng, có ví dụ cụ thể
- Sử dụng thuật ngữ chuyên môn chính xác

{f"Lịch sử trò chuyện gần đây:\\n{history_text}" if history_text else ""}

Context từ tài liệu:
{context}

Câu hỏi của học sinh: {question}

Trả lời chuyên gia:"""

            response = self._call_llm_with_fallback(expert_prompt)
            
            # Post-process response
            response = doc_reader.normalize_math_text(response)
            
            return response
            
        except Exception as e:
            print(f"❌ Lỗi trong act_as_expert: {str(e)}")
            return "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."

    def suggest_research_topics(self, num_topics: int = 3) -> str:
        """
        Đề xuất các chủ đề nghiên cứu dựa trên nội dung tài liệu.
        """
        try:
            if not self.retriever:
                return "Lỗi: Chưa có tài liệu nào được xử lý. Vui lòng tải lên tài liệu trước."
            
            # Lấy tất cả nội dung liên quan
            docs = self.retriever.invoke("", config={"search_kwargs": {"k": 10}})
            
            if not docs:
                return "Xin lỗi, không thể đề xuất chủ đề nghiên cứu vì không tìm thấy nội dung phù hợp."
            
            # Xử lý LaTeX trong context
            from .document_reader import DocumentReader
            doc_reader = DocumentReader()
            
            all_content = "\n".join([doc_reader.normalize_math_text(doc.page_content) for doc in docs])
            
            # Tạo prompt
            topics_prompt = f"""Dựa trên nội dung tài liệu, hãy đề xuất {num_topics} chủ đề nghiên cứu thú vị và có tính thực tiễn.

QUY TẮC:
- LUÔN sử dụng ký hiệu Unicode: λ, α, β, ℝ, ℂ, ∈, ∉, ⊆
- KHÔNG sử dụng LaTeX commands
- Mỗi chủ đề nên có mô tả ngắn gọn về tại sao nó thú vị
- Ưu tiên các chủ đề có tính ứng dụng thực tế

Nội dung tài liệu:
{all_content[:3000]}

Đề xuất {num_topics} chủ đề nghiên cứu:"""

            response = self._call_llm_with_fallback(topics_prompt)
            
            # Post-process response
            response = doc_reader.normalize_math_text(response)
            
            return response
            
        except Exception as e:
            print(f"❌ Lỗi trong suggest_research_topics: {str(e)}")
            return "Đã xảy ra lỗi khi đề xuất chủ đề nghiên cứu. Vui lòng thử lại."

    def _call_llm_with_fallback(self, prompt: str, **kwargs) -> str:
        """
        Gọi LLM với hệ thống fallback 3 tầng.
        """
        try:
            # Tầng 1: LLM chính (OpenRouter)
            print("🤖 Đang gọi OpenRouter LLM...")
            response = self.llm.invoke(prompt, **kwargs)
            
            print(f"📝 OpenRouter response length: {len(response) if response else 0}")
            print(f"📝 OpenRouter response preview: {response[:100] if response else 'EMPTY'}")
            
            # Kiểm tra nếu response rỗng hoặc None
            if not response or response.strip() == "":
                print("⚠️ OpenRouter trả về response rỗng, chuyển sang fallback...")
                raise Exception("Empty response from OpenRouter")
            
            # Kiểm tra nếu response cho thấy lỗi kết nối
            if any(error_text in response.lower() for error_text in [
                "không thể kết nối", "connection error", "timeout", 
                "network error", "api error", "rate limit", "error"
            ]):
                print("⚠️ OpenRouter có vấn đề, chuyển sang fallback...")
                raise Exception(f"OpenRouter error: {response[:100]}")
            
            print("✅ OpenRouter response thành công")
            return response
            
        except Exception as e:
            print(f"❌ OpenRouter LLM lỗi: {str(e)}")
            
            try:
                # Tầng 2: Fallback LLM
                print("🔄 Đang thử fallback LLM...")
                fallback_response = self.fallback_llm.invoke(prompt)
                
                if fallback_response and fallback_response.strip():
                    print("✅ Fallback LLM thành công")
                    return fallback_response
                else:
                    print("⚠️ Fallback LLM trả về rỗng")
                    raise Exception("Fallback LLM returned empty")
                    
            except Exception as e2:
                print(f"❌ Fallback LLM lỗi: {str(e2)}")
                
                try:
                    # Tầng 3: Offline LLM
                    print("🔄 Đang thử offline LLM...")
                    offline_response = self.offline_llm.invoke(prompt)
                    
                    if offline_response and offline_response.strip():
                        print("✅ Offline LLM thành công")
                        return offline_response
                    else:
                        print("⚠️ Offline LLM trả về rỗng")
                        raise Exception("Offline LLM returned empty")
                        
                except Exception as e3:
                    print(f"❌ Offline LLM lỗi: {str(e3)}")
                    
                    # Tầng 4: Hardcoded fallback cho quiz generation
                    if "tạo" in prompt.lower() and "câu hỏi" in prompt.lower():
                        print("🆘 Tất cả LLM đều lỗi, sử dụng template quiz...")
                        return self._generate_template_quiz()
                    
                    return "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."

    def _generate_template_quiz(self):
        """
        Tạo quiz template khi tất cả LLM đều lỗi.
        """
        template_quiz = [
            {
                "question": "Đây là câu hỏi mẫu khi hệ thống gặp sự cố. Bạn có hiểu không?",
                "choices": [
                    "A. Có, tôi hiểu",
                    "B. Không, tôi không hiểu", 
                    "C. Cần giải thích thêm",
                    "D. Tôi muốn thử lại"
                ],
                "correct_answer": 1
            },
            {
                "question": "Hệ thống đang gặp vấn đề tạm thời. Bạn muốn làm gì?",
                "choices": [
                    "A. Thử lại sau",
                    "B. Liên hệ hỗ trợ",
                    "C. Sử dụng chức năng khác", 
                    "D. Thoát khỏi hệ thống"
                ],
                "correct_answer": 1
            }
        ]
        return json.dumps(template_quiz, ensure_ascii=False)
        
    def enhanced_query_with_context(self, question: str) -> str:
        """Advanced query với preprocessing và context awareness"""
        try:
            # 1. Phân loại loại câu hỏi
            question_type = "general"
            if any(word in question.lower() for word in ["toán", "math", "tính", "giải", "phương trình"]):
                question_type = "math"
            elif any(word in question.lower() for word in ["vật lý", "physics", "lực", "điện"]):
                question_type = "physics"
            elif any(word in question.lower() for word in ["hóa", "chemistry", "phản ứng", "chất"]):
                question_type = "chemistry"
            
            # 2. Tìm kiếm với fallback đơn giản (không dùng MMR vì dependency)
            if hasattr(self, 'retriever') and self.retriever:
                docs = self.retriever.invoke(question, config={"search_kwargs": {"k": 5}})
                
                # 3. Filter theo loại câu hỏi nếu có metadata
                relevant_docs = []
                for doc in docs:
                    if hasattr(doc, 'metadata') and doc.metadata:
                        doc_topic = doc.metadata.get('topic', '').lower()
                        if question_type == "math" and any(math_word in doc_topic for math_word in ["toán", "math", "equation"]):
                            relevant_docs.append(doc)
                        elif question_type in doc_topic:
                            relevant_docs.append(doc)
                        else:
                            relevant_docs.append(doc)  # Fallback
                    else:
                        relevant_docs.append(doc)
                
                # 4. Tạo context từ docs
                if relevant_docs:
                    context_parts = []
                    for i, doc in enumerate(relevant_docs[:3]):  # Top 3 docs
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        context_parts.append(f"[Tài liệu {i+1}]: {content}")
                    
                    context = "\n\n".join(context_parts)
                    
                    # 5. Enhanced prompt với context
                    enhanced_prompt = f"""
Dựa trên ngữ cảnh sau, hãy trả lời câu hỏi một cách chính xác và chi tiết.

NGỮ CẢNH:
{context}

CÂUHỎI: {question}

HƯỚNG DẪN:
- Loại câu hỏi: {question_type}
- Trả lời dựa trên ngữ cảnh được cung cấp
- Nếu cần tính toán, hiển thị các bước rõ ràng
- Nếu không đủ thông tin, nói rõ cần thêm gì

TRẢ LỜI:"""
                    
                    # 6. Gọi LLM với enhanced prompt
                    return self.llm_client.generate_response(enhanced_prompt)
                else:
                    return "Không tìm thấy tài liệu phù hợp để trả lời câu hỏi."
            
            # Fallback to basic query if no vector store
            return self.query(question)
            
        except Exception as e:
            print(f"Error in enhanced query: {e}")
            return self.query(question)  # Fallback to basic query