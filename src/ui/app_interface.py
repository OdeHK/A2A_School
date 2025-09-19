# src/ui/app_interface.py
# Lớp định nghĩa và quản lý toàn bộ giao diện Gradio.

import gradio as gr
import logging
import re
from pathlib import Path
import json
from typing import List, Tuple, Optional

# Import các thành phần đã được refactor
from config.settings import AppConfig
from src.core.llm_provider import LLMProvider
from src.core.vector_store import VectorStore
from src.core.professional_pdf_processor import ProfessionalPDFProcessor
from src.agents.context_aware_quiz_agent import ContextAwareQuizAgent
from src.agents.summarization_agent_optimized import SummarizationAgentOptimized
from src.db.database_manager import DatabaseManager
from src.services.document_service import DocumentService
from src.services.analysis_service import AnalysisService
from src.services.quiz_service import QuizService
from src.agents.analysis_agent import AnalysisAgent
from src.agents.card_manager import CardManager

logger = logging.getLogger(__name__)

class A2ASchoolApp:
    """
    Lớp chính của ứng dụng, khởi tạo tất cả các thành phần và
    xây dựng giao diện người dùng bằng Gradio.
    """
    def __init__(self, config: AppConfig):
        logger.info("--- Bắt đầu khởi tạo các thành phần cốt lõi ---")
        self.config = config
        
        # --- Khởi tạo theo nguyên tắc Dependency Injection ---
        # 1. Khởi tạo các thành phần cấp thấp (không phụ thuộc vào ai)
        self.db_manager = DatabaseManager(config.DB_PATH)
        self.llm_provider = LLMProvider(config.OPENROUTER_API_KEY, config.API_URL, config.DEFAULT_MODEL, config.DATA_DIR)
        self.vector_store = VectorStore(config.EMBEDDING_MODEL)
        
        # 2. Khởi tạo các Agents tối ưu
        self.analysis_agent = AnalysisAgent()
        # Khởi tạo Professional processors (thay thế các agent riêng lẻ)
        self.summarization_agent = SummarizationAgentOptimized(self.llm_provider)
        self.quiz_agent = ContextAwareQuizAgent(self.llm_provider)
        self.pdf_processor = ProfessionalPDFProcessor(self.llm_provider)
        self.card_manager = CardManager(config.AGENT_CARDS_DIR, self.llm_provider)

        # 3. Khởi tạo các Services (phụ thuộc vào các thành phần trên)
        self.document_service = DocumentService(config, self.db_manager, self.vector_store, self.pdf_processor)
        self.analysis_service = AnalysisService(self.analysis_agent)
        self.quiz_service = QuizService(config, self.db_manager, self.vector_store, self.quiz_agent)
        
        # --- Trạng thái của ứng dụng ---
        self.active_document_id = None # Lưu ID của tài liệu đang được chọn
        logger.info("--- ✅ Tất cả các thành phần đã được khởi tạo thành công ---")

    # --- Các hàm xử lý sự kiện cho Tab "Hỏi đáp PDF" ---

    def _handle_pdf_upload(self, files: List[gr.File]) -> gr.Dropdown:
        """Xử lý sự kiện upload file PDF với loading state."""
        if not files:
            return gr.Dropdown(choices=[], value=None)
        
        logger.info(f"Nhận được {len(files)} file để xử lý.")
        try:
            for file in files:
                file_path = Path(file.name)
                logger.info(f"Đang xử lý file: {file_path.name}")
                self.document_service.process_document(file_path)
                logger.info(f"✅ Hoàn thành xử lý file: {file_path.name}")
            
            # Tải lại danh sách tài liệu
            return self._update_document_dropdown()
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý file PDF: {e}")
            return gr.Dropdown(choices=[], value=None)

    def _update_document_dropdown(self) -> gr.Dropdown:
        """Cập nhật danh sách tài liệu trong dropdown."""
        docs = self.document_service.get_all_documents()
        choices = [(f"{doc['filename']} ({doc['status']})", doc['id']) for doc in docs]
        return gr.Dropdown(choices=choices, label="Chọn tài liệu đã xử lý", interactive=True)

    def _handle_doc_selection(self, doc_id: str) -> Tuple[gr.Dropdown, gr.Dropdown]:
        """Xử lý khi người dùng chọn một tài liệu từ dropdown."""
        self.active_document_id = doc_id
        if not doc_id:
            empty_dropdown = gr.Dropdown(choices=[], value=None)
            return empty_dropdown, empty_dropdown
        
        doc = self.document_service.get_document(doc_id)
        if not doc:
            empty_dropdown = gr.Dropdown(choices=[], value=None)
            return empty_dropdown, empty_dropdown
        
        # Cập nhật dropdown chương
        chapters = doc.get('chapters', [])
        if chapters is None:
            chapters = []
        chapter_choices = ["Toàn bộ tài liệu"] + [f"{ch.get('number', 'N/A')}. {ch.get('title', 'Không có tiêu đề')}" for ch in chapters]
        
        chapter_dropdown = gr.Dropdown(choices=chapter_choices, value="Toàn bộ tài liệu", interactive=True)
        quiz_dropdown = gr.Dropdown(choices=chapter_choices, value="Toàn bộ tài liệu", interactive=True)
        
        return chapter_dropdown, quiz_dropdown

    def _handle_chat(self, message: str, history: List, chapter_scope: str):
        """Xử lý tin nhắn chat từ người dùng."""
        if not self.active_document_id:
            history.append([message, "Lỗi: Vui lòng chọn một tài liệu trước khi bắt đầu chat."])
            return history
        
        # --- Logic Agentic: Nhận biết ý định tóm tắt ---
        if "tóm tắt" in message.lower() or "summarize" in message.lower():
            history.append([message, "Đã nhận diện yêu cầu tóm tắt. Đang gọi Summarization Agent..."])
            
            chapter_filter = None
            if chapter_scope != "Toàn bộ tài liệu":
                 # Trích xuất tiêu đề chương để lọc
                match = re.search(r'\d+\.\s*(.*)', chapter_scope)
                if match:
                    chapter_filter = match.group(1).strip()

            context_to_summarize = self.document_service.get_context_for_query(
                doc_id=self.active_document_id,
                query=message, # Query vẫn có thể chứa thông tin về nội dung cần tóm tắt
                chapter_filter=chapter_filter
            )
            
            summary = self.document_processor.summarizer.summarize(context_to_summarize)
            history[-1][1] = summary or "Không thể tạo tóm tắt."
            return history

        # Nếu không phải tóm tắt, tiến hành RAG
        chapter_filter = None
        if chapter_scope != "Toàn bộ tài liệu":
            match = re.search(r'\d+\.\s*(.*)', chapter_scope)
            if match:
                chapter_filter = match.group(1).strip()
                
        context = self.document_service.get_context_for_query(self.active_document_id, message, chapter_filter)

        system_prompt = "You are a helpful AI assistant. Answer the user's question based ONLY on the provided context."
        user_prompt = f"CONTEXT:\n---\n{context}\n---\n\nQUESTION: {message}"
        
        response = self.llm_provider.generate(system_prompt, user_prompt)
        history.append([message, response])
        return history

    # --- Các hàm xử lý cho Tab "Tạo Câu hỏi" ---
    
    def _handle_quiz_generation(self, num_q: int, difficulty: str, scope: str) -> str:
        if not self.active_document_id:
            return "Lỗi: Vui lòng chọn một tài liệu ở tab 'Hỏi đáp PDF' trước."
        
        try:
            logger.info(f"🎯 Bắt đầu tạo quiz: {num_q} câu hỏi, độ khó {difficulty}, phạm vi {scope}")
            
            if scope == "Toàn bộ tài liệu":
                # Generate quiz for entire document
                logger.info("📄 Tạo quiz cho toàn bộ tài liệu...")
                
                # Get document content
                doc = self.document_service.get_document(self.active_document_id)
                if not doc:
                    return "Lỗi: Không tìm thấy tài liệu."
                
                # Get all chunks for the document
                all_chunks = self.vector_store.get_chunks_for_document(self.active_document_id)
                if not all_chunks:
                    return "Lỗi: Không tìm thấy nội dung tài liệu."
                
                # Concatenate all content
                full_content = "\n\n".join([chunk.content for chunk in all_chunks])
                
                # Generate quiz using the old method
                result = self.quiz_service.generate_quiz(
                    document_text=full_content,
                    num_questions=int(num_q),
                    difficulty=difficulty,
                    scope=scope
                )
            else:
                # Generate quiz for specific chapter
                logger.info(f"📚 Tạo quiz cho chương: {scope}")
                
                match = re.search(r'\d+\.\s*(.*)', scope)
                if match:
                    chapter_title = match.group(1).strip()
                    result = self.quiz_service.generate_quiz_for_chapter(
                        doc_id=self.active_document_id,
                        chapter_title=chapter_title,
                        num_questions=int(num_q),
                        difficulty=difficulty,
                        document_service=self.document_service
                    )
                else:
                    return "Lỗi: Không thể xác định tên chương."
            
            logger.info("✅ Hoàn thành tạo quiz")
            return result if result else "Không thể tạo quiz. Vui lòng thử lại."
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo quiz: {e}", exc_info=True)
            return f"Lỗi khi tạo quiz: {str(e)}"
        
    # --- Các hàm xử lý cho Tab "Phân tích CSV" ---
    
    def _handle_csv_analysis(self, file: gr.File) -> Tuple[gr.Textbox, gr.Plot, gr.Plot]:
        if not file:
            return "Vui lòng tải lên một file CSV.", None, None
        
        results = self.analysis_service.analyze_csv(file.name)
        if "error" in results:
            return results["error"], None, None
            
        # Định dạng kết quả thống kê để hiển thị
        stats_str = json.dumps(results.get("basic_stats", {}), indent=2, ensure_ascii=False)
        
        # Get Plotly chart JSON data
        visualizations = results.get("visualizations", {})
        dist_plot = visualizations.get("distribution")
        corr_plot = visualizations.get("correlation")
        
        return stats_str, dist_plot, corr_plot

    # --- Các hàm xử lý cho Tab "Quiz History" ---
    
    def _load_quiz_history(self) -> Tuple[gr.Dropdown, str]:
        """Load quiz history and update dropdown."""
        try:
            quizzes = self.quiz_service.get_quiz_history()
            if not quizzes:
                return gr.Dropdown(choices=[], value=None), "Chưa có quiz nào được tạo."
            
            # Create choices for dropdown
            choices = []
            for quiz in quizzes:
                title = f"{quiz.get('filename', 'Unknown')} - {quiz.get('scope', 'Unknown')} ({quiz.get('created_at', 'Unknown date')})"
                choices.append((title, quiz['id']))
            
            return gr.Dropdown(choices=choices, value=None), "Đã tải danh sách quiz."
            
        except Exception as e:
            logger.error(f"Error loading quiz history: {e}")
            return gr.Dropdown(choices=[], value=None), f"Lỗi khi tải lịch sử quiz: {str(e)}"

    def _handle_quiz_selection(self, quiz_id: str) -> str:
        """Handle quiz selection and display content."""
        if not quiz_id:
            return "Vui lòng chọn một quiz để xem."
        
        try:
            quizzes = self.quiz_service.get_quiz_history()
            selected_quiz = next((q for q in quizzes if q['id'] == quiz_id), None)
            
            if not selected_quiz:
                return "Không tìm thấy quiz được chọn."
            
            return selected_quiz.get('quiz_content', 'Nội dung quiz không có sẵn.')
            
        except Exception as e:
            logger.error(f"Error displaying quiz: {e}")
            return f"Lỗi khi hiển thị quiz: {str(e)}"

    # --- Xây dựng giao diện ---
    
    def launch(self):
        """Xây dựng và khởi chạy giao diện Gradio."""
        with gr.Blocks(theme=gr.themes.Soft(), title="A2A School Platform") as interface:
            gr.Markdown("# 🎓 A2A School - Nền tảng AI Giáo dục Thông minh")
            
            with gr.Tabs():
                # --- TAB 1: HỎI ĐÁP PDF ---
                with gr.Tab("📄 Hỏi đáp & Tóm tắt PDF"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            pdf_upload_btn = gr.Files(label="1. Tải lên file PDF", file_types=[".pdf"])
                            doc_dropdown = gr.Dropdown(label="2. Chọn tài liệu đã xử lý", interactive=False)
                            chapter_dropdown = gr.Dropdown(label="3. Chọn phạm vi (Chương)", interactive=False)
                        with gr.Column(scale=2):
                            chatbot = gr.Chatbot(label="Hỏi đáp với AI", height=500, bubble_full_width=True)
                            msg_box = gr.Textbox(label="Nhập câu hỏi hoặc yêu cầu tóm tắt...", show_label=False, placeholder="Ví dụ: Python là gì? hoặc tóm tắt nội dung chính")
                            msg_box.submit(self._handle_chat, [msg_box, chatbot, chapter_dropdown], chatbot)

                # --- TAB 2: TẠO CÂU HỎI ---
                with gr.Tab("📝 Tạo bộ đề kiểm tra"):
                    gr.Markdown("Sử dụng tài liệu đã chọn ở tab 'Hỏi đáp PDF' để tạo câu hỏi.")
                    with gr.Row():
                        with gr.Column():
                            quiz_num_questions = gr.Slider(1, 20, value=5, step=1, label="Số lượng câu hỏi")
                            quiz_difficulty = gr.Radio(["easy", "medium", "hard"], label="Độ khó", value="medium")
                            # Dropdown cho phạm vi sẽ được copy từ tab chat
                            quiz_scope_dropdown = gr.Dropdown(label="Phạm vi (Chương)", choices=["Toàn bộ tài liệu"], value="Toàn bộ tài liệu", interactive=True)
                            quiz_generate_btn = gr.Button("🚀 Tạo bộ đề", variant="primary")
                        with gr.Column(scale=2):
                            quiz_output_box = gr.Markdown(label="Kết quả bộ đề")
                
                # --- TAB 3: PHÂN TÍCH CSV ---
                with gr.Tab("📊 Phân tích CSV"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            csv_upload = gr.File(label="Tải lên file CSV", file_types=[".csv"])
                            csv_analyze_btn = gr.Button("🔍 Phân tích ngay", variant="primary")
                        with gr.Column(scale=2):
                            gr.Markdown("### Kết quả Phân tích")
                            csv_stats_output = gr.Textbox(label="Thống kê cơ bản", lines=10, interactive=False)
                            gr.Markdown("### Biểu đồ trực quan tương tác")
                            csv_dist_plot = gr.Plot(label="Biểu đồ phân phối")
                            csv_corr_plot = gr.Plot(label="Biểu đồ tương quan")

                # --- TAB 4: QUIZ HISTORY ---
                with gr.Tab("📚 Lịch sử Quiz"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Chọn Quiz để xem")
                            quiz_history_dropdown = gr.Dropdown(
                                label="Danh sách Quiz", 
                                choices=[], 
                                value=None,
                                interactive=True
                            )
                            quiz_refresh_btn = gr.Button("🔄 Làm mới danh sách", variant="secondary")
                            quiz_status = gr.Textbox(label="Trạng thái", value="Chưa tải", interactive=False)
                        with gr.Column(scale=2):
                            gr.Markdown("### Nội dung Quiz")
                            quiz_content_display = gr.Markdown(
                                label="Nội dung Quiz", 
                                value="Vui lòng chọn một quiz để xem nội dung.",
                                show_label=False
                            )

                # --- TAB 5: AGENT CARDS (DEMO) ---
                with gr.Tab("🤖 Agent Cards"):
                    gr.Markdown("Thực thi các tác vụ chuyên biệt bằng AI Agent Cards (Demo).")
                    # ... Giao diện cho Agent Cards có thể được thêm ở đây ...


            # Logic kết nối các thành phần giao diện
            pdf_upload_btn.upload(self._handle_pdf_upload, pdf_upload_btn, doc_dropdown)
            doc_dropdown.change(self._handle_doc_selection, doc_dropdown, [chapter_dropdown, quiz_scope_dropdown])
            
            # Quiz generation with loading state
            quiz_generate_btn.click(
                self._handle_quiz_generation,
                [quiz_num_questions, quiz_difficulty, quiz_scope_dropdown],
                quiz_output_box
            )
            
            # CSV analysis
            csv_analyze_btn.click(
                self._handle_csv_analysis,
                csv_upload,
                [csv_stats_output, csv_dist_plot, csv_corr_plot]
            )
            
            # Quiz History handlers
            quiz_refresh_btn.click(
                self._load_quiz_history,
                outputs=[quiz_history_dropdown, quiz_status]
            )
            
            quiz_history_dropdown.change(
                self._handle_quiz_selection,
                quiz_history_dropdown,
                quiz_content_display
            )
            
        interface.launch(
            server_name=self.config.GRADIO_SERVER_NAME,
            server_port=self.config.GRADIO_SERVER_PORT,
            share=self.config.GRADIO_SHARE,
            debug=True
        )

