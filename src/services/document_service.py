# src/services/document_service.py
# Service trung tâm cho mọi thứ liên quan đến tài liệu: xử lý, chunking, indexing, và RAG.

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
import re

# Sử dụng relative import để giữ cấu trúc module gọn gàng và dễ bảo trì
from ..core.document_reader_optimized import OptimizedChapterDetector
from ..core.professional_pdf_processor import ProfessionalPDFProcessor
from ..core.vector_store import VectorStore
from ..core.data_structures import AgenticChunk, ChunkType, ChapterInfo
from ..db.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Là "bộ não" quản lý vòng đời của một tài liệu, từ lúc upload cho đến khi
    sẵn sàng để hỏi đáp. Service này điều phối ProfessionalPDFProcessor và VectorStore.
    """
    def __init__(
        self,
        config, # Đối tượng config chứa các đường dẫn và cài đặt
        db_manager: DatabaseManager,
        vector_store: VectorStore,
        pdf_processor: ProfessionalPDFProcessor = None
    ):
        # Áp dụng Dependency Injection: Các thành phần phụ thuộc được truyền vào từ bên ngoài.
        # Điều này giúp code linh hoạt, dễ test và dễ thay thế các thành phần.
        self.config = config
        self.db_manager = db_manager
        self.vector_store = vector_store
        
        # Khởi tạo các thành phần phụ thuộc mà service này quản lý trực tiếp
        if pdf_processor:
            self.pdf_processor = pdf_processor
        else:
            # Fallback to professional PDF processor
            from ..core.llm_provider import LLMProvider
            fallback_llm = LLMProvider()
            self.pdf_processor = ProfessionalPDFProcessor(fallback_llm)

        # Trạng thái trong bộ nhớ để truy cập nhanh, tránh gọi DB liên tục cho các tác vụ lặp lại
        self.processed_docs_cache: Dict[str, Dict] = {} 
        logger.info("✅ DocumentService đã được khởi tạo.")

    def _convert_bookmarks_to_chapters(self, bookmarks: List[Dict]) -> List[Dict]:
        """
        Convert PDF bookmarks to chapter format expected by chunking logic.
        
        Args:
            bookmarks: List of bookmark dictionaries from ProfessionalPDFProcessor
            
        Returns:
            List of chapter dictionaries with number, title, and level
        """
        chapters = []
        for i, bookmark in enumerate(bookmarks):
            chapters.append({
                'number': i + 1,
                'title': bookmark.get('title', f'Chapter {i + 1}'),
                'level': bookmark.get('level', 1),
                'page': bookmark.get('page', 1),
                'start_position': 0  # Will be calculated during chunking
            })
        return chapters

    def process_document(self, file_path: Path) -> Optional[str]:
        """
        Hàm chính để xử lý một file PDF mới một cách toàn diện.
        Bao gồm đọc, chunking, và xây dựng vector index.
        Đây là một tác vụ có thể tốn thời gian, rất phù hợp để chạy trong nền (background job).
        """
        # Tạo một ID duy nhất cho tài liệu
        doc_id = str(uuid.uuid4())
        filename = file_path.name
        
        try:
            logger.info(f"Bắt đầu xử lý tài liệu '{filename}' với ID: {doc_id}")
            # 1. Ghi nhận vào DB là đang xử lý để UI có thể theo dõi trạng thái
            self.db_manager.add_or_update_document(doc_id, filename, str(file_path), status='processing')
            
            # 2. Extract PDF structure using professional processor
            pdf_structure = self.pdf_processor.extract_pdf_structure(str(file_path))
            if 'error' in pdf_structure:
                raise ValueError(f"PDF processing failed: {pdf_structure['error']}")
            
            # Extract content for chunking
            full_text = pdf_structure.get('full_text', '')
            if not full_text:
                raise ValueError("Không thể trích xuất nội dung từ tài liệu.")
            
            # Create structured content compatible with existing chunking logic
            structured_content = {
                'full_text': full_text,
                'chapters': self._convert_bookmarks_to_chapters(pdf_structure.get('bookmarks', [])),
                'page_mapped_content': pdf_structure.get('page_mapped_content', {}),
                'structure_analysis': pdf_structure.get('structure_analysis', {})
            }

            # 3. Chia nhỏ văn bản một cách thông minh (Agentic Chunking) với page mapping
            chunks = self._create_agentic_chunks_with_pages(structured_content, filename, pdf_structure, doc_id)
            
            # 4. Xây dựng Vector Index cho tài liệu này thông qua VectorStore
            self.vector_store.build_index_for_doc(doc_id, chunks)
            
            # 5. Cập nhật trạng thái "hoàn thành" và thông tin chi tiết vào DB
            doc_info = {
                "id": doc_id,
                "filename": filename,
                "status": "completed",
                "chunks_count": len(chunks),
                "chapters": structured_content['chapters']
            }
            self.db_manager.add_or_update_document(
                doc_id, filename, str(file_path), 'completed', 
                len(chunks), structured_content['chapters']
            )
            
            # Lưu thông tin vào cache trong bộ nhớ để truy cập nhanh ở các lần gọi sau
            self.processed_docs_cache[doc_id] = doc_info
            
            logger.info(f"🎉 Xử lý thành công tài liệu '{filename}' (ID: {doc_id})")
            return doc_id

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi xử lý tài liệu '{filename}': {e}", exc_info=True)
            # Cập nhật trạng thái 'thất bại' vào DB
            self.db_manager.add_or_update_document(doc_id, filename, str(file_path), status='failed')
            return None
    def _create_agentic_chunks_with_pages(
        self, 
        structured_content: Dict, 
        filename: str, 
        pdf_structure: Dict,
        doc_id: str
    ) -> List[AgenticChunk]:
        """
        Create agentic chunks with accurate page-to-chapter mapping.
        
        Args:
            structured_content: Content structure with chapters and full text
            filename: Name of the source file
            pdf_structure: Complete PDF structure from professional processor
            doc_id: Document ID for chunk association
            
        Returns:
            List of AgenticChunk objects with accurate chapter associations
        """
        logger.info(f"🧩 Creating page-mapped Agentic Chunks for file '{filename}'...")
        
        text = structured_content['full_text']
        chapters = structured_content['chapters']
        page_mapped_content = structured_content.get('page_mapped_content', {})
        structure_analysis = pdf_structure.get('structure_analysis', {})
        
        chunks_list = []
        
        # If we have page-mapped content and structure analysis, use page-based chunking
        if page_mapped_content and structure_analysis.get('page_ranges'):
            chunks_list = self._create_page_based_chunks(
                page_mapped_content, 
                structure_analysis['page_ranges'], 
                chapters,
                filename,
                doc_id
            )
        else:
            # Fallback to text-based chunking
            chunks_list = self._create_text_based_chunks(text, chapters, filename, doc_id)
        
        logger.info(f"✅ Created {len(chunks_list)} agentic chunks with chapter mapping")
        return chunks_list

    def _create_page_based_chunks(self, page_mapped_content: Dict, page_ranges: Dict, chapters: List[Dict], filename: str, doc_id: str) -> List[AgenticChunk]:
        """
        Create chunks based on page-mapped content with proper chapter association.
        
        Args:
            page_mapped_content: Dictionary mapping page numbers to content
            page_ranges: Page ranges for different sections
            chapters: List of chapter information
            filename: Source filename
            doc_id: Document ID
            
        Returns:
            List of AgenticChunk objects with accurate page and chapter mapping
        """
        logger.info(f"📄 Creating page-based chunks for '{filename}'...")
        
        agentic_chunks = []
        chunk_position = 0
        
        # Process each page's content
        for page_num, page_content in page_mapped_content.items():
            if not page_content or not page_content.strip():
                continue
                
            # Find which chapter this page belongs to
            current_chapter_info = None
            for chapter in chapters:
                if chapter.get('page', 1) <= int(page_num):
                    current_chapter_info = ChapterInfo(
                        number=chapter.get('number'),
                        title=chapter.get('title')
                    )
            
            # Split page content into smaller chunks if needed
            page_text = page_content.strip()
            if len(page_text) > 800:  # Split large pages
                sentences = re.split(r'(?<=[.!?])\s+', page_text)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > 800 and current_chunk:
                        # Create chunk from accumulated sentences
                        agentic_chunks.append(AgenticChunk(
                            content=current_chunk.strip(),
                            chunk_type=ChunkType.PARAGRAPH,
                            source_file=filename,
                            position_in_document=chunk_position,
                            chapter_info=current_chapter_info,
                            page_number=int(page_num)
                        ))
                        chunk_position += 1
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence
                
                # Add remaining content
                if current_chunk.strip():
                    agentic_chunks.append(AgenticChunk(
                        content=current_chunk.strip(),
                        chunk_type=ChunkType.PARAGRAPH,
                        source_file=filename,
                        position_in_document=chunk_position,
                        chapter_info=current_chapter_info,
                        page_number=int(page_num)
                    ))
                    chunk_position += 1
            else:
                # Small page, use as single chunk
                agentic_chunks.append(AgenticChunk(
                    content=page_text,
                    chunk_type=ChunkType.PARAGRAPH,
                    source_file=filename,
                    position_in_document=chunk_position,
                    chapter_info=current_chapter_info,
                    page_number=int(page_num)
                ))
                chunk_position += 1
        
        logger.info(f"✅ Created {len(agentic_chunks)} page-based chunks")
        return agentic_chunks

    def _create_text_based_chunks(self, text: str, chapters: List[Dict], filename: str, doc_id: str) -> List[AgenticChunk]:
        """
        Tạo các đối tượng AgenticChunk từ nội dung đã được phân tích cấu trúc.
        Mỗi chunk không chỉ là text mà còn chứa metadata quan trọng.
        """
        logger.info(f"🧩 Đang tạo Agentic Chunks cho file '{filename}'...")
        
        # Chia văn bản thành các câu để xử lý, giúp chunk không bị cắt giữa câu
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        
        chunks_text = []
        current_chunk_content = ""
        chunk_size_target = 400 # Kích thước mục tiêu cho mỗi chunk (tính bằng ký tự)

        for sentence in sentences:
            if len(current_chunk_content) + len(sentence) > chunk_size_target and current_chunk_content:
                chunks_text.append(current_chunk_content.strip())
                current_chunk_content = sentence
            else:
                current_chunk_content += " " + sentence

        if current_chunk_content:
            chunks_text.append(current_chunk_content.strip())

        agentic_chunks = []
        
        # Use page-based chapter mapping since we don't have line numbers
        page_to_chapter = {}
        for ch in chapters:
            page_to_chapter[ch.get('page', 1)] = ch
        
        # Sort pages for lookup
        sorted_pages = sorted(page_to_chapter.keys())

        for i, chunk_text in enumerate(chunks_text):
            current_chapter_info = None
            
            # For text-based chunks, we'll assign them to first chapter as fallback
            if chapters:
                # Find the most appropriate chapter (first available or by position)
                if len(chunks_text) > 1:
                    # Distribute chunks across chapters proportionally
                    chapter_index = min(int(i * len(chapters) / len(chunks_text)), len(chapters) - 1)
                    chapter = chapters[chapter_index]
                else:
                    # Single chunk gets first chapter
                    chapter = chapters[0]
                
                current_chapter_info = ChapterInfo(
                    number=chapter.get('number'),
                    title=chapter.get('title')
                )
            
            agentic_chunks.append(AgenticChunk(
                content=chunk_text,
                chunk_type=ChunkType.PARAGRAPH, # Mặc định là đoạn văn, có thể mở rộng để phân loại
                source_file=filename,
                position_in_document=i,
                chapter_info=current_chapter_info
            ))
        
        logger.info(f"Đã tạo {len(agentic_chunks)} chunks.")
        return agentic_chunks

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Lấy thông tin một tài liệu (ưu tiên từ cache, sau đó mới truy vấn DB)."""
        if doc_id in self.processed_docs_cache:
            return self.processed_docs_cache[doc_id]
        
        doc_from_db = self.db_manager.get_document(doc_id)
        if doc_from_db:
            self.processed_docs_cache[doc_id] = doc_from_db
        return doc_from_db

    def get_all_documents(self) -> List[Dict]:
        """Lấy danh sách tóm tắt của tất cả tài liệu."""
        return self.db_manager.get_all_documents()

    def get_context_for_query(self, doc_id: str, query: str, chapter_filter: Optional[str] = None) -> str:
        """
        Thực hiện RAG: tìm kiếm và trả về ngữ cảnh cho một câu hỏi cụ thể,
        có thể lọc theo chương.
        """
        logger.info(f"Đang thực hiện RAG cho doc_id '{doc_id}' với query: '{query[:50] if query else 'None'}...' và bộ lọc chương: '{chapter_filter}'")
        relevant_chunks = self.vector_store.search(query, top_k=5, chapter_filter=chapter_filter, doc_filter=doc_id)
        
        if not relevant_chunks:
            return "Không tìm thấy thông tin liên quan trong tài liệu cho phạm vi đã chọn."

        # Nối các chunk lại thành một ngữ cảnh duy nhất để gửi cho LLM
        context = "\n\n---\n\n".join([
            f"[Trích từ chương: {chunk.chapter_info.title if chunk.chapter_info else 'Không rõ'}]\n{chunk.content}"
            for chunk in relevant_chunks
        ])
        return context

