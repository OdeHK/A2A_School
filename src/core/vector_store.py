# src/core/vector_store.py
# Chịu trách nhiệm tạo vector embedding và quản lý chỉ mục tìm kiếm FAISS.

import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
from .data_structures import AgenticChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Quản lý việc tạo embedding và tìm kiếm tương đồng với FAISS.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.chunks: List[AgenticChunk] = [] # Giữ tham chiếu đến các chunk gốc
        self._load_model()

    def _load_model(self):
        """Tải model embedding. Việc này chỉ thực hiện một lần."""
        try:
            if self.embedding_model is None:
                logger.info(f"💾 Đang tải model embedding: '{self.model_name}'... (có thể mất vài phút)")
                # trust_remote_code=True cần thiết cho một số model trên Hugging Face
                self.embedding_model = SentenceTransformer(self.model_name, trust_remote_code=True)
                logger.info("✅ Model embedding đã tải xong.")
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi tải model embedding: {e}", exc_info=True)
            raise

    def build_index(self, chunks: List[AgenticChunk]):
        """
        Xây dựng hoặc cập nhật chỉ mục FAISS từ một danh sách các chunk.
        """
        if not chunks:
            return
        
        logger.info(f"🛠️ Bắt đầu xây dựng/cập nhật Vector Index cho {len(chunks)} chunks...")
        self.chunks.extend(chunks)
        
        # Chỉ tạo embedding cho các chunk mới
        contents_to_embed = [chunk.content for chunk in chunks]
        new_embeddings = self.embedding_model.encode(
            contents_to_embed,
            show_progress_bar=True,
            batch_size=32 # Xử lý theo lô để hiệu quả hơn
        )

        # Gán lại embedding vào đối tượng chunk
        for chunk, embedding in zip(chunks, new_embeddings):
            chunk.embedding = embedding

        # Cập nhật chỉ mục FAISS
        if self.index is None:
            dimension = new_embeddings.shape[1]
            # IndexFlatIP hiệu quả cho tìm kiếm cosine similarity sau khi đã chuẩn hóa vector
            self.index = faiss.IndexFlatIP(dimension)
        
        # Chuẩn hóa vector (L2 normalization) để cosine similarity tương đương với inner product
        faiss.normalize_L2(new_embeddings)
        self.index.add(new_embeddings.astype('float32'))
        
        logger.info(f"✅ Vector Index được cập nhật. Tổng số vectors: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 5, chapter_filter: Optional[str] = None, doc_filter: Optional[str] = None) -> List[AgenticChunk]:
        """
        Tìm kiếm các chunk liên quan nhất đến câu hỏi.
        Có hỗ trợ lọc theo chương.
        """
        if self.index is None or not self.chunks:
            logger.warning(f"VectorStore chưa được khởi tạo hoặc không có chunks. Index: {self.index is not None}, Chunks: {len(self.chunks) if self.chunks else 0}")
            return []

        logger.info(f"🔍 Thực hiện tìm kiếm vector cho câu hỏi: '{query[:50]}...' trong {len(self.chunks)} chunks")
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2) # Lấy nhiều hơn để lọc

        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Ngưỡng tương đồng để loại bỏ kết quả nhiễu (giảm xuống để tìm được nhiều kết quả hơn)
            if score < 0.3: 
                continue

            chunk = self.chunks[idx]
            
            # Kiểm tra bộ lọc tài liệu trước (nếu có)
            if doc_filter:
                # Tạm thời bỏ qua filter theo doc_id vì cần map với source_file
                # Sẽ implement sau khi có mapping logic
                pass
            
            # Áp dụng bộ lọc chương nếu có
            if chapter_filter:
                if chunk.chapter_info and chapter_filter.lower() in chunk.chapter_info.title.lower():
                    results.append(chunk)
            else:
                # Nếu không có filter chương, thêm chunk vào kết quả
                results.append(chunk)
        
        logger.info(f"Tìm thấy {len(results)} chunks liên quan (sau khi lọc).")
        return results[:top_k]

    def build_index_for_doc(self, doc_id: str, chunks: List[AgenticChunk]) -> None:
        """
        Xây dựng index cho một tài liệu cụ thể.
        """
        logger.info(f"Đang xây dựng vector index cho tài liệu {doc_id} với {len(chunks)} chunks...")
        
        if not chunks:
            logger.warning(f"Không có chunks nào để index cho tài liệu {doc_id}")
            return
        
        # Thêm chunks vào danh sách hiện tại
        self.chunks.extend(chunks)
        
        # Tạo embeddings cho chunks mới
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Chuẩn hóa embeddings
        faiss.normalize_L2(embeddings)
        
        # Thêm vào FAISS index
        if self.index is None:
            # Tạo index mới nếu chưa có
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        
        self.index.add(embeddings.astype('float32'))
        
        # Cập nhật embeddings cho chunks
        start_idx = len(self.chunks) - len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        
        logger.info(f"✅ Đã thêm {len(chunks)} chunks vào vector store. Tổng: {len(self.chunks)} chunks.")

    def get_chunks_for_document(self, doc_id: str) -> List[AgenticChunk]:
        """
        Lấy tất cả chunks thuộc về một document cụ thể.
        """
        # Filter chunks by source file (assuming filename contains doc info)
        # This is a simple implementation - in production, you might want to store doc_id in chunks
        doc_chunks = []
        for chunk in self.chunks:
            # Check if chunk belongs to this document
            # This is a simple heuristic - you might want to improve this
            if hasattr(chunk, 'source_file') and chunk.source_file:
                doc_chunks.append(chunk)
        
        logger.info(f"Retrieved {len(doc_chunks)} chunks for document {doc_id}")
        return doc_chunks

    def get_chunks_by_doc_id(self, doc_id: str) -> List[AgenticChunk]:
        """
        Get all chunks belonging to a specific document ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of AgenticChunk objects for the document
        """
        doc_chunks = []
        for chunk in self.chunks:
            # Check if chunk has source_info with document reference
            if (hasattr(chunk, 'source_info') and 
                chunk.source_info and 
                'doc_id' in chunk.source_info and 
                chunk.source_info['doc_id'] == doc_id):
                doc_chunks.append(chunk)
            # Fallback: check if chunk has any reference to this doc_id
            elif hasattr(chunk, 'id') and doc_id in str(chunk.id):
                doc_chunks.append(chunk)
        
        logger.info(f"📄 Retrieved {len(doc_chunks)} chunks for document {doc_id}")
        return doc_chunks

    def get_chunks_by_chapter(self, doc_id: str, chapter_title: str) -> List[AgenticChunk]:
        """
        Get all chunks belonging to a specific chapter in a document.
        
        Args:
            doc_id: Document identifier
            chapter_title: Title of the chapter
            
        Returns:
            List of AgenticChunk objects for the specified chapter
        """
        chapter_chunks = []
        doc_chunks = self.get_chunks_by_doc_id(doc_id)
        
        for chunk in doc_chunks:
            # Check if chunk belongs to the specified chapter
            if (chunk.chapter_info and 
                chunk.chapter_info.title and 
                chunk.chapter_info.title.strip().lower() == chapter_title.strip().lower()):
                chapter_chunks.append(chunk)
        
        logger.info(f"📚 Retrieved {len(chapter_chunks)} chunks for chapter '{chapter_title}' in document {doc_id}")
        return chapter_chunks

    def get_document_chapters(self, doc_id: str) -> List[str]:
        """
        Get list of unique chapter titles in a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of chapter titles
        """
        chapters = set()
        doc_chunks = self.get_chunks_by_doc_id(doc_id)
        
        for chunk in doc_chunks:
            if chunk.chapter_info and chunk.chapter_info.title:
                chapters.add(chunk.chapter_info.title)
        
        chapter_list = sorted(list(chapters))
        logger.info(f"📑 Found {len(chapter_list)} chapters in document {doc_id}")
        return chapter_list
