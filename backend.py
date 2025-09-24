# backend.py
import os
import json
import hashlib
import pickle
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import nest_asyncio
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from llama_index.core import (
    Document as LlamaDocument,
    DocumentSummaryIndex,
    get_response_synthesizer,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from sentence_transformers import SentenceTransformer

from config import constants
from config.settings import settings
from utils.logging import logger
from template import SYSTEM_MSG, USER_MSG

# --- global setup ---

embed_model = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

llama3 = Ollama(model="gemma3:4b", request_timeout=300)
Settings.llm = llama3
Settings.embed_model = embed_model


# ===================== PDF Utils =====================
def extract_pages_from_pdf(pdf_path: str, page_numbers: List[int]) -> str:
    """
    Cắt các trang chỉ định từ PDF và trả về nội dung text.
    Args:
        pdf_path: đường dẫn file PDF
        page_numbers: danh sách số trang (0-based)
    Returns:
        str: text gộp từ các trang
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in page_numbers:
            if 0 <= page_num < len(doc):
                text += doc[page_num].get_text("text") + "\n"
            else:
                logger.warning(f"Trang {page_num} không tồn tại trong {pdf_path}")
    return text


# ===================== Document Processor =====================
class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self, files: List) -> None:
        """Validate the total size of the uploaded files."""
        total_size = sum(os.path.getsize(f.name) for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE//1024//1024}MB limit")

    def process(self, files: List) -> List:
        """Process files with caching for subsequent queries"""
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()

        for file in files:
            try:
                with open(file.name, "rb") as f:
                    file_hash = self._generate_hash(f.read())
                cache_path = self.cache_dir / f"{file_hash}.pkl"

                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {file.name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {file.name}")
                    chunks = self._process_file(file)
                    self._save_to_cache(chunks, cache_path)

                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    def _process_file(self, file) -> List:
        """Convert file to markdown then split"""
        if not file.name.endswith(('.pdf', '.docx', '.txt', '.md')):
            logger.warning(f"Skipping unsupported file type: {file.name}")
            return []
        converter = DocumentConverter()
        markdown = converter.convert(file.name).document.export_to_markdown()
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)

    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, chunks: List, cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump({"timestamp": datetime.now().timestamp(), "chunks": chunks}, f)

    def _load_from_cache(self, cache_path: Path) -> List:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)


# ===================== Retriever Builder =====================
class RetrieverBuilder:
    def __init__(self):
        self.embeddings =  HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-multilingual-base",
    model_kwargs={"trust_remote_code": True}
)
    def build_hybrid_retriever(self, docs):
        try:
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=settings.CHROMA_DB_PATH
            )
            bm25 = BM25Retriever.from_documents(docs)
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS
            )
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise


# ===================== Quiz Crafter =====================

class QuizCrafter:
    def __init__(self, llm=None, embeddings=None):
        self.system = SYSTEM_MSG
        self.user = USER_MSG
        self.llm = llm or ChatOllama(model="gemma3:4b", temperature=0.7, top_k=80, top_p=0.9, seed=0, base_url="http://localhost:11434",num_ctx=8192)
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-multilingual-base",
            model_kwargs={"trust_remote_code": True}
        )
        self.documents = None
        self.index = None

    def load_docs(self, file_path: str):
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        self.documents = loader.load()
        return self.documents

    def load_text(self, text: str, metadata: dict = None):
        from langchain_core.documents import Document
        if metadata is None:
            metadata = {"source": "user_text"}
        self.documents = [Document(page_content=text, metadata=metadata)]
        return self.documents

    def split_docs(self, documents, chunk_size=700, chunk_overlap=20):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents=documents)

    def create_index(self):
        from langchain_community.vectorstores import FAISS
        if self.documents is None:
            raise ValueError("Bạn phải load_docs() hoặc load_text() trước khi tạo index.")
        docs_split = self.split_docs(self.documents)
        self.index = FAISS.from_documents(documents=docs_split, embedding=self.embeddings)
        return self.index

    def get_similar_docs(self, query: str, k: int = 2):
        if self.index is None:
            raise ValueError("Index chưa được tạo. Gọi create_index() trước.")
        return self.index.similarity_search(query=query, k=k)

    def load_chat_msg(self, topic: str):
        from langchain_core.messages import SystemMessage, HumanMessage
        self.create_index()
        if topic:
            query = self.get_similar_docs(topic, k=4)
        else:
            query = self.documents[:4] if len(self.documents) >= 4 else self.documents
        text = "".join([doc.page_content for doc in query])
        messages = [SystemMessage(content=self.system), HumanMessage(content=self.user.format(context=text))]
        return messages



    def get_questions(self, topic: str = ""):
        # 1️⃣ Tạo message cho LLM dựa trên topic
        messages = self.load_chat_msg(topic)
        
        # 2️⃣ Gọi LLM
        result = self.llm.invoke(messages)
        
        # 3️⃣ Chuẩn hóa output, loại bỏ markdown code block nếu có
        result_text = str(result.content).strip().rstrip()
        result_text = result_text.strip("```json\n").rstrip("```")
        
        # 4️⃣ Chuyển JSON string thành Python object
        try:
            questions = json.loads(result_text)
            
            # 5️⃣ Lưu ra file JSON để kiểm tra nếu cần
            with open("questions.json", "w", encoding="utf-8") as f:
                json.dump(questions, f, indent=4, ensure_ascii=False)
            
            return questions
        except json.JSONDecodeError:
            logger.error("Không parse được JSON từ LLM output")
            return None
    



# ===================== Summarizer =====================
def summarize_chapter_with_llamaindex(chapter_text: str, title: str):
    doc = LlamaDocument(text=chapter_text, doc_id=title)
    splitter = SentenceSplitter(chunk_size=1024)
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", use_async=True
    )
    doc_summary_index = DocumentSummaryIndex.from_documents(
        [doc],
        transformations=[splitter],
        response_synthesizer=response_synthesizer,
        show_progress=True,
        streaming=True
    )

    # Truy vấn tiếng Việt
    query_engine = doc_summary_index.as_query_engine()
    prompt = (
    f"Bạn là trợ lý học tập. "
    f"Hãy tóm tắt chi tiết nội dung của '{title}' bằng TIẾNG VIỆT.\n"
    "- Tóm tắt phải dài tối thiểu 5–7 câu, đầy đủ các ý chính.\n"
    "- Giữ nguyên các khái niệm, thuật ngữ chuyên môn quan trọng.\n"
    "- Nêu rõ các phần chính dưới dạng danh sách gạch đầu dòng.\n"
    "- Sử dụng văn phong dễ hiểu, phù hợp người mới học.\n"
    "- Kết thúc bằng một câu nêu ứng dụng hoặc ý nghĩa của nội dung.\n"
)




    summary = query_engine.query(
        prompt
    )
    summary_text = str(summary)

    return summary_text



