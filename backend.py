# backend.py
import os
import re
import json
import hashlib
import pickle
import logging
import random
import spacy
from nltk import FreqDist
from nltk.corpus import brown
import textdistance
from flashtext import KeywordProcessor
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
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
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

def normalize_code_field(code):
        """Chuyển None hoặc 'null' thành chuỗi rỗng để hiển thị đẹp"""
        if code is None:
            return ""
        if isinstance(code, str) and code.strip().lower() == "null":
            return ""
        return code

def clean_text(text: str) -> str:
    """Chuẩn hóa text: tách từ dính, loại bỏ URL, email, số điện thoại"""
    # tách chữ dính (cơ bản)
    text = re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', text)
    # loại bỏ số điện thoại, URL, email
    text = re.sub(r'\b\d{2,}\b', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

# ===================== Document Processor =====================
class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self, files: List[str]) -> None:
        total_size = sum(os.path.getsize(f) for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE//1024//1024}MB limit")

    def process(self, files: List[str]) -> List:
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()

        for file in files:
            try:
                with open(file, "rb") as f:
                    file_hash = self._generate_hash(f.read())
                cache_path = self.cache_dir / f"{file_hash}.pkl"

                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {file}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {file}")
                    chunks = self._process_file(file)
                    self._save_to_cache(chunks, cache_path)

                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                logger.error(f"Failed to process {file}: {str(e)}")
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")

        # ===== LƯU JSON METADATA =====
        metadata_list = [chunk.metadata for chunk in all_chunks]
        json_path = self.cache_dir / "all_chunks_metadata.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadata saved to {json_path}")


        return all_chunks

    def _process_file(self, file_path: str) -> List:
        ext = Path(file_path).suffix.lower()
        if ext not in ('.pdf', '.docx', '.txt', '.md'):
            logger.warning(f"Skipping unsupported file type: {file_path}")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=20,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        docs = []

        if ext == ".pdf":
            loader = PyMuPDF4LLMLoader(file_path,mode="page")
            raw_docs = loader.load()
            for d in raw_docs:
                page_chunks = text_splitter.split_documents([d])
                for chunk in page_chunks:
                    # copy toàn bộ metadata từ page gốc
                    chunk.metadata = {**d.metadata, **chunk.metadata}
                    # gắn thêm source và page cho chắc
                    chunk.metadata.update({
                        "source": file_path,
                        "page": d.metadata.get("page", None)
                    })
                docs.extend(page_chunks)

        else:
            converter = DocumentConverter()
            markdown = converter.convert(file_path).document.export_to_markdown()
            chunks = text_splitter.create_documents([markdown])
            for chunk in chunks:
                chunk.metadata.update({"source": file_path})
            docs.extend(chunks)

        return docs

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

# backend.py


class QuizCrafter:
    def __init__(self, documents=None, llm=None, embeddings=None):
        self.system = SYSTEM_MSG
        self.user = USER_MSG
        self.documents = documents or []
        self.llm = llm or ChatOllama(
            model="gemma3:4b",
            temperature=0.7,
            top_k=80,
            top_p=0.9,
            seed=0,
            base_url="http://localhost:11434",
            num_ctx=8192,
        )
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-multilingual-base",
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.index = None

        # 🔹 NLP toolkits
        self.nlp = spacy.load("en_core_web_sm")
        self.fdist = FreqDist(brown.words())
        
        self.levenshtein_similarity = textdistance.levenshtein.normalized_similarity
        self.levenshtein_distance = textdistance.levenshtein.normalized_distance
    def create_index(self):
        if not self.documents:
            raise ValueError("Chưa có documents để tạo index")
        if self.index is None:
            self.index = FAISS.from_documents(
                documents=self.documents, embedding=self.embeddings
            )
        return self.index

    def get_similar_docs(self, query: str, k: int = 2):
        if self.index is None:
            raise ValueError("Index chưa được tạo. Gọi create_index() trước.")
        return self.index.similarity_search(query=query, k=k)

    # -------------------- 🔹 Keyword Extraction --------------------
    def get_keywords_from_index(self, topic: str, max_keywords: int = 5):
        """Trích xuất keywords từ FAISS index dựa vào topic"""
        if self.index is None:
            self.create_index()

        if topic:
            docs = self.get_similar_docs(topic, k=max_keywords*2)
        else:
            docs = self.documents

        raw_text = "\n".join(doc.page_content for doc in docs)
        text = clean_text(raw_text)

        doc = self.nlp(text)

        # lấy noun chunks
        phrases = {}
        for np in doc.noun_chunks:
            phrase = np.text.strip()
            if len(phrase.split()) > 1:
                phrases[phrase] = phrases.get(phrase, 0) + 1

        phrase_keys = sorted(phrases.keys(), key=lambda x: len(x), reverse=True)

        # lọc trùng bằng Levenshtein
        filtered = []
        for ph in phrase_keys:
            # loại bỏ các keyword có similarity ≥ 0.7 với keyword đã chọn
            if all(self.levenshtein_similarity(ph, f) < 0.7 for f in filtered):
                filtered.append(ph)
            if len(filtered) >= max_keywords:
                break

        json_file = "all_keywords.json"
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                all_keywords = json.load(f)
        else:
            all_keywords = {}

        all_keywords[topic] = filtered  # cập nhật hoặc thêm mới

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(all_keywords, f, ensure_ascii=False, indent=2)

        return filtered

    # -------------------- 🔹 Parse JSON --------------------
    def parse_json(self, text: str):
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            logger.warning("Không tìm thấy JSON object trong output: %s", text)
            return None

        try:
            return json.loads(match.group(0))
        except Exception as e:
            logger.error("Parse JSON fail: %s", e)
            return None

    # -------------------- 🔹 Sinh câu hỏi --------------------
    

    def get_questions(self, topic: str = "", count: int = 5):
        """
        Sinh câu hỏi dựa trên keyword. Nếu thất bại thì fallback sang context.
        """
        keywords = self.get_keywords_from_index(topic, max_keywords=count)

        if not keywords:
            logger.warning("Không có keyword nào, fallback sang context.")
            return self._get_questions_from_context(topic, count)

        questions = []
        for idx, kw in enumerate(keywords, start=1):
            system_msg = self.system.replace("{count}", "1")
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=self.user.format(context=kw)),
            ]

            try:
                result = self.llm.invoke(messages)
                q = self.parse_json(str(result.content).strip())

                if q:
                    if isinstance(q, dict):
                        q["code"] = normalize_code_field(q.get("code"))
                        q["id"] = idx
                        questions.append(q)
                    elif isinstance(q, list) and len(q) > 0:
                        q[0]["code"] = normalize_code_field(q[0].get("code"))
                        q[0]["id"] = idx
                        questions.append(q[0])

                else:
                    questions.append({
                        "id": idx,
                        "question": f"Câu hỏi về: {kw}",
                        "options": [],
                        "answer": None,
                        "code": ""
                    })

            except Exception as e:
                logger.error("Lỗi sinh câu hỏi %s: %s", idx, e)
                questions.append({
                    "id": idx,
                    "question": f"Câu hỏi về: {kw}",
                    "options": [],
                    "answer": None,
                    "code": ""
                })

        if len(questions) < count:
            extra_qs = self._get_questions_from_context(topic, count - len(questions))
            # normalize luôn code trong extra_qs
            for q in extra_qs:
                q["code"] = normalize_code_field(q.get("code"))
            questions.extend(extra_qs)

        # lưu ra JSON
        with open("questions.json", "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=4, ensure_ascii=False)

        return questions[:count]

    # -------------------- 🔹 Fallback Context Mode --------------------
    def _get_questions_from_context(self, topic: str, count: int):
        if self.index is None:
            self.create_index()

        if topic:
            query_docs = self.get_similar_docs(topic, k=4)
        else:
            query_docs = self.documents[:4]

        text = "\n\n".join(doc.page_content for doc in query_docs)

        system_msg = self.system.replace("{count}", str(count))
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=self.user.format(context=text)),
        ]

        result = self.llm.invoke(messages)
        raw_text = str(result.content).strip()

        try:
            data = json.loads(re.search(r"\[.*\]", raw_text, re.S).group(0))
        except Exception as e:
            logger.error("Parse context JSON fail: %s", e)
            return []

        return data[:count]




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







