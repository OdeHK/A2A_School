# 🚀 A2A_SCHOOL - ĐỀ XUẤT CÁC TÍNH NĂNG NÂNG CAO

## 📋 Mục Lục
1. [Memory System - Conversation History](#1-memory-system)
2. [PDF Dài - Xử Lý Hiệu Quả](#2-pdf-dài)
3. [Tối Ưu Hóa Dữ Liệu Lớn](#3-tối-ưu-hóa)
4. [Xử Lý Tài Liệu Không Có TOC](#4-no-toc-handling)
5. [Implementation Plan](#5-implementation-plan)

---

## 1️⃣ MEMORY SYSTEM - Conversation History

### 🎯 Mục Đích
Lưu trữ lịch sử hội thoại để:
- Hiểu ngữ cảnh câu hỏi tiếp theo
- Tránh lặp lại thông tin
- Cá nhân hóa trải nghiệm người dùng
- Theo dõi learning path của học sinh

---

### 🏗️ Kiến Trúc Đề Xuất

#### Option 1: **LangChain Memory (Đơn Giản)** ⭐ RECOMMENDED

**Cấu trúc**:
```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import HumanMessage, AIMessage

class ChatMemoryService:
    """
    Service quản lý memory cho chat
    """
    
    def __init__(self, vector_service, llm_service):
        self.vector_service = vector_service
        self.llm_service = llm_service
        
        # Memory strategies
        self.buffer_memory = None      # Short-term (last 10 messages)
        self.summary_memory = None     # Mid-term (summarized)
        self.vector_memory = None      # Long-term (semantic search)
    
    def create_memory_for_session(
        self, 
        session_id: str, 
        memory_type: str = "hybrid"
    ):
        """
        Tạo memory cho session
        
        Args:
            session_id: Session identifier
            memory_type: "buffer" | "summary" | "vector" | "hybrid"
        """
        if memory_type == "buffer":
            return ConversationBufferWindowMemory(
                k=10,  # Giữ 10 message gần nhất
                return_messages=True,
                memory_key="chat_history"
            )
        
        elif memory_type == "summary":
            return ConversationSummaryMemory(
                llm=self.llm_service.llm,
                return_messages=True,
                memory_key="chat_history"
            )
        
        elif memory_type == "vector":
            # Lưu vào ChromaDB để semantic search
            retriever = self.vector_service.get_retriever(
                collection_name=f"memory_{session_id}",
                k=5
            )
            return VectorStoreRetrieverMemory(
                retriever=retriever,
                memory_key="chat_history"
            )
        
        elif memory_type == "hybrid":
            # Kết hợp cả 3!
            return HybridMemory(
                buffer=ConversationBufferWindowMemory(k=5),
                summary=ConversationSummaryMemory(
                    llm=self.llm_service.llm
                ),
                vector_retriever=self.vector_service.get_retriever(
                    collection_name=f"memory_{session_id}"
                )
            )
```

**Ưu điểm**:
- ✅ Tích hợp sẵn với LangChain
- ✅ Dễ implement (< 100 lines)
- ✅ Hỗ trợ nhiều loại memory
- ✅ Tự động format messages

**Nhược điểm**:
- ⚠️ Tốn RAM nếu conversation dài
- ⚠️ Không persist khi restart app

---

#### Option 2: **MongoDB Memory (Persistent)** 

**Cấu trúc Database**:
```python
# Collection: conversation_history
{
    "_id": ObjectId("..."),
    "session_id": "session_123",
    "user_name": "student_A",
    "document_id": "doc_456",
    "timestamp": ISODate("2025-10-31T10:30:00Z"),
    "role": "human",  # "human" | "ai"
    "content": "Giải thích về Transformer là gì?",
    "metadata": {
        "intent": "question",
        "topic": "transformer_basics",
        "referenced_sections": ["ch1_intro"]
    }
}

# Collection: conversation_summary
{
    "_id": ObjectId("..."),
    "session_id": "session_123",
    "user_name": "student_A",
    "document_id": "doc_456",
    "summary": "Học sinh đã hỏi về kiến trúc Transformer...",
    "topics_discussed": ["transformer", "attention", "encoder"],
    "total_messages": 15,
    "last_updated": ISODate("2025-10-31T11:00:00Z")
}
```

**Implementation**:
```python
class MongoDBMemoryService:
    """
    Memory service với MongoDB persistence
    """
    
    def __init__(self, database_service):
        self.db = database_service
        self.history_collection = self.db.db["conversation_history"]
        self.summary_collection = self.db.db["conversation_summary"]
    
    def save_message(
        self,
        session_id: str,
        user_name: str,
        role: str,  # "human" | "ai"
        content: str,
        metadata: Dict[str, Any] = None
    ):
        """Lưu message vào database"""
        message = {
            "session_id": session_id,
            "user_name": user_name,
            "timestamp": datetime.now(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.history_collection.insert_one(message)
    
    def get_recent_messages(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Lấy N messages gần nhất"""
        cursor = self.history_collection.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(limit)
        
        messages = list(cursor)
        messages.reverse()  # Đảo ngược để có thứ tự đúng
        return messages
    
    def get_conversation_context(
        self,
        session_id: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Lấy context từ conversation history
        Tự động truncate nếu quá dài
        """
        messages = self.get_recent_messages(session_id, limit=20)
        
        context_parts = []
        total_tokens = 0
        
        for msg in reversed(messages):  # Ưu tiên messages mới
            msg_text = f"{msg['role']}: {msg['content']}"
            msg_tokens = len(msg_text.split())  # Rough estimate
            
            if total_tokens + msg_tokens > max_tokens:
                break
            
            context_parts.insert(0, msg_text)
            total_tokens += msg_tokens
        
        return "\n".join(context_parts)
    
    def summarize_conversation(
        self,
        session_id: str,
        llm_service
    ):
        """
        Tạo summary cho conversation dài
        Chạy định kỳ (mỗi 10 messages)
        """
        messages = self.get_recent_messages(session_id, limit=50)
        
        if len(messages) < 5:
            return  # Chưa đủ để summarize
        
        # Format messages
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        
        # LLM summarize
        summary_prompt = f"""
        Tóm tắt cuộc hội thoại sau về học tập:
        
        {conversation_text}
        
        Tóm tắt nên bao gồm:
        - Chủ đề chính đã thảo luận
        - Các khái niệm quan trọng
        - Câu hỏi chưa được trả lời
        """
        
        summary = llm_service.llm.invoke(summary_prompt).content
        
        # Save summary
        self.summary_collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "summary": summary,
                    "total_messages": len(messages),
                    "last_updated": datetime.now()
                }
            },
            upsert=True
        )
        
        return summary
```

**Ưu điểm**:
- ✅ Persistent - không mất khi restart
- ✅ Có thể query theo nhiều tiêu chí
- ✅ Hỗ trợ analytics (topic tracking)
- ✅ Scalable (MongoDB sharding)

**Nhược điểm**:
- ⚠️ Cần thêm database queries
- ⚠️ Phức tạp hơn LangChain memory

---

#### Option 3: **Hybrid Approach** ⭐⭐ BEST PRACTICE

**Kết hợp cả 2**:
```python
class HybridMemoryService:
    """
    Kết hợp LangChain Memory (runtime) + MongoDB (persistent)
    """
    
    def __init__(self, database_service, vector_service, llm_service):
        self.db_memory = MongoDBMemoryService(database_service)
        self.langchain_memory = {}  # Dict[session_id, Memory]
        self.vector_service = vector_service
        self.llm_service = llm_service
    
    def get_or_create_memory(self, session_id: str):
        """
        Get LangChain memory cho session
        Load từ MongoDB nếu đã tồn tại
        """
        if session_id not in self.langchain_memory:
            # Create new memory
            memory = ConversationBufferWindowMemory(k=10)
            
            # Load history từ MongoDB
            messages = self.db_memory.get_recent_messages(
                session_id, limit=10
            )
            
            for msg in messages:
                if msg['role'] == 'human':
                    memory.chat_memory.add_user_message(msg['content'])
                else:
                    memory.chat_memory.add_ai_message(msg['content'])
            
            self.langchain_memory[session_id] = memory
        
        return self.langchain_memory[session_id]
    
    def save_exchange(
        self,
        session_id: str,
        user_name: str,
        user_message: str,
        ai_response: str
    ):
        """
        Lưu cả vào LangChain memory VÀ MongoDB
        """
        # Save to LangChain (runtime)
        memory = self.get_or_create_memory(session_id)
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(ai_response)
        
        # Save to MongoDB (persistent)
        self.db_memory.save_message(
            session_id, user_name, "human", user_message
        )
        self.db_memory.save_message(
            session_id, user_name, "ai", ai_response
        )
        
        # Check if need to summarize
        if self._should_summarize(session_id):
            self.db_memory.summarize_conversation(
                session_id, self.llm_service
            )
    
    def _should_summarize(self, session_id: str) -> bool:
        """Summarize mỗi 10 messages"""
        messages = self.db_memory.get_recent_messages(session_id, limit=1)
        if not messages:
            return False
        
        total_count = self.db_memory.history_collection.count_documents(
            {"session_id": session_id}
        )
        return total_count % 10 == 0
```

**Ưu điểm**:
- ✅✅ Fast access (LangChain in-memory)
- ✅✅ Persistent storage (MongoDB)
- ✅✅ Best of both worlds
- ✅ Tự động summarization

**Nhược điểm**:
- ⚠️ Phức tạp nhất
- ⚠️ Cần quản lý sync giữa 2 systems

---

### 🔌 Tích Hợp Vào Chatbot

**Before (Không có memory)**:
```python
def chat(user_message: str, document_id: str, username: str):
    # Retrieve context
    context = rag_service.retrieve(user_message, document_id, username)
    
    # Generate response
    response = llm.invoke(
        f"Context: {context}\nQuestion: {user_message}"
    )
    
    return response
```

**After (Có memory)**:
```python
def chat(
    user_message: str, 
    document_id: str, 
    username: str,
    session_id: str
):
    # Get memory
    memory_service = HybridMemoryService(...)
    memory = memory_service.get_or_create_memory(session_id)
    
    # Get conversation context
    chat_history = memory.load_memory_variables({})['chat_history']
    
    # Retrieve context với conversation awareness
    context = rag_service.retrieve_with_context(
        query=user_message,
        document_id=document_id,
        username=username,
        chat_history=chat_history  # ← NEW!
    )
    
    # Generate response với memory
    response = llm.invoke(
        f"Chat History:\n{chat_history}\n\n"
        f"Context: {context}\n\n"
        f"Current Question: {user_message}"
    )
    
    # Save to memory
    memory_service.save_exchange(
        session_id, username, user_message, response
    )
    
    return response
```

---

### 📊 So Sánh Memory Options

| **Feature** | **LangChain Only** | **MongoDB Only** | **Hybrid** |
|-------------|-------------------|------------------|------------|
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡⚡⚡ Fast |
| **Persistence** | ❌ No | ✅ Yes | ✅ Yes |
| **Scalability** | ⚠️ RAM limited | ✅ Good | ✅ Good |
| **Complexity** | 🟢 Low | 🟡 Medium | 🔴 High |
| **Analytics** | ❌ No | ✅ Yes | ✅ Yes |
| **Implementation** | 1-2 days | 2-3 days | 3-5 days |

**Recommendation**: 
- Small project: **LangChain Only**
- Production: **Hybrid Approach**

---

## 2️⃣ PDF DÀI - Xử Lý Hiệu Quả

### 🎯 Vấn Đề Hiện Tại

**Scenario**: PDF 500+ trang (sách giáo khoa, luận văn)

**Challenges**:
1. ❌ **Token Limit**: LLM chỉ xử lý ~128k tokens (~200 pages)
2. ❌ **Processing Time**: TOC extraction + chunking mất 10-15 phút
3. ❌ **Memory Usage**: Load toàn bộ PDF vào RAM (>1GB)
4. ❌ **Vector Storage**: ChromaDB slow khi có >10,000 chunks

---

### 🏗️ Giải Pháp Đề Xuất

#### Strategy 1: **Hierarchical Processing** ⭐ RECOMMENDED

**Concept**: Chia PDF thành các "books" nhỏ, xử lý song song

```python
class HierarchicalPDFProcessor:
    """
    Xử lý PDF dài theo hierarchical approach
    """
    
    def __init__(self, chunk_size: int = 50):
        """
        Args:
            chunk_size: Số trang mỗi "book" (default: 50 trang)
        """
        self.chunk_size = chunk_size
    
    def process_large_pdf(
        self,
        pdf_path: str,
        document_id: str,
        username: str
    ) -> ProcessingResult:
        """
        Chia PDF thành nhiều sub-books và xử lý song song
        """
        # Step 1: Get total pages
        total_pages = self._get_total_pages(pdf_path)
        logger.info(f"📄 Total pages: {total_pages}")
        
        # Step 2: Chia thành sub-books
        sub_books = self._split_into_books(total_pages, self.chunk_size)
        # sub_books = [(0, 49), (50, 99), (100, 149), ...]
        
        logger.info(f"📚 Split into {len(sub_books)} sub-books")
        
        # Step 3: Process song song với ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self._process_sub_book,
                    pdf_path, start_page, end_page, 
                    document_id, username, idx
                ): idx
                for idx, (start_page, end_page) in enumerate(sub_books)
            }
            
            for future in as_completed(futures):
                book_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"✅ Sub-book {book_idx} completed")
                except Exception as e:
                    logger.error(f"❌ Sub-book {book_idx} failed: {e}")
        
        # Step 4: Merge results
        merged_result = self._merge_results(results)
        
        return merged_result
    
    def _process_sub_book(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int,
        document_id: str,
        username: str,
        book_idx: int
    ):
        """
        Xử lý 1 sub-book (50 trang)
        """
        logger.info(f"🔄 Processing pages {start_page}-{end_page}...")
        
        # Extract pages
        pages = self._extract_pages(pdf_path, start_page, end_page)
        
        # TOC extraction (chỉ cho sub-book này)
        toc_sections = self._extract_toc_for_range(
            pdf_path, start_page, end_page
        )
        
        # Chunking
        chunks = self._chunk_pages(pages, document_id, username)
        
        # Vector store (với sub-collection)
        sub_collection = f"{document_id}_book_{book_idx}"
        self.vector_service.add_documents(chunks, sub_collection)
        
        return {
            "book_idx": book_idx,
            "page_range": (start_page, end_page),
            "toc_sections": toc_sections,
            "chunk_count": len(chunks),
            "collection_name": sub_collection
        }
    
    def _merge_results(self, results: List[Dict]):
        """
        Merge kết quả từ nhiều sub-books
        """
        # Merge TOC
        all_toc_sections = []
        for result in sorted(results, key=lambda x: x['book_idx']):
            all_toc_sections.extend(result['toc_sections'])
        
        # Merge chunk counts
        total_chunks = sum(r['chunk_count'] for r in results)
        
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            toc_sections=all_toc_sections,
            total_chunks=total_chunks,
            sub_collections=[r['collection_name'] for r in results]
        )
```

**Ưu điểm**:
- ✅ **Parallel Processing**: 4x faster với 4 workers
- ✅ **Memory Efficient**: Chỉ load 50 trang mỗi lần
- ✅ **Fault Tolerant**: 1 sub-book fail không ảnh hưởng toàn bộ
- ✅ **Progress Tracking**: Có thể show progress bar

**Nhược điểm**:
- ⚠️ TOC có thể bị split giữa 2 books
- ⚠️ Cần merge logic phức tạp

**Cách Tối Ưu**:
```python
# Fix TOC splitting issue
def _extract_toc_with_overlap(
    self,
    pdf_path: str,
    start_page: int,
    end_page: int
):
    """
    Extract TOC với overlap 5 trang để tránh miss sections
    """
    overlap = 5
    actual_start = max(0, start_page - overlap)
    actual_end = end_page + overlap
    
    toc = self._extract_toc_for_range(pdf_path, actual_start, actual_end)
    
    # Filter sections chỉ trong range chính
    filtered = [
        section for section in toc
        if start_page <= section.page_number <= end_page
    ]
    
    return filtered
```

---

#### Strategy 2: **Streaming Processing**

**Concept**: Xử lý page-by-page thay vì load toàn bộ

```python
class StreamingPDFProcessor:
    """
    Xử lý PDF theo streaming mode
    """
    
    def process_streaming(
        self,
        pdf_path: str,
        document_id: str,
        username: str,
        batch_size: int = 10
    ):
        """
        Process PDF page-by-page với batching
        """
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        batch = []
        chunk_count = 0
        
        for page_num in range(total_pages):
            # Read 1 page
            page = reader.pages[page_num]
            text = page.extract_text()
            
            # Create document
            doc = Document(
                page_content=text,
                metadata={
                    "page": page_num + 1,
                    "document_id": document_id,
                    "username": username
                }
            )
            
            batch.append(doc)
            
            # Process khi batch đủ size
            if len(batch) >= batch_size:
                chunks = self._chunk_batch(batch)
                self.vector_service.add_documents(chunks)
                chunk_count += len(chunks)
                
                logger.info(
                    f"✅ Processed pages {page_num-batch_size+1}-{page_num+1} "
                    f"({chunk_count} chunks so far)"
                )
                
                batch = []  # Clear batch
                
                # Optional: Yield progress
                yield {
                    "progress": (page_num + 1) / total_pages,
                    "pages_processed": page_num + 1,
                    "chunks_created": chunk_count
                }
        
        # Process remaining batch
        if batch:
            chunks = self._chunk_batch(batch)
            self.vector_service.add_documents(chunks)
            chunk_count += len(chunks)
        
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            total_chunks=chunk_count
        )
```

**Ưu điểm**:
- ✅ **Constant Memory**: Chỉ ~50MB RAM dù PDF bao lớn
- ✅ **Real-time Progress**: Có thể update UI live
- ✅ **Interruptible**: Có thể pause/resume

**Nhược điểm**:
- ⚠️ Slower (sequential processing)
- ⚠️ Khó extract TOC (cần full document)

---

#### Strategy 3: **Lazy Loading + Caching** ⭐⭐

**Concept**: Load on-demand + cache kết quả

```python
from functools import lru_cache
import hashlib

class LazyPDFProcessor:
    """
    Lazy loading với intelligent caching
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, pdf_path: str, page_num: int) -> str:
        """Generate cache key cho page"""
        pdf_hash = hashlib.md5(
            Path(pdf_path).read_bytes()
        ).hexdigest()[:8]
        return f"{pdf_hash}_page_{page_num}"
    
    @lru_cache(maxsize=100)
    def get_page(self, pdf_path: str, page_num: int) -> Document:
        """
        Get page với caching
        LRU cache giữ 100 pages gần nhất trong RAM
        """
        cache_key = self._get_cache_key(pdf_path, page_num)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Check disk cache
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return Document(**data)
        
        # Extract from PDF
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        page = reader.pages[page_num]
        text = page.extract_text()
        
        doc = Document(
            page_content=text,
            metadata={"page": page_num + 1}
        )
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(doc.dict(), f)
        
        return doc
    
    def process_with_lazy_loading(
        self,
        pdf_path: str,
        document_id: str,
        username: str
    ):
        """
        Process PDF với lazy loading
        Chỉ load pages khi cần
        """
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        # Create TOC first (requires full scan)
        toc = self._extract_toc_lazy(pdf_path)
        
        # Process pages on-demand
        for section in toc.sections:
            # Chỉ load pages trong section này
            pages = [
                self.get_page(pdf_path, p)
                for p in range(
                    section.start_page, 
                    section.end_page + 1
                )
            ]
            
            # Chunk và store
            chunks = self._chunk_pages(pages)
            self.vector_service.add_documents(chunks)
```

**Ưu điểm**:
- ✅ **Memory Efficient**: Chỉ load cần thiết
- ✅ **Fast Reprocessing**: Cache giúp reprocess nhanh
- ✅ **Flexible**: Có thể load bất kỳ page nào

**Nhược điểm**:
- ⚠️ Disk space cho cache
- ⚠️ Cache invalidation phức tạp

---

### 📊 So Sánh Strategies

| **Strategy** | **Speed** | **Memory** | **Complexity** | **Best For** |
|-------------|-----------|------------|----------------|--------------|
| **Hierarchical** | ⚡⚡⚡ | 🟢 Low | 🟡 Medium | PDF 200-1000 trang |
| **Streaming** | ⚡ | 🟢🟢 Very Low | 🟢 Low | PDF >1000 trang |
| **Lazy + Cache** | ⚡⚡ | 🟢 Low | 🟡 Medium | Reprocessing nhiều |

**Recommendation**: 
- **< 200 pages**: Current approach OK
- **200-500 pages**: **Hierarchical Processing**
- **> 500 pages**: **Streaming + Hierarchical Hybrid**

---

## 3️⃣ TỐI ƯU HÓA DỮ LIỆU LỚNS

### 🎯 Chunking Strategy Optimization

#### Current Issues
```python
# Current: One-page-per-chunk
# Vấn đề: 500 pages = 500 chunks → Slow retrieval
```

#### Optimized Approaches

**Option 1: Semantic Chunking** ⭐ RECOMMENDED

```python
from langchain.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Chunk dựa trên semantic similarity
    Tự động chia tại điểm topic thay đổi
    """
    
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",  # Hoặc "standard_deviation"
            breakpoint_threshold_amount=0.75
        )
    
    def chunk(self, pages: Iterator[Document], document_id: str, username: str):
        """
        Chunk theo semantic boundaries
        """
        # Merge pages thành text lớn
        full_text = "\n\n".join([page.page_content for page in pages])
        
        # Semantic split
        chunks = self.text_splitter.create_documents([full_text])
        
        # Add metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                "document_id": document_id,
                "username": username,
                "chunk_id": idx,
                "chunk_type": "semantic"
            })
        
        return chunks
```

**Ưu điểm**:
- ✅ Intelligent splitting (không cắt giữa concept)
- ✅ Better retrieval quality
- ✅ Fewer chunks (10x reduction)

**Nhược điểm**:
- ⚠️ Requires embedding API (cost $)
- ⚠️ Slower processing

---

**Option 2: TOC-Aware Chunking** ⭐⭐ BEST

```python
class TOCAwareChunkingStrategy(ChunkingStrategy):
    """
    Chunk dựa trên TOC structure
    Mỗi section = 1 hoặc nhiều chunks
    """
    
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200
        )
    
    def chunk_with_toc(
        self,
        pages: List[Document],
        toc_sections: List[TOCSection],
        document_id: str,
        username: str
    ):
        """
        Chunk theo TOC sections
        """
        chunks = []
        
        for section in toc_sections:
            # Get pages cho section này
            section_pages = [
                p for p in pages
                if section.start_page <= p.metadata['page'] <= section.end_page
            ]
            
            # Merge text
            section_text = "\n\n".join([
                p.page_content for p in section_pages
            ])
            
            # Check size
            if len(section_text.split()) < self.max_chunk_size:
                # Small section → 1 chunk
                chunks.append(Document(
                    page_content=section_text,
                    metadata={
                        "document_id": document_id,
                        "username": username,
                        "section_id": section.section_id,
                        "section_title": section.section_title,
                        "page_range": f"{section.start_page}-{section.end_page}",
                        "chunk_type": "toc_section"
                    }
                ))
            else:
                # Large section → Multiple chunks
                sub_chunks = self.text_splitter.create_documents([section_text])
                
                for idx, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.metadata.update({
                        "document_id": document_id,
                        "username": username,
                        "section_id": section.section_id,
                        "section_title": section.section_title,
                        "sub_chunk_idx": idx,
                        "chunk_type": "toc_subsection"
                    })
                    chunks.append(sub_chunk)
        
        return chunks
```

**Ưu điểm**:
- ✅✅ Perfect context preservation
- ✅✅ Better retrieval (section-aware)
- ✅ Hierarchical search support

**Nhược điểm**:
- ⚠️ Requires TOC extraction

---

### 🚀 Vector Store Optimization

**Current**: ChromaDB single collection

**Problem**: >10k chunks → Slow search

**Solution: Hierarchical Vector Store**

```python
class HierarchicalVectorStore:
    """
    Multi-level vector store cho large documents
    """
    
    def __init__(self, vector_service):
        self.vector_service = vector_service
    
    def create_hierarchical_index(
        self,
        document_id: str,
        toc_sections: List[TOCSection],
        chunks: List[Document]
    ):
        """
        Create 2-level index:
        Level 1: Section summaries (coarse search)
        Level 2: Detailed chunks (fine search)
        """
        # Level 1: Section-level collection
        section_summaries = []
        for section in toc_sections:
            # Get chunks thuộc section này
            section_chunks = [
                c for c in chunks
                if c.metadata.get('section_id') == section.section_id
            ]
            
            # Tạo summary
            summary_text = self._create_section_summary(section_chunks)
            
            section_summaries.append(Document(
                page_content=summary_text,
                metadata={
                    "document_id": document_id,
                    "section_id": section.section_id,
                    "section_title": section.section_title,
                    "chunk_count": len(section_chunks),
                    "level": "section"
                }
            ))
        
        # Store level 1 (fast search)
        self.vector_service.add_documents(
            section_summaries,
            collection_name=f"{document_id}_L1_sections"
        )
        
        # Store level 2 (detailed search)
        self.vector_service.add_documents(
            chunks,
            collection_name=f"{document_id}_L2_chunks"
        )
    
    def hierarchical_search(
        self,
        query: str,
        document_id: str,
        k: int = 5
    ):
        """
        2-step search:
        1. Find relevant sections (L1)
        2. Search within those sections (L2)
        """
        # Step 1: Coarse search
        relevant_sections = self.vector_service.similarity_search(
            query=query,
            collection_name=f"{document_id}_L1_sections",
            k=3  # Top 3 sections
        )
        
        # Step 2: Fine search within sections
        section_ids = [
            s.metadata['section_id']
            for s in relevant_sections
        ]
        
        detailed_chunks = []
        for section_id in section_ids:
            # Search in L2 với filter
            chunks = self.vector_service.similarity_search(
                query=query,
                collection_name=f"{document_id}_L2_chunks",
                filter={"section_id": section_id},
                k=k // len(section_ids)  # Chia đều
            )
            detailed_chunks.extend(chunks)
        
        return detailed_chunks[:k]
```

**Ưu điểm**:
- ✅ 10x faster search
- ✅ Better context (section-aware)
- ✅ Scalable to millions of chunks

---

## 4️⃣ XỬ LÝ TÀI LIỆU KHÔNG CÓ TOC

### 🎯 Vấn Đề

**Scenario**: PDF không có TOC structure (slides, handouts, papers)

**Current Behavior**: Fail hoặc fallback to page-by-page

---

### 🏗️ Giải Pháp: Auto TOC Generation

#### Option 1: **LLM-Based TOC Generation** ⭐ RECOMMENDED

```python
class AutoTOCGenerator:
    """
    Tự động generate TOC cho PDF không có sẵn
    """
    
    def __init__(self, llm_service):
        self.llm = llm_service.llm
    
    def generate_toc_from_content(
        self,
        pdf_path: str,
        sample_size: int = 50
    ) -> List[TOCSection]:
        """
        Tạo TOC bằng LLM
        
        Strategy:
        1. Sample N pages đều khắp document
        2. LLM phân tích và tạo structure
        3. Validate và refine
        """
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        # Sample pages (đều nhau)
        sample_indices = self._get_sample_indices(total_pages, sample_size)
        sampled_pages = []
        
        for idx in sample_indices:
            page_text = reader.pages[idx].extract_text()
            sampled_pages.append({
                "page": idx + 1,
                "content": page_text[:500]  # First 500 chars
            })
        
        # LLM prompt
        prompt = f"""
        Analyze this document and create a Table of Contents structure.
        
        Document has {total_pages} pages.
        Here are samples from various pages:
        
        {json.dumps(sampled_pages, indent=2)}
        
        Generate a hierarchical TOC with:
        - Section titles
        - Estimated page ranges
        - Hierarchical levels
        
        Output as JSON:
        {{
            "sections": [
                {{
                    "title": "Introduction",
                    "start_page": 1,
                    "end_page": 10,
                    "level": 1,
                    "subsections": [...]
                }}
            ]
        }}
        """
        
        response = self.llm.invoke(prompt)
        toc_data = json.loads(response.content)
        
        # Convert to TOCSection objects
        sections = self._parse_toc_response(toc_data)
        
        return sections
    
    def _get_sample_indices(self, total_pages: int, sample_size: int):
        """Lấy indices đều khắp document"""
        import numpy as np
        return np.linspace(0, total_pages-1, sample_size, dtype=int)
```

**Ưu điểm**:
- ✅ Fully automatic
- ✅ Smart structure detection
- ✅ Works with any document

**Nhược điểm**:
- ⚠️ Requires LLM (cost)
- ⚠️ May not be 100% accurate

---

#### Option 2: **Heuristic-Based TOC** ⭐

```python
class HeuristicTOCGenerator:
    """
    Generate TOC dựa trên heuristics (không cần LLM)
    """
    
    def generate_toc_heuristic(self, pdf_path: str):
        """
        Tìm headings dựa trên:
        - Font size (lớn hơn = heading)
        - Font weight (bold = heading)
        - Position (đầu trang)
        - Numbering pattern (1., 1.1, etc.)
        """
        from pypdf import PdfReader
        import fitz  # PyMuPDF for font analysis
        
        doc = fitz.open(pdf_path)
        sections = []
        
        for page_num, page in enumerate(doc):
            # Extract blocks with formatting
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            font_size = span.get("size", 0)
                            font_flags = span.get("flags", 0)
                            
                            # Heuristics
                            is_bold = font_flags & 2 ** 4  # Bold flag
                            is_large = font_size > 12
                            is_numbered = self._is_numbered_heading(text)
                            
                            if (is_bold and is_large) or is_numbered:
                                # Likely a heading
                                level = self._determine_level(
                                    text, font_size, is_numbered
                                )
                                
                                sections.append(TOCSection(
                                    section_id=f"auto_{len(sections)}",
                                    section_title=text,
                                    parent_section_id=None,
                                    level=level,
                                    page_number=page_num + 1,
                                    children=[]
                                ))
        
        # Post-process: Assign parents based on levels
        sections = self._build_hierarchy(sections)
        
        return sections
    
    def _is_numbered_heading(self, text: str) -> bool:
        """Check if text matches numbering pattern"""
        import re
        patterns = [
            r'^\d+\.',           # 1., 2., 3.
            r'^\d+\.\d+',        # 1.1, 1.2
            r'^Chapter \d+',     # Chapter 1
            r'^Section \d+',     # Section 1
        ]
        return any(re.match(p, text) for p in patterns)
    
    def _determine_level(self, text: str, font_size: float, is_numbered: bool):
        """Determine heading level"""
        if font_size > 18:
            return 1  # Main chapter
        elif font_size > 14 or (is_numbered and '.' not in text[5:]):
            return 2  # Section
        else:
            return 3  # Subsection
```

**Ưu điểm**:
- ✅ No LLM cost
- ✅ Fast
- ✅ Works offline

**Nhược điểm**:
- ⚠️ Less accurate
- ⚠️ Depends on PDF formatting

---

#### Option 3: **Hybrid Fallback Strategy** ⭐⭐ BEST

```python
class SmartTOCExtractor:
    """
    Try multiple methods với fallback
    """
    
    def extract_toc_smart(self, pdf_path: str):
        """
        Hierarchy:
        1. Try built-in TOC
        2. Try heuristic detection
        3. Try LLM generation
        4. Fallback to page-based sections
        """
        logger.info("🔍 Attempting TOC extraction...")
        
        # Method 1: Built-in TOC
        try:
            toc = self._extract_builtin_toc(pdf_path)
            if toc and len(toc.sections) > 0:
                logger.info("✅ Found built-in TOC")
                return toc
        except Exception as e:
            logger.warning(f"No built-in TOC: {e}")
        
        # Method 2: Heuristic
        try:
            toc = self.heuristic_generator.generate_toc_heuristic(pdf_path)
            if len(toc) > 3:  # At least 3 sections
                logger.info("✅ Generated TOC via heuristics")
                return toc
        except Exception as e:
            logger.warning(f"Heuristic failed: {e}")
        
        # Method 3: LLM
        try:
            toc = self.auto_generator.generate_toc_from_content(pdf_path)
            if toc:
                logger.info("✅ Generated TOC via LLM")
                return toc
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
        
        # Method 4: Fallback - Page-based
        logger.warning("⚠️ Using fallback: page-based sections")
        return self._create_page_based_toc(pdf_path)
    
    def _create_page_based_toc(self, pdf_path: str):
        """
        Fallback: Chia document thành sections 10 trang
        """
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        sections = []
        section_size = 10
        
        for i in range(0, total_pages, section_size):
            sections.append(TOCSection(
                section_id=f"section_{i//section_size + 1}",
                section_title=f"Section {i//section_size + 1} (Pages {i+1}-{min(i+section_size, total_pages)})",
                parent_section_id=None,
                level=1,
                page_number=i + 1,
                children=[]
            ))
        
        return sections
```

---

## 5️⃣ IMPLEMENTATION PLAN

### 📅 Roadmap (4-6 tuần)

#### **Phase 1: Memory System** (1-2 tuần)

**Week 1**:
- [ ] Implement `MongoDBMemoryService`
- [ ] Add conversation_history collection
- [ ] Basic save/load messages

**Week 2**:
- [ ] Integrate LangChain memory
- [ ] Implement hybrid approach
- [ ] Add to chatbot UI

**Deliverables**:
- ✅ Memory working in chat
- ✅ History persisted in MongoDB
- ✅ Conversation context in responses

---

#### **Phase 2: Large PDF Support** (2 tuần)

**Week 3**:
- [ ] Implement `HierarchicalPDFProcessor`
- [ ] Add parallel processing
- [ ] Test với PDF 200-500 trang

**Week 4**:
- [ ] Add streaming support
- [ ] Optimize chunking strategy
- [ ] Hierarchical vector store

**Deliverables**:
- ✅ Handle 500+ page PDFs
- ✅ 4x faster processing
- ✅ Memory efficient

---

#### **Phase 3: TOC Auto-Generation** (1 tuần)

**Week 5**:
- [ ] Implement heuristic TOC generator
- [ ] LLM-based generator
- [ ] Smart fallback logic

**Deliverables**:
- ✅ Works với PDFs không có TOC
- ✅ Auto-structure detection

---

#### **Phase 4: Optimization** (1 tuần)

**Week 6**:
- [ ] TOC-aware chunking
- [ ] Semantic chunking tests
- [ ] Performance benchmarks
- [ ] Documentation

**Deliverables**:
- ✅ Production-ready
- ✅ Full documentation
- ✅ Performance report

---

### 🧪 Testing Strategy

```python
# tests/test_memory.py
def test_memory_persistence():
    """Test memory saves to MongoDB"""
    memory_service = HybridMemoryService(...)
    
    memory_service.save_exchange(
        session_id="test_123",
        user_name="test_user",
        user_message="Hello",
        ai_response="Hi there!"
    )
    
    # Retrieve
    context = memory_service.db_memory.get_conversation_context("test_123")
    assert "Hello" in context
    assert "Hi there" in context

# tests/test_large_pdf.py
def test_500_page_processing():
    """Test hierarchical processing"""
    processor = HierarchicalPDFProcessor(chunk_size=50)
    
    result = processor.process_large_pdf(
        pdf_path="test_data/large_book_500p.pdf",
        document_id="test_doc",
        username="test_user"
    )
    
    assert result.status == ProcessingStatus.SUCCESS
    assert len(result.sub_collections) == 10  # 500/50 = 10

# tests/test_auto_toc.py
def test_toc_generation_no_builtin():
    """Test TOC generation for PDF without built-in TOC"""
    extractor = SmartTOCExtractor(...)
    
    toc = extractor.extract_toc_smart("test_data/no_toc.pdf")
    
    assert len(toc.sections) > 0
    assert all(s.section_title for s in toc.sections)
```

---

### 📊 Expected Improvements

| **Metric** | **Current** | **After Phase 2** | **After Phase 4** |
|-----------|-------------|-------------------|-------------------|
| **Max PDF Size** | ~200 pages | 1000+ pages | Unlimited |
| **Processing Time (500p)** | ~15 min | ~4 min | ~2 min |
| **Memory Usage** | ~1.5 GB | ~300 MB | ~200 MB |
| **Chunk Quality** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Search Speed** | ~500ms | ~200ms | ~50ms |
| **Context Awareness** | ❌ No | ✅ Yes | ✅✅ Yes |

---

### 💰 Cost Analysis

**Memory System**:
- Development: 40 hours
- LLM API (summarization): ~$5/month
- MongoDB storage: Free (< 512MB)

**Large PDF Support**:
- Development: 60 hours
- Infrastructure: Same (no extra cost)

**Auto TOC Generation**:
- Development: 20 hours
- LLM API (TOC gen): ~$10/month (one-time per doc)

**Total**:
- Development: ~120 hours (3 tuần full-time)
- Monthly cost: ~$15 (LLM API)

---

## 🎯 KẾT LUẬN VÀ KHUYẾN NGHỊ

### ⭐ Top Priorities

1. **Memory System (MUST HAVE)**
   - Implement: Hybrid Approach
   - Timeline: 2 tuần
   - Impact: ⭐⭐⭐⭐⭐

2. **Large PDF Support (HIGH PRIORITY)**
   - Implement: Hierarchical Processing
   - Timeline: 2 tuần
   - Impact: ⭐⭐⭐⭐

3. **Auto TOC (NICE TO HAVE)**
   - Implement: Smart Fallback
   - Timeline: 1 tuần
   - Impact: ⭐⭐⭐

### 🚀 Quick Wins (Implement First)

1. **MongoDB Memory** - Easy, high impact
2. **Streaming Processing** - Solves memory issues now
3. **Heuristic TOC** - No LLM cost, works 70%

### 📈 Long-term Vision

**6 tháng**: 
- Handle 10,000 page documents
- Multi-language support
- Real-time collaboration
- Advanced analytics

---

**Bạn muốn bắt đầu implement feature nào trước?**
1. Memory system? 
2. Large PDF support?
3. Auto TOC generation?

Tôi có thể tạo code chi tiết cho bất kỳ feature nào! 🚀
