"""
🚀 FINAL OPTIMIZED - NO RE-EMBEDDING!
======================================

Fix: Storage đang RE-EMBED tất cả chunks!
Solution: Pre-compute embeddings → Direct ChromaDB insert

Target: 2620 pages in 3-4 MINUTES!
"""

import sys
import logging
import time
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

import fitz
from langchain.schema.document import Document
from sentence_transformers import SentenceTransformer

from services.rag.embedding_service import EmbeddingService, EmbeddingType
from services.rag.vector_service import VectorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FinalStats:
    """Processing statistics"""
    total_pages: int
    total_chunks: int
    pdf_time: float
    chunk_time: float
    embed_time: float
    store_time: float
    total_time: float
    pages_per_second: float


def progress_callback(current, total, message):
    """Progress bar"""
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r[{bar}] {current:3d}% - {message}", end='', flush=True)


def read_page_range(pdf_path: str, start_page: int, end_page: int) -> List[str]:
    """Read pages in parallel"""
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        
        for page_num in range(start_page, end_page):
            if page_num >= doc.page_count:
                break
            page = doc[page_num]
            text = page.get_text()
            pages_text.append(text)
        
        doc.close()
        return pages_text
    except Exception as e:
        logger.error(f"Error reading pages {start_page}-{end_page}: {e}")
        return []


def process_final_optimized(pdf_path: str) -> FinalStats:
    """
    FINAL OPTIMIZED PROCESSING
    
    Steps:
    1. Parallel PDF reading
    2. Fast chunking
    3. Batch embedding (ONCE!)
    4. Direct ChromaDB insert (NO RE-EMBED!)
    
    Target: 2620 pages in 3-4 minutes
    """
    start_time = time.time()
    
    print("=" * 80)
    print("🚀 FINAL OPTIMIZED PROCESSING - NO RE-EMBEDDING!")
    print("=" * 80)
    
    pdf_file = Path(pdf_path)
    print(f"\n📄 File: {pdf_file.name}")
    print(f"   Path: {pdf_file}")
    print(f"   Size: {pdf_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # ============================================
    # STEP 1: PARALLEL PDF READING
    # ============================================
    print("\n📖 STEP 1: Parallel PDF reading...")
    pdf_start = time.time()
    
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()
    
    print(f"   Total pages: {total_pages}")
    
    # Split into chunks
    pages_per_worker = 50
    page_ranges = []
    for i in range(0, total_pages, pages_per_worker):
        end = min(i + pages_per_worker, total_pages)
        page_ranges.append((i, end))
    
    print(f"   Work chunks: {len(page_ranges)} ({pages_per_worker} pages each)")
    
    # Read in parallel
    all_pages_text = []
    max_workers = 16
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(read_page_range, pdf_path, start, end): (start, end)
            for start, end in page_ranges
        }
        
        completed = 0
        for future in as_completed(futures):
            start, end = futures[future]
            try:
                pages_data = future.result()
                all_pages_text.extend(pages_data)
                
                completed += 1
                progress = int(completed / len(page_ranges) * 100)
                progress_callback(progress // 5, 100, f"Reading pages {start}-{end}")
            except Exception as e:
                logger.error(f"Error: {e}")
    
    pdf_time = time.time() - pdf_start
    print(f"\n   ✅ Read {total_pages} pages in {pdf_time:.2f}s ({total_pages/pdf_time:.1f} pages/s)")
    
    # ============================================
    # STEP 2: FAST CHUNKING
    # ============================================
    print("\n✂️ STEP 2: Fast chunking...")
    chunk_start = time.time()
    
    chunk_size = 1000
    chunk_overlap = 200
    
    all_chunks = []
    for page_num, page_text in enumerate(all_pages_text):
        if not page_text.strip():
            continue
        
        text_length = len(page_text)
        start = 0
        
        while start < text_length:
            end = start + chunk_size
            chunk_text = page_text[start:end]
            
            if chunk_text.strip():
                all_chunks.append({
                    'text': chunk_text,
                    'page': page_num + 1,
                    'chunk_index': len(all_chunks),
                    'source': pdf_file.name
                })
            
            start = end - chunk_overlap
    
    chunk_time = time.time() - chunk_start
    print(f"   ✅ Created {len(all_chunks)} chunks in {chunk_time:.2f}s")
    progress_callback(25, 100, f"Created {len(all_chunks)} chunks")
    
    # ============================================
    # STEP 3: BATCH EMBEDDING (ONCE!)
    # ============================================
    print("\n🧠 STEP 3: Batch embedding (ONCE, NO RE-EMBED!)...")
    embed_start = time.time()
    
    # Load Vietnamese multilingual model
    print("   Loading model: AITeamVN/Vietnamese_Embedding_v2 (multilingual)")
    print("   → Supports Vietnamese + English")
    print("   → Dimension: 1024, Max seq: 2048")
    model = SentenceTransformer('AITeamVN/Vietnamese_Embedding_v2')
    model.max_seq_length = 2048  # Set max sequence length
    
    # Extract texts
    texts = [chunk['text'] for chunk in all_chunks]
    
    # Embed in mega-batches
    batch_size = 500
    all_embeddings = []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        
        # Encode batch
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        all_embeddings.extend(batch_embeddings.tolist())
        
        current_batch = (batch_idx // batch_size) + 1
        progress = 25 + int(current_batch / total_batches * 60)
        progress_callback(progress, 100, f"Embedding batch {current_batch}/{total_batches}")
    
    embed_time = time.time() - embed_start
    print(f"\n   ✅ Embedded {len(all_embeddings)} chunks in {embed_time:.2f}s")
    print(f"   Speed: {len(all_embeddings)/embed_time:.1f} chunks/second")
    
    # ============================================
    # STEP 4: DIRECT CHROMADB INSERT (NO RE-EMBED!)
    # ============================================
    print("\n💾 STEP 4: Direct ChromaDB insert (NO RE-EMBED!)...")
    store_start = time.time()
    
    progress_callback(90, 100, "Initializing vector store...")
    
    # IMPORTANT: Use SAME model as embedding (Vietnamese_Embedding_v2, dim=1024)
    # Create a custom embedding service with Vietnamese model
    from langchain_huggingface import HuggingFaceEmbeddings
    
    vietnamese_embedding = HuggingFaceEmbeddings(
        model_name='AITeamVN/Vietnamese_Embedding_v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store with Vietnamese embeddings
    from langchain_chroma.vectorstores import Chroma
    from config.constants import DatabaseConstants
    
    vectorstore = Chroma(
        persist_directory=DatabaseConstants.VECTOR_STORE_CONFIGS["chroma"]["persist_directory"],
        embedding_function=vietnamese_embedding
    )
    
    # Create Document objects
    documents = []
    for chunk in all_chunks:
        doc = Document(
            page_content=chunk['text'],
            metadata={
                'page': chunk['page'],
                'chunk_index': chunk['chunk_index'],
                'source': chunk['source']
            }
        )
        documents.append(doc)
    
    progress_callback(95, 100, f"Inserting {len(documents)} documents...")
    
    # Generate IDs
    import uuid
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    
    # Extract data
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    # Direct insert in batches (ChromaDB limit: 5461 per batch)
    max_batch_size = 5000
    total_batches = (len(documents) + max_batch_size - 1) // max_batch_size
    
    for batch_idx in range(0, len(documents), max_batch_size):
        end_idx = min(batch_idx + max_batch_size, len(documents))
        
        batch_embeddings = all_embeddings[batch_idx:end_idx]
        batch_texts = texts[batch_idx:end_idx]
        batch_metadatas = metadatas[batch_idx:end_idx]
        batch_ids = ids[batch_idx:end_idx]
        
        # Direct ChromaDB insert (NO RE-EMBED!)
        vectorstore._collection.add(
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        
        current_batch = (batch_idx // max_batch_size) + 1
        logger.info(f"✅ Inserted batch {current_batch}/{total_batches} ({len(batch_ids)} docs)")
    
    store_time = time.time() - store_start
    print(f"\n   ✅ Stored {len(documents)} documents in {store_time:.2f}s (NO RE-EMBED!)")
    
    # ============================================
    # FINAL STATS
    # ============================================
    total_time = time.time() - start_time
    
    progress_callback(100, 100, "✅ Complete!")
    print("\n")
    
    stats = FinalStats(
        total_pages=total_pages,
        total_chunks=len(documents),
        pdf_time=pdf_time,
        chunk_time=chunk_time,
        embed_time=embed_time,
        store_time=store_time,
        total_time=total_time,
        pages_per_second=total_pages / total_time if total_time > 0 else 0
    )
    
    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_final_optimized.py <pdf_path>")
        print('Example: python demo_final_optimized.py "D:\\Downloads\\Machine-Learning-Systems_1.pdf"')
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)
    
    # Process
    stats = process_final_optimized(pdf_path)
    
    # Display results
    print("\n" + "=" * 80)
    print("✅ FINAL OPTIMIZED PROCESSING COMPLETE!")
    print("=" * 80)
    
    print(f"\n📊 STATISTICS:")
    print(f"   Total pages:      {stats.total_pages:,}")
    print(f"   Total chunks:     {stats.total_chunks:,}")
    print(f"   Chunks/page:      {stats.total_chunks/stats.total_pages:.1f}")
    
    print(f"\n⏱️ TIME BREAKDOWN:")
    print(f"   PDF reading:      {stats.pdf_time:7.2f}s ({stats.pdf_time/stats.total_time*100:5.1f}%)")
    print(f"   Chunking:         {stats.chunk_time:7.2f}s ({stats.chunk_time/stats.total_time*100:5.1f}%)")
    print(f"   Embedding:        {stats.embed_time:7.2f}s ({stats.embed_time/stats.total_time*100:5.1f}%)")
    print(f"   Storage:          {stats.store_time:7.2f}s ({stats.store_time/stats.total_time*100:5.1f}%)")
    print(f"   " + "-" * 70)
    print(f"   TOTAL:            {stats.total_time:7.2f}s = {stats.total_time/60:.1f} minutes")
    
    print(f"\n🚀 PERFORMANCE:")
    print(f"   Speed:            {stats.pages_per_second:.1f} pages/second")
    print(f"   Time per page:    {stats.total_time/stats.total_pages:.2f}s")
    
    print("\n" + "=" * 80)
    print("✅ SUCCESS! Vector database ready for queries!")
    print("=" * 80)
    
    # Compare with old method
    print("\n💡 COMPARISON:")
    print(f"   Old method (2 hours):     7200s")
    print(f"   This method:              {stats.total_time:.0f}s")
    print(f"   Speedup:                  {7200/stats.total_time:.1f}x faster! 🔥")


if __name__ == "__main__":
    main()
