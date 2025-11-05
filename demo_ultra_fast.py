"""
Demo: Ultra-Fast PDF Processing
================================

Test ultra-fast processor with large PDFs

Usage:
    python demo_ultra_fast.py [pdf_path]
"""

import sys
import logging
import asyncio
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from services.document_processing.ultra_fast_processor import UltraFastPDFProcessor
from services.rag.embedding_service import EmbeddingService, EmbeddingType
from services.rag.vector_service import VectorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_callback(current, total, message):
    """Progress bar callback"""
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = int(current / total * 100)
    print(f"\r[{bar}] {percent:3d}% - {message}", end='', flush=True)


async def demo_ultra_fast():
    """Demo ultra-fast PDF processing"""
    
    print("=" * 80)
    print("⚡ ULTRA-FAST PDF PROCESSING DEMO - DETAILED PROFILING")
    print("=" * 80)
    
    # Get PDF path from args or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "example_data/large_document.pdf"
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"\n❌ PDF not found: {pdf_path}")
        print("\nUsage:")
        print("  python demo_ultra_fast.py [pdf_path]")
        print("\nExample:")
        print("  python demo_ultra_fast.py example_data/textbook.pdf")
        return
    
    print(f"\n📄 Processing: {pdf_file.name}")
    print(f"   Path: {pdf_file}")
    print(f"   Size: {pdf_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("-" * 80)
    
    # Initialize services
    print("\n🔧 Initializing services...")
    embedding_service = EmbeddingService.create_with_type(EmbeddingType.HUGGINGFACE)
    vector_service = VectorService(embedding_service=embedding_service)
    vector_service.init_vectorstore()
    print("   ✅ Services initialized")
    
    # Initialize processor
    processor = UltraFastPDFProcessor(
        batch_size=200,      # Embed 200 chunks at once
        max_workers=8,       # 8 parallel PDF readers
        chunk_size=1000,     # 1000 chars per chunk
        chunk_overlap=200    # 200 chars overlap
    )
    
    # Process PDF
    print(f"\n⚡ Processing with ultra-fast processor...")
    print("-" * 80)
    
    import time
    start_time = time.time()
    
    stats = processor.process_pdf(
        pdf_path=str(pdf_file),
        embedding_service=embedding_service,
        vector_service=vector_service,
        force_reprocess=False,
        progress_callback=progress_callback
    )
    
    total_time = time.time() - start_time
    
    print("\n\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 80)
    
    # Display statistics
    print(f"\n📊 STATISTICS:")
    print(f"   Total pages:      {stats.total_pages:,}")
    print(f"   Total chunks:     {stats.total_chunks:,}")
    print(f"   Avg chunk size:   {stats.total_chunks / stats.total_pages if stats.total_pages > 0 else 0:.1f} chunks/page")
    
    print(f"\n⏱️ TIME BREAKDOWN:")
    print(f"   PDF reading:      {stats.pdf_reading_time:>6.2f}s ({stats.pdf_reading_time/total_time*100:>5.1f}%)")
    print(f"   Chunking:         {stats.chunking_time:>6.2f}s ({stats.chunking_time/total_time*100:>5.1f}%)")
    print(f"   Embedding:        {stats.embedding_time:>6.2f}s ({stats.embedding_time/total_time*100:>5.1f}%)")
    print(f"   Storage:          {stats.storage_time:>6.2f}s ({stats.storage_time/total_time*100:>5.1f}%)")
    print(f"   " + "-" * 40)
    print(f"   TOTAL:            {total_time:>6.2f}s (100.0%)")
    
    print(f"\n💾 CACHE:")
    print(f"   Cache hit:        {'Yes ⚡' if stats.cache_hit else 'No (first run)'}")
    
    # Performance comparison
    print(f"\n🚀 PERFORMANCE:")
    
    # Estimate old processing time
    # Assumptions:
    # - Sequential PDF read: 0.15s/page
    # - Sequential embedding: 0.06s/chunk
    old_read_time = stats.total_pages * 0.15
    old_embed_time = stats.total_chunks * 0.06
    old_total = old_read_time + old_embed_time + stats.chunking_time + stats.storage_time
    
    speedup = old_total / total_time if total_time > 0 else 0
    
    print(f"   Estimated old time: {old_total/60:>6.1f} min")
    print(f"   New time:           {total_time/60:>6.1f} min")
    print(f"   Speedup:            {speedup:>6.1f}x faster! 🎉")
    
    # Pages per second
    pps = stats.total_pages / total_time if total_time > 0 else 0
    print(f"\n📈 THROUGHPUT:")
    print(f"   Pages/second:     {pps:.1f}")
    print(f"   Time per page:    {total_time/stats.total_pages if stats.total_pages > 0 else 0:.3f}s")
    
    print("\n" + "=" * 80)
    
    # Test cache (re-run)
    if not stats.cache_hit:
        print("\n🔄 Testing cache (re-running same file)...")
        print("-" * 80)
        
        cache_start = time.time()
        
        stats_cached = processor.process_pdf(
            pdf_path=str(pdf_file),
            embedding_service=embedding_service,
            vector_service=vector_service,
            force_reprocess=False,
            progress_callback=progress_callback
        )
        
        cache_time = time.time() - cache_start
        
        print("\n\n" + "=" * 80)
        print("✅ CACHE TEST COMPLETE!")
        print("=" * 80)
        print(f"\n💾 CACHE PERFORMANCE:")
        print(f"   First run:        {total_time:.2f}s")
        print(f"   Cached run:       {cache_time:.2f}s")
        print(f"   Cache speedup:    {total_time/cache_time if cache_time > 0 else 0:.1f}x faster! ⚡")
        print("=" * 80)


def main():
    """Main entry point"""
    try:
        asyncio.run(demo_ultra_fast())
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
