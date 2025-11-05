"""
🔥 BLAZING FAST DEMO
====================

Xử lý PDF 2600 trang trong 2-5 PHÚT (không phải 2 giờ!)

Usage:
    python demo_blazing_fast.py "D:\Downloads\Machine-Learning-Systems_1.pdf"
"""

import sys
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from services.document_processing.blazing_fast_processor import BlazingFastProcessor
from services.rag.embedding_service import EmbeddingService, EmbeddingType
from services.rag.vector_service import VectorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_callback(current, total, message):
    """Progress bar"""
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r[{bar}] {current:3d}% - {message}", end='', flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_blazing_fast.py <pdf_path>")
        print('Example: python demo_blazing_fast.py "D:\\Downloads\\Machine-Learning-Systems_1.pdf"')
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("🔥 BLAZING FAST PDF PROCESSING")
    print("=" * 80)
    print(f"\n📄 File: {pdf_file.name}")
    print(f"   Path: {pdf_file}")
    print(f"   Size: {pdf_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("-" * 80)
    
    # Initialize services
    print("\n🔧 Initializing services...")
    embedding_service = EmbeddingService.create_with_type(EmbeddingType.HUGGINGFACE)
    vector_service = VectorService(embedding_service=embedding_service)
    vector_service.init_vectorstore()
    print("   ✅ Services ready")
    
    # Initialize processor
    processor = BlazingFastProcessor(
        batch_size=1000,    # MEGA batch!
        max_workers=16,     # More workers!
        chunk_size=800,
        chunk_overlap=100
    )
    
    # Estimate time
    import fitz
    doc = fitz.open(str(pdf_file))
    total_pages = doc.page_count
    doc.close()
    
    estimate = processor.estimate_time(total_pages)
    print(f"\n📊 Processing {total_pages} pages")
    print(f"   Estimated time: {estimate['total_minutes']:.1f} minutes")
    print("-" * 80)
    
    # Process
    print("\n⚡ Processing...")
    import time
    start = time.time()
    
    stats = processor.process_pdf(
        pdf_path=str(pdf_file),
        embedding_service=embedding_service,
        vector_service=vector_service,
        progress_callback=progress_callback
    )
    
    actual_time = time.time() - start
    
    # Results
    print("\n\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 80)
    
    print(f"\n📊 STATISTICS:")
    print(f"   Total pages:      {stats.total_pages:,}")
    print(f"   Total chunks:     {stats.total_chunks:,}")
    print(f"   Chunks/page:      {stats.total_chunks/stats.total_pages:.1f}")
    
    print(f"\n⏱️ TIME BREAKDOWN:")
    print(f"   PDF reading:      {stats.pdf_time:6.1f}s ({stats.pdf_time/stats.total_time*100:5.1f}%)")
    print(f"   Chunking:         {stats.chunk_time:6.1f}s ({stats.chunk_time/stats.total_time*100:5.1f}%)")
    print(f"   Embedding:        {stats.embed_time:6.1f}s ({stats.embed_time/stats.total_time*100:5.1f}%)")
    print(f"   Storage:          {stats.store_time:6.1f}s ({stats.store_time/stats.total_time*100:5.1f}%)")
    print(f"   " + "-" * 60)
    print(f"   TOTAL:            {stats.total_time:6.1f}s = {stats.total_time/60:.1f} minutes")
    
    print(f"\n🚀 PERFORMANCE:")
    print(f"   Speed:            {stats.pages_per_second:.1f} pages/second")
    print(f"   Time per page:    {stats.total_time/stats.total_pages:.2f}s")
    
    # Compare with estimate
    if estimate['total_time'] > 0:
        accuracy = (stats.total_time / estimate['total_time']) * 100
        print(f"\n   Estimated:        {estimate['total_minutes']:.1f} min")
        print(f"   Actual:           {stats.total_time/60:.1f} min")
        print(f"   Accuracy:         {accuracy:.0f}%")
    
    print("\n" + "=" * 80)
    print("✅ DONE! Vector database is ready for queries.")
    print("=" * 80)


if __name__ == "__main__":
    main()
