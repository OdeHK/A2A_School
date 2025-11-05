"""
🔍 DIAGNOSTIC DEMO - Find the REAL bottleneck!
==============================================

Chạy test này TRƯỚC để biết chính xác vấn đề ở đâu:
- PDF reading chậm?
- Embedding chậm?
- Database chậm?

Usage:
    python demo_diagnostic.py "D:\Downloads\Machine-Learning-Systems_1.pdf"
"""

import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from services.document_processing.blazing_fast_processor import diagnose_performance

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_diagnostic.py <pdf_path>")
        print("Example: python demo_diagnostic.py \"D:\\Downloads\\Machine-Learning-Systems_1.pdf\"")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)
    
    # Run diagnostic (test 100 pages)
    results = diagnose_performance(pdf_path, sample_pages=100)
    
    print("\n" + "=" * 80)
    print("✅ DIAGNOSTIC COMPLETE!")
    print("=" * 80)
    print("\nNow run the blazing fast processor:")
    print(f'python demo_blazing_fast.py "{pdf_path}"')
