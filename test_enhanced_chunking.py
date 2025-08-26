#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Enhanced Math-Aware Chunking
"""

import sys
import os
sys.path.append('.')

from src.rag import RAGManager

def test_enhanced_chunking():
    """Test the enhanced math-aware chunking functionality"""
    
    # Test content with math formulas
    test_content = '''
Toán học căn bản:

1. Phương trình bậc hai: ax² + bx + c = 0
   Nghiệm: x = (-b ± √(b²-4ac)) / 2a

2. Tích phân: ∫ x² dx = x³/3 + C

3. Ma trận định thức:
   det(A) = |a b|
            |c d| = ad - bc

4. Lượng giác:
   sin²(x) + cos²(x) = 1
   
5. Giới hạn:
   lim(x→0) sin(x)/x = 1

6. Vector trong không gian:
   v⃗ = (x, y, z) ∈ ℝ³
   ||v⃗|| = √(x² + y² + z²)

7. Phép biến đổi LaTeX:
   $$\\begin{equation}
   E = mc²
   \\end{equation}$$

8. Lý thuyết tập hợp:
   A ∪ B = {x | x ∈ A hoặc x ∈ B}
   A ∩ B = {x | x ∈ A và x ∈ B}
'''

    print('📝 Testing Enhanced Math-Aware Chunking...')
    
    # Create a dummy RAGManager just to access the method
    class DummyChunker:
        pass
    class DummyEmbedder:
        pass
    class DummyLLM:
        pass
    
    rag = RAGManager(DummyChunker(), DummyEmbedder(), DummyLLM())
    
    # Test chunking (method doesn't take chunk_size parameter)
    print(f'\n🔍 Testing enhanced math-aware chunking')
    chunks = rag.enhanced_math_aware_chunking(test_content)
    
    print(f'✅ Generated {len(chunks)} chunks')
    
    for i, chunk in enumerate(chunks):
        print(f'\n--- Chunk {i+1} (length: {len(chunk)}) ---')
        # Hiển thị 150 ký tự đầu
        display_text = chunk[:150]
        if len(chunk) > 150:
            display_text += "..."
        print(f'Content: {display_text}')
        
        # Đếm math patterns
        math_patterns = ['=', '±', '√', '∫', '∈', '∪', '∩', '²', '³']
        math_count = sum(chunk.count(pattern) for pattern in math_patterns)
        if math_count > 0:
            print(f'🔢 Math patterns found: {math_count}')

if __name__ == "__main__":
    test_enhanced_chunking()
