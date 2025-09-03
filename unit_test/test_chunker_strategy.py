"""
Test script for the new Strategy Pattern implementation of DocumentChunker.
This script demonstrates how to use the different chunking strategies.
"""

import logging
from typing import List
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the new implementation
from services.document_chunker import (
    DocumentChunker,
    ChunkingStrategyType,
    create_recursive_chunker,
    create_page_chunker,
    create_llm_chunker,
    OnePagePerChunkStrategy,
    RecursiveTextSplitStrategy,
    NvidiaLLMService
)

def create_sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is the first page with some content about introduction to AI.",
            metadata={"page": 1, "source": "test.pdf"}
        ),
        Document(
            page_content="This is the second page containing detailed explanation of machine learning algorithms and their applications in various domains.",
            metadata={"page": 2, "source": "test.pdf"}
        ),
        Document(
            page_content="The third page discusses deep learning neural networks and how they revolutionize the field of artificial intelligence.",
            metadata={"page": 3, "source": "test.pdf"}
        )
    ]

def test_page_chunker():
    """Test the one-page-per-chunk strategy."""
    print("\n=== Testing Page Chunker ===")
    
    chunker = create_page_chunker()
    documents = create_sample_documents()
    
    chunks = chunker.chunk(iter(documents))
    
    print(f"Original documents: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Strategy: {chunker.strategy.strategy_name}")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.page_content[:50]}...")
        print(f"Metadata: {chunk.metadata}")

def test_recursive_chunker():
    """Test the recursive text splitting strategy."""
    print("\n=== Testing Recursive Chunker ===")
    
    chunker = create_recursive_chunker(chunk_size=100, chunk_overlap=20)
    documents = create_sample_documents()
    
    chunks = chunker.chunk(iter(documents))
    
    print(f"Original documents: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Strategy: {chunker.strategy.strategy_name}")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.page_content[:50]}...")
        print(f"Metadata: {chunk.metadata}")

def test_strategy_switching():
    """Test switching strategies at runtime."""
    print("\n=== Testing Strategy Switching ===")
    
    # Start with page chunker
    chunker = create_page_chunker()
    documents = create_sample_documents()
    
    print(f"Initial strategy: {chunker.strategy.strategy_name}")
    chunks1 = chunker.chunk(iter(documents))
    print(f"Chunks with page strategy: {len(chunks1)}")
    
    # Switch to recursive strategy
    recursive_strategy = RecursiveTextSplitStrategy(chunk_size=80, chunk_overlap=10)
    chunker.set_strategy(recursive_strategy)
    
    print(f"New strategy: {chunker.strategy.strategy_name}")
    chunks2 = chunker.chunk(iter(documents))
    print(f"Chunks with recursive strategy: {len(chunks2)}")

def test_direct_strategy_usage():
    """Test using strategies directly."""
    print("\n=== Testing Direct Strategy Usage ===")
    
    documents = create_sample_documents()
    
    # Test OnePagePerChunkStrategy directly
    page_strategy = OnePagePerChunkStrategy()
    page_chunks = page_strategy.chunk(iter(documents))
    print(f"Page strategy chunks: {len(page_chunks)}")
    
    # Test RecursiveTextSplitStrategy directly
    recursive_strategy = RecursiveTextSplitStrategy(chunk_size=120, chunk_overlap=15)
    recursive_chunks = recursive_strategy.chunk(iter(documents))
    print(f"Recursive strategy chunks: {len(recursive_chunks)}")

def test_llm_chunker():
    """Test LLM-based chunking (if API key is available)."""
    print("\n=== Testing LLM Chunker ===")
    
    try:
        llm_chunker = create_llm_chunker()
        documents = create_sample_documents()
        
        chunks = llm_chunker.chunk(iter(documents))
        
        print(f"Original documents: {len(documents)}")
        print(f"LLM chunks created: {len(chunks)}")
        print(f"Strategy: {llm_chunker.strategy.strategy_name}")
        
        for i, chunk in enumerate(chunks):
            print(f"LLM Chunk {i+1}:")
            print(f"  Title: {chunk.metadata.get('summary_title', 'N/A')}")
            print(f"  Content: {chunk.page_content[:100]}...")
            print(f"  Pages: {chunk.metadata.get('start_page_index', 'N/A')}-{chunk.metadata.get('end_page_index', 'N/A')}")
            
    except ValueError as e:
        print(f"LLM chunker could not be created: {e}")
        print("This is expected if NVIDIA API key is not configured.")

if __name__ == "__main__":
    print("Testing Strategy Pattern Implementation for DocumentChunker")
    print("="*60)
    
    # Test basic functionality
    test_page_chunker()
    test_recursive_chunker()
    test_strategy_switching()
    test_direct_strategy_usage()
    test_llm_chunker()
    
    print("\n" + "="*60)
    print("All tests completed!")
