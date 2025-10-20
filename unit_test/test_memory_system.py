"""
Unit Test cho Short-Term Memory System
Test các tính năng của memory và summarization context
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.agent.memory_manager import ShortTermMemory, MemoryEntry
from services.agent.summarization_context import (
    SummarizationContextTracker,
    SummaryScope,
    SummaryStyle
)
import time


def test_short_term_memory():
    """Test ShortTermMemory basic functionality"""
    print("\n" + "="*60)
    print("TEST 1: Short-Term Memory Basic Operations")
    print("="*60)
    
    memory = ShortTermMemory(max_entries=10, decay_minutes=5)
    
    # Test 1: Add user queries
    print("\n--- Adding user queries ---")
    memory.add_user_query(
        query="Tóm tắt sách Python rất là cơ bản",
        task_type="summarization"
    )
    memory.add_user_query(
        query="Tóm tắt chương 1 của sách Python",
        task_type="summarization"
    )
    
    # Test 2: Add agent responses
    print("\n--- Adding agent responses ---")
    memory.add_agent_response(
        response="Đây là tóm tắt sách Python rất là cơ bản...",
        task_type="summarization"
    )
    
    # Test 3: Add document context
    print("\n--- Adding document context ---")
    memory.add_document_context(
        document_id="doc_001",
        document_title="Python rất là cơ bản - Võ Duy Tuấn",
        section_title="Chương 1. Hello World"
    )
    
    # Test 4: Get recent entries
    print("\n--- Recent entries ---")
    recent = memory.get_recent_entries(max_count=5)
    for i, entry in enumerate(recent, 1):
        print(f"{i}. [{entry.entry_type}] {entry.content[:50]}...")
    
    # Test 5: Get conversation history
    print("\n--- Conversation history ---")
    memory.add_agent_response(
        response="Đây là tóm tắt chương 1...",
        task_type="summarization"
    )
    
    history = memory.get_conversation_history(max_turns=3)
    for i, (query, response) in enumerate(history, 1):
        print(f"\nTurn {i}:")
        print(f"  User: {query[:60]}...")
        print(f"  Agent: {response[:60]}...")
    
    # Test 6: Get context for LLM
    print("\n--- Context for LLM ---")
    context = memory.get_context_for_llm(task_type="summarization", max_tokens=500)
    print(context[:300] + "..." if len(context) > 300 else context)
    
    # Test 7: Statistics
    print("\n--- Memory Statistics ---")
    stats = memory.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Test 1 completed successfully!")


def test_summarization_context():
    """Test SummarizationContextTracker functionality"""
    print("\n" + "="*60)
    print("TEST 2: Summarization Context Tracker")
    print("="*60)
    
    tracker = SummarizationContextTracker(max_history=10)
    
    # Test 1: Start document session
    print("\n--- Starting document session ---")
    tracker.start_document_session(
        document_id="doc_001",
        document_title="Python rất là cơ bản - Võ Duy Tuấn"
    )
    print(f"Current document: {tracker.current_document_id}")
    
    # Test 2: Add summaries to history
    print("\n--- Adding summaries to history ---")
    
    # Summary 1: Chương 1
    tracker.add_summary_to_history(
        document_id="doc_001",
        document_title="Python rất là cơ bản - Võ Duy Tuấn",
        section_path=["Chương 1", "Hello World"],
        summary_scope=SummaryScope.SECTION,
        summary_style=SummaryStyle.CONCISE,
        summary_content="Chương này giới thiệu về Python, cách cài đặt và chương trình Hello World đầu tiên.",
        user_query="Tóm tắt ngắn gọn chương 1",
        tokens_used=150
    )
    
    # Summary 2: Chương 2
    tracker.add_summary_to_history(
        document_id="doc_001",
        document_title="Python rất là cơ bản - Võ Duy Tuấn",
        section_path=["Chương 2", "Biến và kiểu dữ liệu"],
        summary_scope=SummaryScope.SECTION,
        summary_style=SummaryStyle.DETAILED,
        summary_content="Chương này trình bày chi tiết về biến, các kiểu dữ liệu cơ bản trong Python như int, float, string, boolean...",
        user_query="Tóm tắt chi tiết chương 2",
        tokens_used=200
    )
    
    print(f"Total summaries: {len(tracker.summary_history)}")
    
    # Test 3: Get recent summaries
    print("\n--- Recent summaries ---")
    recent_summaries = tracker.get_recent_summaries(document_id="doc_001", max_count=5)
    for i, entry in enumerate(recent_summaries, 1):
        path_str = " > ".join(entry.section_path)
        print(f"{i}. {path_str}")
        print(f"   Scope: {entry.summary_scope.value}, Style: {entry.summary_style.value}")
        print(f"   Summary: {entry.summary_content[:80]}...")
    
    # Test 4: Get document summary context
    print("\n--- Document summary context ---")
    doc_context = tracker.get_document_summary_context(
        document_id="doc_001",
        include_summaries=True,
        max_summaries=2
    )
    print(f"Document: {doc_context.get('document_title', 'N/A')}")
    print(f"Visited sections: {doc_context.get('visited_sections_count', 0)}")
    print(f"Recent summaries count: {len(doc_context.get('recent_summaries', []))}")
    
    # Test 5: Build context for summarization
    print("\n--- Context for summarization ---")
    context_str = tracker.build_context_for_summarization(
        document_id="doc_001",
        target_section=["Chương 3", "Vòng lặp"],
        max_context_summaries=2
    )
    print(context_str)
    
    # Test 6: Detect summary scope and style
    print("\n--- Detecting summary scope and style ---")
    
    test_queries = [
        "Tóm tắt toàn bộ sách",
        "Tóm tắt ngắn gọn chương 1",
        "Cho tôi xem chi tiết về phần cài đặt",
        "Liệt kê các điểm chính trong mục này"
    ]
    
    for query in test_queries:
        scope = tracker.detect_summary_scope(query, ["Chương 1"])
        style = tracker.detect_summary_style(query)
        print(f"\nQuery: {query}")
        print(f"  → Scope: {scope.value}, Style: {style.value}")
    
    # Test 7: Statistics
    print("\n--- Tracker Statistics ---")
    stats = tracker.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Test 2 completed successfully!")


def test_memory_integration():
    """Test integration between Memory and Summarization Context"""
    print("\n" + "="*60)
    print("TEST 3: Memory Integration")
    print("="*60)
    
    memory = ShortTermMemory(max_entries=20)
    tracker = SummarizationContextTracker(max_history=10)
    
    # Simulate a multi-turn summarization conversation
    print("\n--- Simulating multi-turn conversation ---")
    
    # Turn 1: User asks for document summary
    print("\n[Turn 1]")
    user_query_1 = "Tóm tắt sách Python rất là cơ bản"
    memory.add_user_query(user_query_1, task_type="summarization")
    
    tracker.start_document_session(
        document_id="doc_001",
        document_title="Python rất là cơ bản - Võ Duy Tuấn"
    )
    
    response_1 = "Sách Python rất là cơ bản là một giáo trình nhập môn về Python..."
    memory.add_agent_response(response_1, task_type="summarization")
    
    tracker.add_summary_to_history(
        document_id="doc_001",
        document_title="Python rất là cơ bản - Võ Duy Tuấn",
        section_path=["Toàn bộ sách"],
        summary_scope=SummaryScope.FULL_DOCUMENT,
        summary_style=SummaryStyle.CONCISE,
        summary_content=response_1,
        user_query=user_query_1,
        tokens_used=100
    )
    
    print(f"User: {user_query_1}")
    print(f"Agent: {response_1[:60]}...")
    
    # Turn 2: User asks for specific chapter
    print("\n[Turn 2]")
    user_query_2 = "Cho tôi xem chi tiết hơn về chương 1"
    memory.add_user_query(user_query_2, task_type="summarization")
    
    # Build context
    context = tracker.build_context_for_summarization(
        document_id="doc_001",
        target_section=["Chương 1"],
        max_context_summaries=1
    )
    
    response_2 = "Chương 1 giới thiệu về Python, bao gồm lịch sử, ưu điểm và cách cài đặt..."
    memory.add_agent_response(response_2, task_type="summarization")
    
    tracker.add_summary_to_history(
        document_id="doc_001",
        document_title="Python rất là cơ bản - Võ Duy Tuấn",
        section_path=["Chương 1"],
        summary_scope=SummaryScope.CHAPTER,
        summary_style=SummaryStyle.DETAILED,
        summary_content=response_2,
        user_query=user_query_2,
        tokens_used=150
    )
    
    print(f"User: {user_query_2}")
    print(f"Context used:\n{context[:200]}...")
    print(f"Agent: {response_2[:60]}...")
    
    # Turn 3: User asks for another chapter
    print("\n[Turn 3]")
    user_query_3 = "Tiếp theo, tóm tắt chương 2"
    memory.add_user_query(user_query_3, task_type="summarization")
    
    # Get conversation history
    conv_history = memory.get_conversation_history(max_turns=2)
    print(f"\nConversation history (last 2 turns):")
    for i, (q, r) in enumerate(conv_history, 1):
        print(f"  Turn {i}: {q[:40]}... → {r[:40]}...")
    
    # Get LLM context
    llm_context = memory.get_context_for_llm(task_type="summarization", max_tokens=500)
    print(f"\nLLM Context length: {len(llm_context)} chars")
    
    response_3 = "Chương 2 trình bày về biến và các kiểu dữ liệu trong Python..."
    memory.add_agent_response(response_3, task_type="summarization")
    
    tracker.add_summary_to_history(
        document_id="doc_001",
        document_title="Python rất là cơ bản - Võ Duy Tuấn",
        section_path=["Chương 2"],
        summary_scope=SummaryScope.CHAPTER,
        summary_style=SummaryStyle.CONCISE,
        summary_content=response_3,
        user_query=user_query_3,
        tokens_used=120
    )
    
    print(f"User: {user_query_3}")
    print(f"Agent: {response_3[:60]}...")
    
    # Final statistics
    print("\n--- Final Statistics ---")
    print("\nMemory:")
    mem_stats = memory.get_statistics()
    for key, value in mem_stats.items():
        print(f"  {key}: {value}")
    
    print("\nSummarization Context:")
    sum_stats = tracker.get_statistics()
    for key, value in sum_stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Test 3 completed successfully!")


def test_temporal_decay():
    """Test temporal decay functionality"""
    print("\n" + "="*60)
    print("TEST 4: Temporal Decay")
    print("="*60)
    
    memory = ShortTermMemory(max_entries=10, decay_minutes=1)  # 1 minute for testing
    
    print("\n--- Adding entries with delays ---")
    
    # Add first entry
    entry1 = memory.add_user_query("Query 1", task_type="test")
    print(f"Added entry 1 at {entry1.timestamp.strftime('%H:%M:%S')}")
    print(f"  Initial importance: {entry1.importance_score:.2f}")
    
    # Wait a bit
    time.sleep(2)
    
    # Add second entry
    entry2 = memory.add_user_query("Query 2", task_type="test")
    print(f"\nAdded entry 2 at {entry2.timestamp.strftime('%H:%M:%S')}")
    print(f"  Initial importance: {entry2.importance_score:.2f}")
    
    # Check decayed importance
    print("\n--- Checking decayed importance ---")
    from datetime import datetime
    
    for entry in memory.entries:
        decayed = memory._calculate_decayed_importance(entry)
        age_seconds = (datetime.now() - entry.timestamp).total_seconds()
        print(f"Entry: {entry.content[:30]}")
        print(f"  Age: {age_seconds:.1f} seconds")
        print(f"  Original importance: {entry.importance_score:.2f}")
        print(f"  Decayed importance: {decayed:.2f}")
    
    print("\n✓ Test 4 completed successfully!")


def test_save_load_memory():
    """Test saving and loading memory"""
    print("\n" + "="*60)
    print("TEST 5: Save and Load Memory")
    print("="*60)
    
    # Create and populate memory
    memory1 = ShortTermMemory(max_entries=10)
    
    print("\n--- Creating memory ---")
    memory1.add_user_query("Tóm tắt sách Python", task_type="summarization")
    memory1.add_agent_response("Đây là tóm tắt...", task_type="summarization")
    memory1.add_document_context(
        document_id="doc_001",
        document_title="Python rất là cơ bản"
    )
    
    stats1 = memory1.get_statistics()
    print(f"Memory 1 entries: {stats1['current_entries']}")
    
    # Save to file
    test_file = "test_memory.json"
    print(f"\n--- Saving to {test_file} ---")
    memory1.save_to_file(test_file)
    
    # Load into new memory instance
    print(f"\n--- Loading from {test_file} ---")
    memory2 = ShortTermMemory(max_entries=10)
    memory2.load_from_file(test_file)
    
    stats2 = memory2.get_statistics()
    print(f"Memory 2 entries: {stats2['current_entries']}")
    
    # Verify
    print("\n--- Verifying loaded entries ---")
    for entry in memory2.entries:
        print(f"  [{entry.entry_type}] {entry.content[:50]}...")
    
    # Cleanup
    import os
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n✓ Cleaned up {test_file}")
    
    print("\n✓ Test 5 completed successfully!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print(" "*20 + "SHORT-TERM MEMORY SYSTEM TESTS")
    print("="*80)
    
    try:
        test_short_term_memory()
        test_summarization_context()
        test_memory_integration()
        test_temporal_decay()
        test_save_load_memory()
        
        print("\n" + "="*80)
        print(" "*25 + "ALL TESTS PASSED! ✓")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
