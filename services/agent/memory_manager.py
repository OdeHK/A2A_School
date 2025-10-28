import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """
    Đơn vị lưu trữ memory cơ bản
    
    Attributes:
        timestamp: Thời điểm tạo entry
        entry_type: Loại entry (user_query, agent_response)
        content: Nội dung chính
        metadata: Thông tin bổ sung
        importance_score: Điểm quan trọng (0.0 - 1.0)
        task_type: Loại task (summary, quiz, rag, general)
    """
    timestamp: datetime
    entry_type: str  # user_query, agent_response
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5
    task_type: str = "general"  # summary, quiz, rag, general
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "entry_type": self.entry_type,
            "content": self.content,
            "metadata": self.metadata,
            "importance_score": self.importance_score,
            "task_type": self.task_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class ShortTermMemory:
    """
    Short-Term Memory Manager đơn giản cho Agent
    
    Tính năng:
    1. Sliding window với max_entries
    2. Temporal decay - giảm importance theo thời gian
    3. Priority-based retrieval
    """
    
    def __init__(
        self,
        max_entries: int = 50,
        decay_minutes: int = 30,
        min_importance_threshold: float = 0.1
    ):
        """
        Khởi tạo Short-Term Memory
        
        Args:
            max_entries: Số lượng entries tối đa (sliding window)
            decay_minutes: Thời gian để importance giảm xuống 50%
            min_importance_threshold: Ngưỡng importance tối thiểu để giữ lại
        """
        self.max_entries = max_entries
        self.decay_minutes = decay_minutes
        self.min_importance_threshold = min_importance_threshold
        
        # Lưu trữ memory entries với sliding window
        self.entries: deque[MemoryEntry] = deque(maxlen=max_entries)
        
        # Statistics
        self.total_entries_added = 0
        self.total_entries_evicted = 0
        
        logger.info(f"ShortTermMemory initialized: max_entries={max_entries}, "
                   f"decay_minutes={decay_minutes}")
        
        # Statistics
        self.total_entries_added = 0
        self.total_entries_evicted = 0
        
        logger.info(f"ShortTermMemory initialized: max_entries={max_entries}, "
                   f"decay_minutes={decay_minutes}")
    
    def add_entry(
        self,
        entry_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: float = 0.5,
        document_id: Optional[str] = None,
        task_type: str = "general"
    ) -> MemoryEntry:
        """
        Thêm một entry mới vào memory
        
        Args:
            entry_type: Loại entry (user_query, agent_response)
            content: Nội dung
            metadata: Metadata bổ sung
            importance_score: Điểm quan trọng ban đầu (0.0 - 1.0)
            document_id: Id của tài liệu liên quan
            task_type: Loại task (summary, quiz, rag, general)
            
        Returns:
            MemoryEntry đã tạo
        """
        entry_metadata = metadata.copy() if metadata else {}
        if document_id:
            entry_metadata['document_id'] = document_id
        entry = MemoryEntry(
            timestamp=datetime.now(),
            entry_type=entry_type,
            content=content,
            metadata=entry_metadata,
            importance_score=importance_score,
            task_type=task_type
        )
        
        self.entries.append(entry)
        self.total_entries_added += 1
        
        # Cleanup old entries nếu cần
        self._cleanup_old_entries()
        
        logger.debug(f"Added memory entry: type={entry_type}, "
                    f"importance={importance_score:.2f}")
        
        return entry
    
    def add_user_query(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        task_type: str = "general"
    ) -> MemoryEntry:
        """
        Thêm user query vào memory (importance cao)
        
        Args:
            query: Câu hỏi/yêu cầu của user
            metadata: Metadata bổ sung
            document_id: Id của tài liệu liên quan
            task_type: Loại task (summary, quiz, rag, general)
        
        Returns:
            MemoryEntry đã tạo
        """
        return self.add_entry(
            entry_type="user_query",
            content=query,
            metadata=metadata,
            importance_score=0.9,  
            document_id=document_id,
            task_type=task_type
        )    
    def add_agent_response(
        self,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        task_type: str = "general"
    ) -> MemoryEntry:
        """
        Thêm agent response vào memory
        
        Args:
            response: Câu trả lời của agent
            metadata: Metadata bổ sung
            document_id: Id của tài liệu liên quan
            task_type: Loại task (summary, quiz, rag, general)
        
        Returns:
            MemoryEntry đã tạo
        """
        return self.add_entry(
            entry_type="agent_response",
            content=response,
            metadata=metadata,
            importance_score=0.7,  # Responses quan trọng vừa
            document_id=document_id,
            task_type=task_type
        )
    
    def get_recent_entries(
        self,
        max_count: Optional[int] = None,
        entry_type: Optional[str] = None,
        min_importance: Optional[float] = None,
        document_id: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """
        Lấy các entries gần đây với filtering
        
        Args:
            max_count: Số lượng entries tối đa
            entry_type: Lọc theo entry type
            min_importance: Importance tối thiểu (sau khi apply decay)
            document_id: Lọc theo document id
            task_type: Lọc theo task type (summary, quiz, rag, general)
        
        Returns:
            List các MemoryEntry phù hợp
        """
        # Apply decay và filter
        filtered_entries = []
        
        for entry in reversed(self.entries):  # Từ mới nhất đến cũ nhất
            # Apply temporal decay
            current_importance = self._calculate_decayed_importance(entry)
            # Check minimum importance
            if min_importance and current_importance < min_importance:
                continue
            # Check entry type filter
            if entry_type and entry.entry_type != entry_type:
                continue
            # Check document_id filter
            if document_id:
                entry_doc_id = entry.metadata.get('document_id')
                if entry_doc_id != document_id:
                    continue
            # Check task_type filter
            if task_type and entry.task_type != task_type:
                continue
            filtered_entries.append(entry)
            # Check max count
            if max_count and len(filtered_entries) >= max_count:
                break
        return filtered_entries
    
    def get_context_for_llm(
        self,
        max_tokens: int = 2000,
        document_id: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> str:
        """
        Tạo context string cho LLM từ memory
        
        Args:
            max_tokens: Số tokens tối đa (ước lượng)
            document_id: Lọc context theo document id
            task_type: Lọc context theo task type (summary, quiz, rag, general)
        
        Returns:
            Context string formatted cho LLM
        """
        entries = self.get_recent_entries(
            max_count=self.max_entries,
            min_importance=self.min_importance_threshold,
            document_id=document_id,
            task_type=task_type
        )
        
        if not entries:
            return ""
        
        # Build context string
        context_parts = []
        
        estimated_tokens = 0
        max_estimated_tokens = max_tokens - 50  
        
        for entry in entries:
            # Format entry
            entry_str = self._format_entry_for_llm(entry)
            
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            entry_tokens = len(entry_str) // 4
            
            if estimated_tokens + entry_tokens > max_estimated_tokens:
                break
            
            context_parts.append(entry_str)
            estimated_tokens += entry_tokens
        
        return "\n".join(context_parts)
    
    
    def _calculate_decayed_importance(self, entry: MemoryEntry) -> float:
        """
        Tính importance sau khi apply temporal decay
        
        Args:
            entry: Memory entry
            
        Returns:
            Decayed importance score
        """
        time_diff = datetime.now() - entry.timestamp
        minutes_passed = time_diff.total_seconds() / 60
        
        # Exponential decay: importance * 0.5^(minutes_passed / decay_minutes)
        decay_factor = 0.5 ** (minutes_passed / self.decay_minutes)
        
        return entry.importance_score * decay_factor
    
    def _cleanup_old_entries(self) -> None:
        """
        Xóa các entries cũ có importance quá thấp
        """
        # Convert to list để có thể modify
        entries_list = list(self.entries)
        
        # Filter out entries with too low importance
        filtered = []
        for entry in entries_list:
            current_importance = self._calculate_decayed_importance(entry)
            if current_importance >= self.min_importance_threshold:
                filtered.append(entry)
            else:
                self.total_entries_evicted += 1
        
        # Update deque
        self.entries = deque(filtered, maxlen=self.max_entries)
    
    def _format_entry_for_llm(self, entry: MemoryEntry) -> str:
        """
        Format một entry cho LLM context
        
        Args:
            entry: Memory entry
            
        Returns:
            Formatted string
        """
        timestamp_str = entry.timestamp.strftime("%H:%M:%S")
        
        if entry.entry_type == "user_query":
            return f"[{timestamp_str}] User: {entry.content}"
        elif entry.entry_type == "agent_response":
            return f"[{timestamp_str}] Agent: {entry.content}"
        else:
            return f"[{timestamp_str}] {entry.content}"
