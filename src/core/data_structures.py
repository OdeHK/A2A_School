# src/core/data_structures.py
# Định nghĩa các cấu trúc dữ liệu cốt lõi, giúp code sạch sẽ và nhất quán.

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np

class ChunkType(Enum):
    """Phân loại các loại nội dung có thể có trong tài liệu."""
    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"
    CODE = "code"

@dataclass
class ChapterInfo:
    """Lưu trữ thông tin về chương mà một chunk thuộc về."""
    number: Optional[int] = None
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dạng dictionary để lưu trữ."""
        return {'number': self.number, 'title': self.title}

@dataclass
class AgenticChunk:
    """
    Cấu trúc chunk nâng cao (Agentic Chunk).
    Không chỉ chứa nội dung, mà còn chứa rất nhiều metadata (siêu dữ liệu)
    giúp cho việc tìm kiếm và xử lý thông minh hơn.
    """
    content: str
    chunk_type: ChunkType
    source_file: str
    position_in_document: int # Vị trí (thứ tự) của chunk trong tài liệu
    chapter_info: Optional[ChapterInfo] = None
    page_number: Optional[int] = None  # Số trang của chunk (để mapping chính xác)
    embedding: Optional[np.ndarray] = field(default=None, repr=False) # Vector embedding, không in ra khi debug

    # Các metadata có thể được thêm vào sau bởi các agent khác
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dạng dictionary."""
        return {
            'content': self.content,
            'chunk_type': self.chunk_type.value,
            'source_file': self.source_file,
            'position_in_document': self.position_in_document,
            'chapter_info': self.chapter_info.to_dict() if self.chapter_info else None,
            'page_number': self.page_number,
            'summary': self.summary,
            'keywords': self.keywords,
        }
