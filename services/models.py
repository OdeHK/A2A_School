"""
Data models for document processing and session management.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from enum import Enum


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Document metadata model"""
    document_id: str
    file_name: str
    file_path: str
    file_size: int
    upload_date: datetime
    processing_status: ProcessingStatus
    chunk_count: Optional[int] = None
    page_count: Optional[int] = None
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TocSection(BaseModel):
    """Table of Contents section model"""
    section_id: str
    section_title: str
    parent_section_id: Optional[str] = None
    level: int
    page_number: Optional[int] = None
    children: List['TocSection'] = []
    
    class Config:
        # Enable forward references for recursive model
        validate_assignment = True


class TableOfContents(BaseModel):
    """Complete Table of Contents model"""
    document_id: str
    extraction_method: str  # "library" or "llm"
    extraction_date: datetime
    sections: List[TocSection]
    raw_text: Optional[str] = None  # Original extracted text
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionMetadata(BaseModel):
    """Session metadata model"""
    session_id: str
    created_date: datetime
    last_accessed: datetime
    documents: List[str] = []  # List of document IDs
    vector_store_path: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingResult(BaseModel):
    """Result of document processing"""
    status: ProcessingStatus
    document_id: str
    file_name: str
    message: str
    metadata: Optional[DocumentMetadata] = None
    table_of_contents: Optional[TableOfContents] = None
    error: Optional[str] = None


# Enable forward references for TocSection
TocSection.model_rebuild()