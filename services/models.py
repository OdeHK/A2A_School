

from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum


# =============================
# Quiz Generation Models
# =============================

class PlanTaskOutput(BaseModel):
    section_id: str = Field(..., description="A unique identifier for the section")
    section_title: str = Field(..., description="The official title of the section as listed in the Table of Contents")
    number_of_questions: int = Field(..., description="The number of questions allocated to this section")
    question_requirements: str = Field(
        default="Multiple choice questions with 4 options, containing 1 correct answer, designed for university-level students.",
        description="A brief description of the expected question format and audience. This is derived from the teacher’s instructions"
    )
    query_string: str = Field(
        ...,
        description="A descriptive sentence that explains the context and focus of this section, based on the ToC"
    )

class PlanTaskOutputList(BaseModel):
    tasks: List[PlanTaskOutput]

class QuizQuestion(BaseModel):
    type: str = Field(..., description="Type of question: 'multiple_choice' or 'essay'.")
    title: str = Field(..., description="The question text")
    options: Optional[List[str]] = Field(default=None, description="Multiple choice options (only for multiple_choice type)")
    answer: Optional[str] = Field(default=None, description="Correct answer (only for multiple_choice type)")
    answer_explanation: Optional[str] = Field(default=None, description="Explanation for the answer (only for multiple_choice type)")

class QuizQuestionOutput(BaseModel):
    questions: List[QuizQuestion] = Field(..., description="List of questions in the quiz")

    
# =============================
# Document Processing Models
# =============================

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


# =============================
# Session Management Models
# =============================

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


# =============================
# Processing Result Models
# =============================

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