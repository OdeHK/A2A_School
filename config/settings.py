# config/settings.py
"""
Quản lý cấu hình ứng dụng sử dụng Pydantic Settings.
Hỗ trợ load từ environment variables và .env file.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """
    Main settings class sử dụng Pydantic để validate và manage config
    """
    
    # === API Configuration ===
    nvidia_api_key: Optional[str] = Field(
        default=None,
        description="NVIDIA API key for LLM access"
    )
    
    google_api_key: Optional[str] = Field(
        default=None, 
        description="Google API key for LLM access"
    )
    
    # === Model Configuration ===
    default_llm_provider: str = Field(
        default="nvidia",
        description="Default LLM provider (nvidia, openai, anthropic)"
    )
    
    default_model_name: str = Field(
        default="openai/gpt-oss-20b",
        description="Default model name"
    )
    
    max_tokens: int = Field(
        default=15000,
        description="Maximum tokens for LLM response"
    )
    
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM creativity"
    )
    
    # === Vector Store Configuration ===
    vector_store_type: str = Field(
        default="chroma",
        description="Vector store type (chroma, faiss, pinecone)"
    )
    
    embedding_model: str = Field(
        default="models/gemini-embedding-001",
        description="Embedding model name"
    )
    
    chunk_size: int = Field(
        default=3000,
        description="Document chunk size for splitting"
    )
    
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between document chunks"
    )
    
    # === File and Directory Paths ===
    documents_dir: str = Field(
        default="./documents",
        description="Directory to store uploaded documents"
    )
    
    vector_db_dir: str = Field(
        default="./vector_db",
        description="Directory to store vector database"
    )
    
    logs_dir: str = Field(
        default="./logs",
        description="Directory to store application logs"
    )
    
    # === UI Configuration ===
    app_title: str = Field(
        default="AI Teacher Assistant",
        description="Application title"
    )
    
    app_description: str = Field(
        default="Trợ lý AI thông minh cho giáo viên",
        description="Application description"
    )
    
    max_file_size_mb: int = Field(
        default=25,
        description="Maximum file upload size in MB"
    )
    
    allowed_file_types: List[str] = Field(
        default=[".pdf"],
        description="Allowed file types for upload"
    )
    
    # === Google Services Configuration ===
    google_credentials_file: Optional[str] = Field(
        default=None,
        description="Path to Google service account credentials JSON"
    )
    
    google_drive_folder_id: Optional[str] = Field(
        default=None,
        description="Google Drive folder ID for document storage"
    )
    
    # === Database Configuration ===
    database_url: str = Field(
        default="sqlite:///./teacher_assistant.db",
        description="Database URL for storing sessions and metadata"
    )
    
    # === Logging Configuration ===
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    enable_debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = "allow"
        
    def get_vector_db_path(self) -> str:
        """Lấy đường dẫn đầy đủ đến thư mục vector database"""
        os.makedirs(self.vector_db_dir, exist_ok=True)
        return self.vector_db_dir
    
    def get_documents_path(self) -> str:
        """Lấy đường dẫn đầy đủ đến thư mục documents"""
        os.makedirs(self.documents_dir, exist_ok=True)
        return self.documents_dir
    
    def get_logs_path(self) -> str:
        """Lấy đường dẫn đầy đủ đến thư mục logs"""
        os.makedirs(self.logs_dir, exist_ok=True)
        return self.logs_dir
    
    def is_api_key_configured(self, provider: str) -> bool:
        """Kiểm tra xem API key có được cấu hình hay không"""
        if provider.lower() == "nvidia":
            return self.nvidia_api_key is not None
        elif provider.lower() == "google_gen_ai":
            return self.google_api_key is not None
        return False
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Lấy API key cho provider cụ thể"""
        if provider.lower() == "nvidia":
            return self.nvidia_api_key
        elif provider.lower() == "google_gen_ai":
            return self.google_api_key
        return None


@lru_cache()
def get_settings() -> Settings:
    """Lấy instance Settings với caching"""
    return Settings()


def update_api_key(provider: str, api_key: str) -> bool:
    """
    Cập nhật API key trong runtime.
    Trả về True nếu thành công, False nếu provider không hỗ trợ.
    """
    settings = get_settings()
    
    if provider.lower() == "nvidia":
        settings.nvidia_api_key = api_key
        return True
    elif provider.lower() == "google_gen_ai":
        settings.google_api_key = api_key
        return True
    return False


# Validate settings khi import module
def validate_settings():
    """Validate cấu hình cơ bản khi khởi động ứng dụng"""
    settings = get_settings()
    
    # Tạo các thư mục cần thiết
    settings.get_documents_path()
    settings.get_vector_db_path() 
    settings.get_logs_path()
    
    # Warning nếu không có API key nào được cấu hình
    if not any([
        settings.nvidia_api_key,
        settings.google_api_key,
    ]):
        print("⚠️ Warning: Không có API key nào được cấu hình. Vui lòng set API key trong .env file hoặc environment variables.")
    
    return True


# Auto validate khi import
validate_settings()
