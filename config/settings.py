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
    
    # === Vector Database Configuration
    vector_db_dir: str = Field(
        default="./vector_db",
        description="Directory to store vector database"
    )
    
    # === Embedding Configuration
    embedding_chunk_batch_size: int = Field(
        default=20,
        description="Batch size used when adding embedding chunks to the vector store"
    )

    # === MongoDB Configuration ===
    mongodb_uri: str = Field(
        default="",
    )
    
    mongodb_database_name: str = Field(
        default="agent_for_teacher",
        description="MongoDB database name"
    )

    logs_dir: str = Field(
        default="./logs",
        description="Directory to store application logs"
    )
    
    # === UI Configuration ===
    
    max_file_size_mb: int = Field(
        default=25,
        description="Maximum file upload size in MB"
    )
    
    allowed_file_types: List[str] = Field(
        default=[".pdf"],
        description="Allowed file types for upload"
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
