# config/constants.py
"""
Định nghĩa các hằng số được sử dụng trong toàn bộ ứng dụng.
Tách riêng constants ra khỏi settings để dễ quản lý.
"""

from typing import Dict, List


class ModelConstants:
    """Hằng số liên quan đến các model AI"""
    
    # === LLM Provider mappings ===
    SUPPORTED_LLM_PROVIDERS = {
        "nvidia": "NVIDIA NIM inference services",
        "google_gen_ai": "Google Generative AI"

    }
    
    # Default provider: switch to Google GenAI to avoid NVIDIA 403 for users without access
    DEFAULT_LLM_PROVIDER = "google_gen_ai"

    # === Default models cho từng provider ===
    DEFAULT_MODELS = {
        "nvidia": "openai/gpt-oss-20b",
        "google_gen_ai": "models/gemini-2.5-flash-lite",
    }
    
    # === Embedding models ===
    EMBEDDING_MODELS = {
        "nvidia": "nvidia/llama-3.2-nemoretriever-300m-embed-v1",
        "google_gen_ai": "models/gemini-embedding-001",
        "huggingface": "Alibaba-NLP/gte-multilingual-base"
    }

    # === Model limitations ===
    MAX_CONTEXT_LENGTHS = {
        "openai/gpt-oss-20b": 128000
    }
    
    # === Cache directory reference ===
    @staticmethod
    def get_huggingface_cache_dir() -> str:
        """Lấy đường dẫn cache HuggingFace từ StorageConstants"""
        return StorageConstants.HUGGINGFACE_CACHE_DIR
    



class UIConstants:
    """Hằng số liên quan đến giao diện người dùng"""
    
   
    # === Gradio theme colors ===
    THEME_COLORS = {
        "primary": "#2563eb",
        "secondary": "#64748b", 
        "success": "#059669",
        "warning": "#d97706",
        "error": "#dc2626"
    }


class FileConstants:
    """Hằng số liên quan đến xử lý file"""
    
    # === Supported file formats ===
    SUPPORTED_DOCUMENT_TYPES = {
        ".pdf": "PDF Document",
    }
    
    # === File type categorization ===
    TEXT_FILES = [".txt", ".md", ".csv", ".json"]
    DOCUMENT_FILES = [".pdf", ".docx", ".doc", ".pptx"]  
    SPREADSHEET_FILES = [".csv", ".xlsx"]
    WEB_FILES = [".html", ".htm"]
    
    # === MIME types ===
    MIME_TYPES = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".json": "application/json",
        ".html": "text/html"
    }
    
    # === Chunking strategies ===
    CHUNKING_STRATEGIES = {
        "ONE_PAGE": "Một chunk per trang",
        "RECURSIVE": "Chia nhỏ đệ quy theo ký tự",
        "LLM": "Sử dụng LLM phân chia các đoạn văn và tiền xử lý",
    }
    

# === Document Repository folder names ===
class DocumentRepositoryConstants:
    """Hằng số liên quan đến các folder trong DocumentRepository"""
    BASE_DOCUMENTS_DIR = "session_data"
    SESSIONS_DIR = "sessions"
    TEMP_DIR = "temp"
    RAW_FILES_DIR = "raw_files"
    METADATA_DIR = "metadata"
    TOC_DIR = "toc"
    CONTENT_DIR = "content"  # Thêm thư mục cho content data từ toc_extractor
    DOCUMENT_LIBRARY_DIR = "document_library"  # Thêm thư mục cho document library
    VECTOR_STORE_DIR = "vector_store"


class AgentConstants:
    """Hằng số liên quan đến AI Agents"""
    
    # === Agent types ===
    # AGENT_TYPES = {
    #     "main": "Main Teacher Agent",
    #     "exam_creator": "Exam Creation Agent", 
    #     "document_analyzer": "Document Analysis Agent",
    #     "classroom_manager": "Classroom Management Agent",
    #     "lesson_planner": "Lesson Planning Agent"
    # }
    
    # # === Agent capabilities ===
    # AGENT_CAPABILITIES = {
    #     "exam_creator": [
    #         "Tạo câu hỏi trắc nghiệm",
    #         "Tạo câu hỏi tự luận", 
    #         "Phân loại độ khó",
    #         "Export đề thi ra Google Forms"
    #     ],
    #     "document_analyzer": [
    #         "Tóm tắt tài liệu",
    #         "Trích xuất điểm chính",
    #         "Phân tích nội dung",
    #         "So sánh tài liệu"
    #     ],
    #     "classroom_manager": [
    #         "Quản lý Google Classroom",
    #         "Tạo assignment",
    #         "Theo dõi progress",
    #         "Gửi thông báo"
    #     ]
    # }
    
    # # === Tool categories ===
    # TOOL_CATEGORIES = {
    #     "document": "Document Processing Tools",
    #     "google": "Google Services Tools", 
    #     "exam": "Exam Creation Tools",
    #     "analysis": "Analysis Tools",
    #     "utility": "Utility Tools"
    # }
    
    # # === Default agent prompts ===
    # DEFAULT_SYSTEM_PROMPTS = {
    #     "main": """Bạn là một trợ lý AI thông minh cho giáo viên. 
    #     Bạn có thể giúp tạo đề thi, phân tích tài liệu, quản lý lớp học và hỗ trợ giảng dạy.
    #     Hãy sử dụng các công cụ có sẵn để giúp đỡ giáo viên một cách hiệu quả nhất.""",
        
    #     "exam_creator": """Bạn là chuyên gia tạo đề thi và câu hỏi.
    #     Nhiệm vụ của bạn là tạo ra các câu hỏi chất lượng cao, phù hợp với nội dung học tập.""",
        
    #     "document_analyzer": """Bạn là chuyên gia phân tích tài liệu giáo dục.
    #     Hãy giúp giáo viên hiểu rõ nội dung, trích xuất thông tin quan trọng và tóm tắt hiệu quả."""
    # }


class DatabaseConstants:
    """Hằng số liên quan đến database và storage"""
    
    # === Vector store configurations ===
    VECTOR_STORE_CONFIGS = {
        "chroma": {
            "persist_directory": "./vector_db/chroma",
            "collection_name": "teacher_documents"
        },
        "faiss": {
            "index_path": "./vector_db/faiss/index.faiss",
            "metadata_path": "./vector_db/faiss/metadata.json"
        }
    }


class StorageConstants:
    """Hằng số liên quan đến lưu trữ file tạm và session"""
    
    # === Temporary storage paths ===
    TEMP_DATA_DIR = "temp_data"
    SESSION_TEMP_DIR = "temp_data/session_temp"
    HUGGINGFACE_CACHE_DIR = "temp_data/huggingface_cache"
    
    # === Authentication files ===
    CLIENT_SECRET_FILE = "temp_data/client_secret_vscode.json"
    TOKEN_FILE_NAME = "token.json"
    
    @staticmethod
    def get_user_session_dir(username: str) -> str:
        """Lấy đường dẫn folder session của user cụ thể"""
        return f"{StorageConstants.SESSION_TEMP_DIR}/{username}"
    
    @staticmethod
    def get_user_token_path(username: str) -> str:
        """Lấy đường dẫn file token của user cụ thể"""
        return f"{StorageConstants.get_user_session_dir(username)}/{StorageConstants.TOKEN_FILE_NAME}"


# === Global configuration mappings ===
def get_supported_file_extensions() -> List[str]:
    """Lấy danh sách extension được hỗ trợ"""
    return list(FileConstants.SUPPORTED_DOCUMENT_TYPES.keys())


def get_llm_providers() -> List[str]:
    """Lấy danh sách LLM providers được hỗ trợ"""
    return list(ModelConstants.SUPPORTED_LLM_PROVIDERS.keys())


def get_chunking_strategies() -> List[str]:
    """Lấy danh sách chunking strategies"""
    return list(FileConstants.CHUNKING_STRATEGIES.keys())
