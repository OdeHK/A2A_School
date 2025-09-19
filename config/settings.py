# config/settings.py
# Quản lý cấu hình tập trung, đọc từ file .env

import os
from dotenv import load_dotenv
from pathlib import Path

# Tải các biến từ file .env ở thư mục gốc của dự án
# Điều này giúp bảo mật các key và dễ dàng thay đổi cấu hình mà không cần sửa code.
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class AppConfig:
    """
    Lớp chứa tất cả các biến cấu hình cho ứng dụng.
    """
    def __init__(self):
        # --- Cấu hình API ---
        # Lấy API key từ biến môi trường, nếu không có sẽ trả về None
        self.OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
        self.API_URL: str = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")

        # --- Cấu hình Model ---
        # Đọc model từ file .env, nếu không có sẽ dùng model mặc định
        self.DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b:free")
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")

        # --- Cấu hình Đường dẫn (Path) ---
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.DB_PATH = self.DATA_DIR / "a2a_school.db"
        self.UPLOADS_DIR = self.DATA_DIR / "uploads"
        self.AGENT_CARDS_DIR = self.DATA_DIR / "agent_cards"
        self.CONFIG_DIR = self.BASE_DIR / "config"
        self.DETECTOR_PROFILES_PATH = self.CONFIG_DIR / "detector_profiles.yml"

        # Tạo các thư mục cần thiết nếu chúng chưa tồn tại
        self.DATA_DIR.mkdir(exist_ok=True)
        self.UPLOADS_DIR.mkdir(exist_ok=True)
        self.AGENT_CARDS_DIR.mkdir(exist_ok=True)

        # --- Cấu hình Gradio ---
        self.GRADIO_SERVER_NAME: str = "127.0.0.1"
        self.GRADIO_SERVER_PORT: int = 7860
        # Mặc định không chia sẻ public link, an toàn hơn
        self.GRADIO_SHARE: bool = False
