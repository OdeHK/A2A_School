# main.py
# ĐÂY LÀ ĐIỂM BẮT ĐẦU (ENTRY POINT) ĐỂ CHẠY TOÀN BỘ ỨNG DỤNG GRADIO
# Chạy file này từ terminal: python main.py

import logging
from dotenv import load_dotenv
load_dotenv()
from src.ui.app_interface import A2ASchoolApp
from config.settings import AppConfig

# Cấu hình logging cơ bản cho toàn bộ ứng dụng
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Chào mừng người dùng và kiểm tra cấu hình
def run_app():
    """
    Hàm chính để khởi tạo và chạy ứng dụng.
    """
    logger = logging.getLogger("MainApp")
    logger.info("🚀 Bắt đầu khởi tạo A2A School Platform...")

    # Tải cấu hình ứng dụng từ biến môi trường
    config = AppConfig()

    # Kiểm tra API Key quan trọng
    if not config.OPENROUTER_API_KEY or "your_" in config.OPENROUTER_API_KEY:
        logger.warning("="*60)
        logger.warning("⚠️ CẢNH BÁO: OPENROUTER_API_KEY chưa được cấu hình!")
        logger.warning("Các tính năng liên quan đến LLM sẽ chạy ở chế độ MOCK (giả lập).")
        logger.warning("Vui lòng tạo file .env và thêm key: OPENROUTER_API_KEY='sk-or-...'")
        logger.warning("="*60)
    else:
        logger.info("✅ OPENROUTER_API_KEY đã được tải thành công.")

    # Khởi tạo và chạy giao diện Gradio
    try:
        app = A2ASchoolApp(config)
        app.launch()
    except Exception as e:
        logger.error(f"❌ Đã xảy ra lỗi nghiêm trọng khi khởi chạy ứng dụng: {e}", exc_info=True)
        print("\n[LỖI] Không thể khởi chạy ứng dụng. Vui lòng kiểm tra log để biết chi tiết.")

if __name__ == "__main__":
    run_app()
