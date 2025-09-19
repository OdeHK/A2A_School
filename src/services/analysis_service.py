# src/services/analysis_service.py
# Lớp service điều phối logic cho chức năng phân tích CSV.

import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Sử dụng relative import để giữ cấu trúc module gọn gàng
from ..agents.analysis_agent import AnalysisAgent

logger = logging.getLogger(__name__)

class AnalysisService:
    """
    Service chịu trách nhiệm điều phối các tác vụ liên quan đến phân tích file CSV.
    Nó hoạt động như một lớp trung gian giữa UI và AnalysisAgent, giúp tách biệt logic.
    """
    def __init__(self, analysis_agent: AnalysisAgent):
        # Áp dụng Dependency Injection: nhận AnalysisAgent từ bên ngoài.
        # Điều này giúp code dễ dàng kiểm thử (testing) và bảo trì.
        self.analysis_agent = analysis_agent
        self.current_analysis_results: Optional[Dict[str, Any]] = None
        logger.info("✅ AnalysisService đã được khởi tạo.")

    def analyze_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Thực hiện một quy trình phân tích đầy đủ trên một file CSV.
        
        Args:
            file_path (str): Đường dẫn đến file CSV.
        
        Returns:
            Dict[str, Any]: Một dictionary chứa kết quả phân tích hoặc thông báo lỗi.
        """
        logger.info(f"Bắt đầu quy trình phân tích cho file: {file_path}")
        
        # Bước 1: Sử dụng agent để tải dữ liệu
        success = self.analysis_agent.load_data(file_path)
        if not success:
            logger.error(f"Analysis agent không thể tải dữ liệu từ {file_path}")
            return {"error": "Không thể tải hoặc đọc file CSV. Vui lòng kiểm tra định dạng và nội dung file."}

        # Bước 2: Thực hiện các bước phân tích tuần tự
        logger.info("Đang thực hiện thống kê cơ bản...")
        basic_stats = self.analysis_agent.get_basic_statistics()
        
        logger.info("Đang tạo các biểu đồ trực quan...")
        visualizations = self.analysis_agent.generate_visualizations()
        
        # Bước 3: Tập hợp kết quả và trả về
        self.current_analysis_results = {
            "file_name": Path(file_path).name,
            "basic_stats": basic_stats,
            "visualizations": visualizations,
        }
        
        logger.info(f"Hoàn tất phân tích cho file: {file_path}")
        return self.current_analysis_results

