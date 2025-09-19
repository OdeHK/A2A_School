# src/agents/analysis_agent.py
# Agent chuyên biệt cho việc phân tích dữ liệu từ file CSV với Plotly visualizations.

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AnalysisAgent:
    """
    Agent thực hiện các tác vụ phân tích dữ liệu học tập từ DataFrame của Pandas.
    """
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        logger.info("✅ AnalysisAgent đã được khởi tạo.")

    def load_data(self, file_path: str) -> bool:
        """
        Tải dữ liệu từ file CSV với khả năng thử nhiều encoding.
        Trả về True nếu thành công, False nếu thất bại.
        """
        logger.info(f"🔄 Đang tải dữ liệu từ file CSV: {file_path}")
        try:
            # Thử các encoding phổ biến để tăng khả năng tương thích
            for encoding in ['utf-8', 'cp1252', 'latin1']:
                try:
                    self.data = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Tải CSV thành công với encoding: {encoding}")
                    # Xóa các khoảng trắng thừa trong tên cột
                    self.data.columns = self.data.columns.str.strip()
                    return True
                except UnicodeDecodeError:
                    continue
            
            logger.error("Không thể giải mã file CSV với các encoding được hỗ trợ.")
            return False
        except Exception as e:
            logger.error(f"Lỗi khi tải file CSV: {e}", exc_info=True)
            return False

    def get_basic_statistics(self) -> Dict[str, Any]:
        """Thực hiện thống kê mô tả cơ bản trên dữ liệu."""
        if self.data is None: return {"error": "Chưa có dữ liệu."}
        
        logger.info("📊 Đang thực hiện thống kê cơ bản...")
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        
        return {
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "numeric_stats": self.data[numeric_cols].describe().to_dict() if numeric_cols else {}
        }

    def generate_visualizations(self) -> Dict[str, Any]:
        """
        Tạo các biểu đồ trực quan tương tác sử dụng Plotly.
        """
        if self.data is None: return {"error": "Chưa có dữ liệu."}
        
        logger.info("🎨 Đang tạo biểu đồ trực quan tương tác...")
        visualizations = {}
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols: return {}

        # Biểu đồ phân phối
        try:
            if len(numeric_cols) == 1:
                # Single column histogram
                fig = px.histogram(
                    self.data, 
                    x=numeric_cols[0], 
                    nbins=20,
                    title=f'Phân phối của cột {numeric_cols[0]}',
                    labels={numeric_cols[0]: numeric_cols[0], 'count': 'Tần suất'}
                )
            else:
                # Multiple columns - create subplots
                fig = make_subplots(
                    rows=(len(numeric_cols) + 1) // 2, 
                    cols=2,
                    subplot_titles=numeric_cols
                )
                
                for i, col in enumerate(numeric_cols):
                    row = (i // 2) + 1
                    col_num = (i % 2) + 1
                    fig.add_trace(
                        go.Histogram(x=self.data[col], name=col, nbinsx=20),
                        row=row, col=col_num
                    )
                
                fig.update_layout(
                    title_text="Phân phối của các cột số",
                    showlegend=False,
                    height=400 * ((len(numeric_cols) + 1) // 2)
                )
            
            visualizations['distribution'] = fig.to_json()
        except Exception as e:
            logger.warning(f"Không thể tạo biểu đồ phân phối: {e}")

        # Biểu đồ tương quan (nếu có nhiều hơn 1 cột số)
        if len(numeric_cols) > 1:
            try:
                corr_matrix = self.data[numeric_cols].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title='Ma trận tương quan',
                    xaxis_title="Cột",
                    yaxis_title="Cột",
                    width=600,
                    height=500
                )
                
                visualizations['correlation'] = fig.to_json()
            except Exception as e:
                logger.warning(f"Không thể tạo biểu đồ tương quan: {e}")

        # Biểu đồ scatter plot cho 2 cột số đầu tiên
        if len(numeric_cols) >= 2:
            try:
                fig = px.scatter(
                    self.data, 
                    x=numeric_cols[0], 
                    y=numeric_cols[1],
                    title=f'Tương quan giữa {numeric_cols[0]} và {numeric_cols[1]}',
                    labels={numeric_cols[0]: numeric_cols[0], numeric_cols[1]: numeric_cols[1]}
                )
                
                # Add trend line
                fig.add_trace(go.Scatter(
                    x=self.data[numeric_cols[0]], 
                    y=self.data[numeric_cols[1]],
                    mode='markers',
                    name='Dữ liệu',
                    opacity=0.6
                ))
                
                visualizations['scatter'] = fig.to_json()
            except Exception as e:
                logger.warning(f"Không thể tạo biểu đồ scatter: {e}")

        return visualizations
