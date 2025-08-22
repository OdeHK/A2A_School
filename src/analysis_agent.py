# src/analysis_agent.py
import pandas as pd

class AnalysisAgent:
    def __init__(self, data_path='data/grades.csv'):
        try:
            # Skip rows that start with # and properly read the header
            self.df = pd.read_csv(
                data_path,
                comment='#',  # Skip lines that start with #
                skipinitialspace=True,  # Skip leading spaces
                encoding='utf-8'  # Handle UTF-8 encoding for Vietnamese characters
            )
        except FileNotFoundError:
            self.df = pd.DataFrame()

    def get_class_overview(self, subject: str) -> dict:
        """Phân tích tổng quan kết quả kiểm tra của lớp theo môn."""
        if self.df.empty: return {}
        subject_df = self.df[self.df['subject'] == subject]
        if subject_df.empty: return {}
        # Trả về dictionary thay vì Series để dễ dàng chuyển thành JSON
        return subject_df['grade'].describe().to_dict()

    def detect_students_needing_support(self, subject: str, threshold: float = 5.0) -> list:
        """Phát hiện học sinh cần hỗ trợ và trả về list of dicts."""
        if self.df.empty: return []
        subject_df = self.df[self.df['subject'] == subject]
        low_performers = subject_df[subject_df['grade'] <= threshold]
        return low_performers.to_dict('records')

    def group_students_by_level(self, subject: str) -> dict:
        """Gom nhóm học sinh theo trình độ."""
        if self.df.empty: return {}
        subject_df = self.df[self.df['subject'] == subject]
        
        bins = [0, 5, 6.5, 8, 10]
        labels = ['Yếu', 'Trung bình', 'Khá', 'Giỏi']
        subject_df['level'] = pd.cut(subject_df['grade'], bins=bins, labels=labels, right=True)
        
        groups = {level: data[['student_id', 'student_name']].to_dict('records') 
                  for level, data in subject_df.groupby('level')}
        return groups

