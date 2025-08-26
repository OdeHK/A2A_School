"""
Simplified Multi-Agent RAG System for School Management
Không cần langchain dependencies - chỉ dùng pandas và Python built-ins
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re


class SimpleDocument:
    """Simple document class thay thế langchain Document"""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class SimpleLLM:
    """Mock LLM cho demo - thay thế OpenRouter/LangChain LLM"""
    
    def invoke(self, prompt: str) -> str:
        """Mock response dựa trên patterns trong prompt"""
        prompt_lower = prompt.lower()
        
        # Student support responses
        if "eigenvalue" in prompt_lower or "ma trận" in prompt_lower:
            return """Eigenvalue (giá trị riêng) của ma trận A là số λ sao cho:
            A·v = λ·v (với v là vector riêng khác 0)
            
            Cách tìm:
            1. Giải phương trình đặc trưng: det(A - λI) = 0
            2. Tìm λ từ phương trình trên
            3. Với mỗi λ, tìm vector riêng v từ (A - λI)v = 0
            
            Ví dụ: Ma trận 2x2 [[3,1],[0,2]] có eigenvalue λ₁=3, λ₂=2"""
        
        elif "phương trình bậc hai" in prompt_lower:
            return """Phương trình bậc hai ax² + bx + c = 0 (a≠0)
            
            Công thức nghiệm: x = (-b ± √Δ)/2a
            Với Δ = b² - 4ac
            
            - Δ > 0: 2 nghiệm phân biệt
            - Δ = 0: 1 nghiệm kép  
            - Δ < 0: vô nghiệm (trong R)"""
        
        elif "hóa học" in prompt_lower:
            return """Một số khái niệm cơ bản trong hóa học:
            - Nguyên tử: đơn vị cơ bản của vật chất
            - Phân tử: nhóm nguyên tử liên kết
            - Ion: nguyên tử/phân tử tích điện
            - Liên kết: cách nguyên tử kết hợp (ion, cộng hóa trị, kim loại)"""
        
        # Teacher support responses  
        elif "chia nhóm" in prompt_lower or "group" in prompt_lower:
            return """Đề xuất chia nhóm học sinh theo tiêu chí được cung cấp.
            Đã phân tích dữ liệu học sinh và tạo nhóm cân bằng."""
        
        elif "phương pháp dạy" in prompt_lower or "teaching method" in prompt_lower:
            return """Dựa trên môn học và chủ đề, đề xuất phương pháp dạy phù hợp:
            - Sử dụng trực quan hóa cho các khái niệm trừu tượng
            - Thực hành qua bài tập có hướng dẫn
            - Thảo luận nhóm để tăng tương tác"""
        
        # Data analysis responses
        elif "phân tích" in prompt_lower or "analyze" in prompt_lower:
            return """Đã phân tích dữ liệu điểm số và đưa ra thống kê chi tiết."""
        
        else:
            return "Xin lỗi, tôi cần thêm thông tin để trả lời câu hỏi này."


class BaseRAGAgent:
    """Base class cho tất cả agents"""
    
    def __init__(self, agent_type: str, knowledge_base: List[SimpleDocument] = None):
        self.agent_type = agent_type
        self.knowledge_base = knowledge_base or []
        self.llm = SimpleLLM()
        
    def add_documents(self, documents: List[SimpleDocument]):
        """Thêm documents vào knowledge base"""
        self.knowledge_base.extend(documents)
    
    def search_knowledge(self, query: str, k: int = 3) -> List[SimpleDocument]:
        """Simple keyword search trong knowledge base"""
        query_words = set(query.lower().split())
        
        # Score documents dựa trên keyword overlap
        scored_docs = []
        for doc in self.knowledge_base:
            content_words = set(doc.page_content.lower().split())
            score = len(query_words.intersection(content_words))
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort và return top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]
    
    def generate_response(self, query: str, context: str = "") -> str:
        """Generate response sử dụng LLM"""
        prompt = f"""Dựa trên context sau:
{context}

Trả lời câu hỏi: {query}

Trả lời:"""
        return self.llm.invoke(prompt)


class StudentSupportAgent(BaseRAGAgent):
    """Agent hỗ trợ học sinh"""
    
    def __init__(self):
        super().__init__("student_support")
        self.subjects = ["toán", "lý", "hóa", "anh", "văn"]
        self.reminders = []  # Store reminders locally
    
    def setup_knowledge_base(self, documents: List[SimpleDocument]):
        """Setup knowledge base với documents"""
        self.add_documents(documents)
        print(f"✅ Setup knowledge base với {len(documents)} documents")
    
    def ask_question(self, question: str, subject: str = None, student_id: str = None) -> Dict[str, Any]:
        """Trả lời câu hỏi học thuật"""
        
        # Detect subject if not provided
        detected_subject = self._detect_subject(question) if not subject else subject
        
        # Search relevant knowledge
        relevant_docs = self.search_knowledge(question)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate answer
        answer = self.generate_response(question, context)
        
        return {
            "answer": answer,
            "detected_subject": detected_subject,
            "confidence": "high",
            "sources": len(relevant_docs)
        }
    
    def answer_academic_question(self, question: str, subject: str = None) -> str:
        """Backend method - Trả lời câu hỏi học thuật (chỉ trả về answer string)"""
        result = self.ask_question(question, subject)
        return result["answer"]
    
    def recommend_materials(self, subject: str, difficulty: str = "medium") -> Dict[str, Any]:
        """Đề xuất tài liệu học tập"""
        
        materials_db = {
            "toán": {
                "easy": ["Sách giáo khoa lớp 10", "Bài tập cơ bản", "Video Khan Academy"],
                "medium": ["Sách nâng cao", "Đề thi thử", "Bài tập ứng dụng"],
                "hard": ["Sách chuyên đề", "Đề Olympic", "Nghiên cứu khoa học"]
            },
            "lý": {
                "easy": ["Thí nghiệm cơ bản", "Video minh họa", "Sách giáo khoa"],
                "medium": ["Bài tập tổng hợp", "Thí nghiệm nâng cao", "Sách bổ trợ"],
                "hard": ["Chuyên đề vật lý", "Nghiên cứu ứng dụng", "Olympic Vật lý"]
            },
            "hóa": {
                "easy": ["Bảng tuần hoàn", "Thí nghiệm an toàn", "Sách cơ bản"],
                "medium": ["Phản ứng hóa học", "Bài tập định lượng", "Thực hành lab"],
                "hard": ["Hóa học hữu cơ", "Phân tích định lượng", "Nghiên cứu chuyên sâu"]
            }
        }
        
        materials = materials_db.get(subject, {}).get(difficulty, ["Tài liệu chung"])
        
        # Generate study plan
        study_plan = [
            {"step": 1, "activity": "Đọc lý thuyết cơ bản", "duration": "30 phút"},
            {"step": 2, "activity": "Làm bài tập mẫu", "duration": "45 phút"},
            {"step": 3, "activity": "Thực hành tự làm", "duration": "60 phút"},
            {"step": 4, "activity": "Ôn tập và kiểm tra", "duration": "30 phút"}
        ]
        
        return {
            "recommendations": {
                "materials": materials,
                "study_plan": study_plan,
                "estimated_time": "2.5 giờ"
            }
        }
    
    def recommend_study_materials(self, subject: str, difficulty: str = "medium"):
        """Backend method - Đề xuất tài liệu học tập"""
        return self.recommend_materials(subject, difficulty)["recommendations"]
    
    def set_reminder(self, student_id: str, reminder_type: str, subject: str, 
                    datetime_str: str, note: str = "") -> Dict[str, Any]:
        """Thiết lập nhắc nhở"""
        
        reminder = {
            "id": len(self.reminders) + 1,
            "student_id": student_id,
            "type": reminder_type,
            "subject": subject,
            "datetime": datetime_str,
            "note": note,
            "created_at": datetime.now().isoformat()
        }
        
        self.reminders.append(reminder)
        
        return {
            "result": {
                "message": f"Đã thiết lập nhắc nhở {reminder_type} cho môn {subject}",
                "reminder_id": reminder["id"]
            }
        }
    
    def set_study_reminder(self, student_id: str, reminder_type: str, subject: str, 
                          datetime_str: str, note: str = ""):
        """Backend method - Thiết lập nhắc nhở học tập"""
        return self.set_reminder(student_id, reminder_type, subject, datetime_str, note)["result"]
    
    def get_reminders(self, student_id: str, days_ahead: int = 7) -> Dict[str, Any]:
        """Lấy danh sách nhắc nhở"""
        
        now = datetime.now()
        future_date = now + timedelta(days=days_ahead)
        
        student_reminders = []
        for reminder in self.reminders:
            if reminder["student_id"] == student_id:
                reminder_date = datetime.fromisoformat(reminder["datetime"])
                if now <= reminder_date <= future_date:
                    student_reminders.append(reminder)
        
        # Sort by datetime
        student_reminders.sort(key=lambda x: x["datetime"])
        
        return {
            "reminders": student_reminders
        }
    
    def get_upcoming_reminders(self, student_id: str, days_ahead: int = 7):
        """Backend method - Lấy nhắc nhở sắp tới"""
        return self.get_reminders(student_id, days_ahead)["reminders"]
    
    def _detect_subject(self, question: str) -> str:
        """Phát hiện môn học từ câu hỏi"""
        question_lower = question.lower()
        
        math_keywords = ["phương trình", "đạo hàm", "tích phân", "ma trận", "hình học", "eigenvalue"]
        physics_keywords = ["lực", "điện", "từ", "ánh sáng", "nhiệt", "sóng"]
        chemistry_keywords = ["phản ứng", "nguyên tử", "phân tử", "axit", "bazơ", "hóa học"]
        english_keywords = ["grammar", "vocabulary", "speaking", "writing", "english"]
        
        if any(keyword in question_lower for keyword in math_keywords):
            return "toán"
        elif any(keyword in question_lower for keyword in physics_keywords):
            return "lý"
        elif any(keyword in question_lower for keyword in chemistry_keywords):
            return "hóa"
        elif any(keyword in question_lower for keyword in english_keywords):
            return "anh"
        else:
            return "tổng hợp"


class TeacherSupportAgent(BaseRAGAgent):
    """Agent hỗ trợ giáo viên"""
    
    def __init__(self):
        super().__init__("teacher_support")
    
    def group_students(self, student_data: List[Dict], criteria: str = "academic_level") -> Dict[str, Any]:
        """Chia nhóm học sinh"""
        
        df = pd.DataFrame(student_data)
        
        if criteria == "academic_level":
            # Chia theo điểm số
            df['performance_group'] = pd.cut(df['average_score'], 
                                           bins=[0, 5, 7, 8.5, 10], 
                                           labels=['Cần hỗ trợ', 'Trung bình', 'Khá', 'Giỏi'])
            
            groups = {}
            for group_name, group_df in df.groupby('performance_group'):
                groups[str(group_name)] = {
                    "count": len(group_df),
                    "students": group_df['name'].tolist(),
                    "average_score": round(group_df['average_score'].mean(), 2),
                    "activities": self._suggest_activities(str(group_name))
                }
        
        elif criteria == "learning_style":
            # Chia theo phong cách học
            groups = {}
            for style, group_df in df.groupby('learning_style'):
                groups[style] = {
                    "count": len(group_df),
                    "students": group_df['name'].tolist(),
                    "average_score": round(group_df['average_score'].mean(), 2),
                    "activities": self._suggest_activities_by_style(style)
                }
        
        return {
            "grouping": {
                "total_students": len(df),
                "grouping_method": criteria,
                "groups": groups,
                "recommendations": [
                    "Tạo hoạt động nhóm đa dạng để phù hợp với từng nhóm",
                    "Kết hợp học sinh giỏi với học sinh yếu để hỗ trợ lẫn nhau",
                    "Điều chỉnh phương pháp dạy theo đặc điểm từng nhóm"
                ]
            }
        }
    
    def suggest_student_grouping(self, student_data: List[Dict], criteria: str):
        """Backend method - Đề xuất chia nhóm học sinh"""
        return self.group_students(student_data, criteria)["grouping"]
    
    def suggest_teaching_method(self, subject: str, class_level: str, topic: str) -> Dict[str, Any]:
        """Đề xuất phương pháp dạy"""
        
        methods_db = {
            "toán": {
                "ma trận": "Sử dụng biểu diễn trực quan và thao tác cụ thể. Bắt đầu với ma trận 2x2 đơn giản.",
                "phương trình": "Áp dụng phương pháp giải từng bước, có hướng dẫn chi tiết.",
                "hình học": "Sử dụng hình vẽ, mô hình 3D và phần mềm hỗ trợ."
            },
            "lý": {
                "điện": "Thí nghiệm thực tế với mạch điện đơn giản, đo đạc cụ thể.",
                "quang học": "Sử dụng laser pointer, gương, thấu kính để minh họa.",
                "cơ học": "Thí nghiệm với các vật thể quen thuộc."
            }
        }
        
        method = methods_db.get(subject, {}).get(topic, 
            f"Phương pháp tích hợp: lý thuyết + thực hành + thảo luận cho chủ đề {topic}")
        
        activities = [
            f"Giới thiệu khái niệm {topic} qua ví dụ thực tế",
            "Thực hành có hướng dẫn từng bước",
            "Hoạt động nhóm để củng cố kiến thức",
            "Đánh giá và phản hồi"
        ]
        
        return {
            "method": {
                "subject": subject,
                "class_level": class_level,
                "topic": topic,
                "teaching_method": method,
                "suggested_activities": activities,
                "estimated_duration": "45 phút"
            }
        }
    
    def find_teaching_materials(self, subject: str, topic: str, material_type: str = "all") -> Dict[str, Any]:
        """Tìm tài liệu giảng dạy"""
        
        materials = {
            "subject": subject,
            "topic": topic,
            "lesson_plans": [
                f"Giáo án {topic} - Lý thuyết cơ bản",
                f"Giáo án {topic} - Bài tập thực hành",
                f"Giáo án {topic} - Ôn tập tổng hợp"
            ],
            "presentations": [
                f"Slide bài giảng {topic}",
                f"Presentation tương tác {topic}",
                f"Video minh họa {topic}"
            ],
            "worksheets": [
                f"Phiếu bài tập {topic} - Cơ bản",
                f"Phiếu bài tập {topic} - Nâng cao",
                f"Đề kiểm tra {topic}"
            ]
        }
        
        if material_type != "all" and material_type in materials:
            return {"materials": {material_type: materials[material_type]}}
        
        return {"materials": materials}
    
    def suggest_teaching_materials(self, subject: str, topic: str, material_type: str = "all"):
        """Backend method - Đề xuất tài liệu giảng dạy"""
        return self.find_teaching_materials(subject, topic, material_type)["materials"]
    
    def _suggest_activities(self, group_name: str) -> List[str]:
        """Đề xuất hoạt động theo nhóm năng lực"""
        activities_map = {
            "Cần hỗ trợ": ["Ôn tập cơ bản", "Bài tập có hướng dẫn", "Hỗ trợ 1-1"],
            "Trung bình": ["Bài tập thực hành", "Thảo luận nhóm", "Dự án nhỏ"],
            "Khá": ["Bài tập ứng dụng", "Thuyết trình", "Hướng dẫn bạn"],
            "Giỏi": ["Bài tập nâng cao", "Nghiên cứu độc lập", "Dẫn dắt nhóm"]
        }
        return activities_map.get(group_name, ["Hoạt động chung"])
    
    def _suggest_activities_by_style(self, style: str) -> List[str]:
        """Đề xuất hoạt động theo phong cách học"""
        style_activities = {
            "visual": ["Sơ đồ tư duy", "Biểu đồ", "Video minh họa"],
            "auditory": ["Thảo luận", "Thuyết trình", "Nghe giảng"],
            "kinesthetic": ["Thí nghiệm", "Mô hình", "Hoạt động thực hành"],
            "reading": ["Đọc tài liệu", "Viết báo cáo", "Nghiên cứu"]
        }
        return style_activities.get(style, ["Hoạt động đa phương thức"])


class DataAnalysisAgent(BaseRAGAgent):
    """Agent phân tích dữ liệu"""
    
    def __init__(self):
        super().__init__("data_analysis")
    
    def analyze_class_performance(self, class_data: List[Dict]) -> Dict[str, Any]:
        """Phân tích hiệu suất lớp học"""
        
        df = pd.DataFrame(class_data)
        
        # Basic statistics
        stats = {
            "total_students": len(df),
            "average_score": round(df['score'].mean(), 2),
            "max_score": df['score'].max(),
            "min_score": df['score'].min(),
            "std_deviation": round(df['score'].std(), 2)
        }
        
        # Performance distribution
        performance_dist = {
            "excellent": len(df[df['score'] >= 8.5]),
            "good": len(df[(df['score'] >= 7) & (df['score'] < 8.5)]),
            "average": len(df[(df['score'] >= 5) & (df['score'] < 7)]),
            "below_average": len(df[df['score'] < 5])
        }
        
        # Recommendations
        recommendations = []
        if performance_dist["below_average"] > stats["total_students"] * 0.3:
            recommendations.append("Cần tăng cường hỗ trợ học sinh yếu")
        if performance_dist["excellent"] < stats["total_students"] * 0.2:
            recommendations.append("Cần thêm bài tập nâng cao cho học sinh giỏi")
        if stats["std_deviation"] > 2:
            recommendations.append("Lớp có sự chênh lệch lớn, cần phân nhóm dạy học")
        
        return {
            "analysis": {
                "class_statistics": stats,
                "performance_summary": performance_dist,
                "recommendations": recommendations
            }
        }
    
    def identify_at_risk_students(self, class_data: List[Dict]) -> Dict[str, Any]:
        """Xác định học sinh cần hỗ trợ"""
        
        df = pd.DataFrame(class_data)
        
        # Define at-risk criteria
        mean_score = df['score'].mean()
        std_score = df['score'].std()
        threshold = mean_score - std_score
        
        at_risk_students = df[df['score'] < threshold]
        
        student_details = []
        for _, student in at_risk_students.iterrows():
            risk_level = "high" if student['score'] < 5 else "medium"
            
            reasons = []
            if student['score'] < 5:
                reasons.append("Điểm số thấp")
            if student['score'] < mean_score - std_score:
                reasons.append("Dưới mức trung bình lớp")
            
            actions = [
                "Tăng cường ôn tập cá nhân",
                "Hỗ trợ thêm từ giáo viên",
                "Ghép với học sinh giỏi",
                "Điều chỉnh phương pháp học"
            ]
            
            student_details.append({
                "student_id": student['student_id'],
                "current_score": student['score'],
                "risk_level": risk_level,
                "reasons": reasons,
                "actions": actions
            })
        
        return {
            "at_risk_students": {
                "total_checked": len(df),
                "need_support": len(at_risk_students),
                "high_risk": len(df[df['score'] < 5]),
                "student_details": student_details
            }
        }
    
    def identify_students_need_support(self, class_data: List[Dict]):
        """Backend method - Xác định học sinh cần hỗ trợ"""
        return self.identify_at_risk_students(class_data)["at_risk_students"]
    
    def predict_trends(self, class_data: List[Dict], prediction_period: int = 6) -> Dict[str, Any]:
        """Dự đoán xu hướng điểm số"""
        
        df = pd.DataFrame(class_data)
        
        # Simple trend analysis
        current_avg = df['score'].mean()
        recent_data = df.tail(len(df)//2) if len(df) > 4 else df
        earlier_data = df.head(len(df)//2) if len(df) > 4 else df
        
        recent_avg = recent_data['score'].mean()
        earlier_avg = earlier_data['score'].mean()
        
        trend_direction = "tăng" if recent_avg > earlier_avg else "giảm" if recent_avg < earlier_avg else "ổn định"
        change_rate = recent_avg - earlier_avg
        
        # Simple prediction (linear trend)
        predictions = []
        for i in range(1, prediction_period + 1):
            predicted_score = current_avg + (change_rate * i / 3)  # Smooth the change
            predicted_score = max(0, min(10, predicted_score))  # Clamp to valid range
            predictions.append(round(predicted_score, 2))
        
        confidence = "cao" if abs(change_rate) < 0.5 else "trung bình"
        
        return {
            "trends": {
                "overall_trend": trend_direction,
                "confidence": confidence,
                "trend_analysis": {
                    "recent_average": round(recent_avg, 2),
                    "earlier_average": round(earlier_avg, 2),
                    "change": round(change_rate, 2)
                },
                "predictions": predictions
            }
        }
    
    def predict_learning_trends(self, class_data: List[Dict], prediction_period: int = 6):
        """Backend method - Dự đoán xu hướng học tập"""
        return self.predict_trends(class_data, prediction_period)["trends"]


class SimpleMultiAgentRAGSystem:
    """Hệ thống Multi-Agent RAG đơn giản"""
    
    def __init__(self):
        self.student_agent = StudentSupportAgent()
        self.teacher_agent = TeacherSupportAgent()
        self.data_agent = DataAnalysisAgent()
        
        # Load sample knowledge base
        self._load_sample_knowledge()
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Tự động định tuyến query đến agent phù hợp"""
        
        query_lower = query.lower()
        
        # Student support keywords
        student_keywords = ["học", "bài tập", "giải", "làm sao", "cách nào", "eigenvalue", "phương trình"]
        teacher_keywords = ["dạy", "giảng", "chia nhóm", "phương pháp", "học sinh", "lớp học"]
        data_keywords = ["phân tích", "thống kê", "điểm", "xu hướng", "báo cáo", "dữ liệu"]
        
        # Detect user type and route to appropriate agent
        if any(keyword in query_lower for keyword in student_keywords):
            return {
                "query": query,
                "detected_user_type": "học sinh",
                "selected_agent": "student_support",
                "confidence": "high"
            }
        elif any(keyword in query_lower for keyword in teacher_keywords):
            return {
                "query": query,
                "detected_user_type": "giáo viên", 
                "selected_agent": "teacher_support",
                "confidence": "high"
            }
        elif any(keyword in query_lower for keyword in data_keywords):
            return {
                "query": query,
                "detected_user_type": "quản lý",
                "selected_agent": "data_analysis", 
                "confidence": "high"
            }
        else:
            return {
                "query": query,
                "detected_user_type": "không xác định",
                "selected_agent": "student_support",  # Default fallback
                "confidence": "low"
            }
    
    def _detect_user_type(self, query: str) -> str:
        """Backend method - Phát hiện loại người dùng từ query"""
        route_result = self.route_query(query)
        return route_result["detected_user_type"]
    
    def _load_sample_knowledge(self):
        """Load sample knowledge base cho demo"""
        
        sample_docs = [
            SimpleDocument(
                "Eigenvalue và eigenvector là khái niệm quan trọng trong đại số tuyến tính. "
                "Eigenvalue λ của ma trận A thỏa mãn Av = λv với v là eigenvector.",
                {"subject": "toán", "topic": "đại số tuyến tính"}
            ),
            SimpleDocument(
                "Phương trình bậc hai ax² + bx + c = 0 có nghiệm x = (-b ± √Δ)/2a "
                "với Δ = b² - 4ac là biệt số.",
                {"subject": "toán", "topic": "phương trình"}
            ),
            SimpleDocument(
                "Định luật Ohm: V = I × R, trong đó V là hiệu điện thế, I là cường độ dòng điện, "
                "R là điện trở.",
                {"subject": "lý", "topic": "điện học"}
            ),
            SimpleDocument(
                "Chia nhóm học sinh hiệu quả cần dựa trên năng lực học tập, phong cách học tập "
                "và tính cách của từng em.",
                {"subject": "giáo dục", "topic": "quản lý lớp học"}
            )
        ]
        
        # Add to all agents
        for agent in [self.student_agent, self.teacher_agent, self.data_agent]:
            agent.add_documents(sample_docs)


def create_sample_data():
    """Tạo dữ liệu mẫu cho test - Trả về 3 values cho backend"""
    
    # Create sample documents for knowledge base
    sample_docs = [
        SimpleDocument(
            "Eigenvalue và eigenvector là khái niệm quan trọng trong đại số tuyến tính. "
            "Eigenvalue λ của ma trận A thỏa mãn Av = λv với v là eigenvector.",
            {"subject": "toán", "topic": "đại số tuyến tính"}
        ),
        SimpleDocument(
            "Phương trình bậc hai ax² + bx + c = 0 có nghiệm x = (-b ± √Δ)/2a "
            "với Δ = b² - 4ac là biệt số.",
            {"subject": "toán", "topic": "phương trình"}
        ),
        SimpleDocument(
            "Định luật Ohm: V = I × R, trong đó V là hiệu điện thế, I là cường độ dòng điện, "
            "R là điện trở.",
            {"subject": "lý", "topic": "điện học"}
        ),
        SimpleDocument(
            "Chia nhóm học sinh hiệu quả cần dựa trên năng lực học tập, phong cách học tập "
            "và tính cách của từng em.",
            {"subject": "giáo dục", "topic": "quản lý lớp học"}
        )
    ]
    
    sample_students = [
        {"student_id": "HS001", "name": "Nguyễn Văn A", "average_score": 8.5, "learning_style": "visual"},
        {"student_id": "HS002", "name": "Trần Thị B", "average_score": 6.2, "learning_style": "auditory"},
        {"student_id": "HS003", "name": "Lê Văn C", "average_score": 4.8, "learning_style": "kinesthetic"},
        {"student_id": "HS004", "name": "Phạm Thị D", "average_score": 9.1, "learning_style": "reading"},
        {"student_id": "HS005", "name": "Hoàng Văn E", "average_score": 7.3, "learning_style": "visual"},
        {"student_id": "HS006", "name": "Võ Thị F", "average_score": 5.9, "learning_style": "auditory"},
        {"student_id": "HS007", "name": "Đặng Văn G", "average_score": 8.8, "learning_style": "kinesthetic"},
        {"student_id": "HS008", "name": "Bùi Thị H", "average_score": 7.1, "learning_style": "reading"}
    ]
    
    sample_grades = []
    subjects = ["toán", "lý", "hóa", "anh", "văn"]
    
    for student in sample_students:
        for subject in subjects:
            # Generate some variation around their average
            base_score = student["average_score"]
            variation = (hash(student["student_id"] + subject) % 21 - 10) / 10  # -1 to +1
            score = max(0, min(10, base_score + variation))
            
            sample_grades.append({
                "student_id": student["student_id"],
                "subject": subject,
                "score": round(score, 1),
                "class": "10A1",
                "exam_date": "2024-12-01"
            })
    
    # Return 3 separate values as expected by backend
    return sample_docs, sample_students, sample_grades


# Test function
def test_simple_multi_agent():
    """Test function để verify system hoạt động"""
    
    print("🚀 Testing Simple Multi-Agent RAG System...")
    
    # Initialize system
    system = SimpleMultiAgentRAGSystem()
    sample_docs, sample_students, sample_grades = create_sample_data()
    
    # Add documents to system
    system.student_agent.add_documents(sample_docs)
    
    # Test student agent
    print("\n📚 Testing Student Agent:")
    result = system.student_agent.ask_question("Eigenvalue của ma trận là gì?", subject="toán")
    print(f"Answer: {result['answer'][:100]}...")
    
    # Test teacher agent  
    print("\n👩‍🏫 Testing Teacher Agent:")
    grouping = system.teacher_agent.group_students(sample_students, "academic_level")
    print(f"Groups created: {len(grouping['grouping']['groups'])}")
    
    # Test data agent
    print("\n📊 Testing Data Agent:")
    analysis = system.data_agent.analyze_class_performance(sample_grades[:25])  # First 25 records
    print(f"Average score: {analysis['analysis']['class_statistics']['average_score']}")
    
    # Test routing
    print("\n🔀 Testing Auto Routing:")
    route_result = system.route_query("Làm sao giải phương trình bậc hai?")
    print(f"Routed to: {route_result['selected_agent']}")
    
    print("\n✅ All tests completed successfully!")
    return True


if __name__ == "__main__":
    test_simple_multi_agent()
