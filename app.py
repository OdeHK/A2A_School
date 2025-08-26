# app.py (phiên bản cập nhật cuối cùng)

import streamlit as st
import requests
import pandas as pd
import altair as alt

# --- Cấu hình ---
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="A2A School Client", layout="wide")
st.title("👨‍🏫 Hệ thống Agent Thông minh cho Trường học")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

# --- Giao diện Sidebar ---
with st.sidebar:
    st.header("📤 Tải lên Tài liệu")
    uploaded_file = st.file_uploader("Chọn file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        if st.button("Xử lý Tài liệu"):
            with st.spinner("⏳ Đang xử lý file..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/process-document/", files=files)
                    
                    if response.status_code == 200:
                        st.session_state.document_processed = True
                        st.session_state.uploaded_filename = uploaded_file.name
                        st.success("✅ Xử lý thành công!")
                        st.rerun()
                    else:
                        st.error(f"❌ Lỗi: {response.text}")
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối: {str(e)}")

# --- Giao diện chính với các Tab ---
tab_student, tab_teacher, tab_analysis, tab_multi_agent = st.tabs([
    "🎓 Dành cho Học sinh", 
    "👩‍🏫 Dành cho Giáo viên", 
    "📊 Dành cho Quản lý",
    "🤖 Multi-Agent RAG"
])

# --- Tab Học sinh ---
with tab_student:
    st.header("Trợ lý Học tập Cá nhân")
    if not st.session_state.document_processed:
        st.warning("Vui lòng tải lên một tài liệu (giáo trình) ở sidebar để các chức năng hoạt động.")
    else:
        student_task = st.selectbox("Chọn một chức năng:", 
                                    ["Hỏi đáp nhanh (RAG)", "Trò chuyện với Chuyên gia", "Gợi ý Đề tài Nghiên cứu"])

        if student_task == "Hỏi đáp nhanh (RAG)":
            # ... (code chat cũ của bạn) ...
            pass
        
        elif student_task == "Trò chuyện với Chuyên gia":
            st.info("Ở chế độ này, AI sẽ sử dụng toàn bộ kiến thức trong tài liệu để trả lời sâu hơn.")
            
            # Hiển thị lịch sử chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
                st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Đang suy nghĩ..."):
                        try:
                            response = requests.post(
                                f"{BACKEND_URL}/student/expert-chat/",
                                json={
                                    "question": prompt,
                                    "chat_history": st.session_state.messages
                                }
                            )
                            if response.status_code == 200:
                                answer = response.json()["answer"]
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                st.write(answer)
                            else:
                                st.error(f"❌ Lỗi: {response.text}")
                        except Exception as e:
                            st.error(f"❌ Lỗi kết nối: {str(e)}")
        
        elif student_task == "Gợi ý Đề tài Nghiên cứu":
            if st.button("Tìm kiếm ý tưởng"):
                with st.spinner("🔍 Đang phân tích tài liệu..."):
                    response = requests.post(f"{BACKEND_URL}/student/suggest-topics/", json={"num_topics": 3})
                    if response.status_code == 200:
                        st.markdown(response.json().get("topics"))
                    else:
                        st.error(f"Lỗi: {response.text}")

# --- Tab Giáo viên ---
with tab_teacher:
    st.header("Trợ lý Giảng dạy")
    teacher_task = st.selectbox("Chọn một chức năng:", ["Tạo Quiz từ Tài liệu", "Gợi ý Phương pháp dạy"])

    if teacher_task == "Tạo Quiz từ Tài liệu":
        if not st.session_state.document_processed:
            st.warning("⚠️ Vui lòng tải lên và xử lý tài liệu trước!")
        else:
            num_questions = st.number_input("Số lượng câu hỏi:", min_value=1, max_value=10, value=5)
            subject = st.text_input("Môn học:", "Quiz chung")
            
            if st.button("Tạo Quiz"):
                with st.spinner("🤔 Đang tạo quiz..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/generate-quiz/",
                            json={"num_questions": num_questions, "subject": subject}
                        )
                        if response.status_code == 200:
                            quiz_data = response.json().get("quiz_data")
                            st.session_state.quiz_data = quiz_data
                            
                            # Hiển thị quiz
                            st.success("✅ Đã tạo quiz thành công!")
                            for i, question in enumerate(quiz_data, 1):
                                st.subheader(f"Câu {i}:")
                                st.write(question["question"])
                                
                                # Hiển thị các lựa chọn
                                choices = question.get("choices", [])
                                for choice in choices:
                                    st.write(choice)  # Đã bao gồm A., B., C., D.
                                    
                                # Hiển thị đáp án
                                correct_idx = int(question.get("correct_answer", 1))
                                correct_choice = choices[correct_idx - 1] if 0 < correct_idx <= len(choices) else "Không xác định"
                                st.write(f"📍 Đáp án đúng: {correct_choice}")
                                st.write("---")
                        else:
                            st.error(f"❌ Lỗi: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối: {str(e)}")

    elif teacher_task == "Gợi ý Phương pháp dạy":
        st.info("Chức năng này sẽ phân tích dữ liệu điểm số của lớp (từ file grades.csv) để đưa ra gợi ý.")
        subject = st.text_input("Nhập môn học:", "Toán")
        lesson_topic = st.text_input("Nhập chủ đề bài học sắp tới:", "Lượng giác")
        if st.button("Tạo Gợi ý Giảng dạy"):
            with st.spinner("Agent đang phân tích và suy luận..."):
                payload = {"subject": subject, "lesson_topic": lesson_topic}
                response = requests.post(f"{BACKEND_URL}/teacher/generate-suggestions/", json=payload)
                if response.status_code == 200:
                    st.markdown(response.json().get("suggestion"))
                else:
                    st.error(f"Lỗi: {response.text}")

# --- Tab RAG Chat ---
tab_main, tab_teacher, tab_analysis, tab_chat = st.tabs(["🎓 Học sinh", "👨‍🏫 Giáo viên", "📊 Phân tích", "💬 Trò chuyện"])

# In tab_chat
with tab_chat:
    st.header("💬 Trò chuyện với Chuyên gia")
    
    if not st.session_state.document_processed:
        st.warning("⚠️ Vui lòng tải lên và xử lý tài liệu trước!")
    else:
        # Hiển thị lịch sử chat
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            with st.chat_message(role):
                st.write(content)
        
        # Chat input
        if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
            # Thêm tin nhắn của người dùng
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Gọi API expert-chat
            with st.chat_message("assistant"):
                with st.spinner("🤔 Đang suy nghĩ..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/student/expert-chat/",
                            json={
                                "question": prompt,
                                "chat_history": st.session_state.messages
                            }
                        )
                        if response.status_code == 200:
                            answer = response.json()["answer"]
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.write(answer)
                        else:
                            st.error(f"❌ Lỗi: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối: {str(e)}")

# --- Tab Phân tích ---
with tab_analysis:
    st.header("📊 Phân tích Dữ liệu Học sinh")
    
    subject_input = st.text_input("Nhập môn học để phân tích:", "Toán")
    if st.button("Phân tích"):
        with st.spinner("Đang phân tích dữ liệu..."):
            try:
                # Gọi API để lấy thống kê tổng quan
                response = requests.get(f"{BACKEND_URL}/teacher/class-overview/{subject_input}")
                if response.status_code == 200:
                    data = response.json()
                    
                    # Hiển thị thống kê
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Điểm trung bình", f"{data['mean']:.2f}")
                    with col2:
                        st.metric("Điểm cao nhất", f"{data['max']:.2f}")
                    with col3:
                        st.metric("Điểm thấp nhất", f"{data['min']:.2f}")
                    
                    # Hiển thị biểu đồ phân phối điểm
                    if 'distribution' in data:
                        chart_data = pd.DataFrame(data['distribution'])
                        st.line_chart(chart_data)
                else:
                    st.error(f"❌ Lỗi: {response.text}")
            except Exception as e:
                st.error(f"❌ Lỗi kết nối: {str(e)}")

# --- Tab Multi-Agent RAG ---
with tab_multi_agent:
    st.header("🤖 Multi-Agent RAG System")
    st.markdown("Hệ thống AI đa agent chuyên biệt cho giáo dục")
    
    # Agent selector
    agent_type = st.selectbox(
        "Chọn Agent:",
        ["🎓 Student Support", "👩‍🏫 Teacher Support", "📊 Data Analysis", "🔀 Auto Route"]
    )
    
    if agent_type == "🎓 Student Support":
        st.subheader("🎓 Student Support Agent")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            student_action = st.radio(
                "Chọn chức năng:",
                ["Hỏi đáp học thuật", "Đề xuất tài liệu", "Thiết lập nhắc nhở", "Xem nhắc nhở"]
            )
        
        with col2:
            student_id = st.text_input("Student ID:", value="HS001")
        
        if student_action == "Hỏi đáp học thuật":
            question = st.text_area("Câu hỏi của bạn:", placeholder="VD: Eigenvalue của ma trận là gì?")
            subject = st.selectbox("Môn học:", ["", "toán", "lý", "hóa", "anh", "văn"])
            
            if st.button("Hỏi Agent", key="student_ask"):
                if question:
                    with st.spinner("🤔 Agent đang suy nghĩ..."):
                        try:
                            response = requests.post(f"{BACKEND_URL}/agent/student/ask", 
                                json={
                                    "question": question,
                                    "subject": subject if subject else None,
                                    "student_id": student_id
                                })
                            
                            if response.status_code == 200:
                                data = response.json()
                                st.success("✅ Trả lời từ Student Agent:")
                                st.write(data["answer"])
                                st.info(f"📚 Môn học được phát hiện: {data['detected_subject']}")
                            else:
                                st.error(f"❌ Lỗi: {response.text}")
                        except Exception as e:
                            st.error(f"❌ Lỗi kết nối: {str(e)}")
        
        elif student_action == "Đề xuất tài liệu":
            col1, col2 = st.columns(2)
            with col1:
                subject = st.selectbox("Môn học:", ["toán", "lý", "hóa", "anh", "văn"])
            with col2:
                difficulty = st.selectbox("Mức độ:", ["easy", "medium", "hard"])
            
            if st.button("Lấy đề xuất", key="student_recommend"):
                with st.spinner("📚 Đang tìm tài liệu phù hợp..."):
                    try:
                        response = requests.post(f"{BACKEND_URL}/agent/student/recommend", 
                            params={"subject": subject, "difficulty": difficulty})
                        
                        if response.status_code == 200:
                            data = response.json()
                            recommendations = data["recommendations"]
                            
                            st.success("✅ Đề xuất tài liệu:")
                            
                            st.write("📖 **Tài liệu đề xuất:**")
                            for material in recommendations.get("materials", []):
                                st.write(f"- {material}")
                            
                            st.write("📋 **Kế hoạch học tập:**")
                            for step in recommendations.get("study_plan", []):
                                st.write(f"{step['step']}. {step['activity']} ({step['duration']})")
                        else:
                            st.error(f"❌ Lỗi: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối: {str(e)}")
        
        elif student_action == "Thiết lập nhắc nhở":
            col1, col2 = st.columns(2)
            with col1:
                reminder_type = st.selectbox("Loại nhắc nhở:", ["class", "exam", "assignment"])
                subject = st.selectbox("Môn học:", ["toán", "lý", "hóa", "anh", "văn"])
            with col2:
                reminder_date = st.date_input("Ngày:")
                reminder_time = st.time_input("Giờ:")
            
            note = st.text_area("Ghi chú:", placeholder="VD: Kiểm tra chương ma trận")
            
            if st.button("Thiết lập nhắc nhở", key="student_reminder"):
                datetime_str = f"{reminder_date}T{reminder_time}"
                
                try:
                    response = requests.post(f"{BACKEND_URL}/agent/student/reminder",
                        json={
                            "student_id": student_id,
                            "reminder_type": reminder_type,
                            "subject": subject,
                            "datetime_str": datetime_str,
                            "note": note
                        })
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ {data['result']['message']}")
                    else:
                        st.error(f"❌ Lỗi: {response.text}")
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối: {str(e)}")
        
        elif student_action == "Xem nhắc nhở":
            days_ahead = st.slider("Xem nhắc nhở trong bao nhiêu ngày tới:", 1, 30, 7)
            
            if st.button("Lấy nhắc nhở", key="student_get_reminders"):
                try:
                    response = requests.get(f"{BACKEND_URL}/agent/student/reminders/{student_id}",
                        params={"days_ahead": days_ahead})
                    
                    if response.status_code == 200:
                        data = response.json()
                        reminders = data["reminders"]
                        
                        if reminders:
                            st.success(f"✅ Có {len(reminders)} nhắc nhở sắp tới:")
                            for reminder in reminders:
                                st.write(f"⏰ **{reminder['type'].upper()}** - {reminder['subject']}")
                                st.write(f"   📅 {reminder['datetime']}")
                                if reminder['note']:
                                    st.write(f"   📝 {reminder['note']}")
                                st.write("---")
                        else:
                            st.info("📅 Không có nhắc nhở nào trong thời gian này")
                    else:
                        st.error(f"❌ Lỗi: {response.text}")
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối: {str(e)}")
    
    elif agent_type == "👩‍🏫 Teacher Support":
        st.subheader("👩‍🏫 Teacher Support Agent")
        
        teacher_action = st.radio(
            "Chọn chức năng:",
            ["Chia nhóm học sinh", "Đề xuất phương pháp dạy", "Tài liệu giảng dạy"]
        )
        
        if teacher_action == "Chia nhóm học sinh":
            st.write("📝 **Nhập dữ liệu học sinh:**")
            
            # Sample data for demo
            sample_students = [
                {"student_id": "HS001", "name": "Nguyễn Văn A", "average_score": 8.5, "learning_style": "visual"},
                {"student_id": "HS002", "name": "Trần Thị B", "average_score": 6.2, "learning_style": "auditory"},
                {"student_id": "HS003", "name": "Lê Văn C", "average_score": 4.8, "learning_style": "kinesthetic"},
                {"student_id": "HS004", "name": "Phạm Thị D", "average_score": 9.1, "learning_style": "reading"},
                {"student_id": "HS005", "name": "Hoàng Văn E", "average_score": 7.3, "learning_style": "visual"}
            ]
            
            use_sample = st.checkbox("Sử dụng dữ liệu mẫu", value=True)
            
            if use_sample:
                student_data = sample_students
                st.dataframe(pd.DataFrame(student_data))
            else:
                st.info("Tính năng upload dữ liệu thực sẽ được bổ sung")
                student_data = sample_students
            
            criteria = st.selectbox("Tiêu chí chia nhóm:", ["academic_level", "learning_style"])
            
            if st.button("Chia nhóm", key="teacher_group"):
                with st.spinner("👥 Đang phân tích và chia nhóm..."):
                    try:
                        response = requests.post(f"{BACKEND_URL}/agent/teacher/group",
                            json={
                                "student_data": student_data,
                                "criteria": criteria
                            })
                        
                        if response.status_code == 200:
                            data = response.json()
                            grouping = data["grouping"]
                            
                            st.success("✅ Đề xuất chia nhóm:")
                            st.write(f"📊 **Tổng số học sinh:** {grouping['total_students']}")
                            st.write(f"🎯 **Phương pháp:** {grouping['grouping_method']}")
                            
                            for group_name, info in grouping["groups"].items():
                                st.write(f"\n**📌 Nhóm {group_name}:**")
                                st.write(f"- Số lượng: {info['count']} học sinh")
                                st.write(f"- Học sinh: {', '.join(info['students'])}")
                                if 'average_score' in info:
                                    st.write(f"- Điểm TB: {info['average_score']}")
                                if 'activities' in info:
                                    st.write(f"- Hoạt động: {', '.join(info['activities'][:2])}")
                            
                            if grouping.get("recommendations"):
                                st.write("\n💡 **Khuyến nghị:**")
                                for rec in grouping["recommendations"]:
                                    st.write(f"- {rec}")
                        else:
                            st.error(f"❌ Lỗi: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối: {str(e)}")
        
        elif teacher_action == "Đề xuất phương pháp dạy":
            col1, col2, col3 = st.columns(3)
            with col1:
                subject = st.selectbox("Môn học:", ["toán", "lý", "hóa", "anh", "văn"])
            with col2:
                class_level = st.selectbox("Lớp:", ["6", "7", "8", "9", "10", "11", "12"])
            with col3:
                topic = st.text_input("Chủ đề:", placeholder="VD: ma trận")
            
            if st.button("Lấy đề xuất", key="teacher_method"):
                if topic:
                    with st.spinner("🎯 Đang phân tích và đề xuất..."):
                        try:
                            response = requests.post(f"{BACKEND_URL}/agent/teacher/method",
                                json={
                                    "subject": subject,
                                    "class_level": class_level,
                                    "topic": topic
                                })
                            
                            if response.status_code == 200:
                                data = response.json()
                                method = data["method"]
                                
                                st.success("✅ Đề xuất phương pháp dạy:")
                                st.write(f"📚 **Môn:** {method['subject']}")
                                st.write(f"🎓 **Lớp:** {method['class_level']}")
                                st.write(f"📝 **Chủ đề:** {method['topic']}")
                                st.write(f"⏱️ **Thời gian:** {method.get('estimated_duration', 'N/A')}")
                                
                                st.write("🎯 **Phương pháp:**")
                                st.write(method['teaching_method'])
                                
                                if 'suggested_activities' in method:
                                    st.write("📋 **Hoạt động đề xuất:**")
                                    for activity in method['suggested_activities']:
                                        st.write(f"- {activity}")
                            else:
                                st.error(f"❌ Lỗi: {response.text}")
                        except Exception as e:
                            st.error(f"❌ Lỗi kết nối: {str(e)}")
                else:
                    st.warning("⚠️ Vui lòng nhập chủ đề")
        
        elif teacher_action == "Tài liệu giảng dạy":
            col1, col2 = st.columns(2)
            with col1:
                subject = st.selectbox("Môn học:", ["toán", "lý", "hóa", "anh", "văn"])
                topic = st.text_input("Chủ đề:", placeholder="VD: ma trận")
            with col2:
                material_type = st.selectbox("Loại tài liệu:", ["all", "lesson_plans", "presentations", "worksheets"])
            
            if st.button("Tìm tài liệu", key="teacher_materials"):
                if topic:
                    with st.spinner("📚 Đang tìm tài liệu..."):
                        try:
                            response = requests.post(f"{BACKEND_URL}/agent/teacher/materials",
                                params={
                                    "subject": subject,
                                    "topic": topic,
                                    "material_type": material_type
                                })
                            
                            if response.status_code == 200:
                                data = response.json()
                                materials = data["materials"]
                                
                                st.success("✅ Tài liệu đề xuất:")
                                st.write(f"📚 **Môn:** {materials['subject']}")
                                st.write(f"📝 **Chủ đề:** {materials['topic']}")
                                
                                for category, items in materials.items():
                                    if category not in ['subject', 'topic'] and items:
                                        st.write(f"\n**{category.replace('_', ' ').title()}:**")
                                        for item in items[:3]:  # Show first 3 items
                                            st.write(f"- {item}")
                            else:
                                st.error(f"❌ Lỗi: {response.text}")
                        except Exception as e:
                            st.error(f"❌ Lỗi kết nối: {str(e)}")
                else:
                    st.warning("⚠️ Vui lòng nhập chủ đề")
    
    elif agent_type == "📊 Data Analysis":
        st.subheader("📊 Data Analysis Agent")
        
        data_action = st.radio(
            "Chọn chức năng:",
            ["Phân tích lớp học", "Học sinh cần hỗ trợ", "Dự đoán xu hướng"]
        )
        
        # Sample grade data
        sample_grade_data = [
            {"student_id": "HS001", "subject": "toán", "score": 8.5, "class": "10A1", "exam_date": "2024-12-01"},
            {"student_id": "HS002", "subject": "toán", "score": 6.2, "class": "10A1", "exam_date": "2024-12-01"},
            {"student_id": "HS003", "subject": "toán", "score": 4.8, "class": "10A1", "exam_date": "2024-12-01"},
            {"student_id": "HS004", "subject": "toán", "score": 9.1, "class": "10A1", "exam_date": "2024-12-01"},
            {"student_id": "HS005", "subject": "toán", "score": 7.3, "class": "10A1", "exam_date": "2024-12-01"},
        ] * 5  # Duplicate for more data
        
        use_sample = st.checkbox("Sử dụng dữ liệu mẫu", value=True, key="data_sample")
        
        if use_sample:
            class_data = sample_grade_data
            st.dataframe(pd.DataFrame(class_data).head())
        else:
            st.info("Tính năng upload dữ liệu thực sẽ được bổ sung")
            class_data = sample_grade_data
        
        if data_action == "Phân tích lớp học":
            if st.button("Phân tích", key="data_analyze"):
                with st.spinner("📈 Đang phân tích dữ liệu..."):
                    try:
                        response = requests.post(f"{BACKEND_URL}/agent/data/analyze",
                            json={"class_data": class_data})
                        
                        if response.status_code == 200:
                            data = response.json()
                            analysis = data["analysis"]
                            
                            st.success("✅ Kết quả phân tích:")
                            
                            # Class statistics
                            if "class_statistics" in analysis:
                                stats = analysis["class_statistics"]
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Tổng HS", stats.get("total_students", 0))
                                with col2:
                                    st.metric("Điểm TB", f"{stats.get('average_score', 0):.2f}")
                                with col3:
                                    st.metric("Cao nhất", f"{stats.get('max_score', 0):.1f}")
                                with col4:
                                    st.metric("Thấp nhất", f"{stats.get('min_score', 0):.1f}")
                            
                            # Performance summary
                            if "performance_summary" in analysis:
                                perf = analysis["performance_summary"]
                                st.write("📊 **Phân phối điểm:**")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Giỏi (≥8.5)", perf.get("excellent", 0))
                                with col2:
                                    st.metric("Khá (7-8.5)", perf.get("good", 0))
                                with col3:
                                    st.metric("TB (5-7)", perf.get("average", 0))
                                with col4:
                                    st.metric("Yếu (<5)", perf.get("below_average", 0))
                            
                            # Recommendations
                            if analysis.get("recommendations"):
                                st.write("💡 **Khuyến nghị:**")
                                for rec in analysis["recommendations"]:
                                    st.write(f"- {rec}")
                        else:
                            st.error(f"❌ Lỗi: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối: {str(e)}")
        
        elif data_action == "Học sinh cần hỗ trợ":
            if st.button("Phát hiện", key="data_at_risk"):
                with st.spinner("🔍 Đang phân tích học sinh..."):
                    try:
                        response = requests.post(f"{BACKEND_URL}/agent/data/at-risk",
                            json={"class_data": class_data})
                        
                        if response.status_code == 200:
                            data = response.json()
                            at_risk = data["at_risk_students"]
                            
                            st.success("✅ Kết quả phân tích:")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Tổng HS kiểm tra", at_risk.get("total_checked", 0))
                            with col2:
                                st.metric("Cần hỗ trợ", at_risk.get("need_support", 0))
                            with col3:
                                st.metric("Mức độ cao", at_risk.get("high_risk", 0))
                            
                            if at_risk.get("student_details"):
                                st.write("⚠️ **Danh sách học sinh cần hỗ trợ:**")
                                for student in at_risk["student_details"][:5]:
                                    st.write(f"**👤 {student['student_id']}** (Mức: {student['risk_level']})")
                                    st.write(f"- Điểm hiện tại: {student['current_score']}")
                                    st.write(f"- Lý do: {', '.join(student['reasons'])}")
                                    st.write(f"- Hành động: {', '.join(student['actions'][:2])}")
                                    st.write("---")
                            else:
                                st.info("🎉 Không có học sinh nào cần hỗ trợ đặc biệt!")
                        else:
                            st.error(f"❌ Lỗi: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối: {str(e)}")
        
        elif data_action == "Dự đoán xu hướng":
            prediction_period = st.slider("Dự đoán cho bao nhiêu kỳ tới:", 1, 12, 6)
            
            if st.button("Dự đoán", key="data_trends"):
                with st.spinner("🔮 Đang phân tích xu hướng..."):
                    try:
                        response = requests.post(f"{BACKEND_URL}/agent/data/trends",
                            json={"class_data": class_data},
                            params={"prediction_period": prediction_period})
                        
                        if response.status_code == 200:
                            data = response.json()
                            trends = data["trends"]
                            
                            st.success("✅ Dự đoán xu hướng:")
                            
                            st.write(f"📈 **Xu hướng tổng thể:** {trends.get('overall_trend', 'N/A')}")
                            st.write(f"🎯 **Độ tin cậy:** {trends.get('confidence', 'N/A')}")
                            
                            if trends.get("trend_analysis"):
                                ta = trends["trend_analysis"]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Gần đây", f"{ta.get('recent_average', 0):.2f}")
                                with col2:
                                    st.metric("Trước đó", f"{ta.get('earlier_average', 0):.2f}")
                                with col3:
                                    st.metric("Thay đổi", f"{ta.get('change', 0):+.2f}")
                            
                            if trends.get("predictions"):
                                st.write(f"🔮 **Dự đoán {prediction_period} kỳ tới:**")
                                pred_df = pd.DataFrame({
                                    "Kỳ": range(1, len(trends["predictions"]) + 1),
                                    "Điểm dự đoán": trends["predictions"]
                                })
                                st.line_chart(pred_df.set_index("Kỳ"))
                        else:
                            st.error(f"❌ Lỗi: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối: {str(e)}")
    
    elif agent_type == "🔀 Auto Route":
        st.subheader("🔀 Auto Route Query")
        st.markdown("Nhập câu hỏi và hệ thống sẽ tự động chọn agent phù hợp")
        
        query = st.text_area("Câu hỏi của bạn:", 
            placeholder="VD: Làm sao giải phương trình bậc hai?\nCách chia nhóm học sinh hiệu quả?\nPhân tích xu hướng điểm lớp 10A1?")
        
        if st.button("Phân tích Query", key="auto_route"):
            if query:
                with st.spinner("🤖 Đang phân tích và định tuyến..."):
                    try:
                        response = requests.get(f"{BACKEND_URL}/agent/route",
                            params={"query": query})
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            st.success("✅ Kết quả phân tích:")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**📝 Query:** {data['query']}")
                                st.write(f"**👤 Loại người dùng:** {data['detected_user_type']}")
                            with col2:
                                st.write(f"**🤖 Agent được chọn:** {data['selected_agent']}")
                                st.write(f"**🎯 Độ chính xác:** Cao")
                            
                            # Show recommendation
                            agent_map = {
                                "student_support": "🎓 Student Support Agent",
                                "teacher_support": "👩‍🏫 Teacher Support Agent", 
                                "data_analysis": "📊 Data Analysis Agent"
                            }
                            
                            selected = data['selected_agent']
                            if selected in agent_map:
                                st.info(f"💡 **Khuyến nghị:** Sử dụng {agent_map[selected]} để được hỗ trợ tốt nhất cho câu hỏi này.")
                            
                        else:
                            st.error(f"❌ Lỗi: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối: {str(e)}")
            else:
                st.warning("⚠️ Vui lòng nhập câu hỏi")

