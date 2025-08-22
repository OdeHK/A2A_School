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
tab_student, tab_teacher, tab_analysis = st.tabs(["🎓 Dành cho Học sinh", "👩‍🏫 Dành cho Giáo viên", "📊 Dành cho Quản lý"])

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

