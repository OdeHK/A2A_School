import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import gradio as gr
from typing import List
import logging
from datetime import datetime

# Import our services
from services.ui_integration_service import UIIntegrationService

# Import new utilities
from utils.performance import measure_time, Timer, get_performance_report
from utils.error_handler import ErrorHandler
from models.responses import TextResponse, FileDownloadResponse, ErrorResponse
from models.session import SessionState
from config.exceptions import ValidationError, AuthenticationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the UI integration service
ui_service = UIIntegrationService()

# Function to process uploaded document and add new URL
@measure_time("document_upload")
def process_uploaded_document(file_path: str, session_state: dict):
    """Process the selected document through RAG pipeline"""
    try:
        with Timer("extract_username"):
            user_name = session_state.get("user_name", "default_user")
        
        with Timer("process_document_core"):
            status_msg = ui_service.process_uploaded_document(file_path, user_name)
        
        logger.info(f"✅ Document processing completed: {status_msg}")
        return status_msg
        
    except Exception as e:
        error_msg, is_critical = ErrorHandler.handle_error(e, {
            "operation": "process_uploaded_document",
            "user": session_state.get("user_name"),
            "file_path": file_path
        })
        return error_msg
    
def add_url_and_clear(new_url, current_file_list: List):
    # """Handle URL input and add to current list"""
    # try:
    #     updated_list, cleared_url, status_msg = ui_service.handle_url_input(new_url)
    #     logger.info(f"URL input status: {status_msg}")
    #     return updated_list, cleared_url
    # except Exception as e:
    #     logger.error(f"Error in add_url_and_clear: {str(e)}")
    #     return current_file_list, ""
    pass


# ==== Function to process file list =====
def convert_file_list_to_checkbox(file_list: List) -> gr.CheckboxGroup:
    """Convert list of files to gr.CheckboxGroup with single selection enforced"""
    if not file_list:
        return gr.CheckboxGroup(choices=[], value=[])

    # Return CheckboxGroup with single selection enforced
    return gr.CheckboxGroup(choices=file_list, value=[file_list[0]])

def update_file_list_choices(session_state: dict) -> gr.CheckboxGroup:
    """Get the current list of files for the user"""
    with Timer("get_user_files"):
        user_name = session_state.get("user_name", "default_user")
        current_files = ui_service.get_user_files(user_name)
        file_list_checkbox = convert_file_list_to_checkbox(current_files)
        logger.info(f"Current files for {user_name}: {current_files}")
        return file_list_checkbox

def handle_single_selection(selected_items: List[str]) -> List[str]:
    """Đảm bảo chỉ có thể chọn một nguồn duy nhất"""
    logger.info(f"Selected items before enforcing single selection: {selected_items}")
    if len(selected_items) > 1:
        # Chỉ giữ lại item được chọn cuối cùng
        return [selected_items[-1]]
    return selected_items

def handle_document_selection(selected_items: List[str], session_state: dict):
    """Handle document selection and update UI service"""
    user_name = session_state.get("user_name", "default_user")
    try:
        if selected_items and len(selected_items) > 0:
            selected_filename = selected_items[0]  # Get the first (and only) selected item
            
            with Timer("find_document_id"):
                # Find document_id using ui_service
                document_id, status_msg = ui_service.find_document_id_by_filename(user_name, selected_filename)
            
            logger.info(f"Document selection status: {status_msg}")
            
            # Update session state with selected document information
            updated_session_state = session_state.copy()
            updated_session_state["selected_document_id"] = document_id
            
            return status_msg, updated_session_state
        else:
            # No document selected - clear the session state
            updated_session_state = session_state.copy()
            updated_session_state["selected_document_id"] = None
            
            return "Chưa chọn tài liệu nào", updated_session_state
    except Exception as e:
        error_msg, _ = ErrorHandler.handle_error(e, {
            "operation": "document_selection",
            "user": user_name,
            "selected_items": selected_items
        })
        return error_msg, session_state


# Function to handle loader and chunker dropdown changes
def on_loader_change(loader_value):
    """Handle loader dropdown change"""
    try:
        with Timer("update_loader"):
            status_msg = ui_service.update_loader_strategy(loader_value)
        logger.info(f"✅ Loader change status: {status_msg}")
        return loader_value
    except Exception as e:
        error_msg, _ = ErrorHandler.handle_error(e)
        logger.error(error_msg)
        return loader_value

def on_chunker_change(chunker_value):
    """Handle chunker dropdown change"""
    try:
        with Timer("update_chunker"):
            status_msg = ui_service.update_chunker_strategy(chunker_value)
        logger.info(f"✅ Chunker change status: {status_msg}")
        return chunker_value
    except Exception as e:
        error_msg, _ = ErrorHandler.handle_error(e)
        logger.error(error_msg)
        return chunker_value

# Function to handle chat input and return response
def add_user_message_first(user_input, chat_history):
    """Add user message to chat history first before processing.
    
    Args:
        user_input (str): The input text from the user.
        chat_history (List[gr.ChatMessage]): The current chat history.
    Returns:
        Tuple[List[gr.ChatMessage], ""]: Updated chat history with user message and cleared input box.
    """
    if user_input.strip():
        chat_history.append(gr.ChatMessage(role="user", content=user_input))
    return chat_history, ""


def handle_chat_input(chat_history, session_state: dict):
    """Handle chat input and return response. 
    It receives the user input from the textbox and the current chat history, 
    then returns the updated chat history and clears the input box.

    Args:
        chat_history (List[gr.ChatMessage]): The current chat history.
        session_state (dict): Session state containing user information.
    Returns:
        Tuple[List[gr.ChatMessage], gr.File]: Updated chat history and file component

    """
    # Get the last user input from chat history
    user_input = chat_history[-1].get("content") if chat_history else ""
    user_name = session_state.get("user_name", "default_user")
    selected_document_id = session_state.get("selected_document_id")
    try:
        response = ui_service.handle_chat_query(user_input, chat_history, user_name, selected_document_id)
        
        # Check if response contains file download marker
        if response.startswith("FILE_DOWNLOAD:"):
            file_path = response.replace("FILE_DOWNLOAD:", "").strip()
            chat_history.append(gr.ChatMessage(role="assistant", content="✅ File Word đã được tạo xong! Click vào file bên dưới để tải xuống."))
            # Return file with visible=True
            return chat_history, gr.File(value=file_path, visible=True, label="📥 Tải xuống file Word")
        else:
            chat_history.append(gr.ChatMessage(role="assistant", content=response))
            # Return hidden file component
            return chat_history, gr.File(visible=False, label="📥 Tải xuống file Word")
    except Exception as e:
        error_msg = f"Error in chat: {str(e)}"
        logger.error(error_msg)
        chat_history.append(gr.ChatMessage(role="assistant", content=f"🤖 Xin lỗi, đã có lỗi xảy ra: {error_msg}"))
        return chat_history, gr.File(visible=False, label="📥 Tải xuống file Word")

def update_download_file(file_path):
    """Update the download file component based on the file path.
    
    Args:
        file_path: Path to the file to download, or None if no file
        
    Returns:
        Updated gr.File component
    """
    if file_path:
        return gr.File(value=file_path, visible=True, label="📥 Tải xuống file Word")
    else:
        return gr.File(value=None, visible=False, label="📥 Tải xuống file Word")

# Function to handle Google Authentication
def handle_google_authentication():
    """Handle Google Authentication and open sign-in website."""
    try:
        # Get authentication result with user name (force re-auth to allow account switching)
        auth_result, user_name = ui_service.open_sign_in_website(force_reauth=True)
        logger.info("Google authentication process completed.")

        # Check if authentication was successful
        if auth_result:
            gr.Info(message=f"Đăng nhập thành công với tài khoản: {user_name}", duration=5, title="✅ Thành công")
            return (
                gr.Button(value=user_name, interactive=False, variant="secondary"),  # Login button
                gr.Button(value="Đăng xuất Google", visible=True, variant="secondary")  # Logout button
            )
        else:
            # Authentication failed
            logger.error(f"Authentication failed: {auth_result}")
            gr.Info(message="Đăng nhập không thành công, vui lòng thử lại sau.", duration=5, title="Lỗi đăng nhập")
            return (
                gr.Button(value="Đăng nhập tài khoản Google", interactive=True, variant="primary"),
                gr.Button(visible=False)
            )
            
    except Exception as e:
        logger.error(f"Error during Google authentication: {str(e)}")
        gr.Info(message="Đăng nhập không thành công, vui lòng thử lại sau.", duration=5, title="Lỗi đăng nhập")
        return (
            gr.Button(value="Đăng nhập tài khoản Google", interactive=True, variant="primary"),
            gr.Button(visible=False)
        )

def handle_google_logout():
    """Handle Google Logout."""
    try:
        success, message = ui_service.logout_google()
        
        if success:
            gr.Info(message="Đã đăng xuất Google. Có thể đăng nhập tài khoản khác.", duration=5, title="✅ Đăng xuất")
            return (
                gr.Button(value="Đăng nhập tài khoản Google", interactive=True, variant="primary"),  # Login button
                gr.Button(visible=False)  # Logout button (hidden)
            )
        else:
            gr.Info(message=f"Lỗi: {message}", duration=5, title="❌ Lỗi")
            return (
                gr.Button(value="Đăng nhập tài khoản Google", interactive=True, variant="primary"),
                gr.Button(visible=False)
            )
            
    except Exception as e:
        logger.error(f"Error during Google logout: {str(e)}")
        gr.Info(message="Lỗi khi đăng xuất", duration=5, title="❌ Lỗi")
        return (
            gr.Button(value="Đăng nhập tài khoản Google", interactive=True, variant="primary"),
            gr.Button(visible=False)
        )

def authenticate(username, password):
    """
    Authenticate user credentials using UI integration service.
    """
    try:
        # Use UI service to authenticate
        is_authenticated = ui_service.authenticate_user(username, password)
        
        if is_authenticated:
            logger.info(f"User {username} authenticated successfully")
            return True
        else:
            logger.warning(f"Authentication failed for user: {username}")
            return False
            
    except Exception as e:
        logger.error(f"Error during authentication: {str(e)}")
        # Fallback to default admin credentials in case of error
        if username == "admin" and password == "admin":
            logger.info("Used fallback admin credentials")
            return True
        return False

def register_new_user(username, password, confirm_password, email, full_name):
    """
    Register a new user.
    
    Returns:
        Tuple of (success_html, error_html, clear_fields...)
    """
    try:
        # Validation
        if not username or len(username) < 3:
            return (
                "",
                "❌ Username phải có ít nhất 3 ký tự",
                username, password, confirm_password, email, full_name
            )
        
        if not password or len(password) < 6:
            return (
                "",
                "❌ Password phải có ít nhất 6 ký tự",
                username, password, confirm_password, email, full_name
            )
        
        if password != confirm_password:
            return (
                "",
                "❌ Password không khớp",
                username, "", "", email, full_name
            )
        
        if email and "@" not in email:
            return (
                "",
                "❌ Email không hợp lệ",
                username, password, confirm_password, email, full_name
            )
        
        # Register user
        success, message = ui_service.register_user(username, password, email, full_name)
        
        if success:
            logger.info(f"User {username} registered successfully")
            return (
                f"✅ {message}",
                "",
                "", "", "", "", ""  # Clear all fields
            )
        else:
            return (
                "",
                f"❌ {message}",
                username, password, confirm_password, email, full_name
            )
            
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        return (
            "",
            f"❌ Lỗi hệ thống: {str(e)}",
            username, password, confirm_password, email, full_name
        )

def save_user_name(request: gr.Request):
    """Save the authenticated user's name for session tracking."""
    return {"user_name": request.username}

def create_greeting_message(session_state):
    """Create a greeting message based on the user's name."""
    user_name = session_state.get("user_name", "Người dùng")
    gr.Info(message=f"Xin chào, {user_name}!", duration=5, title="Chào mừng")

# Gradio UI setup
with gr.Blocks(fill_width=True, theme=gr.themes.Soft()) as demo: #type: ignore
    session_state = gr.State()
    with gr.Sidebar(open=False):
        side_bar_title = gr.Markdown(value="**Developer Setting**")

        # Chọn phương thức loader
        loader_dropdown = gr.Dropdown(label="Loader",
                                        choices=['Base', 'OCR', 'Base+OCR'],
                                        value='Base',  
                                        multiselect=False,
                                        interactive=True)  
        chunker_dropdown = gr.Dropdown(label="Chunker",
                                        choices=['ONE_PAGE', 'RECURSIVE_CHARACTER_TEXT_SPLITTER', 'LLM_SPLITTER'],
                                        value='ONE_PAGE', 
                                        multiselect=False,
                                        interactive=True)  
        

    app_title = gr.Markdown(value="<h1 style='text-align: center; font-weight: bold;'>TRỢ LÝ AI ĐẮC LỰC CỦA MỌI GIẢNG VIÊN</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            
            #file_list_state = gr.State([]) # Using the file list in ui_integration_service to maintain state
            file_list_checkbox = gr.CheckboxGroup(
                label="📂 Nguồn dữ liệu đã tải",
                choices=[],
                value=[],
                info="Chọn một nguồn dữ liệu để phân tích (chỉ được chọn 1)",
                interactive=True
            )
            
            # Status display
            status_display = gr.Textbox(
                label="📊 Trạng thái xử lý",
                value="Chưa có tài liệu nào được xử lý",
                interactive=False,
                lines=3
            )
            
            url_input = gr.Textbox(label="Nhập đường dẫn Google Drive", submit_btn=True)
            file_upload_btn = gr.UploadButton(
                label="Upload a File"
            )
        
        with gr.Column(scale=2):
            # Tin nhắn giới thiệu ban đầu
            initial_message = [
                gr.ChatMessage(role="assistant", content="👋 Xin chào! Tôi là trợ lý AI đắc lực của bạn!\n\n🔸 Tôi có thể giúp bạn:\n• Soạn bộ đề kiểm tra một cách chính xác\n• Tổng hợp và phân tích bài làm của học sinh\n• Quản lý lớp học thông qua Google Classroom\n\n**Để bắt đầu:** Upload tài liệu ở bên trái 📂 hoặc kết nối với dịch vụ Google ở bên phải 🔗")
            ]
            chatbot = gr.Chatbot(
                value=initial_message, # type: ignore
                type="messages",
                label="💬 Trò chuyện với AI",
                show_label=True,
                height=600,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ]
            )
            with gr.Row(equal_height=True):
                user_input_textbox = gr.Textbox(scale=5, show_label=False, placeholder="Nhập yêu cầu của bạn...")
                input_submit_btn = gr.Button("Gửi", scale=1, variant="primary")
            
            # File download component (hidden by default)
            download_file = gr.File(label="📥 Tải xuống file Word", visible=False, type="filepath")

        with gr.Column(scale=1):
            with gr.Tab("Công cụ"):
                google_auth_btn = gr.Button(value="Đăng nhập tài khoản Google", variant="primary")
                google_logout_btn = gr.Button(value="Đăng xuất Google", variant="secondary", visible=False)
                gr.Markdown("---")
                gr.Markdown("💡 **Ghi chú:** Nếu muốn đổi tài khoản Google, hãy đăng xuất trước rồi đăng nhập lại.")
        
    # WHen loading the app,
    demo.load(
        fn=save_user_name,
        inputs=[],
        outputs=[session_state]
    ).then(
        fn=create_greeting_message,
        inputs=[session_state],
        outputs=[]
    ).then(
        fn=update_file_list_choices,
        inputs=[session_state],
        outputs=[file_list_checkbox]
    )

    # Process file upload
    file_upload_btn.upload(
        fn=process_uploaded_document,
        inputs=[file_upload_btn, session_state],
        outputs=[status_display]
    ).success(
        fn=update_file_list_choices,
        inputs=[session_state],
        outputs=[file_list_checkbox]
    )

    # Process when user chooses a uploaded source
    file_list_checkbox.change(
        fn=handle_single_selection,
        inputs=file_list_checkbox,
        outputs=file_list_checkbox
    ).then(
        fn=handle_document_selection,
        inputs=[file_list_checkbox, session_state],
        outputs=[status_display, session_state]
    )

    # Chat functionality
    user_input_textbox.submit(
        fn=add_user_message_first,
        inputs=[user_input_textbox, chatbot],
        outputs=[chatbot, user_input_textbox]
    ).then(
        fn=handle_chat_input,
        inputs=[chatbot, session_state],
        outputs=[chatbot, download_file]
    )

    input_submit_btn.click(
        fn=add_user_message_first,
        inputs=[user_input_textbox, chatbot],
        outputs=[chatbot, user_input_textbox]
    ).then(
        fn=handle_chat_input,
        inputs=[chatbot, session_state],
        outputs=[chatbot, download_file]
    )

    # Sign in to Google
    google_auth_btn.click(
        fn=handle_google_authentication,
        inputs=[],
        outputs=[google_auth_btn, google_logout_btn]
    )
    
    # Sign out from Google
    google_logout_btn.click(
        fn=handle_google_logout,
        inputs=[],
        outputs=[google_auth_btn, google_logout_btn]
    )

    # Thêm event handlers cho dropdowns
    loader_dropdown.change(
        fn=on_loader_change,
        inputs=loader_dropdown,
        outputs=[]
    )

    chunker_dropdown.change(
        fn=on_chunker_change,
        inputs=chunker_dropdown,
        outputs=[]
    )
            
if __name__ == "__main__":
    # Create registration interface
    with gr.Blocks(theme=gr.themes.Soft()) as register_demo:
        gr.Markdown("# 📝 Đăng ký tài khoản mới")
        gr.Markdown("Tạo tài khoản để sử dụng Trợ lý AI cho giảng viên")
        
        with gr.Row():
            with gr.Column(scale=1):
                pass  # Empty column for centering
            
            with gr.Column(scale=2):
                reg_username = gr.Textbox(label="👤 Tên đăng nhập *", placeholder="Ít nhất 3 ký tự")
                reg_email = gr.Textbox(label="📧 Email", placeholder="email@example.com (tùy chọn)")
                reg_full_name = gr.Textbox(label="📛 Họ và tên", placeholder="Nguyễn Văn A (tùy chọn)")
                reg_password = gr.Textbox(label="🔒 Mật khẩu *", placeholder="Ít nhất 6 ký tự", type="password")
                reg_confirm_password = gr.Textbox(label="🔒 Xác nhận mật khẩu *", placeholder="Nhập lại mật khẩu", type="password")
                
                with gr.Row():
                    register_btn = gr.Button("Đăng ký", variant="primary", scale=2)
                    cancel_btn = gr.Button("Hủy", scale=1)
                
                success_msg = gr.Markdown(visible=True)
                error_msg = gr.Markdown(visible=True)
                
                gr.Markdown("---")
                gr.Markdown("**Lưu ý:** Các trường có dấu * là bắt buộc")
            
            with gr.Column(scale=1):
                pass  # Empty column for centering
        
        # Register button handler
        register_btn.click(
            fn=register_new_user,
            inputs=[reg_username, reg_password, reg_confirm_password, reg_email, reg_full_name],
            outputs=[success_msg, error_msg, reg_username, reg_password, reg_confirm_password, reg_email, reg_full_name]
        )
    
    try:
        # Launch both interfaces
        demo.queue()
        
        # Check if user wants to register (via command line argument or environment variable)
        import sys
        if "--register" in sys.argv or len(sys.argv) > 1 and sys.argv[1] == "register":
            logger.info("Launching registration interface...")
            register_demo.launch(share=False)
        else:
            logger.info("=" * 80)
            logger.info("🚀 A2A_SCHOOL APPLICATION STARTED")
            logger.info("=" * 80)
            logger.info(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
            
            demo.launch(auth=authenticate, share=True)
            
    finally:
        logger.info("\n" + "=" * 80)
        logger.info("🛑 SHUTTING DOWN APPLICATION")
        logger.info("=" * 80)
        
        # Print performance report
        logger.info("\n" + get_performance_report())
        
        logger.info("\n" + "=" * 80)
        logger.info("🧹 Cleaning up resources...")
        ui_service.cleanup()
        logger.info("✅ Application has been shut down gracefully.")
        logger.info(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)