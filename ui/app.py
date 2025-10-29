import time
import gradio as gr
from typing import List
import logging

# Import our services
from services.ui_integration_service import UIIntegrationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the UI integration service
ui_service = UIIntegrationService()

# Function to process uploaded document and add new URL
def process_uploaded_document(file_path: str, session_state: dict, loader_version: str, source_type: str = "Upload"):
    """Process the selected document through RAG pipeline.

    Now accepts an explicit `source_type` selected by the user ("Upload" or "Link").
    """
    try:
        user_name = session_state.get("user_name", "default_user")

        # Normalize source_type and file_path for the service
        normalized_source = file_path
        normalized_type = source_type.lower() if source_type else "upload"
        status_msg = ui_service.process_uploaded_document(
            normalized_source,
            user_name,
            source_type=normalized_type,
            loader_version=loader_version
        )
        logger.info(f"Document processing status: {status_msg}")
        return status_msg
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        logger.error(error_msg)
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
        error_msg = f"Error in document selection: {str(e)}"
        logger.error(error_msg)
        return error_msg, session_state


# Function to handle loader and chunker dropdown changes
def on_loader_version_change(loader_version):
    """Handle loader version radio button change"""
    try:
        status_msg = ui_service.update_loader_version(loader_version)
        logger.info(f"Loader version change status: {status_msg}")
        return status_msg
    except Exception as e:
        logger.error(f"Error in on_loader_version_change: {str(e)}")
        return f"❌ Error: {str(e)}"

def on_chunker_change(chunker_value):
    """Handle chunker dropdown change"""
    try:
        status_msg = ui_service.update_chunker_strategy(chunker_value)
        logger.info(f"Chunker change status: {status_msg}")
        return chunker_value
    except Exception as e:
        logger.error(f"Error in on_chunker_change: {str(e)}")
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
        List[gr.ChatMessage]: Updated chat history

    """
    # Get the last user input from chat history
    user_input = chat_history[-1].get("content") if chat_history else ""
    user_name = session_state.get("user_name", "default_user")
    selected_document_id = session_state.get("selected_document_id")
    try:
        response = ui_service.handle_chat_query(user_input, chat_history, user_name, selected_document_id)
        chat_history.append(gr.ChatMessage(role="assistant", content=response))
        return chat_history
    except Exception as e:
        error_msg = f"Error in chat: {str(e)}"
        logger.error(error_msg)
        chat_history.append(gr.ChatMessage(role="assistant", content=f"🤖 Xin lỗi, đã có lỗi xảy ra: {error_msg}"))
        return chat_history

# Function to handle Google Authentication
def handle_google_authentication(session_state: dict) -> gr.Button:
    """Handle Google Authentication and open sign-in website."""
    try:
        user_name = session_state.get("user_name")
        assert user_name is not None, "User name not found in session state"
       
        # Get authentication result with user name
        auth_result, google_account_name = ui_service.open_sign_in_website(username=user_name)
        logger.info("Google authentication process completed.")

        # Check if authentication was successful
        if auth_result:
            return gr.Button(value=google_account_name, interactive=False)
        else:
            # Authentication failed
            logger.error(f"Authentication failed: {auth_result}")
            gr.Info(message="Đăng nhập không thành công, vui lòng thử lại sau.", duration=5, title="Lỗi đăng nhập")
            return gr.Button(value="Đăng nhập tài khoản Google", interactive=True)
            
    except Exception as e:
        logger.error(f"Error during Google authentication: {str(e)}")
        gr.Info(message="Đăng nhập không thành công, vui lòng thử lại sau.", duration=5, title="Lỗi đăng nhập")
        return gr.Button(value="Đăng nhập tài khoản Google", interactive=True)

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

def save_user_name(request: gr.Request):
    """Save the authenticated user's name for session tracking."""
    return {"user_name": request.username}

def create_greeting_message(session_state):
    """Create a greeting message based on the user's name."""
    user_name = session_state.get("user_name", "Người dùng")
    gr.Info(message=f"Xin chào, {user_name}!", duration=5, title="Chào mừng")

def toggle_source_input(source_type):
    """Toggle visibility of upload button and URL input based on source type."""
    if source_type == "Upload":
        return gr.update(visible=True), gr.update(visible=False)  
    else:  # Link
        return gr.update(visible=False), gr.update(visible=True)

# Gradio UI setup
with gr.Blocks(fill_width=True, theme=gr.themes.Soft()) as demo: #type: ignore
    session_state = gr.State()
    with gr.Accordion(label="⚙️ Developer Setting", open=False):

        # Chọn version cho document loader
        loader_version_radio = gr.Radio(
            label="Chọn tính năng xử lý tài liệu",
            choices=['Version 1', 'Version 2'],
            value='Version 1',
        )
        
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
            
            # Let the user explicitly choose whether they are using an upload or a link
            source_type_radio = gr.Radio(
                label="Chọn loại nguồn",
                choices=['Upload', 'Link'],
                value='Upload',
                info="Chọn Upload để tải file hoặc Link để nhập URL"
            )
            
            # Upload button (visible by default)
            file_upload_btn = gr.UploadButton(
                label="📤 Upload a File",
                visible=True
            )
            
            # URL input (hidden by default)
            url_input = gr.Textbox(
                label="🔗 Nhập đường dẫn (URL)", 
                submit_btn=True,
                visible=False
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

        with gr.Column(scale=1):
            with gr.Tab("Công cụ"):
                google_auth_btn = gr.Button(value="Đăng nhập tài khoản Google")
        
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

    # Toggle visibility when source type changes
    source_type_radio.change(
        fn=toggle_source_input,
        inputs=[source_type_radio],
        outputs=[file_upload_btn, url_input]
    )

    # Process file upload
    file_upload_btn.upload(
        fn=process_uploaded_document,
        inputs=[file_upload_btn, session_state, loader_version_radio, source_type_radio],
        outputs=[status_display]
    ).success(
        fn=update_file_list_choices,
        inputs=[session_state],
        outputs=[file_list_checkbox]
    )

    # Process URL submission from the textbox when user presses Enter
    url_input.submit(
        fn=process_uploaded_document,
        inputs=[url_input, session_state, loader_version_radio, source_type_radio],
        outputs=[status_display]
    ).then(
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
        outputs=[chatbot]
    )

    input_submit_btn.click(
        fn=add_user_message_first,
        inputs=[user_input_textbox, chatbot],
        outputs=[chatbot, user_input_textbox]
    ).then(
        fn=handle_chat_input,
        inputs=[chatbot, session_state],
        outputs=[chatbot]
    )

    # Sign in to Google
    google_auth_btn.click(
        fn=handle_google_authentication,
        inputs=[session_state],
        outputs=[google_auth_btn]
    )

    # Loader version change handler
    loader_version_radio.change(
        fn=on_loader_version_change,
        inputs=loader_version_radio,
        outputs=[status_display]
    )

    # Thêm event handlers cho dropdowns
    chunker_dropdown.change(
        fn=on_chunker_change,
        inputs=chunker_dropdown,
        outputs=[]
    )
            
if __name__ == "__main__":
    try:
        demo.queue()
        demo.launch(auth=authenticate, share=True)  # Enable authentication with a simple username/password prompt
    finally:
        logger.info("Shutting down the application...")
        ui_service.cleanup()  # Perform any necessary cleanup actions
        logger.info("Application has been shut down gracefully.")