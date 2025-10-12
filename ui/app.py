import time
import gradio as gr
from typing import List
import logging
from pymongo import MongoClient

# Import our services
from services.ui_integration_service import UIIntegrationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the UI integration service
ui_service = UIIntegrationService()

# Function to process uploaded document and add new URL
def process_uploaded_document(file_path:str):
    """Process the selected document through RAG pipeline"""
    try:
        status_msg = ui_service.process_uploaded_document(file_path)
        logger.info(f"Document processing status: {status_msg}")
        return status_msg
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        logger.error(error_msg)
        return error_msg
    
def add_url_and_clear(new_url, current_file_list: List):
    """Handle URL input and add to current list"""
    try:
        updated_list, cleared_url, status_msg = ui_service.handle_url_input(new_url)
        logger.info(f"URL input status: {status_msg}")
        return updated_list, cleared_url
    except Exception as e:
        logger.error(f"Error in add_url_and_clear: {str(e)}")
        return current_file_list, ""


# Function to process file list 
def convert_file_list_to_checkbox(file_list: List):
    # Chuyển đổi danh sách file thành choices cho CheckboxGroup
    if not file_list:
        return gr.CheckboxGroup(choices=[], value=[])
    
    choices = []
    for idx, item in enumerate(file_list):
        if hasattr(item, 'name'):  # File upload
            choices.append(f"hello{item.name}")
        else:  # URL
            choices.append(f"{item}")
    
    return gr.CheckboxGroup(choices=choices, value=[])

def update_file_list_choices():
    """Get the current list of files"""

    current_files = ui_service.get_current_files()
    file_list_checkbox = convert_file_list_to_checkbox(current_files)
    logger.info(f"Current files: {current_files}")
    return file_list_checkbox

def handle_single_selection(selected_items):
    """Đảm bảo chỉ có thể chọn một nguồn duy nhất"""
    if len(selected_items) > 1:
        # Chỉ giữ lại item được chọn cuối cùng
        return [selected_items[-1]]
    return selected_items

def handle_document_selection(selected_items):
    """Handle document selection and update UI service"""
    try:
        if selected_items and len(selected_items) > 0:
            selected_filename = selected_items[0]  # Get the first (and only) selected item
            status_msg = ui_service.set_selected_document(selected_filename)
            logger.info(f"Document selection status: {status_msg}")
            return status_msg
        else:
            # No document selected
            ui_service.set_selected_document("")
            return "Chưa chọn tài liệu nào"
    except Exception as e:
        error_msg = f"Error in document selection: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Function to handle loader and chunker dropdown changes
def on_loader_change(loader_value):
    """Handle loader dropdown change"""
    try:
        status_msg = ui_service.update_loader_strategy(loader_value)
        logger.info(f"Loader change status: {status_msg}")
        return loader_value
    except Exception as e:
        logger.error(f"Error in on_loader_change: {str(e)}")
        return loader_value

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


def handle_chat_input(chat_history):
    """Handle chat input and return response. 
    It receives the user input from the textbox and the current chat history, 
    then returns the updated chat history and clears the input box.

    Args:
        user_input (str): The input text from the user.
        chat_history (List[Tuple[str, str]]): The current chat history as a list of tuples.
    Returns:
        List[gr.ChatMessage]: Updated chat history

    """
    # Get the last user input from chat history
    user_input = chat_history[-1].get("content") if chat_history else ""

    try:
        response = ui_service.handle_chat_query(user_input, chat_history)
        chat_history.append(gr.ChatMessage(role="assistant", content=response))
        return chat_history
    except Exception as e:
        error_msg = f"Error in chat: {str(e)}"
        logger.error(error_msg)
        chat_history.append(gr.ChatMessage(role="assistant", content=f"🤖 Xin lỗi, đã có lỗi xảy ra: {error_msg}"))
        return chat_history

# Function to handle Google Authentication
def handle_google_authentication():
    """Handle Google Authentication and open sign-in website."""
    try:
        # Get authentication result with user name
        auth_result, user_name = ui_service.open_sign_in_website()
        logger.info("Google authentication process completed.")

        # Check if authentication was successful
        if auth_result:
            return gr.Button(value=user_name, interactive=False)
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
    Authenticate user credentials.
    This is a placeholder function. Replace with actual authentication logic.
    """
    uri = "mongodb://A4Teacher_application:A4Teacher_application@127.0.0.1:27017/?authSource=admin"

    try:
        client = MongoClient(uri)
        database = client.get_database(name="agent_for_teacher")
        collection = database.get_collection(name="users")
        user = collection.find_one({"username": username, "password": password})
        client.close()

        if user:
            return True
        else:
            logger.warning("Invalid username or password.")
            return False
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        if username == "admin" and password == "admin":
            return True
        return False
    
# Gradio UI setup
with gr.Blocks(fill_width=True, theme=gr.themes.Soft()) as demo: #type: ignore
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
                value=initial_message,
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

    # Process file upload
    file_upload_btn.upload(
        fn=process_uploaded_document,
        inputs=[file_upload_btn],
        outputs=[status_display]
    ).success(
        fn=update_file_list_choices,
        inputs=[],
        outputs=[file_list_checkbox]
    )

    # Process when user chooses a uploaded source
    file_list_checkbox.change(
        fn=handle_single_selection,
        inputs=file_list_checkbox,
        outputs=file_list_checkbox
    ).then(
        fn=handle_document_selection,
        inputs=file_list_checkbox,
        outputs=status_display
    )

    # Chat functionality
    user_input_textbox.submit(
        fn=add_user_message_first,
        inputs=[user_input_textbox, chatbot],
        outputs=[chatbot, user_input_textbox]
    ).then(
        fn=handle_chat_input,
        inputs=[chatbot],
        outputs=[chatbot]
    )

    input_submit_btn.click(
        fn=add_user_message_first,
        inputs=[user_input_textbox, chatbot],
        outputs=[chatbot, user_input_textbox]
    ).then(
        fn=handle_chat_input,
        inputs=[chatbot],
        outputs=[chatbot]
    )

    # Sign in to Google
    google_auth_btn.click(
        fn=handle_google_authentication,
        inputs=[],
        outputs=[google_auth_btn]
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
    demo.launch(auth=authenticate)  # Enable authentication with a simple username/password prompt