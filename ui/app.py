import gradio as gr
import pymupdf
from typing import List
import logging

# Import our services
from services.ui_integration_service import UIIntegrationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the UI integration service
ui_service = UIIntegrationService()

# def add_file(new_file_path:str, current_file_list: List):
#     """Handle file upload and add to current list"""
#     try:
#         updated_list, status_msg = ui_service.handle_file_upload(new_file_path)
#         logger.info(f"File upload status: {status_msg}")
#         return updated_list
#     except Exception as e:
#         logger.error(f"Error in add_file: {str(e)}")
#         return current_file_list
# def process_uploaded_document(file_path:str):
#     """Process the selected document through RAG pipeline"""
#     try:
#         status_msg = ui_service.process_selected_document(file_path)
#         logger.info(f"Document processing status: {status_msg}")
#         return status_msg
#     except Exception as e:
#         error_msg = f"Error processing document: {str(e)}"
#         logger.error(error_msg)
#         return error_msg

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

def handle_chat_input(user_input, chat_history):
    """Handle chat input and return response. 
    It receives the user input from the textbox and the current chat history, 
    then returns the updated chat history and clears the input box.

    Args:
        user_input (str): The input text from the user.
        chat_history (List[Tuple[str, str]]): The current chat history as a list of tuples.
    Returns:
        Tuple[List[Tuple[str, str]], ""]: Updated chat history and cleared input box.

    """

    try:
        updated_history = ui_service.handle_chat_query(user_input, chat_history)
        return updated_history, ""
    except Exception as e:
        error_msg = f"Error in chat: {str(e)}"
        logger.error(error_msg)
        chat_history.append((user_input, f"🤖 Xin lỗi, đã có lỗi xảy ra: {error_msg}"))
        return chat_history, ""

with gr.Blocks(fill_width=True, theme=gr.themes.Soft()) as demo:
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
                ("","👋 Xin chào! Tôi là trợ lý AI đắc lực của bạn!\n\n🔸 Tôi có thể giúp bạn:\n• Soạn bộ đề kiểm tra một cách chính xác\n• Tổng hợp và phân tích bài làm của học sinh\n• Quản lý lớp học thông qua Google Classroom\n\n� **Để bắt đầu:** Upload tài liệu ở bên trái 📂 hoặc kết nối với dịch vụ Google ở bên phải 🔗")
            ]
            
            chatbot = gr.Chatbot(
                value=initial_message,
                label="💬 Trò chuyện với AI",
                show_label=True,
                height=600
            )
            with gr.Row(equal_height=True):
                user_input_textbox = gr.Textbox(scale=5, show_label=False, placeholder="Nhập yêu cầu của bạn...")
                input_submit_btn = gr.Button("Gửi", scale=1)

        with gr.Column(scale=1):
            with gr.Tab("Công cụ"):
                sign_in_drive_btn = gr.Button(value="Đăng nhập Google Drive")
                sign_in_form_btn = gr.Button(value="Đăng nhập Google Form")
                sign_in_classroom_btn = gr.Button(value="Đăng nhập Google Classroom")


    # Event handlers
    # file_upload_btn.upload(
    #     fn=add_file,
    #     inputs=[file_upload_btn, file_list_state],
    #     outputs=[file_list_state]
    # ).success(
    #     fn=process_uploaded_document,
    #     inputs=[file_upload_btn],
    #     outputs=[status_display]
    # )

    file_upload_btn.upload(
        fn=process_uploaded_document,
        inputs=[file_upload_btn],
        outputs=[status_display]
    ).success(
        fn=update_file_list_choices,
        inputs=[],
        outputs=[file_list_checkbox]
    )


    # url_input.submit(
    #     fn=add_url_and_clear,
    #     inputs=[url_input, file_list_state],
    #     outputs=[file_list_state, url_input]
    # )

    # Xử lý khi người dùng chọn nguồn dữ liệu
    file_list_checkbox.change(
        fn=handle_single_selection,
        inputs=file_list_checkbox,
        outputs=file_list_checkbox
    )

    # Chat functionality
    user_input_textbox.submit(
        fn=handle_chat_input,
        inputs=[user_input_textbox, chatbot],
        outputs=[chatbot, user_input_textbox]
    )

    input_submit_btn.click(
        fn=handle_chat_input,
        inputs=[user_input_textbox, chatbot],
        outputs=[chatbot, user_input_textbox]
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
    demo.launch()