import gradio as gr
import pymupdf
from typing import List

def add_file(new_file, current_file_list:List):
    current_file_list.append(new_file)
    return current_file_list

def add_url_and_clear(new_url, current_file_list:List):
    current_file_list.append(new_url)
    return current_file_list, ""

def refresh_file_list(file_list):
    # Chuyển đổi danh sách file thành choices cho CheckboxGroup
    if not file_list:
        return gr.CheckboxGroup(choices=[], value=[])
    
    choices = []
    for idx, item in enumerate(file_list):
        if hasattr(item, 'name'):  # File upload
            choices.append(f"{item.name}")
        else:  # URL
            choices.append(f"{item}")
    
    return gr.CheckboxGroup(choices=choices, value=[])

def handle_single_selection(selected_items):
    """Đảm bảo chỉ có thể chọn một nguồn duy nhất"""
    if len(selected_items) > 1:
        # Chỉ giữ lại item được chọn cuối cùng
        return [selected_items[-1]]
    return selected_items

def on_loader_change(loader_value):
    print(f"Loader được chọn: {loader_value}")
    return loader_value

def on_chunker_change(chunker_value):
    print(f"Chunker được chọn: {chunker_value}")
    return chunker_value

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
        
        # Thiết lập API key
        nvidia_api_key = gr.Textbox(value='Default',
                                    type='password')

    app_title = gr.Markdown(value="<h1 style='text-align: center; font-weight: bold;'>TRỢ LÝ AI ĐẮC LỰC CỦA MỌI GIẢNG VIÊN</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            file_list_state = gr.State([])
            file_list_checkbox = gr.CheckboxGroup(
                label="📂 Nguồn dữ liệu đã tải",
                choices=[],
                value=[],
                info="Chọn một nguồn dữ liệu để phân tích (chỉ được chọn 1)",
                interactive=True
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


    file_upload_btn.upload(
        fn=add_file,
        inputs=[file_upload_btn, file_list_state],
        outputs=[file_list_state]
    )

    url_input.submit(
        fn=add_url_and_clear,
        inputs=[url_input, file_list_state],
        outputs=[file_list_state, url_input]
    )

    file_list_state.change(
        fn=refresh_file_list,
        inputs=file_list_state,
        outputs=file_list_checkbox
    )

    # Xử lý khi người dùng chọn nguồn dữ liệu
    file_list_checkbox.change(
        fn=handle_single_selection,
        inputs=file_list_checkbox,
        outputs=file_list_checkbox
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
            
        




# user_input.submit(
#     fn=add_text,
#     inputs=[chatbot, user_input],
#     outputs=[chatbot]
# )
demo.launch()