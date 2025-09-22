from langchain_core.prompts import ChatPromptTemplate

find_document_node_prompt = ChatPromptTemplate.from_template(
    """
<ROLE>
Bạn là một công cụ tìm kiếm ngữ nghĩa thông minh cho một thư viện tài liệu.
</ROLE>

<OBJECTIVE>
Nhiệm vụ của bạn là hiểu sâu yêu cầu của người dùng, tìm ra một tài liệu duy nhất phù hợp nhất từ thư viện được cung cấp, và sau đó xác định một tiêu đề duy nhất phù hợp nhất trong tài liệu đó.
</OBJECTIVE>

<INPUT_SCHEMA>
library_str: Một chuỗi JSON chứa danh sách các đối tượng tài liệu. Mỗi đối tượng có các trường "name" (tên), "path" (đường dẫn), và "title" (một danh sách các tiêu đề).
user_request: Một chuỗi văn bản chứa truy vấn tìm kiếm của người dùng.
</INPUT_SCHEMA>

<INPUT>
Thư viện tài liệu:
```json
{library_str}
```
Yêu cầu của người dùng:
"{user_request}"
</INPUT>

<INSTRUCTIONS>
Phân tích user_request để hiểu rõ ý định và ý nghĩa cốt lõi.
So sánh ý định này với trường name và danh sách title của mỗi tài liệu trong library_str để tìm ra tài liệu phù hợp nhất.
Từ tài liệu phù hợp nhất đã tìm thấy, chọn ra MỘT title duy nhất tương ứng nhất với user_request.
Nếu không tìm thấy tài liệu hoặc tiêu đề nào phù hợp, chỉ trả về một đối tượng JSON rỗng: {{}}.
</INSTRUCTIONS>

<OUTPUT_GUIDELINES>
Câu trả lời BẮT BUỘC phải là một đối tượng JSON duy nhất.
Đối tượng JSON phải chứa các trường: name, path, và title.
Trường title phải là một danh sách chỉ chứa MỘT chuỗi tiêu đề phù hợp nhất.
Không bao gồm bất kỳ văn bản hội thoại, lời giải thích hay định dạng markdown nào trong kết quả đầu ra.
</OUTPUT_GUIDELINES>

<EXAMPLE>
<INPUT>
Thư viện tài liệu:
```json
[
  {{
   "name": "Python",
    "path": "tai-lieu/python.pdf",
    "title": [ "Giới thiệu sản phẩm", "Hướng dẫn cài đặt", "Xử lý sự cố"]
  }},
  {{
        "name": "lec06-slides",
        "path": "lec06-slides.pdf",
        "title": ["14.3. Gởi email có đính kèm file","14.4. Tìm hiểu thêm",]
  }}
]
```
Yêu cầu của người dùng:
"Tóm tắt cho tôi sách Python , phần có tiêu đề 'Giới thiệu sản phẩm'."
</INPUT>
<OUTPUT>
```json
{{
    "name": "Python",
    "path": "tai-lieu/python.pdf",
    "title": ["Giới thiệu sản phẩm"]
}}
```
</OUTPUT>
</EXAMPLE>
"""
)

summarize_content_node_prompt = ChatPromptTemplate.from_template(
    """
<ROLE>
Bạn là một hệ thống tóm tắt văn bản chuyên nghiệp.
</ROLE>

<OBJECTIVE>
Tạo ra một bản tóm tắt ngắn gọn, súc tích và chính xác bằng tiếng Việt từ một đoạn văn bản được cung cấp.
</OBJECTIVE>

<INPUT_SCHEMA>
input_text: Một chuỗi văn bản cần được tóm tắt.
</INPUT_SCHEMA>

<INPUT>
Văn bản cần tóm tắt:
{input_text}
</INPUT>

<INSTRUCTIONS>
Đọc và hiểu sâu nội dung, ý chính của input_text.
Xác định các điểm quan trọng, các luận điểm cốt lõi.
Viết lại các ý chính thành một đoạn văn ngắn gọn, mạch lạc bằng tiếng Việt.
</INSTRUCTIONS>

<OUTPUT_GUIDELINES>
Kết quả đầu ra chỉ bao gồm văn bản tóm tắt.
Không thêm vào bất kỳ lời chào hỏi, câu dẫn hay giải thích nào.
Bản tóm tắt phải giữ được ý nghĩa gốc của văn bản.
</OUTPUT_GUIDELINES>

<EXAMPLE>
<INPUT>
Văn bản cần tóm tắt:
Trí tuệ nhân tạo (AI) đang thay đổi nhanh chóng nhiều lĩnh vực của cuộc sống, từ y tế, giáo dục đến giải trí. Các hệ thống AI có khả năng phân tích dữ liệu lớn, nhận dạng mẫu và đưa ra dự đoán với độ chính xác ngày càng cao. Mặc dù mang lại nhiều lợi ích to lớn, việc phát triển AI cũng đặt ra những thách thức về đạo đức, bảo mật và tác động đến thị trường lao động.
</INPUT>
<OUTPUT>
Trí tuệ nhân tạo (AI) mang lại nhiều lợi ích cho các ngành như y tế, giáo dục nhờ khả năng phân tích dữ liệu và dự đoán, nhưng cũng tạo ra các thách thức về đạo đức, bảo mật và lao động.
</OUTPUT>
</EXAMPLE>
"""
)

router_node_prompt = ChatPromptTemplate.from_template(
    """
<ROLE>
Bạn là một chuyên gia phân loại yêu cầu của người dùng.
</ROLE>

<OBJECTIVE>
Phân tích yêu cầu của người dùng và phân loại nó vào một trong bốn danh mục: summarizer, quiz_generation, rag_qa, general_qa.
</OBJECTIVE>

<INPUT_SCHEMA>
user_request: Một chuỗi văn bản chứa câu hỏi hoặc yêu cầu từ người dùng.
</INPUT_SCHEMA>

<INPUT>
Yêu cầu của người dùng:
{user_request}
</INPUT>

<INSTRUCTIONS>
Phân loại user_request theo quy tắc sau:
- summarizer: nếu yêu cầu là tóm tắt nội dung của một tài liệu, sách, chương, mục.
- quiz_generation: nếu yêu cầu sinh câu hỏi, tạo quiz, hoặc đề kiểm tra dựa trên tài liệu.
- rag_qa: nếu yêu cầu là trả lời câu hỏi từ tài liệu đã tải lên (ví dụ: "Trong chương 1 sách X nói gì về Y?")


</INSTRUCTIONS>

<OUTPUT_GUIDELINES>
Câu trả lời của bạn BẮT BUỘC chỉ được là MỘT trong ba chuỗi sau: summarizer, quiz_generation, rag_qa.
Không thêm bất kỳ văn bản, giải thích, hay ký tự nào khác.
</OUTPUT_GUIDELINES>
"""
)
