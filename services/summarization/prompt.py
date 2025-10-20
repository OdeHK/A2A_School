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
library_str: Một chuỗi JSON chứa danh sách các đối tượng tài liệu. 
Mỗi đối tượng có các trường: 
- key (tên duy nhất trong dict, ví dụ "machine_learning_can_ban"), 
- document_id (mã định danh của tài liệu), 
- name (tên tài liệu), 
- title (một danh sách các tiêu đề). 

library_length: Số nguyên cho biết số lượng tài liệu có trong thư viện.

user_request: Một chuỗi văn bản chứa truy vấn tìm kiếm của người dùng. 
</INPUT_SCHEMA> 

<INPUT> 
Thư viện tài liệu: ```json 
{library_str}
```
Số lượng tài liệu: {library_length}
Yêu cầu của người dùng: "{user_request}"

</INPUT>

<INSTRUCTIONS> 
Thực hiện theo quy trình nghiêm ngặt sau:

**Bước 1: Tìm kiếm tài liệu dựa trên KEY**
1.1. Phân tích `user_request` để xác định **tên tài liệu** mà người dùng muốn tìm.
1.2. So sánh tên tài liệu này với các **key** trong `library_str`.

1.3. Xử lý kết quả so khớp:
- **NẾU TÌM THẤY một key phù hợp:** Chọn tài liệu tương ứng với key đó và chuyển sang **Bước 2**.
- **NẾU KHÔNG TÌM THẤY key nào phù hợp:**
- Nếu `library_length > 1`: Trả về đối tượng JSON rỗng: `{{}}`. 
- Nếu `library_length == 1`: Chọn tài liệu duy nhất đó và chuyển sang **Bước 2**.
- Nếu `library_length == 0`: Trả về đối tượng JSON rỗng: `{{}}`.

**Bước 2: Tìm kiếm tiêu đề trong tài liệu đã chọn**
2.1. Phân tích `user_request` một lần nữa để xác định **tiêu đề cụ thể** mà người dùng muốn.
2.2. So sánh tiêu đề này với danh sách `title` trong tài liệu đã chọn ở Bước 1.
- **NẾU TÌM THẤY một title phù hợp:** Chọn title đó.
- **NẾU KHÔNG TÌM THẤY title phù hợp** (hoặc người dùng không chỉ định tiêu đề): Trả về `["full_document"]`.
</INSTRUCTIONS>

<OUTPUT_GUIDELINES>
Câu trả lời BẮT BUỘC phải là một đối tượng JSON duy nhất.
Đối tượng JSON phải chứa các trường: document_id và title.
- Trường title phải là một danh sách chỉ chứa MỘT giá trị (chuỗi hoặc null).
- Không bao gồm bất kỳ văn bản hội thoại, lời giải thích hay định dạng markdown nào trong kết quả đầu ra.
</OUTPUT_GUIDELINES>

<EXAMPLE> 
<INPUT> 
Thư viện tài liệu: 
```json {{ 
"machine_learning_can_ban": 
{{ "document_id": "doc_222abc", "name": "machine_learning_can_ban", 
"title": [ "Giới thiệu chung", "Hồi quy tuyến tính", "Phân loại bằng cây quyết định", "Kết luận" ] }}, 
"deep_learning_nang_cao":
{{ "document_id": "doc_333xyz", "name": "deep_learning_nang_cao",
"title": [ "Mạng neuron tích chập (CNN)", "Mạng hồi tiếp (RNN)", "Transformer", "Ứng dụng thực tế" ] }}
}} ```
Yêu cầu của người dùng: "Tôi muốn xem phần về Transformer trong tài liệu deep learning nâng cao." 
</INPUT>
<OUTPUT> 
{{"document_id": "doc_333xyz", "title": ["Transformer"]}}
</OUTPUT> 
</EXAMPLE>
<EXAMPLE> 
<INPUT> 
Thư viện tài liệu: 
```json {{ 
"machine_learning_can_ban": 
{{ "document_id": "doc_222abc", "name": "machine_learning_can_ban", 
"title": [ "Giới thiệu chung", "Hồi quy tuyến tính", "Phân loại bằng cây quyết định", "Kết luận" ] }}, 
"deep_learning_nang_cao":
{{ "document_id": "doc_333xyz", "name": "deep_learning_nang_cao",
"title": [ "Mạng neuron tích chập (CNN)", "Mạng hồi tiếp (RNN)", "Transformer", "Ứng dụng thực tế" ] }}
}} ```
Yêu cầu của người dùng: "Tôi muốn xem phần về Transformer trong tài liệu toán cao cấp." 
</INPUT>
<OUTPUT> 
{{}}
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
Bạn là một chuyên gia phân tích và điều phối yêu cầu của người dùng.
</ROLE>

<OBJECTIVE>
Phân tích yêu cầu **hiện tại** của người dùng, dựa trên **bối cảnh** của cuộc hội thoại, và phân loại nó vào một trong bốn danh mục: summarizer, quiz_generation, rag_qa, create_form.
</OBJECTIVE>

<CONTEXT>
Đây là lịch sử của cuộc trò chuyện. Hãy sử dụng nó để hiểu các yêu cầu mang tính kế thừa hoặc không đầy đủ trong yêu cầu hiện tại.
Ví dụ: Nếu người dùng nói "tóm tắt nó đi", bạn cần xem lại lịch sử để biết "nó" là tài liệu nào.

<CHAT_HISTORY>
{chat_history}
</CHAT_HISTORY>
</CONTEXT>

<INPUT>
Yêu cầu hiện tại của người dùng cần phân loại:
{user_request}
</INPUT>

<INSTRUCTIONS>
Phân tích và phân loại `user_request` dựa vào `CHAT_HISTORY` theo quy tắc sau:
- **summarizer**: nếu yêu cầu là tóm tắt nội dung của một tài liệu, sách, chương, mục đã được đề cập.
- **quiz_generation**: nếu yêu cầu sinh câu hỏi, tạo quiz, hoặc đề kiểm tra dựa trên tài liệu đã có trong bối cảnh.
- **rag_qa**: nếu yêu cầu là trả lời một câu hỏi cụ thể từ tài liệu đã tải lên (ví dụ: "Trong chương 1 sách X nói gì về Y?").
- **create_form**: nếu yêu cầu tạo Google Form, chuyển đổi quiz sang form, hoặc tạo form từ bộ câu hỏi đã có trong bối cảnh.
</INSTRUCTIONS>

<OUTPUT_GUIDELINES>
Câu trả lời của bạn BẮT BUỘC chỉ được là MỘT trong bốn chuỗi sau: summarizer, quiz_generation, rag_qa, create_form.
Không thêm bất kỳ văn bản, giải thích, hay ký tự nào khác.
</OUTPUT_GUIDELINES>
"""
)