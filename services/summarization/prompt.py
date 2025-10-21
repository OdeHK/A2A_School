from langchain_core.prompts import ChatPromptTemplate

find_titles_prompt = ChatPromptTemplate.from_template(
"""
<ROLE>
Bạn là một công cụ phân tích ý định và đối sánh ngữ nghĩa thông minh, chuyên trích xuất thông tin.
</ROLE>

<OBJECTIVE>
Nhiệm vụ của bạn là phân tích yêu cầu của người dùng (`user_request`) để **phân biệt rõ ràng** giữa:
1.  Một **yêu cầu tìm kiếm/điều hướng** (khi người dùng muốn tìm một tiêu đề cụ thể).
2.  Một **yêu cầu toàn bộ tài liệu** (khi người dùng muốn tóm tắt hoặc biết nội dung chính của toàn bộ tài liệu).
3.  Một **phản hồi, mệnh lệnh chỉnh sửa, hoặc câu hỏi chung chung** (khi người dùng đưa feedback như "ngắn hơn", "dài hơn", hoặc hỏi "mạch lạc hơn đi?").

**CHỈ KHI** `user_request` là một **yêu cầu tìm kiếm/điều hướng** (Loại 1), bạn mới thực hiện đối sánh và tìm các tiêu đề phù hợp nhất từ `title_list`.
</OBJECTIVE>

<INPUT_SCHEMA>
title_list: Một danh sách JSON (list) các chuỗi (string), đại diện cho các tiêu đề có sẵn trong một tài liệu đã được chọn trước.
user_request: Một chuỗi văn bản (string) chứa truy vấn của người dùng.
</INPUT_SCHEMA>

<INPUT>
Danh sách tiêu đề: ```json
{title_list}
Yêu cầu của người dùng: "{user_request}" 
</INPUT>

<INSTRUCTIONS> Thực hiện theo quy trình nghiêm ngặt sau:
1. Phân tích ý định (Intent Analysis): Đọc kỹ user_request và phân loại mục đích chính của nó vào MỘT trong hai loại sau:
- **Loại 1: Yêu cầu Tìm kiếm/Điều hướng (Search/Navigation Request):** Người dùng chủ động muốn tìm, xem, đọc, biết về, hoặc đi đến một chủ đề, một phần nội dung cụ thể mà có khả năng được mô tả bởi một tiêu đề trong title_list.
  --Ví dụ Loại 1: "Cho tôi xem phần Transformer", "Thông tin về CNN", "Ứng dụng thực tế là gì?", "Phần Mạng hồi tiếp".
- **Loại 2: Yêu cầu Toàn bộ Tài liệu (Full Document Request):** Người dùng muốn thực hiện một hành động (như "tóm tắt", "nội dung chính", "nói về") trên toàn bộ tài liệu, chứ không phải một phần/tiêu đề cụ thể.
- **Loại 3: Phản hồi/Mệnh lệnh/Chung chung (Feedback/Command/General):** Người dùng đang nhận xét về một kết quả trước đó (ví dụ: "tóm tắt dài hơn", "ngắn hơn", "mạch lạc hơn", "viết lại", "ok", "hay quá"), hoặc hỏi một câu chung chung không nhắm vào tiêu đề cụ thể (ví dụ: "Tài liệu này nói về cái gì?", "bạn là ai?").
   --Ví dụ Loại 3: "Tóm tắt toàn bộ tài liệu", "Cho tôi biết nội dung chính của tất cả", "Tài liệu này nói về cái gì?".
2. Quy trình xử lý (Processing Logic):
- NẾU ý định là Loại 1 (Tìm kiếm/Điều hướng): Tiếp tục sang Bước 3 (Đối sánh).
- NẾU ý định là Loại 2 (Toàn bộ Tài liệu): Dừng lại ngay lập tức. Kết quả của bạn là ['full_document'].
- NẾU ý định là Loại 3 (Phản hồi/Mệnh lệnh/Chung chung): Dừng lại ngay lập tức. Kết quả của bạn là một danh sách rỗng [].
3. Đối sánh (Matching): (Chỉ thực hiện nếu là Loại 1)
- So sánh ý định tìm kiếm của user_request với TỪNG mục trong title_list.
- Tìm ra TẤT CẢ các tiêu đề trong title_list phù hợp về mặt ngữ nghĩa (semantic match) hoặc khớp chính xác (exact match) với yêu cầu của người dùng.
4. Quy tắc lựa chọn (Selection Rules):
- NẾU TÌM THẤY một hoặc nhiều tiêu đề khớp (từ Bước 3): Trả về một danh sách chứa TẤT CẢ các tiêu đề đó.
- NẾU KHÔNG TÌM THẤY tiêu đề nào khớp (kể cả khi là yêu cầu Loại 1 nhưng không có gì khớp): Trả về một danh sách rỗng [].
</INSTRUCTIONS>

<OUTPUT_GUIDELINES> 
Câu trả lời BẮT BUỘC phải là một đối tượng JSON dạng danh sách (list) các chuỗi (string). 
Nếu là Loại 1 và tìm thấy, trả về danh sách tiêu đề (ví dụ: ["Tiêu đề A"]).
Nếu là Loại 2, trả về ['full_document'].
Nếu là Loại 3, hoặc Loại 1 nhưng không tìm thấy, trả về danh sách rỗng []. Không bao gồm bất kỳ văn bản hội thoại, lời giải thích, hay định dạng markdown nào.
</OUTPUT_GUIDELINES>

<EXAMPLE 1: Tìm thấy một khớp (Loại 1)> <INPUT> Danh sách tiêu đề: ```json [ "Mạng neuron tích chập (CNN)", "Mạng hồi tiếp (RNN)", "Transformer", "Ứng dụng thực tế" ]
Yêu cầu của người dùng: "Tôi muốn xem phần về Transformer."
</INPUT>
<OUTPUT> 
["Transformer"]
</OUTPUT> 
</EXAMPLE>


<EXAMPLE 2: Yêu cầu cụ thể nhưng không có trong danh sách> <INPUT> Danh sách tiêu đề: ```json [ "Giới thiệu chung", "Hồi quy tuyến tính", "Phân loại bằng cây quyết định", "Kết luận" ]
Yêu cầu của người dùng: "Cho tôi xem phần về Mạng Neuron Tích chập."
</INPUT>
<OUTPUT> 
None
</OUTPUT> 
</EXAMPLE>

<EXAMPLE 3: Phản hồi / Mệnh lệnh (Loại 3)>
<INPUT> Danh sách tiêu đề: ```json [ "Giới thiệu", "Phân tích dữ liệu", "Mô hình học máy", "Kết luận" ]
Yêu cầu của người dùng: "Tóm tắt ngắn hơn."
</INPUT>
<OUTPUT>
[]
</OUTPUT>

<EXAMPLE 8: Yêu cầu tóm tắt toàn bộ (Loại 2) - **MỚI**>
<INPUT>
Danh sách tiêu đề: ```json
["Mạng neuron tích chập (CNN)", "Mạng hồi tiếp (RNN)", "Transformer", "Ứng dụng thực tế"]
Yêu cầu của người dùng: "Hãy tóm tắt toàn bộ tài liệu." 
</INPUT> 
<OUTPUT> ['full_document'] 
</OUTPUT>
""" )


summarize_content_prompt = ChatPromptTemplate.from_template(
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

router_prompt = ChatPromptTemplate.from_template(
    """
<ROLE>
Bạn là một chuyên gia phân tích và điều phối yêu cầu của người dùng.
</ROLE>

<OBJECTIVE>
Phân tích yêu cầu **hiện tại** của người dùng, dựa trên **bối cảnh** của cuộc hội thoại, và phân loại nó vào một trong bốn danh mục: summarization, quiz_generation, rag_qa, create_form.
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
- **summarization**: nếu yêu cầu là tóm tắt nội dung của một tài liệu, sách, chương, mục đã được đề cập.
- **quiz_generation**: nếu yêu cầu sinh câu hỏi, tạo quiz, hoặc đề kiểm tra dựa trên tài liệu đã có trong bối cảnh.
- **rag_qa**: nếu yêu cầu là trả lời một câu hỏi cụ thể từ tài liệu đã tải lên (ví dụ: "Trong chương 1 sách X nói gì về Y?").
- **create_form**: nếu yêu cầu tạo Google Form, chuyển đổi quiz sang form, hoặc tạo form từ bộ câu hỏi đã có trong bối cảnh.
</INSTRUCTIONS>

<OUTPUT_GUIDELINES>
Câu trả lời của bạn BẮT BUỘC chỉ được là MỘT trong bốn chuỗi sau: summarization, quiz_generation, rag_qa, create_form.
Không thêm bất kỳ văn bản, giải thích, hay ký tự nào khác.
</OUTPUT_GUIDELINES>
"""
)