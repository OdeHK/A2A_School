
from pypdf import PdfReader
from io import BytesIO
from langchain.schema import Document

class PDFReader:
    """
    Một lớp đơn giản để đọc văn bản từ file PDF.
    """
    def read_pdf(self, uploaded_file: BytesIO) -> str:
        """
        Đọc một đối tượng file được tải lên từ Streamlit và trả về nội dung text.

        Args:
            uploaded_file: Đối tượng file từ st.file_uploader.

        Returns:
            Một chuỗi chứa toàn bộ văn bản từ file PDF.
        """
        try:
            pdf = PdfReader(uploaded_file)
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            return full_text
        except Exception as e:
            print(f"Lỗi khi đọc file PDF: {e}")
            return None
    def read_pdf_as_documents(self, uploaded_file: BytesIO) -> list:
        """
        Đọc file PDF và trả về danh sách Document (langchain.schema.Document).
        """
        text = self.read_pdf(uploaded_file)
        if text:
            return [Document(page_content=text)]
        return []

# Lưu ý: Để có chất lượng trích xuất cao nhất, đặc biệt với các file PDF phức tạp
# có nhiều cột và bảng biểu, bạn nên sử dụng script `preprocess_pdf.py` với 'marker'
# để chuyển đổi PDF thành Markdown trước, sau đó cho ứng dụng đọc file Markdown đó.