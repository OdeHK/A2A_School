
from pypdf import PdfReader
from docx import Document as DocxDocument
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes

class DocumentReader:
    def read(self, uploaded_file: BytesIO, filetype: str) -> str:
        if filetype == "pdf":
            return self.read_pdf(uploaded_file)
        elif filetype == "docx":
            return self.read_docx(uploaded_file)
        elif filetype == "txt":
            return self.read_txt(uploaded_file)
        return None

    def read_pdf(self, uploaded_file: BytesIO) -> str:
        full_text = ""
        try:
            pdf = PdfReader(uploaded_file)
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            
            if not full_text.strip():
                print("⚠️  Phát hiện PDF dạng ảnh, đang thực hiện OCR...")
                images = convert_from_bytes(uploaded_file.getvalue())
                ocr_texts = [pytesseract.image_to_string(image, lang='vie+eng') for image in images]
                full_text = "\n".join(ocr_texts)
                print("✅ OCR hoàn tất.")
            return full_text
        except Exception as e:
            print(f"Lỗi khi đọc file PDF: {e}")
            return None

    def read_docx(self, uploaded_file: BytesIO) -> str:
        try:
            doc = DocxDocument(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Lỗi khi đọc file DOCX: {e}")
            return None

    def read_txt(self, uploaded_file: BytesIO) -> str:
        try:
            return uploaded_file.read().decode("utf-8")
        except Exception as e:
            print(f"Lỗi khi đọc file TXT: {e}")
            return None
