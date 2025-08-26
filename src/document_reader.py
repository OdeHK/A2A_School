
from pypdf import PdfReader
from docx import Document as DocxDocument
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes
import re
import unicodedata

class DocumentReader:
    def __init__(self):
        """Khởi tạo với các mapping ký tự toán học và LaTeX commands"""
        # Mapping cho các LaTeX commands thành ký hiệu Unicode
        self.latex_to_unicode = {
            # Greek letters - LaTeX to Unicode
            r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ', 
            r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
            r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
            r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\rho': 'ρ',
            r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ', r'\phi': 'φ',
            r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
            
            # Capital Greek letters
            r'\Alpha': 'Α', r'\Beta': 'Β', r'\Gamma': 'Γ', r'\Delta': 'Δ',
            r'\Epsilon': 'Ε', r'\Zeta': 'Ζ', r'\Eta': 'Η', r'\Theta': 'Θ',
            r'\Iota': 'Ι', r'\Kappa': 'Κ', r'\Lambda': 'Λ', r'\Mu': 'Μ',
            r'\Nu': 'Ν', r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Rho': 'Ρ',
            r'\Sigma': 'Σ', r'\Tau': 'Τ', r'\Upsilon': 'Υ', r'\Phi': 'Φ',
            r'\Chi': 'Χ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
            
            # Mathematical operators and symbols
            r'\det': 'det', r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
            r'\log': 'log', r'\ln': 'ln', r'\exp': 'exp',
            r'\sum': '∑', r'\prod': '∏', r'\int': '∫',
            r'\partial': '∂', r'\nabla': '∇',
            r'\infty': '∞', r'\pm': '±', r'\mp': '∓',
            r'\times': '×', r'\div': '÷', r'\cdot': '⋅',
            r'\sqrt': '√',
            
            # Bold/italic math (simplified)
            r'\mathbf{([^}]+)}': r'\1', r'\mathit{([^}]+)}': r'\1',
            r'\mathrm{([^}]+)}': r'\1', r'\mathcal{([^}]+)}': r'\1',
            
            # Set theory and logic
            r'\in': '∈', r'\notin': '∉', r'\ni': '∋', r'\notni': '∌',
            r'\subset': '⊂', r'\supset': '⊃', r'\subseteq': '⊆', r'\supseteq': '⊇',
            r'\cup': '∪', r'\cap': '∩', r'\emptyset': '∅',
            r'\land': '∧', r'\lor': '∨', r'\neg': '¬',
            r'\rightarrow': '→', r'\leftarrow': '←', r'\leftrightarrow': '↔',
            r'\Rightarrow': '⇒', r'\Leftarrow': '⇐', r'\Leftrightarrow': '⇔',
            r'\forall': '∀', r'\exists': '∃',
            
            # Number sets (mathbb)
            r'\mathbb{N}': 'ℕ', r'\mathbb{Z}': 'ℤ', r'\mathbb{Q}': 'ℚ',
            r'\mathbb{R}': 'ℝ', r'\mathbb{C}': 'ℂ', r'\mathbb{H}': 'ℍ',
            
            # Brackets and delimiters
            r'\{': '{', r'\}': '}', r'\[': '[', r'\]': ']',
            
            # Common mathematical expressions
            r'\neq': '≠', r'\leq': '≤', r'\geq': '≥', r'\approx': '≈',
            r'\equiv': '≡', r'\sim': '∼', r'\simeq': '≃',
        }
        
        # Mapping cho các ký hiệu bị broken/corrupted (giữ nguyên từ trước)
        self.math_symbol_map = {
            '&#955;': 'λ', '&lambda;': 'λ', '&#945;': 'α', '&alpha;': 'α',
            '&#8747;': '∫', '&int;': '∫', '&#8721;': '∑', '&sum;': '∑',
            'Î»': 'λ', 'Î±': 'α', 'Î²': 'β', 'Î³': 'γ',
        }

    def normalize_math_text(self, text: str) -> str:
        """Normalize mathematical symbols and LaTeX commands - Convert LaTeX to Unicode"""
        if not text:
            return ""
        
        # Normalize unicode characters để đảm bảo consistency
        text = unicodedata.normalize('NFKC', text)
        
        # Handle LaTeX commands with regex patterns (for complex patterns like \mathbf{v})
        import re
        
        # Process \mathbf{}, \mathit{}, etc. commands
        text = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\mathit\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\mathcal\{([^}]+)\}', r'\1', text)
        
        # Convert LaTeX commands to Unicode symbols
        for latex_cmd, unicode_symbol in self.latex_to_unicode.items():
            text = text.replace(latex_cmd, unicode_symbol)
        
        # Handle mathematical expressions in brackets [ ... ]
        # Convert display math mode to inline và clean up spaces
        text = re.sub(r'\[\s*(.*?)\s*\]', r'[\1]', text, flags=re.DOTALL)
        
        # Handle inline math mode (...) 
        text = re.sub(r'\(\s*(.*?)\s*\)', r'(\1)', text, flags=re.DOTALL)
        
        # CHỈ replace các ký hiệu bị broken/corrupted, GIỮ NGUYÊN ký hiệu chuẩn
        for broken_symbol, correct_symbol in self.math_symbol_map.items():
            text = text.replace(broken_symbol, correct_symbol)
        
        # Clean up extra spaces nhưng giữ cấu trúc toán học
        # Nhưng KHÔNG tách các từ có dấu nối như "véc-tơ"
        text = re.sub(r'(?<![a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])\s+(?![a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])', ' ', text)
        text = text.strip()
        
        # Giữ nguyên spacing xung quanh các operator toán học
        # Đảm bảo có space xung quanh =, +, -, etc. để dễ đọc
        text = re.sub(r'([a-zA-Zα-ωΑ-Ω])([=+\-*/])([a-zA-Zα-ωΑ-Ω0-9])', r'\1 \2 \3', text)
        text = re.sub(r'([0-9])([=+\-*/])([a-zA-Zα-ωΑ-Ω0-9])', r'\1 \2 \3', text)
        
        return text

    def extract_text_with_math_preservation(self, text: str) -> str:
        """Extract and preserve mathematical content - Convert LaTeX and preserve important math expressions"""
        if not text:
            return ""
        
        # Normalize text và convert LaTeX commands
        normalized_text = self.normalize_math_text(text)
        
        # Identify và đánh dấu IMPORTANT mathematical expressions để bảo vệ khi chunking
        # Chỉ đánh dấu những biểu thức toán học quan trọng, không phải mọi thứ
        important_math_patterns = [
            r'\[[^\]]*[=+\-*/∈∉⊆⊇∪∩∅→←↔⇒⇔λαβγδεζηθικμνξπρστυφχψωℝℂℕℤℚ][^\]]*\]',  # Complete math expressions in brackets
            r'det\s*\([^)]+\)',  # Determinant expressions
            r'[a-zA-Zα-ωΑ-Ω]\s*[=∈]\s*[a-zA-Zα-ωΑ-Ωℝℂℕℤℚ0-9\^{}×]+',  # Important assignments like A ∈ ℂ, λ = 5
            r'[a-zA-Zα-ωΑ-Ωℝℂℕℤℚ]+\^{[^}]+}',  # Superscripts like ℂ^{n×n}
            r'[a-zA-Zα-ωΑ-Ωℝℂℕℤℚ]+_{[^}]+}',  # Subscripts
        ]
        
        # Đánh dấu mathematical content để chunking algorithm biết và bảo vệ
        for pattern in important_math_patterns:
            matches = list(re.finditer(pattern, normalized_text, re.IGNORECASE | re.UNICODE))
            # Xử lý từ cuối lên đầu để không ảnh hưởng index
            for match in reversed(matches):
                math_expr = match.group()
                start, end = match.span()
                # Chỉ đánh dấu nếu expression đủ quan trọng (có ít nhất 3 ký tự)
                if len(math_expr.strip()) >= 3:
                    normalized_text = (
                        normalized_text[:start] + 
                        f"[MATH_START]{math_expr}[MATH_END]" + 
                        normalized_text[end:]
                    )
        
        return normalized_text
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
                page_text = page.extract_text()
                if page_text:
                    # Apply math-aware text processing
                    processed_text = self.extract_text_with_math_preservation(page_text)
                    full_text += processed_text + "\n"
            
            if not full_text.strip():
                print("⚠️  Phát hiện PDF dạng ảnh, đang thực hiện OCR...")
                images = convert_from_bytes(uploaded_file.getvalue())
                ocr_texts = []
                for image in images:
                    # OCR với cấu hình tối ưu cho ký tự toán học - GIỮ NGUYÊN ký hiệu
                    custom_config = r'--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵýỷỹ+−×÷=()[]{}^√∫∑∏αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ±∞∂∇∆∴∵∈∉∋∌⊆⊇⊂⊃∪∩∅∧∨¬→↔∀∃←↑↓⇒⇔.,;: -c preserve_interword_spaces=1'
                    ocr_text = pytesseract.image_to_string(
                        image, 
                        lang='vie+eng',
                        config=custom_config
                    )
                    if ocr_text.strip():
                        processed_ocr = self.extract_text_with_math_preservation(ocr_text)
                        ocr_texts.append(processed_ocr)
                
                full_text = "\n".join(ocr_texts)
                print("✅ OCR hoàn tất với hỗ trợ ký tự toán học.")
            
            return full_text
        except Exception as e:
            print(f"Lỗi khi đọc file PDF: {e}")
            return None

    def read_docx(self, uploaded_file: BytesIO) -> str:
        try:
            doc = DocxDocument(uploaded_file)
            full_text = ""
            for para in doc.paragraphs:
                if para.text.strip():
                    processed_text = self.extract_text_with_math_preservation(para.text)
                    full_text += processed_text + "\n"
            return full_text
        except Exception as e:
            print(f"Lỗi khi đọc file DOCX: {e}")
            return None

    def read_txt(self, uploaded_file: BytesIO) -> str:
        try:
            raw_text = uploaded_file.read().decode("utf-8")
            return self.extract_text_with_math_preservation(raw_text)
        except Exception as e:
            print(f"Lỗi khi đọc file TXT: {e}")
            return None
