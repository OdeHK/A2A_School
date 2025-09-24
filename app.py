import re
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import shutil, os

from utils.ask_gpt import ask_gpt
from backend import QuizCrafter, summarize_chapter_with_llamaindex, extract_pages_from_pdf, RetrieverBuilder

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# 🚀 Utility chuẩn hóa LaTeX
# =========================
# Regex cho LaTeX block và inline
LATEX_BLOCK_RE = re.compile(r"\\\[(.*?)\\\]", flags=re.DOTALL)   # \[ ... \]
LATEX_INLINE_RE = re.compile(r"\\\((.*?)\\\)")                   # \( ... \)

def fix_left_right(tex: str) -> str:
    r"""
    Tìm tất cả \left mà chưa có \right, thêm \right. tự động
    """
    # Regex đơn giản: match \left, không quan tâm \right kế tiếp
    pattern = re.compile(r"(\\left[^\\]+)(?<!\\right)")
    
    def repl(m):
        return m.group(1) + r"\right."
    
    return pattern.sub(repl, tex)

def render_for_html(text: str) -> str:
    """
    Chuyển LaTeX sang cú pháp MathJax, giữ nguyên công thức toán học,
    đảm bảo \left...\right an toàn và không có SyntaxWarning.
    """
    if not text:
        return ""
    # Inline: \( ... \) -> $ ... $
    text = LATEX_INLINE_RE.sub(lambda m: f"${m.group(1)}$", text)
    # Block: \[ ... \] -> $$ ... $$
    text = LATEX_BLOCK_RE.sub(lambda m: f"$${m.group(1)}$$", text)
    # Sửa \left...\right nếu thiếu
    text = fix_left_right(text)
    return text

# =========================
# 🚀 Upload file
# =========================
@app.post("/upload")
async def upload_file(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"success": True, "file_path": file_path}

# =========================
# 🚀 Quiz
# =========================
@app.post("/generate_quiz", response_class=PlainTextResponse)
async def generate_quiz(
    file_path: str = Form(...),
    scope: str = Form("all"),
    count: int = Form(5),
    start: int = Form(None),
    end: int = Form(None)
):
    crafter = QuizCrafter()
    crafter.load_docs(file_path)
    crafter.create_index()

    if scope == "part" and start and end:
        text = extract_pages_from_pdf(file_path, list(range(start - 1, end)))
        crafter.load_text(text)
        crafter.create_index()

    questions = crafter.get_questions("") or []
    questions = questions[:count]

   
    """Tạo HTML quiz đẹp, đáp án xuống dòng rõ ràng"""
    quiz_html = "<h2>📖 Quiz</h2><br>"
    letters = ["A", "B", "C", "D"]
    for idx, q in enumerate(questions, 1):
        quiz_html += f"<b>Câu {idx}:</b> {q.get('question','') }<br>"
        options = q.get("options", [])  # tránh KeyError
        for i, opt in enumerate(options):
            quiz_html += f"&nbsp;&nbsp;<b>{letters[i]}.</b> {opt}<br>"
        quiz_html += f"<b>👉 Đáp án đúng:</b> {q.get('correct_answer','')}<br><br>"



    return quiz_html
    

# =========================
# 🚀 Summary
# =========================
@app.post("/generate_summary", response_class=PlainTextResponse)
async def generate_summary(
    file_path: str = Form(...),
    scope: str = Form("all"),
    start: int = Form(None),
    end: int = Form(None)
):
    if scope == "part" and start and end:
        text = extract_pages_from_pdf(file_path, list(range(start - 1, end)))
    else:
        text = extract_pages_from_pdf(file_path, list(range(0, 5)))

    summary = summarize_chapter_with_llamaindex(text, "Summary")
    # ✅ Chuẩn hóa LaTeX & Markdown
    summary_html = render_for_html(summary)
    return summary_html


# =========================
# 🚀 Chat
# =========================
@app.post("/chat_with_doc", response_class=PlainTextResponse)
async def chat_with_doc(
    file_path: str = Form(...),
    question: str = Form(...),
    conversation: str = Form("")
):
    from langchain.schema import Document

    text = extract_pages_from_pdf(file_path, list(range(0, 20)))
    docs = [Document(page_content=text, metadata={"source": file_path})]

    builder = RetrieverBuilder()
    retriever = builder.build_hybrid_retriever(docs)
    retrieved_docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in retrieved_docs])

    prompt = f"""
Bạn là một hệ thống HỎI–ĐÁP dựa trên tài liệu.

QUY TẮC:
0. Bất kể ngôn ngữ của Context, bạn **PHẢI trả lời bằng TIẾNG VIỆT**. Không được trả lời bằng tiếng Anh.
1. Chỉ sử dụng thông tin từ phần "Context" để trả lời.
2. KHÔNG bịa, KHÔNG thêm kiến thức ngoài tài liệu.
3. Nếu thông tin không có trong "Context", chỉ nói:
"Tôi không tìm thấy trong tài liệu."
Nhưng nếu câu trả lời có thể **diễn giải từ dữ liệu sẵn có trong Context**, hãy trình bày đầy đủ, rõ ràng.
4. Trả lời phải có cấu trúc rõ ràng, có trình tự logic. Có thể dùng gạch đầu dòng, đánh số, xuống dòng.
5. Giữ nguyên **công thức, số liệu, ký hiệu, dấu câu, định dạng LaTeX, code** nếu có trong Context.
6. Nếu phát hiện công thức toán học, **PHẢI trích nguyên văn, KHÔNG được thay đổi hay paraphrase**, luôn viết trong cú pháp MathJax chuẩn để MathJax hiển thị đúng:
   - Inline: `$ ... $`
   - Block: `$$ ... $$`
7. KHÔNG tự động thêm giải thích dài dòng ngoài Context; chỉ tóm tắt và giữ nguyên công thức/định nghĩa/code.
8. KHÔNG thay đổi cú pháp công thức/code; chỉ copy nguyên văn từ Context.

----------------
Context:
{context}
----------------

Lịch sử hội thoại:
{conversation}

Câu hỏi: {question}

Trả lời:
""".strip()
    answer = ask_gpt("openai/gpt-oss-20b", prompt=prompt, temperature=0.3, max_tokens=1024)
    # ✅ Chuẩn hóa LaTeX & Markdown
    answer_html = render_for_html(answer)
    return answer_html