# app.py
import re
import os
import shutil
import time
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse, HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from authlib.integrations.starlette_client import OAuth

from utils.ask_gpt import ask_gpt
from utils.render import render_quiz_html
from backend import (  
    DocumentProcessor,
    QuizCrafter,
    summarize_chapter_with_llamaindex,
    RetrieverBuilder,
)

from dotenv import load_dotenv

# load file google.env or .env
load_dotenv("google.env")
load_dotenv()  # allow .env as fallback

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret")
)

# Directories
UPLOAD_DIR = "uploads"
CACHE_DIR = "cache"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# In-memory caches (keyed by "email:file_path")
retriever_cache = {}
faiss_index_cache = {}
QUIZ_CACHE = {}

# OAuth Google
oauth = OAuth()
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_id=os.environ.get("GOOGLE_CLIENT_ID", ""),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET", ""),
    client_kwargs={
        'scope': 'openid email profile https://www.googleapis.com/auth/forms.body https://www.googleapis.com/auth/forms.responses.readonly'
    },
)


@app.get("/")
def read_index():
    return FileResponse("frontend.html")


# =========================
# Auth routes
# =========================
@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth")  # callback URL
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth")
async def auth(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = await oauth.google.userinfo(token=token)

    # Lưu session: lấy email làm key (bắt buộc)
    email = user_info.get("email")
    if not email:
        return PlainTextResponse("Không tìm thấy email từ Google.", status_code=400)

    request.session["user"] = dict(user_info)
    request.session["token"] = token

    # tạo thư mục cá nhân nếu chưa có
    user_upload_dir = os.path.join(UPLOAD_DIR, email)
    user_cache_dir = os.path.join(CACHE_DIR, email)
    os.makedirs(user_upload_dir, exist_ok=True)
    os.makedirs(user_cache_dir, exist_ok=True)

    return RedirectResponse(url="/")


# =========================
# LaTeX helpers (unchanged)
# =========================
LATEX_BLOCK_RE = re.compile(r"\\\[(.*?)\\\]", flags=re.DOTALL)
LATEX_INLINE_RE = re.compile(r"\\\((.*?)\\\)")


def fix_left_right(tex: str) -> str:
    if not tex:
        return tex

    left_count = len(re.findall(r"\\left", tex))
    right_count = len(re.findall(r"\\right", tex))

    if left_count <= right_count:
        return tex

    def add_right(match):
        content = match.group(0)
        if r"\right" in content:
            return content
        return content + r"\right."

    pattern = re.compile(r"(\\left(?:(?!\\left).){0,2000}?)($|\\\]|\\\)|\$\$|(?=\n))", flags=re.DOTALL)
    tex = pattern.sub(lambda m: add_right(m), tex)

    left_count = len(re.findall(r"\\left", tex))
    right_count = len(re.findall(r"\\right", tex))
    if left_count > right_count:
        tex = tex + r"\right." * (left_count - right_count)

    return tex


def render_for_html(text: str) -> str:
    if not text:
        return ""
    text = LATEX_INLINE_RE.sub(lambda m: f"${m.group(1)}$", text)
    text = LATEX_BLOCK_RE.sub(lambda m: f"$${m.group(1)}$$", text)
    text = fix_left_right(text)
    return text


# =========================
# Utility: extract pages from pdf (unchanged)
# =========================
def extract_pages_from_pdf(path: str, start: int, end: int) -> str:
    import fitz  # PyMuPDF
    texts = []
    if not os.path.exists(path):
        return ""
    with fitz.open(path) as pdf:
        n_pages = len(pdf)
        start_idx = max(1, int(start))
        end_idx = min(int(end), n_pages)
        if start_idx > end_idx:
            return ""
        for i in range(start_idx - 1, end_idx):
            try:
                text = pdf[i].get_text("text")
            except Exception:
                text = ""
            if text and text.strip():
                texts.append(text)
    return "\n".join(texts)


# =========================
# Helpers: conversation logging
# =========================
def _get_user_dirs(email: str):
    user_upload_dir = os.path.join(UPLOAD_DIR, email)
    user_cache_dir = os.path.join(CACHE_DIR, email)
    os.makedirs(user_upload_dir, exist_ok=True)
    os.makedirs(user_cache_dir, exist_ok=True)
    return user_upload_dir, user_cache_dir


def append_conversation(email: str, file_path: str, question: str, answer: str):
    user_cache_dir = os.path.join(CACHE_DIR, email)
    os.makedirs(user_cache_dir, exist_ok=True)
    conv_file = os.path.join(user_cache_dir, "conversations.json")
    data = {}
    if os.path.exists(conv_file):
        try:
            with open(conv_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    key = Path(file_path).name
    convs = data.get(key, [])
    convs.append({"question": question, "answer": answer, "ts": time.time()})
    data[key] = convs
    with open(conv_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# Upload endpoint (per-user)
# =========================
@app.post("/upload")
async def upload_file(request: Request, file: UploadFile):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"success": False, "message": "Vui lòng đăng nhập trước khi upload."}, status_code=401)

    email = user.get("email")
    user_upload_dir, user_cache_dir = _get_user_dirs(email)

    # prevent path traversal by using filename only
    filename = Path(file.filename).name
    file_path = os.path.join(user_upload_dir, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Xử lý document -> chunks
    processor = DocumentProcessor()
    chunks = processor.process([file_path])  # all_chunks

    if chunks:
        from langchain.schema import Document

        docs = [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in chunks]

        # Build retriever and cache keyed by email + filepath
        key = f"{email}:{file_path}"
        builder = RetrieverBuilder()
        retriever = builder.build_hybrid_retriever(docs)
        retriever_cache[key] = retriever

        # Build vector index FAISS and lưu vào cache per-user
        embeddings = HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-multilingual-base",
            model_kwargs={"trust_remote_code": True},
        )
        index = FAISS.from_documents(docs, embeddings)
        base = Path(file_path).stem
        faiss_dir = os.path.join(user_cache_dir, f"{base}_faiss")
        index.save_local(faiss_dir)
        faiss_index_cache[key] = index

        # Cache chunks per-user
        with open(os.path.join(user_cache_dir, f"{base}_chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)

        with open(os.path.join(user_cache_dir, "uploaded_file.txt"), "w", encoding="utf-8") as f:
            f.write(file_path)

        return {"success": True, "file_path": file_path, "message": f"✅ Indexed {len(chunks)} chunks."}
    else:
        return {"success": False, "file_path": file_path, "message": "⚠️ Không có chunk nào."}


# =========================
# Generate quiz (per-user)
# =========================
@app.post("/generate_quiz", response_class=PlainTextResponse)
async def generate_quiz(
    request: Request,
    file_path: str = Form(...),
    scope: str = Form("all"),
    count: int = Form(5),
    start: Optional[int] = Form(None),
    end: Optional[int] = Form(None),
):
    user = request.session.get("user")
    if not user:
        return PlainTextResponse("⚠️ Vui lòng đăng nhập trước.", status_code=401)
    email = user.get("email")
    _, user_cache_dir = _get_user_dirs(email)

    base = Path(file_path).stem
    chunks_pkl = os.path.join(user_cache_dir, f"{base}_chunks.pkl")
    if not os.path.exists(chunks_pkl):
        return PlainTextResponse("⚠️ Không tìm thấy chunks đã index. Vui lòng upload file trước.", status_code=400)
    with open(chunks_pkl, "rb") as f:
        all_docs = pickle.load(f)

    # Load FAISS index per-user if cần
    index = None
    key = f"{email}:{file_path}"
    if key in faiss_index_cache:
        index = faiss_index_cache[key]
    else:
        faiss_dir = os.path.join(user_cache_dir, f"{base}_faiss")
        if os.path.exists(faiss_dir):
            index = FAISS.load_local(
                faiss_dir,
                HuggingFaceEmbeddings(
                    model_name="Alibaba-NLP/gte-multilingual-base",
                    model_kwargs={"trust_remote_code": True},
                ),
                allow_dangerous_deserialization=True,
            )
            faiss_index_cache[key] = index

    # Nếu chọn part + file PDF → cắt trang thủ công
    if scope == "part" and start is not None and end is not None:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            try:
                s = int(start)
                e = int(end)
            except Exception:
                return PlainTextResponse("⚠️ start/end phải là số nguyên.", status_code=400)
            text = extract_pages_from_pdf(file_path, s, e)
            from langchain_core.documents import Document
            docs = [Document(page_content=text, metadata={"source": file_path})]
        else:
            docs = all_docs
    else:
        docs = all_docs

    # Debug logs (server console)
    print("Docs length:", len(docs) if docs else 0)
    if docs:
        example = getattr(docs[0], "page_content", docs[0])[:300]
        print("Sample context:", example)

    crafter = QuizCrafter(documents=docs)
    if index:
        crafter.index = index
    questions = crafter.get_questions("", count=int(count))

    # Lưu quiz vào cache theo user để publish sau
    QUIZ_CACHE[email] = questions

    return render_quiz_html(questions)


# =========================
# Publish quiz -> Google Forms (per-user)
# =========================
@app.post("/publish_quiz")
async def publish_quiz(request: Request):
    user = request.session.get("user")
    token = request.session.get("token")
    if not user or not token:
        return RedirectResponse(url="/login?next=/publish_quiz")

    email = user.get("email")
    questions = QUIZ_CACHE.get(email)
    if not questions:
        return JSONResponse({"error": "Chưa có quiz để xuất bản"})

    # Use env vars for client id/secret
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")

    creds = Credentials(
        token=token.get('access_token'),
        refresh_token=token.get('refresh_token'),
        token_uri='https://oauth2.googleapis.com/token',
        client_id=client_id,
        client_secret=client_secret,
        scopes=['https://www.googleapis.com/auth/forms.body', 'https://www.googleapis.com/auth/forms.responses.readonly']
    )
    service = build('forms', 'v1', credentials=creds)

    form_body = {
        "info": {
            "title": "Bài kiểm tra",
            "documentTitle": "Bài kiểm tra"
        }
    }
    form = service.forms().create(body=form_body).execute()
    form_id = form['formId']

    for idx, q in enumerate(questions):
        question_body = {
            "requests": [
                {
                    "createItem": {
                        "item": {
                            "title": q["question"],
                            "questionItem": {
                                "question": {
                                    "required": True,
                                    "choiceQuestion": {
                                        "type": "RADIO",
                                        "options": [{"value": opt} for opt in q["options"]],
                                        "shuffle": True
                                    }
                                }
                            }
                        },
                        "location": {"index": idx}
                    }
                }
            ]
        }
        service.forms().batchUpdate(formId=form_id, body=question_body).execute()

    form_url = f"https://docs.google.com/forms/d/{form_id}/edit"
    return JSONResponse({"quiz_link": form_url})


# =========================
# Generate summary (unchanged behavior, but require login optionally)
# =========================
@app.post("/generate_summary", response_class=PlainTextResponse)
async def generate_summary(
    request: Request,
    file_path: str = Form(...),
    scope: str = Form("all"),
    start: Optional[int] = Form(None),
    end: Optional[int] = Form(None)
):
    # Xử lý start/end an toàn
    if scope == "part" and start is not None and end is not None:
        try:
            s = int(start)
            e = int(end)
        except Exception:
            return PlainTextResponse("⚠️ start/end phải là số nguyên.", status_code=400)
        text = extract_pages_from_pdf(file_path, s, e)
    else:
        text = extract_pages_from_pdf(file_path, 1, 5)

    if not text:
        return PlainTextResponse("⚠️ Không có nội dung để tóm tắt hoặc file không tồn tại.", status_code=400)

    summary = summarize_chapter_with_llamaindex(text, "Summary")
    summary_html = render_for_html(summary)
    return summary_html


# =========================
# Chat with document (per-user) + save conversation
# =========================
@app.post("/chat_with_doc", response_class=PlainTextResponse)
async def chat_with_doc(
    request: Request,
    file_path: str = Form(...),
    question: str = Form(...),
    conversation: str = Form("")
):
    user = request.session.get("user")
    if not user:
        return PlainTextResponse("⚠️ Vui lòng đăng nhập trước.", status_code=401)
    email = user.get("email")

    base = Path(file_path).stem
    chunks_pkl = os.path.join(CACHE_DIR, email, f"{base}_chunks.pkl")
    if not os.path.exists(chunks_pkl):
        return PlainTextResponse("⚠️ Không tìm thấy chunks đã index cho user này. Vui lòng upload file trước.", status_code=400)
    with open(chunks_pkl, "rb") as f:
        docs = pickle.load(f)

    key = f"{email}:{file_path}"
    state_retriever = retriever_cache.get(key)
    if not state_retriever:
        builder = RetrieverBuilder()
        state_retriever = builder.build_hybrid_retriever(docs)
        retriever_cache[key] = state_retriever

    top_docs = state_retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in top_docs])

    prompt = f"""
You are a DOCUMENT-BASED Q&A system designed to provide DETAILED answers.

RULES:
0. Regardless of the language in the Context, you **MUST answer in Vietnamese**. Do not answer in English.
1. Use ONLY information from the "Context" section. Do NOT invent, add, or assume knowledge outside the document.
2. If the answer is not explicitly in the Context, reply exactly:
"I could not find it in the document."
However, if the answer can be logically inferred from the Context, explain it clearly and fully.
3. Always provide structured and detailed answers. Use bullet points, numbered lists, headings, or line breaks when appropriate.
4. Preserve **all formulas, numbers, symbols, punctuation, LaTeX, and code** exactly as they appear in the Context.
5. If mathematical formulas exist, copy them exactly in proper MathJax syntax:
- Inline: `$ ... $`
- Block: `$$ ... $$`
6. Do NOT change formula syntax, code, or technical notation; copy exactly from the Context.
7. Do NOT shorten or summarize excessively; always provide full, detailed answers based on the Context.
8. Maintain the original formatting as much as possible.
9. Always answer in Vietnamese, regardless of the language in the Context or question.

----------------
Context:
{context}
----------------

Conversation history:
{conversation}

Question: {question}

Answer in detail in Vietnamese:
""".strip()

    answer = ask_gpt("openai/gpt-oss-20b",
                 prompt=prompt,
                 temperature=0.3, max_tokens=1024)

    answer_html = render_for_html(answer)

    # Lưu lịch sử hội thoại per-user per-file
    try:
        append_conversation(email, file_path, question, answer_html)
    except Exception as e:
        print("Error saving conversation:", e)

    return answer_html


# =========================
# Helper endpoints: list user uploads & get conversations
# =========================
@app.get("/my_uploads")
async def my_uploads(request: Request):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Vui lòng đăng nhập."}, status_code=401)
    email = user.get("email")
    user_upload_dir, _ = _get_user_dirs(email)
    files = []
    for f in os.listdir(user_upload_dir):
        fp = os.path.join(user_upload_dir, f)
        if os.path.isfile(fp):
            files.append({"name": f, "path": fp})
    return JSONResponse({"uploads": files})


@app.get("/conversations")
async def get_conversations(request: Request, file_path: Optional[str] = None):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Vui lòng đăng nhập."}, status_code=401)
    email = user.get("email")
    conv_file = os.path.join(CACHE_DIR, email, "conversations.json")
    if not os.path.exists(conv_file):
        return JSONResponse({"conversations": {}})
    with open(conv_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if file_path:
        key = Path(file_path).name
        return JSONResponse({"conversations": data.get(key, [])})
    return JSONResponse({"conversations": data})

# =========================
# =========================
# Conversation persistence (load/save)
# =========================
@app.post("/save_message")
async def save_message(request: Request):
    """Lưu tin nhắn (user hoặc bot) vào file JSON của người dùng."""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Chưa đăng nhập"}, status_code=401)
    email = user["email"]

    data = await request.json()
    sender = data.get("sender")
    message = data.get("message")
    if not sender or not message:
        return JSONResponse({"error": "Thiếu dữ liệu"}, status_code=400)

    user_dir = os.path.join(CACHE_DIR, email)
    os.makedirs(user_dir, exist_ok=True)
    history_path = os.path.join(user_dir, "history.json")

    history = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append({"sender": sender, "message": message, "ts": time.time()})
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return {"status": "ok"}


@app.get("/load_conversation")
async def load_conversation(request: Request):
    """Trả lại lịch sử chat + file PDF trước đó của người dùng."""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Chưa đăng nhập"}, status_code=401)
    email = user["email"]

    user_dir = os.path.join(CACHE_DIR, email)
    history_path = os.path.join(user_dir, "history.json")
    file_path_txt = os.path.join(user_dir, "uploaded_file.txt")

    history = []
    file_path = None
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    if os.path.exists(file_path_txt):
        with open(file_path_txt, "r", encoding="utf-8") as f:
            file_path = f.read().strip()

    return {"history": history, "file_path": file_path}


