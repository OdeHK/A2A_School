# src/db/database_manager.py
# Quản lý tất cả các tương tác với cơ sở dữ liệu SQLite.

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Lớp quản lý cơ sở dữ liệu SQLite cho toàn bộ ứng dụng.
    Tạo bảng và cung cấp các phương thức CRUD (Create, Read, Update, Delete).
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        # Đảm bảo thư mục chứa file DB tồn tại
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"✅ DatabaseManager kết nối tới DB tại: {self.db_path}")

    def _get_connection(self):
        """Tạo và trả về một kết nối đến DB."""
        # Bật foreign key constraints cho mỗi kết nối
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_database(self):
        """Khởi tạo cấu trúc bảng trong DB nếu chưa tồn tại."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Bảng lưu thông tin tài liệu đã xử lý
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL, -- (e.g., 'processing', 'completed', 'failed')
                        chunks_count INTEGER,
                        chapters TEXT, -- Lưu danh sách chương dưới dạng JSON string
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Migration: Thêm column updated_at nếu chưa có
                cursor.execute("PRAGMA table_info(documents)")
                columns = [row[1] for row in cursor.fetchall()]
                if 'updated_at' not in columns:
                    cursor.execute("ALTER TABLE documents ADD COLUMN updated_at TIMESTAMP")
                    # Cập nhật tất cả record hiện tại với timestamp hiện tại
                    cursor.execute("UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL")
                    logger.info("✅ Đã thêm column updated_at vào bảng documents")

                # Bảng lưu các bộ đề quiz đã tạo
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quizzes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT,
                        quiz_content TEXT NOT NULL,
                        num_questions INTEGER NOT NULL,
                        difficulty TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (id)
                    )
                """)

                # Bảng lưu kết quả phân tích CSV
                cursor.execute("""
                     CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT NOT NULL UNIQUE,
                        analysis_data TEXT NOT NULL, -- Lưu kết quả phân tích dạng JSON
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()
                logger.info("Khởi tạo database và các bảng thành công.")
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi khởi tạo database: {e}", exc_info=True)
            raise

    def add_or_update_document(self, doc_id: str, filename: str, file_path: str, status: str, chunks_count: int = 0, chapters: Optional[List] = None):
        """Thêm mới hoặc cập nhật thông tin một tài liệu."""
        chapters_json = json.dumps(chapters, ensure_ascii=False) if chapters else None
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO documents (id, filename, file_path, status, chunks_count, chapters)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(file_path) DO UPDATE SET
                        id=excluded.id,
                        filename=excluded.filename,
                        status=excluded.status,
                        chunks_count=excluded.chunks_count,
                        chapters=excluded.chapters,
                        updated_at=CURRENT_TIMESTAMP
                """, (doc_id, filename, file_path, status, chunks_count, chapters_json))
                conn.commit()
                logger.info(f"Đã lưu/cập nhật tài liệu '{filename}' với status '{status}'.")
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi lưu tài liệu vào DB: {e}", exc_info=True)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Lấy thông tin một tài liệu bằng ID."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                if row:
                    doc = dict(row)
                    if doc.get('chapters'):
                        doc['chapters'] = json.loads(doc['chapters'])
                    return doc
                return None
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi lấy tài liệu từ DB: {e}", exc_info=True)
            return None

    def get_all_documents(self) -> List[Dict]:
        """Lấy danh sách tất cả các tài liệu đã được xử lý."""
        documents = []
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT id, filename, status, chunks_count, created_at FROM documents ORDER BY created_at DESC")
                for row in cursor.fetchall():
                    documents.append(dict(row))
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi lấy danh sách tài liệu: {e}", exc_info=True)
        return documents

    def save_quiz(self, quiz_data: Dict[str, Any]) -> bool:
        """
        Save quiz data to database.
        
        Args:
            quiz_data: Dictionary containing quiz information
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # First, check if we need to update the table schema
                cursor.execute("PRAGMA table_info(quizzes)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Update table schema if needed
                if 'chapter_title' not in columns:
                    cursor.execute("ALTER TABLE quizzes ADD COLUMN chapter_title TEXT")
                if 'quiz_id' not in columns:
                    cursor.execute("ALTER TABLE quizzes ADD COLUMN quiz_id TEXT UNIQUE")
                if 'questions_data' not in columns:
                    cursor.execute("ALTER TABLE quizzes ADD COLUMN questions_data TEXT")
                if 'metadata' not in columns:
                    cursor.execute("ALTER TABLE quizzes ADD COLUMN metadata TEXT")
                
                # Insert quiz data
                cursor.execute("""
                    INSERT INTO quizzes (
                        quiz_id, document_id, chapter_title, quiz_content, 
                        questions_data, num_questions, difficulty, scope, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    quiz_data.get('id'),
                    quiz_data.get('document_id'),
                    quiz_data.get('chapter_title'),
                    quiz_data.get('formatted_content'),
                    json.dumps(quiz_data.get('questions', []), ensure_ascii=False),
                    quiz_data.get('metadata', {}).get('num_questions', 0),
                    quiz_data.get('metadata', {}).get('difficulty', 'medium'),
                    quiz_data.get('chapter_title', ''),
                    json.dumps(quiz_data.get('metadata', {}), ensure_ascii=False)
                ))
                
                conn.commit()
                logger.info(f"✅ Successfully saved quiz {quiz_data.get('id')} to database")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error saving quiz to database: {e}", exc_info=True)
            return False

    def get_quiz_history(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get quiz history from database.
        
        Args:
            doc_id: Optional document ID to filter quizzes
            
        Returns:
            List of quiz metadata dictionaries
        """
        quizzes = []
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if doc_id:
                    cursor.execute("""
                        SELECT quiz_id, document_id, chapter_title, num_questions, 
                               difficulty, scope, created_at 
                        FROM quizzes 
                        WHERE document_id = ? 
                        ORDER BY created_at DESC
                    """, (doc_id,))
                else:
                    cursor.execute("""
                        SELECT quiz_id, document_id, chapter_title, num_questions, 
                               difficulty, scope, created_at 
                        FROM quizzes 
                        ORDER BY created_at DESC
                    """)
                
                for row in cursor.fetchall():
                    quizzes.append(dict(row))
                    
        except sqlite3.Error as e:
            logger.error(f"Error retrieving quiz history: {e}", exc_info=True)
        
        return quizzes

    def get_quiz_by_id(self, quiz_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific quiz by ID.
        
        Args:
            quiz_id: Quiz identifier
            
        Returns:
            Quiz data dictionary or None if not found
        """
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT quiz_id, document_id, chapter_title, quiz_content, 
                           questions_data, num_questions, difficulty, scope, 
                           metadata, created_at
                    FROM quizzes 
                    WHERE quiz_id = ?
                """, (quiz_id,))
                
                row = cursor.fetchone()
                if row:
                    quiz_data = dict(row)
                    # Parse JSON fields
                    if quiz_data.get('questions_data'):
                        quiz_data['questions'] = json.loads(quiz_data['questions_data'])
                    if quiz_data.get('metadata'):
                        quiz_data['metadata'] = json.loads(quiz_data['metadata'])
                    return quiz_data
                    
        except sqlite3.Error as e:
            logger.error(f"Error retrieving quiz {quiz_id}: {e}", exc_info=True)
        
        return None

