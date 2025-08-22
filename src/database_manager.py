
import sqlite3
import json
from datetime import datetime

DB_FILE = "quizzes.db"

def create_connection():
    try:
        return sqlite3.connect(DB_FILE)
    except sqlite3.Error as e:
        print(e)
    return None

def create_tables():
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (id INTEGER PRIMARY KEY, subject TEXT, created_at TEXT);
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY, quiz_id INTEGER, question_text TEXT, options TEXT, answer TEXT, FOREIGN KEY (quiz_id) REFERENCES quizzes (id));
            """)
            conn.commit()
        finally:
            conn.close()

def save_quiz(quiz_data: list, subject: str):
    if not quiz_data:
        raise ValueError("Quiz data is empty")
        
    # Validate quiz data structure
    for q in quiz_data:
        if not isinstance(q, dict):
            raise ValueError(f"Invalid question format: {q}")
        if not all(key in q for key in ['question', 'choices', 'correct_answer']):
            raise ValueError(f"Missing required fields in question: {q}")
            
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            created_at = datetime.now().isoformat()
            
            # Create quiz record
            cursor.execute("INSERT INTO quizzes (subject, created_at) VALUES (?, ?)", 
                         (subject, created_at))
            quiz_id = cursor.lastrowid
            
            # Insert questions
            for q in quiz_data:
                # Convert choices list to JSON string
                choices_json = json.dumps(q.get('choices', []), ensure_ascii=False)
                
                # Convert correct_answer (1-4) to actual answer text
                correct_idx = int(q.get('correct_answer', 1)) - 1  # Convert to 0-based index
                correct_answer = q['choices'][correct_idx] if 0 <= correct_idx < len(q['choices']) else None
                
                if not correct_answer:
                    raise ValueError(f"Invalid correct_answer for question: {q}")
                
                cursor.execute("""
                    INSERT INTO questions (quiz_id, question_text, options, answer) 
                    VALUES (?, ?, ?, ?)""",
                    (quiz_id, q['question'], choices_json, correct_answer))
                    
            conn.commit()
            return quiz_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()