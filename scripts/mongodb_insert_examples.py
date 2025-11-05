"""
MongoDB Atlas Insert Examples
==============================
Các ví dụ về cách insert dữ liệu vào MongoDB Atlas cho project A2A_School.

Để chạy các ví dụ này:
1. Đảm bảo đã cài đặt pymongo: pip install pymongo
2. Đảm bảo file .env có MONGODB_URI và MONGODB_DATABASE_NAME
3. Chạy: python scripts/mongodb_insert_examples.py
"""

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import dns.resolver
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure DNS
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "agent_for_teacher")

def get_db_connection():
    """Kết nối tới MongoDB Atlas"""
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    # Test connection
    client.admin.command('ismaster')
    db = client.get_database(DATABASE_NAME)
    return client, db


# ============================================
# VÍ DỤ 1: INSERT USER
# ============================================
def insert_user_example():
    """
    Insert một user mới vào collection 'users'
    """
    client, db = get_db_connection()
    users_collection = db.get_collection("users")
    
    # Tạo user document
    user_doc = {
        "username": "teacher_nguyen",
        "password": "hashed_password_here",  # Nên hash password trước khi lưu
        "email": "nguyen@example.com",
        "full_name": "Nguyễn Văn A",
        "role": "teacher",
        "created_at": datetime.now(),
        "last_login": None
    }
    
    # Insert vào MongoDB
    result = users_collection.insert_one(user_doc)
    print(f"✅ Inserted user with _id: {result.inserted_id}")
    
    client.close()


# ============================================
# VÍ DỤ 2: INSERT DOCUMENT METADATA
# ============================================
def insert_document_metadata_example():
    """
    Insert metadata của một tài liệu vào collection 'documents'
    """
    client, db = get_db_connection()
    documents_collection = db.get_collection("documents")
    
    # Tạo document metadata
    doc_metadata = {
        "document_id": "doc_12345",
        "username": "demo",
        "file_name": "Giao_trinh_Python.pdf",
        "file_path": "/uploads/Giao_trinh_Python.pdf",
        "file_size": 2048576,  # bytes
        "upload_date": datetime.now(),
        "processing_status": "completed",
        "page_count": 150,
        "chunk_count": 300,
        "error_message": None,
        
        # Document library info
        "document_library": {
            "Giao_trinh_Python": {
                "document_id": "doc_12345",
                "name": "Giao_trinh_Python",
                "title": [
                    "Chương 1: Giới thiệu Python",
                    "Chương 2: Biến và kiểu dữ liệu",
                    "Chương 3: Cấu trúc điều khiển"
                ],
                "added_date": datetime.now().isoformat()
            }
        },
        
        # Table of contents structure
        "table_of_contents": {
            "sections": [
                {
                    "title": "Chương 1: Giới thiệu Python",
                    "page": 10,
                    "children": [
                        {
                            "title": "1.1 Lịch sử Python",
                            "page": 11,
                            "children": []
                        },
                        {
                            "title": "1.2 Ứng dụng của Python",
                            "page": 15,
                            "children": []
                        }
                    ]
                },
                {
                    "title": "Chương 2: Biến và kiểu dữ liệu",
                    "page": 20,
                    "children": []
                }
            ]
        },
        
        # Short content (summaries)
        "short_content": {
            "content": [
                {
                    "title": "Chương 1: Giới thiệu Python",
                    "content": "Python là ngôn ngữ lập trình bậc cao, dễ học và mạnh mẽ. Được phát triển bởi Guido van Rossum vào năm 1991..."
                },
                {
                    "title": "Chương 2: Biến và kiểu dữ liệu",
                    "content": "Biến trong Python không cần khai báo kiểu. Python hỗ trợ nhiều kiểu dữ liệu như int, float, string, list, dict..."
                }
            ]
        }
    }
    
    # Insert vào MongoDB
    result = documents_collection.insert_one(doc_metadata)
    print(f"✅ Inserted document metadata with _id: {result.inserted_id}")
    
    client.close()


# ============================================
# VÍ DỤ 3: UPDATE EXISTING DOCUMENT
# ============================================
def update_document_example():
    """
    Update một document đã tồn tại
    """
    client, db = get_db_connection()
    documents_collection = db.get_collection("documents")
    
    # Update document với document_id = "doc_12345"
    result = documents_collection.update_one(
        {"document_id": "doc_12345", "username": "demo"},  # Filter
        {
            "$set": {
                "processing_status": "completed",
                "page_count": 150,
                "last_updated": datetime.now()
            }
        }
    )
    
    print(f"✅ Matched {result.matched_count} document(s)")
    print(f"✅ Modified {result.modified_count} document(s)")
    
    client.close()


# ============================================
# VÍ DỤ 4: UPSERT (UPDATE OR INSERT)
# ============================================
def upsert_document_example():
    """
    Upsert: Nếu document tồn tại thì update, không thì insert mới
    """
    client, db = get_db_connection()
    documents_collection = db.get_collection("documents")
    
    # Upsert document
    result = documents_collection.update_one(
        {"document_id": "doc_67890", "username": "demo"},  # Filter
        {
            "$set": {
                "file_name": "Machine_Learning_Basics.pdf",
                "upload_date": datetime.now(),
                "processing_status": "processing"
            }
        },
        upsert=True  # Nếu không tìm thấy thì insert mới
    )
    
    if result.upserted_id:
        print(f"✅ Inserted new document with _id: {result.upserted_id}")
    else:
        print(f"✅ Updated existing document")
    
    client.close()


# ============================================
# VÍ DỤ 5: INSERT NHIỀU DOCUMENTS CÙNG LÚC
# ============================================
def insert_many_documents_example():
    """
    Insert nhiều documents cùng một lúc (bulk insert)
    """
    client, db = get_db_connection()
    documents_collection = db.get_collection("documents")
    
    # Danh sách các documents
    documents = [
        {
            "document_id": "doc_001",
            "username": "demo",
            "file_name": "Document_1.pdf",
            "upload_date": datetime.now()
        },
        {
            "document_id": "doc_002",
            "username": "demo",
            "file_name": "Document_2.pdf",
            "upload_date": datetime.now()
        },
        {
            "document_id": "doc_003",
            "username": "demo",
            "file_name": "Document_3.pdf",
            "upload_date": datetime.now()
        }
    ]
    
    # Insert nhiều documents
    result = documents_collection.insert_many(documents)
    print(f"✅ Inserted {len(result.inserted_ids)} documents")
    print(f"   IDs: {result.inserted_ids}")
    
    client.close()


# ============================================
# VÍ DỤ 6: QUERY DOCUMENTS
# ============================================
def query_documents_example():
    """
    Query và đọc dữ liệu từ MongoDB
    """
    client, db = get_db_connection()
    documents_collection = db.get_collection("documents")
    
    # Query 1: Find all documents của user "demo"
    print("\n📚 All documents for user 'demo':")
    for doc in documents_collection.find({"username": "demo"}):
        print(f"   - {doc.get('file_name')} (ID: {doc.get('document_id')})")
    
    # Query 2: Find one specific document
    print("\n📄 Finding specific document:")
    doc = documents_collection.find_one({"document_id": "doc_12345"})
    if doc:
        print(f"   Found: {doc.get('file_name')}")
        print(f"   Status: {doc.get('processing_status')}")
    
    # Query 3: Count documents
    count = documents_collection.count_documents({"username": "demo"})
    print(f"\n🔢 Total documents for 'demo': {count}")
    
    client.close()


# ============================================
# VÍ DỤ 7: DELETE DOCUMENTS
# ============================================
def delete_document_example():
    """
    Xóa documents từ MongoDB
    """
    client, db = get_db_connection()
    documents_collection = db.get_collection("documents")
    
    # Delete one document
    result = documents_collection.delete_one({"document_id": "doc_001"})
    print(f"✅ Deleted {result.deleted_count} document(s)")
    
    # Delete many documents
    result = documents_collection.delete_many({
        "username": "demo",
        "processing_status": "failed"
    })
    print(f"✅ Deleted {result.deleted_count} failed document(s)")
    
    client.close()


# ============================================
# MAIN - CHẠY CÁC VÍ DỤ
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("MongoDB Atlas Insert Examples for A2A_School")
    print("=" * 60)
    
    try:
        # Uncomment các dòng dưới để chạy ví dụ tương ứng:
        
        # print("\n1️⃣ Inserting user...")
        # insert_user_example()
        
        # print("\n2️⃣ Inserting document metadata...")
        # insert_document_metadata_example()
        
        # print("\n3️⃣ Updating document...")
        # update_document_example()
        
        # print("\n4️⃣ Upserting document...")
        # upsert_document_example()
        
        # print("\n5️⃣ Inserting many documents...")
        # insert_many_documents_example()
        
        print("\n6️⃣ Querying documents...")
        query_documents_example()
        
        # print("\n7️⃣ Deleting documents...")
        # delete_document_example()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
