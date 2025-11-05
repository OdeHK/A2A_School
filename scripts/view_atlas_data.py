"""
Script to view detailed data in MongoDB Atlas database
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import dns.resolver
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure DNS
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "agent_for_teacher")

print("=" * 70)
print("MONGODB ATLAS - DETAILED DATA VIEW")
print("=" * 70)

try:
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    db = client[DATABASE_NAME]
    
    # View Users collection
    print("\n👥 USERS COLLECTION:")
    print("-" * 70)
    users = list(db.users.find({}, {"password": 0}))  # Hide password for security
    for i, user in enumerate(users, 1):
        print(f"\n   User {i}:")
        print(f"   - Username: {user.get('username')}")
        print(f"   - Email: {user.get('email', 'N/A')}")
        print(f"   - Created: {user.get('created_at', 'N/A')}")
    
    # View Documents collection
    print("\n\n📄 DOCUMENTS COLLECTION:")
    print("-" * 70)
    documents = list(db.documents.find({}))
    for i, doc in enumerate(documents, 1):
        print(f"\n   Document {i}:")
        print(f"   - Document ID: {doc.get('document_id')}")
        print(f"   - Username: {doc.get('username')}")
        print(f"   - File Name: {doc.get('file_name')}")
        print(f"   - Upload Date: {doc.get('upload_date')}")
        print(f"   - Processing Status: {doc.get('processing_status')}")
        print(f"   - Page Count: {doc.get('page_count')}")
        print(f"   - Chunk Count: {doc.get('chunk_count')}")
        
        # Check if has document library
        if 'document_library' in doc:
            lib = doc['document_library']
            print(f"   - Document Library: {len(lib)} items")
            for doc_name, doc_info in list(lib.items())[:2]:  # Show first 2
                print(f"     • {doc_name}: {len(doc_info.get('title', []))} chapters")
        
        # Check if has TOC
        if 'table_of_contents' in doc:
            toc = doc['table_of_contents']
            if 'sections' in toc:
                print(f"   - TOC Sections: {len(toc['sections'])}")
        
        # Check if has content
        if 'short_content' in doc:
            content = doc['short_content']
            if 'content' in content:
                print(f"   - Content Items: {len(content['content'])}")
    
    client.close()
    
    print("\n" + "=" * 70)
    print("✅ DATA VIEW COMPLETED")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
