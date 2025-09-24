import os
import json
import hashlib
import uuid
from PyPDF2 import PdfReader

# --- Helper Functions ---
def generate_document_id(name, path):
    """
    Generates a unique document ID based on document name and path.
    Uses MD5 hash of the combination to ensure uniqueness.
    """
    # Create a unique string from name and path
    unique_string = f"{name}_{path}"
    # Generate MD5 hash for shorter, consistent ID
    doc_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()[:12]
    return f"doc_{doc_id}"
def extract_document_name(path, remove_extension=True):
    """Extracts the base name of a file from its path."""
    base_name = os.path.basename(path)
    if remove_extension:
        return os.path.splitext(base_name)[0]
    else:
        return base_name

def get_all_bookmark_titles(bookmarks):
    """Recursively extracts all bookmark titles from a PyPDF2 outline object."""
    titles = []
    for item in bookmarks:
        if isinstance(item, list):
            titles.extend(get_all_bookmark_titles(item))
        else:
            titles.append(item.title)
    return titles

def process_pdf(pdf_path):
    """Processes a single PDF file and returns its information as a dictionary with document_id."""
    try:
        reader = PdfReader(pdf_path)
        doc_name = extract_document_name(pdf_path)
        bookmark_titles = get_all_bookmark_titles(reader.outline)
        
        # Generate unique document ID
        doc_id = generate_document_id(doc_name, pdf_path)
        
        return {
            'document_id': doc_id,
            'name': doc_name,
            'path': pdf_path,
            'title': bookmark_titles
        }
    except FileNotFoundError:
        print(f"🚨 Error: The PDF file was not found at '{pdf_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {pdf_path}: {e}")
        return None

# --- Main Function (Modified for Dictionary Structure) ---
def update_document_library(pdf_to_process, json_db_path):
    """
    Loads a library from a JSON file, adds info for a new PDF if it's not
    already present, and saves the updated library as a dictionary.
    """
    # 1. Load existing library or start with an empty dictionary
    doc_library = {}
    if os.path.exists(json_db_path):
        print(f"✅ Found '{json_db_path}'. Loading existing library.")
        with open(json_db_path, 'r', encoding='utf-8') as f:
            # Handle empty file case
            try:
                loaded_data = json.load(f)
                # Check if it's old format (list) and convert to new format (dict)
                if isinstance(loaded_data, list):
                    print("🔄 Converting from old list format to new dictionary format...")
                    doc_library = convert_list_to_dict(loaded_data)
                else:
                    doc_library = loaded_data
            except json.JSONDecodeError:
                doc_library = {} # File is empty, start fresh
    else:
        print(f"❌ '{json_db_path}' not found. A new library will be created.")

    # 2. Check if the current PDF is already in the library
    is_present = any(doc['path'] == pdf_to_process for doc in doc_library.values())

    if is_present:
        print(f"ℹ️ Document '{pdf_to_process}' is already in the library. Skipping.")
    else:
        print(f"➕ Adding info for '{pdf_to_process}' to the library.")
        # 3. If not present, process the PDF and add its data
        new_doc_data = process_pdf(pdf_to_process)
        if new_doc_data:
            doc_id = new_doc_data['document_id']
            # Store document data without the document_id field (since it's the key)
            doc_library[doc_id] = {
                'name': new_doc_data['name'],
                'path': new_doc_data['path'],
                'title': new_doc_data['title']
            }
            # 4. Save the updated library back to the file
            with open(json_db_path, 'w', encoding='utf-8') as f:
                json.dump(doc_library, f, indent=4, ensure_ascii=False)
            print(f"💾 Library updated and saved to '{json_db_path}'.")

    return doc_library

def convert_list_to_dict(doc_list):
    """
    Converts old list format to new dictionary format.
    """
    doc_dict = {}
    for doc in doc_list:
        doc_id = generate_document_id(doc['name'], doc['path'])
        doc_dict[doc_id] = {
            'name': doc['name'],
            'path': doc['path'],
            'title': doc['title']
        }
    return doc_dict

