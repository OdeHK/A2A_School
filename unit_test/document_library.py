import os
import json
from PyPDF2 import PdfReader

# --- Helper Functions (Unchanged) ---
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
    """Processes a single PDF file and returns its information as a dictionary."""
    try:
        reader = PdfReader(pdf_path)
        doc_name = extract_document_name(pdf_path)
        bookmark_titles = get_all_bookmark_titles(reader.outline)
        return {
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

# --- Main Function (Modified for Append Logic) ---
def update_document_library(pdf_to_process, json_db_path):
    """
    Loads a library from a JSON file, adds info for a new PDF if it's not
    already present, and saves the updated library.
    """
    # 1. Load existing library or start with an empty list
    doc_library = []
    if os.path.exists(json_db_path):
        print(f"✅ Found '{json_db_path}'. Loading existing library.")
        with open(json_db_path, 'r', encoding='utf-8') as f:
            # Handle empty file case
            try:
                doc_library = json.load(f)
            except json.JSONDecodeError:
                doc_library = [] # File is empty, start fresh
    else:
        print(f"❌ '{json_db_path}' not found. A new library will be created.")

    # 2. Check if the current PDF is already in the library
    is_present = any(doc['path'] == pdf_to_process for doc in doc_library)

    if is_present:
        print(f"ℹ️ Document '{pdf_to_process}' is already in the library. Skipping.")
    else:
        print(f"➕ Appending info for '{pdf_to_process}' to the library.")
        # 3. If not present, process the PDF and append its data
        new_doc_data = process_pdf(pdf_to_process)
        if new_doc_data:
            doc_library.append(new_doc_data)
            # 4. Save the updated library back to the file
            with open(json_db_path, 'w', encoding='utf-8') as f:
                json.dump(doc_library, f, indent=4, ensure_ascii=False)
            print(f"💾 Library updated and saved to '{json_db_path}'.")

    return doc_library

# --- Script Execution ---
if __name__ == "__main__":
    # Define file paths
    PDF_FILE = "Python rat la co ban - Vo Duy Tuan.pdf"
    JSON_DATABASE_FILE = "document_library.json" # Renamed for clarity

    # Update the library with the specified PDF
    final_library = update_document_library(PDF_FILE, JSON_DATABASE_FILE)

    # Print the final result
    print("\n--- Current Document Library ---")
    if final_library:
        print(json.dumps(final_library, indent=2, ensure_ascii=False))
    else:
        print("Library is empty.")