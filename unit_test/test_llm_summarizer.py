import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# --- Helper Function (Unchanged) ---
def load_json_library(json_path):
    """Loads the document library from a JSON file."""
    if not os.path.exists(json_path):
        print(f"🚨 Error: JSON file not found at '{json_path}'")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"🚨 Error reading or parsing JSON file: {e}")
        return None

# --- New Semantic Search Function ---
def find_relevant_document(document_library, user_request):
    """
    Uses LangChain and Gemini for semantic search to find the most relevant
    document and title from a library based on a user's request.
    """
    if not document_library:
        return "The document library is empty or could not be loaded."

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # This new prompt encourages semantic analysis rather than direct lookup.
    prompt_template = """
    You are an intelligent semantic search engine for a document library.
    Your task is to deeply understand the user's request and find the single most relevant document from the provided library.
    After finding the best document, you must then find the single most relevant 'title' from within that document's title list.

    **Context: Document Library**
    ```json
    {library_str}
    ```

    **User Request:**
    "{user_request}"

    **Your Task & Rules:**
    1.  Analyze the user's request for its core intent and meaning.
    2.  Compare this intent against the 'name' and list of 'title's for each document to find the best match.
    3.  Your response MUST be a single JSON object containing the `name` and `path` of the best matching document, and a `title` field containing a list with ONLY the single best matching title.
    4.  If no relevant document or title is found, respond ONLY with an empty JSON object: {{}}.
    5.  Do not include any conversational text or explanations.
    
    **Example Output Structure:**
    {{"name": "Document Name", "path": "path/to/doc.pdf", "title": ["Most Relevant Title"]}}
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    print("🧠 Invoking semantic search chain...")
    try:
        library_str = json.dumps(document_library, indent=2)
        result = chain.invoke({
            "library_str": library_str,
            "user_request": user_request
        })
        return result
    except Exception as e:
        return f"An error occurred while running the chain: {e}"

# --- Script Execution ---
if __name__ == "__main__":
    JSON_DATABASE_FILE = "document_library.json"
    
    # A request that requires understanding, not just keyword matching.
    request = "summarize the book named 'Python rat la co ban - Vo Duy Tuan' title 'Giới thiệu'"

    print(f"User Request: \"{request}\"")
    
    library = load_json_library(JSON_DATABASE_FILE)
    
    if library:
        # Call the new semantic search function
        result = find_relevant_document(library, request)
        print(result)
        
        # print("\n--- Semantic Search Result ---")
        # if isinstance(result, (dict, list)):
        #     print(json.dumps(result, indent=2, ensure_ascii=False))
        # else:
        #     print(result)