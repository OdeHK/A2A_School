import os
import json
from typing import List, Dict, TypedDict, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from services.rag.llm_service import LLMService
from services.TOC_generator import TOCGenerator, TaskType
from services.prompt import find_document_node_prompt, summarize_content_node_prompt


def find_content_by_title(toc: list, title: str) -> str | None:
    """
    Duyệt TOC theo đệ quy để tìm content của một title.
    """
    for node in toc:
        if node.get("title") == title:
            return node.get("content")
        if node.get("children"):  # nếu có children thì duyệt tiếp
            result = find_content_by_title(node["children"], title)
            if result is not None:
                return result
    return None


def get_content_from_pdf(pdf_list: list, pdf_path: str, title: str) -> str | None:
    """
    pdf_list: danh sách nhiều pdf (mỗi pdf là dict có 'path' và 'table_of_contents')
    pdf_path: đường dẫn pdf cần tìm
    title: tên mục cần lấy content
    """
    for pdf in pdf_list:
        if pdf.get("path") == pdf_path:
            toc = pdf.get("table_of_contents", [])
            return find_content_by_title(toc, title)
    return None

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        user_request: The initial query from the user.
        document_library: The loaded JSON data of all documents.
        relevant_document: The document object found by the search node.
        extracted_content: The text content extracted from the PDF.
        summary: The final summary of the content.
    """
    user_request: str
    document_library: List[Dict]
    table_of_contents: List[Dict]
    relevant_document: Optional[Dict]
    extracted_content: Optional[str]
    summary: Optional[str]

def find_document_node(state: GraphState):
    """
    Finds the most relevant document in the library based on the user request.
    This corresponds to your original find_relevant_document function.
    """
    print("--- 1. FINDING RELEVANT DOCUMENT ---")
    user_request = state["user_request"]
    document_library = state["document_library"]

    llm = LLMService(llm_type="google_gen_ai").get_llm()
    
    parser = JsonOutputParser()
    chain = find_document_node_prompt | llm | parser

    library_str = json.dumps(document_library, indent=2)
    result = chain.invoke({
        "library_str": library_str,
        "user_request": user_request
    })
    
    return {"relevant_document": result}

def extract_content_node(state: GraphState):
    """
    Extracts the relevant content from the PDF using the TOCGenerator.
    """
    print("--- 2. EXTRACTING CONTENT FROM PDF ---")
    relevant_document = state["relevant_document"]
    
    if not relevant_document or not relevant_document.get("path"):
        print("No document found to extract content from.")
        return {"extracted_content": "Error: Document not found."}

    content = get_content_from_pdf(
        state["table_of_contents"], 
        state["relevant_document"]["path"], 
        state["relevant_document"]["title"][0]
    )
    
    return {"extracted_content": content}

def summarize_content_node(state: GraphState):
    """
    Summarizes the extracted content using an LLM.
    """
    print("--- 3. SUMMARIZING EXTRACTED CONTENT ---")
    extracted_content = state["extracted_content"]

    if not extracted_content or "Error" in extracted_content:
         return {"summary": "Could not generate summary because content extraction failed."}

    llm = LLMService(llm_type="google_gen_ai").get_llm()
    chain = summarize_content_node_prompt | llm
    summary = chain.invoke({"input_text": extracted_content})
    
    return {"summary": summary.content}

workflow = StateGraph(GraphState)

workflow.add_node("find_document", find_document_node)
workflow.add_node("extract_content", extract_content_node)
workflow.add_node("summarize_content", summarize_content_node)

workflow.set_entry_point("find_document")


workflow.add_edge("find_document", "extract_content")
workflow.add_edge("extract_content", "summarize_content")
workflow.add_edge("summarize_content", END) 

summarization_graph = workflow.compile()


def load_json(json_path):
    """Helper function to load the initial data."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    request = "Tóm tắt sách Python rất là cơ bản - Võ Duy Tuấn, phần có tiêu đề 'Giới thiệu'."
    
    # Load the library data
    library = load_json("document_library.json")
    table_of_contents = load_json("table_of_contents.json")

    if library:
        # Define the initial inputs for the graph
        inputs = {
            "user_request": request,
            "document_library": library,
            "table_of_contents": table_of_contents
        }
        
        # Invoke the graph and stream the intermediate results
        for output in summarization_graph.stream(inputs):
            # The key is the name of the node that just finished
            node_name = list(output.keys())[0]
            print(f"Output from node '{node_name}':")
            # The value is the state update from that node
            print(output[node_name])
            print("\n---\n")
            
        # The final result is also available in the last output
        final_summary = list(output.values())[0].get('summary')
        print("\n===== FINAL SUMMARY =====\n")
        print(final_summary)