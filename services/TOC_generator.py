from abc import ABC, abstractmethod
from enum import Enum
from PyPDF2 import PdfReader
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import os
import fitz
import logging
import re
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass
from pathlib import Path

# Import enhanced components
from services.token_manager import TokenManager, create_token_manager
from services.textrank_summarizer import HybridSummarizerStrategy, create_hybrid_summarizer

logger = logging.getLogger(__name__)

class TaskType(str, Enum):
    """Supported task types for TOC generation"""
    SUMMARIZE = "summarize"                      # Basic TextRank-only
    HYBRID_SUMMARIZE = "textrank_extract"        # Advanced TextRank for downstream LLM

# Data Models
@dataclass
class BookmarkNode:
    """Data class representing a bookmark node"""
    title: str
    page: Optional[int]
    children: List['BookmarkNode']
    content: Optional[str] = None  # Generated content (summary)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "page": self.page,
            "children": [child.to_dict() for child in self.children],
            "content": self.content
        }

# === STRATEGY PATTERN IMPLEMENTATION ===
class TOCContentStrategy(ABC):
    """Abstract base class for TOC content generation strategies."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return strategy name."""
        pass
    
    @abstractmethod
    def generate_content(self, text: str, title: str, **kwargs) -> str:
        """
        Generate content for a bookmark node.
        
        Args:
            text: Text content to process
            title: Section title
            **kwargs: Additional parameters for content generation
            
        Returns:
            str: Generated content
        """
        pass

class SummarizerStrategy(TOCContentStrategy):
    """Strategy for generating summaries using embedding-based TextRank with token optimization"""
    
    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-base", 
                 cache_folder: str = "./model"):
        """
        Initialize summarizer strategy.
        
        Args:
            model_name: Name of the embedding model to use
            cache_folder: Folder to cache the model
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={"trust_remote_code": True}
        )
        
        # Initialize token manager for optimal chunking
        from .token_manager import TokenManager
        self.token_manager = TokenManager(model_name="gpt-4")  # Default for basic strategy
        
        logger.info(f"Initialized SummarizerStrategy with model: {model_name}")

    @property
    def strategy_name(self) -> str:
        return TaskType.SUMMARIZE
    
    def generate_content(self, text: str, title: str, top_k: int = 5, **kwargs) -> str:
        """Generate summary using TextRank algorithm with token-aware chunking."""
        logger.info(f"Generating summary for: {title}")
        
        if not text or not text.strip():
            return f"Tóm tắt {title}: (không có nội dung)."
        
        # Use token_manager's adaptive chunking for better semantic boundaries
        chunks = self.token_manager.adaptive_chunking(
            text=text,
            max_chunk_tokens=500
        )
        
        if not chunks or len(chunks) == 0:
            return ""
        
        if len(chunks) == 1:
            return chunks[0]
        
        try:
            # Get embeddings
            doc_embeddings = self.embeddings.embed_documents(chunks)
            
            # Similarity matrix
            cooc_matrix = cosine_similarity(doc_embeddings)
            
            # Graph + PageRank
            graph = nx.from_numpy_array(cooc_matrix)
            scores = nx.pagerank(graph)
            
            # Calculate optimal top_k based on token budget
            optimal_top_k = self.token_manager.calculate_optimal_top_k(
                chunks=chunks,
                title=title
            )
            
            # Rank by score
            ranked = sorted(((scores[i], i, chunk) for i, chunk in enumerate(chunks)), reverse=True)
            
            # Select optimal number of chunks
            top_indices = sorted([idx for _, idx, _ in ranked[:optimal_top_k]])
            
            # Reorder chunks by original order
            top_chunks = [chunks[i] for i in top_indices]
            
            # Combine
            summary = " ".join(top_chunks)
            
            logger.debug(f"Selected {optimal_top_k}/{len(chunks)} chunks for summary")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary for {title}: {e}")
            return f"Tóm tắt {title}: Lỗi khi xử lý ({str(e)})."

# === FACTORY PATTERN IMPLEMENTATION ===
class TOCStrategyFactory:
    """Factory for creating TOC content strategies."""
    
    @staticmethod
    def create_hybrid_summarizer_strategy(
        embedding_model: str = "Alibaba-NLP/gte-multilingual-base",
        **kwargs
    ) -> HybridSummarizerStrategy:
        """Create pure TextRank content extractor for downstream LLM processing."""
        return create_hybrid_summarizer(
            embedding_model=embedding_model
        )
    
    @staticmethod
    def create_summarizer_strategy(
        model_name: str = "Alibaba-NLP/gte-multilingual-base", 
        cache_folder: str = "./model",
        **kwargs
    ) -> SummarizerStrategy:
        """Create legacy summarizer strategy with given configuration."""
        return SummarizerStrategy(model_name=model_name, cache_folder=cache_folder)

    @staticmethod
    def create_strategy(
        task_type: TaskType,
        toc_config: Optional[Dict[str, Any]] = None,
    ) -> TOCContentStrategy:
        """Create strategy based on task type and config."""
        config = toc_config or {}
        
        if task_type == TaskType.SUMMARIZE:
            return TOCStrategyFactory.create_summarizer_strategy(**config)
        elif task_type == TaskType.HYBRID_SUMMARIZE:
            return TOCStrategyFactory.create_hybrid_summarizer_strategy(**config)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

# === MAIN TOC GENERATOR CLASS ===
class TOCGenerator:
    """
    Main TOC generator class using Strategy pattern.
    
    Provides a unified interface for generating table of contents
    with different content generation strategies.
    """

    def __init__(self, pdf_path: str = None, strategy: Optional[TOCContentStrategy] = None):
        """
        Initialize TOC generator.
        
        Args:
            pdf_path: Path to PDF file
            strategy: Content generation strategy to use
        """
        self.pdf_path = pdf_path
        self.strategy = strategy
        self.bookmark_tree: List[BookmarkNode] = []
        
        # Validate PDF file
        
        self._validate_pdf_file(pdf_path)
        # Initialize PDF reader
        self.reader = PdfReader(pdf_path)
        self.outlines = self.reader.outline
        self.doc = fitz.open(pdf_path)
        
        # Build the bookmark tree on initialization
        self.bookmark_tree = self.build_bookmark_tree(self.outlines)
        
        if strategy:
            logger.info(f"Initialized TOCGenerator with strategy: {strategy.strategy_name}")
        
        logger.info(f"Loaded PDF: {pdf_path} ({len(self.reader.pages)} pages)")
        logger.info(f"Bookmark tree built with {len(self.bookmark_tree)} root nodes.")

    def set_strategy(self, strategy: TOCContentStrategy) -> None:
        """
        Set or change the content generation strategy.
        
        Args:
            strategy: New content generation strategy
        """
        logger.info(f"Changing strategy to: {strategy.strategy_name}")
        self.strategy = strategy
    
    def generate_toc(self, **strategy_kwargs) -> List[BookmarkNode]:
        """
        Generate table of contents for the entire document.
        
        Args:
            **strategy_kwargs: Additional parameters for strategy
            
        Returns:
            List[BookmarkNode]: Generated bookmark tree with full_document node
        """
        if not self.strategy:
            raise RuntimeError("No content generation strategy set. Use set_strategy() first.")
        
        logger.info(f"Generating full TOC using strategy: {self.strategy.strategy_name}")
        
        # Process all root nodes
        for i, root in enumerate(self.bookmark_tree):
            # Determine next root's page
            next_root_page = None
            if i + 1 < len(self.bookmark_tree):
                next_root_page = self.bookmark_tree[i + 1].page
            
            self.process_node(root, next_root_page, **strategy_kwargs)
        
        # Create full_document node that contains all existing nodes as children
        full_document_node = BookmarkNode(
            title="full_document",
            page=1,  # Start from page 1
            children=self.bookmark_tree.copy()  # All existing nodes become children
        )
        
        # Process the full_document node using the same logic as other nodes
        # This will combine content from all its children (which are the root nodes)
        self.process_node(full_document_node, len(self.reader.pages) + 1, **strategy_kwargs)
        
        # Return the tree with full_document as an additional root node
        result_tree = self.bookmark_tree.copy()
        result_tree.append(full_document_node)
        
        return result_tree

    def generate_toc_up_to_title(self, title: str, **strategy_kwargs) -> List[BookmarkNode]:
        """
        Generates content from leaf nodes up to the specified title node.

        This method processes only the necessary parts of the PDF but returns 
        the entire TOC tree. Nodes not in the target path will have no content.

        Args:
            title (str): The exact title of the bookmark to generate content for.
            **strategy_kwargs: Additional keyword arguments for the content generation strategy.

        Returns:
            List[BookmarkNode]: The full bookmark tree, with content generated only 
                                for the specified title's path. Returns an empty
                                list if the title is not found.
        """
        if not self.strategy:
            raise RuntimeError("No content generation strategy set. Use set_strategy() first.")

        logger.info(f"Generating TOC up to title '{title}' using strategy: {self.strategy.strategy_name}")

        # Special case: handle full_document
        if title == "full_document":
            # First, process all root nodes
            for i, root in enumerate(self.bookmark_tree):
                next_root_page = None
                if i + 1 < len(self.bookmark_tree):
                    next_root_page = self.bookmark_tree[i + 1].page
                self.process_node(root, next_root_page, **strategy_kwargs)
            
            # Create full_document node with all existing nodes as children
            full_document_node = BookmarkNode(
                title="full_document",
                page=1,
                children=self.bookmark_tree.copy()
            )
            
            # Process the full_document node
            self.process_node(full_document_node, len(self.reader.pages) + 1, **strategy_kwargs)
            
            # Return tree with full_document node
            result_tree = self.bookmark_tree.copy()
            result_tree.append(full_document_node)
            return result_tree

        # 1. Find the target node and the path to it
        target_node, path = self._find_node_and_path(title)
        if not target_node:
            logger.warning(f"Title '{title}' not found in bookmarks.")
            return []

        # 2. Determine the page number where the target section ends
        flat_bookmarks = self._flatten_bookmarks()
        try:
            current_index = flat_bookmarks.index(target_node)
        except ValueError:
            logger.error(f"Could not find the target node in the flattened list.")
            return self.bookmark_tree

        next_page = None
        for i in range(current_index + 1, len(flat_bookmarks)):
            if flat_bookmarks[i].page is not None:
                next_page = flat_bookmarks[i].page
                break
        
        if next_page is None:
            next_page = len(self.reader.pages) + 1

        logger.info(f"Processing node '{target_node.title}' (Page: {target_node.page}). Effective end page: {next_page-1}")
        
        # 3. Process the target node (and its children recursively)
        self.process_node(target_node, next_page, **strategy_kwargs)
    
        return self.bookmark_tree

    def _find_node_and_path(self, title: str, nodes: Optional[List[BookmarkNode]] = None, path: Optional[List[BookmarkNode]] = None) -> (Optional[BookmarkNode], Optional[List[BookmarkNode]]):
        """Recursively find a node by title and return the node and its path."""
        if nodes is None:
            nodes = self.bookmark_tree
        if path is None:
            path = []

        for node in nodes:
            current_path = path + [node]
            if node.title == title:
                return node, current_path
            
            if node.children:
                found_node, found_path = self._find_node_and_path(title, node.children, current_path)
                if found_node:
                    return found_node, found_path
        
        return None, None

    def _flatten_bookmarks(self, nodes: Optional[List[BookmarkNode]] = None) -> List[BookmarkNode]:
        """Flatten the bookmark tree into a single list in document order."""
        if nodes is None:
            nodes = self.bookmark_tree
        
        flat_list = []
        for node in nodes:
            flat_list.append(node)
            if node.children:
                flat_list.extend(self._flatten_bookmarks(node.children))
        return flat_list
    
    def build_bookmark_tree(self, bookmarks) -> List[BookmarkNode]:
        """Build bookmark tree from PDF outlines."""
        nodes = []
        
        for item in bookmarks:
            if isinstance(item, list):
                children = self.build_bookmark_tree(item)
                if nodes:
                    nodes[-1].children = children
            else:
                try:
                    page_num = self.reader.get_destination_page_number(item)
                except Exception:
                    page_num = None
                
                node = BookmarkNode(
                    title=item.title,
                    page=page_num + 1 if page_num is not None else None,
                    children=[]
                )
                nodes.append(node)
        
        return nodes

    def get_page_text(self, page_num: int) -> str:
        """Extract text from specific page."""
        try:
            if page_num is None or page_num < 1 or page_num > len(self.reader.pages):
                return ""
            
            page = self.doc[page_num - 1]
            return page.get_text("text") or ""
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num}: {e}")
            return ""
    
    def get_section_text(self, start_page: int, end_page: Optional[int] = None) -> str:
        """Extract text from page range."""
        if start_page is None:
            return ""
        if end_page is None or end_page < start_page:
            end_page = start_page
        
        text = ""
        for page_num in range(start_page, min(end_page + 1, len(self.reader.pages) + 1)):
            text += self.get_page_text(page_num)
        
        return text
    
    def process_node(self, node: BookmarkNode, next_page: Optional[int] = None, **strategy_kwargs):
        """Process a single node using the current strategy."""
        child_contents = []
        
        if node.children:
            for i, child in enumerate(node.children):
                if not child.content:
                    next_child_page = None
                    if i + 1 < len(node.children) and node.children[i+1].page is not None:
                        next_child_page = node.children[i + 1].page
                    else: 
                        next_child_page = next_page
                    
                    self.process_node(child, next_child_page, **strategy_kwargs)

                if child.content:
                    child_contents.append(child.content)
        
        if child_contents:
            combined_text = " ".join(child_contents)
            node.content = self.strategy.generate_content(combined_text, node.title, **strategy_kwargs)
        else:
            if node.page is not None:
                end_page = next_page - 1 if next_page else node.page
                text = self.get_section_text(node.page, end_page)
                node.content = self.strategy.generate_content(text, node.title, **strategy_kwargs)
            else:
                node.content = f"({self.strategy.strategy_name}) {node.title}: (không có trang cụ thể)."
    
    def find_content_by_title(self, title: str, nodes: Optional[List[BookmarkNode]] = None) -> Optional[str]:
        """Find content by title."""
        # Special case: handle full_document
        if title == "full_document":
            if self.strategy:
                # Process all root nodes first
                for i, root in enumerate(self.bookmark_tree):
                    next_root_page = None
                    if i + 1 < len(self.bookmark_tree):
                        next_root_page = self.bookmark_tree[i + 1].page
                    if not root.content:
                        self.process_node(root, next_root_page)
                
                # Create temporary full_document node and process it
                full_document_node = BookmarkNode(
                    title="full_document",
                    page=1,
                    children=self.bookmark_tree.copy()
                )
                self.process_node(full_document_node, len(self.reader.pages) + 1)
                return full_document_node.content
            else:
                return "No strategy set for full_document content generation."
        
        if nodes is None:
            nodes = self.bookmark_tree
        
        for node in nodes:
            if node.title == title:
                return node.content
            
            if node.children:
                result = self.find_content_by_title(title, node.children)
                if result:
                    return result
        
        return None
    
    def get_all_titles(self, nodes: Optional[List[BookmarkNode]] = None) -> List[str]:
        """Get all available titles."""
        if nodes is None:
            nodes = self.bookmark_tree
        
        titles = []
        for node in nodes:
            titles.append(node.title)
            if node.children:
                titles.extend(self.get_all_titles(node.children))
        
        # Add full_document as a special available title
        titles.append("full_document")
        
        return titles
    
    def export_toc(self, filename: str = "table_of_contents.json"):
        """
        Export table of contents to JSON file as list of dictionaries.
        
        Features:
        - Saves data as list of dict format for multiple PDF support
        - Checks if file exists: creates new if not, appends/updates if exists
        - Handles duplicate PDFs by updating existing entries
        - Includes metadata like timestamp and page count
        
        Args:
            filename: Output JSON file path
        """
        if not self.bookmark_tree:
            logger.warning("No bookmark tree to export. Generate TOC first.")
            return
        
        # Create current PDF entry with metadata
        current_entry = {
            "path": self.pdf_path,
            "table_of_contents": [node.to_dict() for node in self.bookmark_tree]
        }
        
        # Check if file exists and handle accordingly
        if os.path.exists(filename):
            try:
                # Read existing data
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Ensure existing_data is a list (handle legacy single dict format)
                if not isinstance(existing_data, list):
                    if isinstance(existing_data, dict):
                        existing_data = [existing_data]  # Convert single dict to list
                    else:
                        existing_data = []  # Invalid format, start fresh
                
                # Check if this PDF already exists in the list
                pdf_exists = False
                for i, entry in enumerate(existing_data):
                    if isinstance(entry, dict) and entry.get("path") == self.pdf_path:
                        # Update existing entry
                        existing_data[i] = current_entry
                        pdf_exists = True
                        logger.info(f"Updated existing entry for PDF: {self.pdf_path}")
                        break
                
                # If PDF doesn't exist, add new entry
                if not pdf_exists:
                    existing_data.append(current_entry)
                    logger.info(f"Added new entry for PDF: {self.pdf_path}")
                
                export_data = existing_data
                
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logger.warning(f"Error reading existing file {filename}: {e}. Creating new file.")
                export_data = [current_entry]
        else:
            # File doesn't exist, create new list
            export_data = [current_entry]
            logger.info(f"Creating new TOC file: {filename}")
        
        # Write data to file with error handling
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Table of Contents exported to {filename}")
            logger.info(f"Total PDFs in collection: {len(export_data)}")
            print(f"✅ Table of Contents exported to {filename}")
            print(f"📚 Total PDFs in collection: {len(export_data)}")
            
        except IOError as e:
            logger.error(f"Error writing to file {filename}: {e}")
            print(f"❌ Error writing to file {filename}: {e}")
            raise
        
    def print_toc(self, nodes: Optional[List[BookmarkNode]] = None, level: int = 0):
        """Print formatted table of contents."""
        if nodes is None:
            nodes = self.bookmark_tree
            strategy_name = self.strategy.strategy_name if self.strategy else "No Strategy"
            print(f"\n{'='*60}")
            print(f"TABLE OF CONTENTS - Strategy: {strategy_name}")
            print(f"PDF: {self.pdf_path}")
            print(f"{'='*60}")
        
        for node in nodes:
            indent = "  " * level
            page_info = f" (Page {node.page})" if node.page else ""
            print(f"{indent}📖 {node.title}{page_info}")
            
            if node.content:
                content_indent = "  " * (level + 1)
                content_lines = node.content.split('\n')
                for line in content_lines:
                    if line.strip():
                        print(f"{content_indent}▶ {line.strip()}")
            
            if node.children:
                self.print_toc(node.children, level + 1)
            
            # Bỏ dòng print() trống để output gọn hơn
    
    def _validate_pdf_file(self, pdf_path: str) -> None:
        """Validate that PDF file exists and is accessible."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {pdf_path}")
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
 
    
    @classmethod
    def create_with_config(
        cls,
        pdf_path: str,
        task_type: TaskType,
        toc_config: Optional[Dict[str, Any]] = None
    ) -> 'TOCGenerator':
        """
        Factory method to create TOCGenerator with specific configuration.
        """
        strategy = TOCStrategyFactory.create_strategy(task_type, toc_config)
        return cls(pdf_path, strategy)

# Usage Example and Testing
def main():
    """Main function to demonstrate TOC generator usage with enhanced features."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 📝 IMPORTANT: Replace this with your actual PDF path
    PDF_PATH = "Python rat la co ban - Vo Duy Tuan.pdf"
    
    # Pre-flight check
    if not Path(PDF_PATH).exists():
        logger.error(f"PDF file not found: {PDF_PATH}")
        logger.error("Please update the 'PDF_PATH' variable with a valid PDF file path.")
        return
    
    print("\n" + "="*60)
    print("🚀 ENHANCED TOC GENERATOR WITH TOKEN OPTIMIZATION 🚀")
    print("="*60 + "\n")
    
    try:
        # === TEST 1: Hybrid Cost-Effective Summarization ===
        print("\n--- TEST 1: Hybrid Summarization with Cost Optimization ---")
        
        hybrid_config = {
            "embedding_model": "Alibaba-NLP/gte-multilingual-base",
        }
        
        toc_generator = TOCGenerator.create_with_config(
            pdf_path=PDF_PATH,
            task_type=TaskType.HYBRID_SUMMARIZE,
            toc_config=hybrid_config
        )
        
        # Generate intelligent TOC with auto-optimized token usage
        toc_generator.generate_toc()
        
        # Export results
        toc_generator.export_toc("table_of_contents.json")
        
        
    
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}", exc_info=True)
    
    print("\n" + "="*60)
    print("🎉 ENHANCED TOC GENERATOR TESTS COMPLETED 🎉")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()