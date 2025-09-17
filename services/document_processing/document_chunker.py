from enum import Enum
import logging
from typing import List, Iterator, Optional, Protocol
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from config.settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class ChunkingStrategyType(str, Enum):
    """Enum defining available chunking strategy types."""
    ONE_PAGE_PER_CHUNK = "one_page_per_chunk"
    RECURSIVE_SPLIT = "recursive_character_split"
    LLM_SPLIT = "llm_split"


class ThematicBlock(BaseModel):
    """Represents a single thematic block in the document."""
    summary_title: str = Field(
        description="The title of the thematic group. Don't add index numbers."
    )
    content: str = Field(
        description="The extracted content of the thematic group."
    )
    start_page_index: int = Field(
        default=0,
        description="The starting page index of the thematic group."
    )
    end_page_index: int = Field(
        default=0,
        description="The ending page index of the thematic group."
    )


class ThematicGroupList(BaseModel):
    """A list of thematic blocks."""
    thematic_group_list: List[ThematicBlock] = Field(
        default_factory=list,
        description="List of thematic blocks extracted from the document"
    )


# === STRATEGY PATTERN IMPLEMENTATION ===

class ChunkingStrategy(ABC):
    """Abstract base class for document chunking strategies."""
    
    @abstractmethod
    def chunk(self, pages: Iterator[Document]) -> List[Document]:
        """
        Chunk documents according to the specific strategy.
        
        Args:
            pages (Iterator[Document]): The pages to process.
            
        Returns:
            List[Document]: The processed chunks.
        """
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this strategy."""
        pass


class LLMService(Protocol):
    """Protocol for LLM services used in chunking."""
    
    def chunk_text(self, text: str) -> ThematicGroupList:
        """Chunk text using LLM and return structured thematic blocks."""
        ...


class OnePagePerChunkStrategy(ChunkingStrategy):
    """Strategy that treats each page as a separate chunk."""
    
    @property
    def strategy_name(self) -> str:
        return ChunkingStrategyType.ONE_PAGE_PER_CHUNK
    
    def chunk(self, pages: Iterator[Document]) -> List[Document]:
        """
        Convert each page to a separate chunk.
        
        Args:
            pages (Iterator[Document]): The pages to process.
            
        Returns:
            List[Document]: List of documents where each represents one page.
        """
        logger.info("Applying one-page-per-chunk strategy")
        processed_chunks = []
        
        for page_number, page in enumerate(pages, start=1):
            # Add page number to metadata
            enhanced_metadata = {**page.metadata, "page_number": page_number}
            chunk = Document(
                page_content=page.page_content,
                metadata=enhanced_metadata
            )
            processed_chunks.append(chunk)
        
        logger.info(f"Created {len(processed_chunks)} chunks (one per page)")
        return processed_chunks


class RecursiveTextSplitStrategy(ChunkingStrategy):
    """Strategy that uses recursive character splitting."""
    
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: Optional[List[str]] = None):
        """
        Initialize the recursive text splitting strategy.
        
        Args:
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Number of characters to overlap between chunks.
            separators (Optional[List[str]]): Custom separators for splitting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
        
        logger.info(f"Initialized RecursiveTextSplitStrategy with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    @property
    def strategy_name(self) -> str:
        return ChunkingStrategyType.RECURSIVE_SPLIT
    
    def chunk(self, pages: Iterator[Document]) -> List[Document]:
        """
        Split documents using recursive character splitting.
        
        Args:
            pages (Iterator[Document]): The pages to process.
            
        Returns:
            List[Document]: The split chunks.
        """
        logger.info("Applying recursive character text splitting strategy")
        
        # Convert iterator to list for processing
        page_list = list(pages)
        logger.info(f"Processing {len(page_list)} pages for recursive splitting")
        
        chunks = self.splitter.split_documents(page_list)
        
        # Enhance metadata with chunk information
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "chunk_strategy": self.strategy_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            })
        
        logger.info(f"Created {len(chunks)} chunks using recursive splitting")
        return chunks


class LLMBasedChunkingStrategy(ChunkingStrategy):
    """Strategy that uses LLM for intelligent semantic chunking."""
    
    def __init__(self, llm_service: LLMService, pages_per_segment: int = 5):
        """
        Initialize the LLM-based chunking strategy.
        
        Args:
            llm_service (LLMService): The LLM service to use for chunking.
            pages_per_segment (int): Number of pages to process in each LLM call.
        """
        self.llm_service = llm_service
        self.pages_per_segment = pages_per_segment
        
        logger.info(f"Initialized LLMBasedChunkingStrategy with pages_per_segment={pages_per_segment}")
    
    @property
    def strategy_name(self) -> str:
        return ChunkingStrategyType.LLM_SPLIT
    
    def chunk(self, pages: Iterator[Document]) -> List[Document]:
        """
        Chunk documents using LLM for semantic understanding.
        
        Args:
            pages (Iterator[Document]): The pages to process.
            
        Returns:
            List[Document]: Semantically coherent chunks.
        """
        logger.info("Applying LLM-based semantic chunking strategy")
        
        chunks = []
        segments = ""
        page_count = 0
        
        # Process pages in segments
        for page_number, page in enumerate(pages, start=1):
            extracted_text = page.page_content
            segments += f"\n\n##PAGE {page_number}##\n{extracted_text}"
            page_count += 1
            
            # Process segment when reaching the limit
            if page_count % self.pages_per_segment == 0:
                logger.debug(f"Processing segment ending at page {page_number}")
                thematic_blocks = self.llm_service.chunk_text(segments)
                chunks.extend(self._convert_thematic_blocks_to_documents(thematic_blocks.thematic_group_list))
                segments = ""
        
        # Process remaining pages if any
        if segments.strip():
            logger.debug(f"Processing final segment with {page_count % self.pages_per_segment} pages")
            thematic_blocks = self.llm_service.chunk_text(segments)
            chunks.extend(self._convert_thematic_blocks_to_documents(thematic_blocks.thematic_group_list))
        
        logger.info(f"Created {len(chunks)} semantic chunks using LLM")
        return chunks
    
    def _convert_thematic_blocks_to_documents(self, thematic_blocks: List[ThematicBlock]) -> List[Document]:
        """Convert ThematicBlock objects to LangChain Document objects."""
        documents = []
        
        for block in thematic_blocks:
            document = Document(
                page_content=block.content,
                metadata={
                    "summary_title": block.summary_title,
                    "start_page_index": block.start_page_index,
                    "end_page_index": block.end_page_index,
                    "chunk_strategy": self.strategy_name
                }
            )
            documents.append(document)
        
        return documents


# === FACTORY PATTERN IMPLEMENTATION ===

class ChunkingStrategyFactory:
    """Factory for creating chunking strategies based on configuration."""
    
    @staticmethod
    def create_strategy(
        strategy_type: ChunkingStrategyType, 
        llm_service: Optional[LLMService] = None,
        **kwargs
    ) -> ChunkingStrategy:
        """
        Create a chunking strategy based on the specified type.
        
        Args:
            strategy_type (ChunkingStrategyType): The type of strategy to create.
            llm_service (Optional[LLMService]): LLM service for LLM-based strategies.
            **kwargs: Additional configuration parameters.
            
        Returns:
            ChunkingStrategy: The created strategy instance.
            
        Raises:
            ValueError: If strategy type is unknown or required parameters are missing.
        """
        settings = get_settings()
        
        if strategy_type == ChunkingStrategyType.ONE_PAGE_PER_CHUNK:
            logger.info("Creating OnePagePerChunkStrategy")
            return OnePagePerChunkStrategy()
        
        elif strategy_type == ChunkingStrategyType.RECURSIVE_SPLIT:
            chunk_size = kwargs.get('chunk_size', settings.chunk_size)
            chunk_overlap = kwargs.get('chunk_overlap', settings.chunk_overlap)
            separators = kwargs.get('separators', None)
            
            logger.info(f"Creating RecursiveTextSplitStrategy with chunk_size={chunk_size}")
            return RecursiveTextSplitStrategy(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators
            )
        
        elif strategy_type == ChunkingStrategyType.LLM_SPLIT:
            if llm_service is None:
                raise ValueError("LLM service is required for LLM-based chunking strategy")
            
            pages_per_segment = kwargs.get('pages_per_segment', 5)
            logger.info(f"Creating LLMBasedChunkingStrategy with pages_per_segment={pages_per_segment}")
            return LLMBasedChunkingStrategy(
                llm_service=llm_service,
                pages_per_segment=pages_per_segment
            )
        
        else:
            raise ValueError(f"Unknown chunking strategy type: {strategy_type}")


# === CONCRETE LLM SERVICE IMPLEMENTATION ===

class NvidiaLLMService:
    """Concrete implementation of LLM service using NVIDIA API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize NVIDIA LLM service.
        
        Args:
            api_key (Optional[str]): NVIDIA API key. If None, will use from settings.
            model (Optional[str]): Model name. If None, will use from settings.
        """
        settings = get_settings()
        
        self.api_key = api_key or settings.nvidia_api_key
        if not self.api_key:
            raise ValueError("NVIDIA API key is required. Set it in environment or settings.")
        
        self.model = model or settings.default_model_name
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        
        self._initialize_llm_chain()
        
        logger.info(f"Initialized NvidiaLLMService with model={self.model}")
    
    def _initialize_llm_chain(self):
        """Initialize the LLM chain for text chunking."""
        try:
            # Initialize the LLM
            self.llm = ChatNVIDIA(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens
            )
            
            # Initialize the parser
            self.parser = PydanticOutputParser(pydantic_object=ThematicGroupList)
            
            # Initialize the prompt template
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("user", self._get_user_prompt())
            ]).partial(format_instruction=self.parser.get_format_instructions())
            
            # Create the chain
            self.llm_chain = self.prompt_template | self.llm | self.parser
            
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA LLM service: {e}")
            raise ValueError(f"Failed to initialize ChatNVIDIA: {e}")
    
    def chunk_text(self, text: str) -> ThematicGroupList:
        """
        Chunk text using the NVIDIA LLM.
        
        Args:
            text (str): The text to chunk.
            
        Returns:
            ThematicGroupList: Structured thematic blocks.
        """
        try:
            result = self.llm_chain.invoke({"text_to_split": text})
            return result
        except Exception as e:
            logger.error(f"Error during LLM text chunking: {e}")
            # Return empty list on error
            return ThematicGroupList(thematic_group_list=[])
    
    @staticmethod
    def _get_system_prompt() -> str:
        """Return the system prompt for LLM chunking."""
        return ("Reasoning:low. Your role includes handling text splitting tasks for the RAG pipeline. "
                "Perfect chunking isn't about blindly cutting at every paragraph break — "
                "it's about finding the sweet spot between semantic coherence and retrieval efficiency.")
    
    @staticmethod
    def _get_user_prompt() -> str:
        """Return the user prompt for chunking instruction."""
        return ("Take a deep breath, this is very important for my career. "
                "Please help me extract and name thematic blocks from a multipage document. "
                "An extracted thematic block can be a single paragraph (if paragraphs are long and dense), "
                "other times it's a few short related paragraphs grouped together. "
                "Ensuring each thematic block groups contain only one clear topic, focused, "
                "relevant and maintains its logical and contextual flow. "
                "Preserve original content, but correct broken sentences, broken math formulas; "
                "remove unnecessary symbols, error words due to extracting method. "
                "Filters out ancillary pages to focus solely on the core textual content. "
                "Using semantically grounded section title for block title. "
                "Start page number is the page of the first paragraph and end page number is the page of the last paragraph.\n"
                "If there is nothing to split, return empty list with provided format\n"
                "Your response must always follow this format: \n{format_instruction}\n"
                "If there is only one paragraph, return a list with one element in the format specified above\n"
                "If there is no text to split, return an empty list in the format specified above.\n"
                "##BEGIN##\n{text_to_split}\n##END##")


# === MAIN DOCUMENT CHUNKER CLASS ===

class DocumentChunker:
    """
    Main document chunker class that uses Strategy pattern for different chunking approaches.
    
    This class provides a clean interface for document chunking while delegating
    the actual chunking logic to specific strategy implementations.
    """
    
    def __init__(self, strategy: ChunkingStrategy):
        """
        Initialize the document chunker with a specific strategy.
        
        Args:
            strategy (ChunkingStrategy): The chunking strategy to use.
        """
        self.strategy = strategy
        self.settings = get_settings()
        logger.info(f"Initialized DocumentChunker with strategy: {self.strategy.strategy_name}")
    
    def set_strategy(self, strategy: ChunkingStrategy) -> None:
        """
        Change the chunking strategy at runtime.
        
        Args:
            strategy (ChunkingStrategy): The new strategy to use.
        """
        logger.info(f"Changing strategy from {self.strategy.strategy_name} to {strategy.strategy_name}")
        self.strategy = strategy
    
    def chunk(self, pages: Iterator[Document]) -> List[Document]:
        """
        Chunk documents using the current strategy.
        
        Args:
            pages (Iterator[Document]): The pages to process.
            
        Returns:
            List[Document]: The chunked documents.
            
        Raises:
            ValueError: If pages iterator is empty or invalid.
        """
        # Convert iterator to list to allow validation and reuse
        page_list = list(pages)
        
        if not page_list:
            raise ValueError("Pages iterator is empty.")
        
        logger.info(f"Chunking {len(page_list)} pages using {self.strategy.strategy_name}")
        
        # Convert back to iterator for strategy processing
        result = self.strategy.chunk(iter(page_list))
        
        logger.info(f"Chunking completed. Created {len(result)} chunks.")
        return result
    
    @classmethod
    def create_with_strategy_type(
        cls,
        strategy_type: ChunkingStrategyType,
        llm_service: Optional[LLMService] = None,
        **kwargs
    ) -> 'DocumentChunker':
        """
        Factory method to create DocumentChunker with specified strategy type.
        
        Args:
            strategy_type (ChunkingStrategyType): Type of strategy to create.
            llm_service (Optional[LLMService]): LLM service for LLM-based strategies.
            **kwargs: Additional strategy configuration.
            
        Returns:
            DocumentChunker: Configured document chunker instance.
        """
        strategy = ChunkingStrategyFactory.create_strategy(
            strategy_type=strategy_type,
            llm_service=llm_service,
            **kwargs
        )
        return cls(strategy)


# === CONVENIENCE FUNCTIONS ===

def create_recursive_chunker(chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> DocumentChunker:
    """
    Create a document chunker with recursive text splitting strategy.
    
    Args:
        chunk_size (Optional[int]): Size of each chunk. Uses settings default if None.
        chunk_overlap (Optional[int]): Overlap between chunks. Uses settings default if None.
        
    Returns:
        DocumentChunker: Configured chunker with recursive strategy.
    """
    return DocumentChunker.create_with_strategy_type(
        strategy_type=ChunkingStrategyType.RECURSIVE_SPLIT,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def create_llm_chunker(api_key: Optional[str] = None, model: Optional[str] = None) -> DocumentChunker:
    """
    Create a document chunker with LLM-based strategy.
    
    Args:
        api_key (Optional[str]): NVIDIA API key. Uses settings if None.
        model (Optional[str]): Model name. Uses settings if None.
        
    Returns:
        DocumentChunker: Configured chunker with LLM strategy.
    """
    llm_service = NvidiaLLMService(api_key=api_key, model=model)
    return DocumentChunker.create_with_strategy_type(
        strategy_type=ChunkingStrategyType.LLM_SPLIT,
        llm_service=llm_service
    )


def create_page_chunker() -> DocumentChunker:
    """
    Create a document chunker with one-page-per-chunk strategy.
    
    Returns:
        DocumentChunker: Configured chunker with page strategy.
    """
    return DocumentChunker.create_with_strategy_type(
        strategy_type=ChunkingStrategyType.ONE_PAGE_PER_CHUNK
    )


# === EXAMPLE USAGE ===

if __name__ == "__main__":
    # Example usage of the new Strategy Pattern implementation
    from langchain_community.document_loaders import PyMuPDFLoader
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # # Example 1: Using convenience function for LLM chunking
    # try:
    #     llm_chunker = create_llm_chunker()
    #     print(f"Created LLM chunker with strategy: {llm_chunker.strategy.strategy_name}")
    # except ValueError as e:
    #     print(f"Could not create LLM chunker: {e}")

    
    # # Example 2: Using convenience function for recursive chunking
    # recursive_chunker = create_recursive_chunker(chunk_size=1000, chunk_overlap=200)
    # print(f"Created recursive chunker with strategy: {recursive_chunker.strategy.strategy_name}")
    
    # # Example 3: Using convenience function for page chunking
    # page_chunker = create_page_chunker()
    # print(f"Created page chunker with strategy: {page_chunker.strategy.strategy_name}")
    
    #Example 4: Loading and processing a document
    loader = PyMuPDFLoader(
        file_path=r"C:\Users\likgn\Repository\RAG\example_data\giao_trinh_php.pdf",
        mode='page'
    )
    pages = loader.lazy_load()
    
    # Process one page per chunk
    chunker = create_page_chunker()
    chunks = chunker.chunk(pages)
    print(f"Processed document into {len(chunks)} chunks")
    
    print("Strategy Pattern implementation completed successfully!")
