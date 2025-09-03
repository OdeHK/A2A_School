from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Iterator, Literal, Optional, Dict, Any
import logging
from pathlib import Path
from urllib.parse import urlparse
from langchain.schema.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from config.settings import get_settings

logger = logging.getLogger(__name__)

class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"

class PDFLoaderType(str, Enum):
    """PDF loader strategies"""
    PYMUPDF = "pymupdf"

class SourceType(str, Enum):
    """Document source types"""
    FILE = "file"

# === STRATEGY PATTERN IMPLEMENTATION ===
class LoadingStrategy(ABC):
    """Abstract base class for document loading strategies."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return strategy name."""
        pass
    
    @abstractmethod
    def load_documents(self, source: str, **kwargs) -> List[Document]:
        """
        Load documents from source.
        
        Args:
            source: File path or URL
            **kwargs: Additional parameters for loading
            
        Returns:
            List[Document]: Loaded documents
        """
        pass
    
    @abstractmethod
    def lazy_load_documents(self, source: str, **kwargs) -> Iterator[Document]:
        """
        Lazily load documents from source.
        
        Args:
            source: File path or URL
            **kwargs: Additional parameters for loading
            
        Returns:
            Iterator[Document]: Document iterator
        """
        pass
    
class PyMuPDFLoadingStrategy(LoadingStrategy):

    def __init__(self, mode: Literal["page", "single"] = "page") -> None:
        """
        Initialize PyMuPDF strategy.
        
        Args:
            mode: Loading mode ('page' or 'single')
        """
        self.mode = mode

    @property
    def strategy_name(self) -> str:
        return PDFLoaderType.PYMUPDF
    
    def load_documents(self, source: str, **kwargs) -> List[Document]:
        """Load PDF using PyMuPDF."""

        logger.info(f"Loading PDF with PyMuPDF: {source}")
        loader = PyMuPDFLoader(file_path=source, mode=self.mode)
        return loader.load()
    
    def lazy_load_documents(self, source: str, **kwargs) -> Iterator[Document]:
        """Lazily load PDF using PyMuPDF."""
        logger.info(f"Lazy loading PDF with PyMuPDF: {source}")
        loader = PyMuPDFLoader(file_path=source, mode=self.mode)
        return loader.lazy_load()

# === FACTORY PATTERN IMPLEMENTATION ===

class LoadingStrategyFactory:
    """Factory for creating loading strategies."""
    
    @staticmethod
    def create_pdf_strategy(loader_type: PDFLoaderType, **kwargs) -> LoadingStrategy:
        """
        Create PDF loading strategy.
        
        Args:
            loader_type: Type of PDF loader
            **kwargs: Additional configuration
            
        Returns:
            LoadingStrategy: PDF loading strategy
        """
        if loader_type == PDFLoaderType.PYMUPDF:
            mode = kwargs.get('mode', 'page')
            return PyMuPDFLoadingStrategy(mode=mode)
        
        else:
            raise ValueError(f"Unknown PDF loader type: {loader_type}")
    
    @staticmethod
    def create_strategy(
        document_type: DocumentType,
        loader_config: Optional[Dict[str, Any]] = None
    ) -> LoadingStrategy:
        """
        Create loading strategy based on document type.
        
        Args:
            document_type: Type of document
            loader_config: Configuration for the loader
            
        Returns:
            LoadingStrategy: Appropriate loading strategy
        """
        config = loader_config or {}
        
        if document_type == DocumentType.PDF:
            pdf_loader_type = config.get('pdf_loader_type', PDFLoaderType.PYMUPDF)
            return LoadingStrategyFactory.create_pdf_strategy(pdf_loader_type, **config)
        
        else:
            raise ValueError(f"Unsupported document type: {document_type}")
        
# === MAIN DOCUMENT LOADER CLASS ===

class DocumentLoader:
    """
    Main document loader class using Strategy pattern.
    
    Provides a unified interface for loading documents from various sources
    and formats using different loading strategies.
    """
    
    def __init__(self, strategy: Optional[LoadingStrategy] = None):
        """
        Initialize document loader.
        
        Args:
            strategy: Loading strategy to use
        """
        self.strategy = strategy
        self.settings = get_settings()
        
        if strategy:
            logger.info(f"Initialized DocumentLoader with strategy: {strategy.strategy_name}")
    
    def set_strategy(self, strategy: LoadingStrategy) -> None:
        """
        Set or change the loading strategy.
        
        Args:
            strategy: New loading strategy
        """
        logger.info(f"Changing strategy to: {strategy.strategy_name}")
        self.strategy = strategy
    
    def load(self, source: str, **kwargs) -> List[Document]:
        """
        Load documents from source.
        
        Args:
            source: File path or URL
            **kwargs: Additional parameters
            
        Returns:
            List[Document]: Loaded documents
        """
        if not self.strategy:
            # Auto-detect strategy if not set
            self._auto_detect_strategy(source, kwargs)
            if not self.strategy:
                raise RuntimeError("No loading strategy could be determined for the given source.")
        
        self._validate_source(source)
        
        logger.info(f"Loading documents from: {source}")
        return self.strategy.load_documents(source, **kwargs)
    
    def lazy_load(self, source: str, **kwargs) -> Iterator[Document]:
        """
        Lazily load documents from source.
        
        Args:
            source: File path or URL
            **kwargs: Additional parameters
            
        Returns:
            Iterator[Document]: Document iterator
        """
        if not self.strategy:
            # Auto-detect strategy if not set
            self._auto_detect_strategy(source, kwargs)
            if not self.strategy:
                raise RuntimeError("No loading strategy could be determined for the given source.")

        self._validate_source(source)
        
        logger.info(f"Lazy loading documents from: {source}")
        return self.strategy.lazy_load_documents(source, **kwargs)
    
    def _auto_detect_strategy(self, source: str, config: Dict[str, Any]) -> None:
        """Auto-detect and set appropriate strategy based on source."""
        document_type = self._detect_document_type(source)
        strategy = LoadingStrategyFactory.create_strategy(document_type, config)
        self.set_strategy(strategy)
    
    def _detect_document_type(self, source: str) -> DocumentType:
        """Detect document type from file extension."""
        if self._is_url(source):
            # For URLs, might need additional logic to detect type
            # For now, assume PDF
            return DocumentType.PDF
        
        file_path = Path(source)
        extension = file_path.suffix.lower().lstrip('.')
        
        try:
            return DocumentType(extension)
        except ValueError:
            raise ValueError(f"Unsupported file extension: {extension}")
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _validate_source(self, source: str) -> None:
        """Validate that source exists and is accessible."""
        if self._is_url(source):
            # For URLs, you might want to add HTTP head request validation
            pass
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {source}")
            if not path.is_file():
                raise ValueError(f"Source is not a file: {source}")
    
    @classmethod
    def create_with_config(
        cls,
        document_type: DocumentType,
        loader_config: Optional[Dict[str, Any]] = None
    ) -> 'DocumentLoader':
        """
        Factory method to create DocumentLoader with specific configuration.
        
        Args:
            document_type: Type of documents to load
            loader_config: Configuration for the loader
            
        Returns:
            DocumentLoader: Configured loader instance
        """
        strategy = LoadingStrategyFactory.create_strategy(document_type, loader_config)
        return cls(strategy)

if __name__ == "__main__":
    document_loader = DocumentLoader.create_with_config(DocumentType.PDF)