
import logging
import io
import pytesseract
import pymupdf

from PIL import Image
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from typing import List, Iterator, Literal, Optional, Dict, Any


from langchain.schema.document import Document
from langchain_community.document_loaders import PyMuPDFLoader, PDFPlumberLoader


#from config.settings import get_settings

logger = logging.getLogger(__name__)

class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"


class PDFLoaderType(str, Enum):
    """PDF loader strategies"""
    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"
    TESSERACT_OCR = "tesseract_ocr"

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
    
# === STRATEGY PATTERN IMPLEMENTATION ===

class PDFPlumberLoadingStrategy(LoadingStrategy):
    @property
    def strategy_name(self) -> str:
        return PDFLoaderType.PDFPLUMBER

    def load_documents(self, source: str, **kwargs) -> List[Document]:

        loader = PDFPlumberLoader(source)
        return loader.load()

    def lazy_load_documents(self, source: str, **kwargs):
        loader = PDFPlumberLoader(source)
        yield from loader.lazy_load()


class TesseractOCR(LoadingStrategy):
    @property
    def strategy_name(self) -> str:
        return PDFLoaderType.TESSERACT_OCR

    def load_documents(self, source: str, **kwargs) -> List[Document]:
        document = pymupdf.open(source)
        docs = []
        for page_num in range(len(document)):
            # Ví dụ Windows: đường dẫn tới tesseract.exe
            pytesseract.pytesseract.tesseract_cmd = r"C:\Users\likgn\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
            page = document.load_page(page_num)
            # Xuất trang dưới dạng ảnh (250dpi)
            pix = page.get_pixmap(dpi=250)
            img = Image.open(io.BytesIO(pix.tobytes(output="png")))

            # OCR với tiếng Việt: lang="vie"
            page_text = pytesseract.image_to_string(img, lang="vie+eng")
            if page_text:
                docs.append(Document(page_content=page_text, metadata={"source": source, "page": page_num + 1}))

        return docs

    def lazy_load_documents(self, source: str, **kwargs):
        # Ví dụ Windows: đường dẫn tới tesseract.exe
        pytesseract.pytesseract.tesseract_cmd = r"C:\Users\likgn\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        document = pymupdf.open(source)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes(output="png")))
            page_text = pytesseract.image_to_string(img, lang="vie+eng")
            if page_text:
                yield Document(page_content=page_text, metadata={"source": source, "page": page_num + 1})

class PyMuPDFLoadingStrategy(LoadingStrategy):

    def __init__(self, mode: Literal["page", "single"] = "page") -> None:
        """
        Initialize PyMuPDF strategy.
        
        Args:
            mode: Loading mode ('page' or 'single')
        """
        if mode not in ("page", "single"):
            raise ValueError("mode must be either 'page' or 'single'")
        self.mode: Literal["page", "single"] = mode

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
        if loader_type == PDFLoaderType.PYMUPDF:
            mode = kwargs.get('mode', 'page')
            return PyMuPDFLoadingStrategy(mode=mode)
        elif loader_type == PDFLoaderType.PDFPLUMBER:
            return PDFPlumberLoadingStrategy()
        elif loader_type == PDFLoaderType.TESSERACT_OCR:
            return TesseractOCR()
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
    sample_pdf = r"C:\Users\likgn\Downloads\chatbot\chatbot\STSV-2024-ONLINE-1-20.pdf"  # Replace with your sample PDF path
    print(f"Testing document loading strategies on: {sample_pdf}\n")

    # PyMuPDF
    # print("--- PyMuPDF Strategy ---")
    # pymupdf_loader = DocumentLoader.create_with_config(DocumentType.PDF, {"pdf_loader_type": PDFLoaderType.PYMUPDF})
    # docs = pymupdf_loader.load(sample_pdf)
    # print(f"Loaded {len(docs)} documents.")
    # if docs:
    #     print("First doc preview:", docs[5].page_content, "...\n")

    # # PDFPlumber
    # print("--- PDFPlumber Strategy ---")
    # pdfplumber_loader = DocumentLoader.create_with_config(DocumentType.PDF, {"pdf_loader_type": PDFLoaderType.PDFPLUMBER})
    # docs = pdfplumber_loader.load(sample_pdf)
    # print(f"Loaded {len(docs)} documents.")
    # if docs:
    #     print("First doc preview:", docs[5].page_content, "...\n")

    # Tesseract OCR
    # print("--- Tesseract OCR Strategy ---")
    # ocr_loader = DocumentLoader.create_with_config(DocumentType.PDF, {"pdf_loader_type": PDFLoaderType.TESSERACT_OCR})
    # docs = ocr_loader.load(sample_pdf)
    # print(f"Loaded {len(docs)} documents.")
    # if docs:
    #     print("First 5 pages preview:")
    #     for doc in docs[:5]:
    #         print(doc.page_content, "...\n")