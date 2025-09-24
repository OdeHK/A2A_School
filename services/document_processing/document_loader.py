from enum import Enum
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from typing import List, Iterator, Literal, Optional, Dict, Any


from langchain.schema.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from pathlib import Path
from typing import Union, BinaryIO
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
logger = logging.getLogger(__name__)

class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    HTML = "html"

class PDFType(str, Enum):
    """Document source types"""
    URL = "url"
    LOCAL = "local"
    STREAM = "stream"
    
class PDFLoaderType(str, Enum):
    """Document source types"""
    PYMUPDF = "pymupdf"
    DOCLING = "docling"

class WebsiteLoaderType(str, Enum):
    """Website (HTML) loader strategies"""
    BEAUTIFULSOUP = "beautiful_soup"
    DOCLING = "docling"


from collections import defaultdict

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

class DoclingPDFLoadingStrategy(LoadingStrategy):
    """Strategy that uses Docling for PDF loading."""

    def __init__(self, pipeline_options: Optional[PdfPipelineOptions] = None):
        """
        Initialize Docling strategy.
        
        Args:
            pipeline_options (Optional[PdfPipelineOptions]): Docling pipeline options.
        """
        self.converter =  DocumentConverter(
            format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
        )

    @property
    def strategy_name(self) -> str:
        return PDFLoaderType.DOCLING
    
    def load_documents(self, source: Union[str, Path, BinaryIO], **kwargs) -> List[Document]:
        """
        Load PDF using Docling.
        
        Args:
            source (Union[str, Path, BinaryIO]): Path to the PDF file or a file-like object.
            **kwargs: Additional parameters for Docling.
            
        Returns:
            List[Document]: Loaded documents.
        """
        logger.info(f"Loading PDF with Docling: {source}")
        
        doc = self.converter.convert(source).document
        
        # collect texts by page number
        pages = defaultdict(list)
        for text in doc.texts: 
            if text.prov:  # provenance tells us page number
                page_no = text.prov[0].page_no
                pages[page_no].append(text.text)

        # total number of pages (highest page number seen)
        total_pages = max(pages.keys())

        # build Document list
        documents = []
        for page_no in sorted(pages.keys()):
            page_text = "\n".join(pages[page_no])
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": str(source),
                        "page": page_no,
                        "total_pages": total_pages,
                    },
                )
            )
                
        return documents
    def lazy_load_documents(self, source: Union[str, Path, BinaryIO], **kwargs) -> Iterator[Document]:
        """Lazily load PDF using Docling (yields from load)."""
        logger.info(f"Lazy loading PDF with Docling: {source}")
        yield from self.load_documents(source, **kwargs)

class BeautifulSoupWebsiteLoadingStrategy(LoadingStrategy):
    """Strategy to load HTML pages with BeautifulSoup."""

    @property
    def strategy_name(self) -> str:
        return WebsiteLoaderType.BEAUTIFULSOUP


    def load_documents(self, source: str, **kwargs) -> List[Document]:
        logger.info(f"Loading website (BeautifulSoup): {source}")
        headers = {
            "User-Agent": "MyResearchBot/1.0 (https://example.com/contact)"
        }
        resp = requests.get(source, timeout=15, headers=headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text()

        return [
            Document(
                page_content=text,
                metadata={
                    "source": source,
                },
            )
        ]

    def lazy_load_documents(self, source: str, **kwargs) -> Iterator[Document]:
        yield from self.load_documents(source, **kwargs)


class DoclingWebsiteLoadingStrategy(LoadingStrategy):
    """Strategy to load HTML pages with Docling."""
    def __init__(self, pipeline_options: Optional[PdfPipelineOptions] = None):
        """
        Initialize Docling strategy.
        
        Args:
            pipeline_options (Optional[PdfPipelineOptions]): Docling pipeline options.
        """
        
        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        })
        
    @property
    def strategy_name(self) -> str:
        return WebsiteLoaderType.DOCLING


    def load_documents(self, source: str, **kwargs) -> List[Document]:
        logger.info(f"Loading website (Docling): {source}")
        
        result = self.converter.convert(source)
        return [
            Document(
                page_content=result.document.export_to_markdown(),
                metadata={
                    "source": source,
                },
            )
        ]
        
    def lazy_load_documents(self, source: str, **kwargs) -> Iterator[Document]:
        yield from self.load_documents(source, **kwargs)

    
# === FACTORY PATTERN IMPLEMENTATION ===

class LoadingStrategyFactory:
    """Factory for creating loading strategies."""

    @staticmethod
    def create_pdf_strategy(loader_type: PDFLoaderType = PDFLoaderType.PYMUPDF, **kwargs) -> LoadingStrategy:
        if loader_type == PDFLoaderType.PYMUPDF:
            return PyMuPDFLoadingStrategy(mode=kwargs.get("mode", "page"))
        elif loader_type == PDFLoaderType.DOCLING:
            return DoclingPDFLoadingStrategy(pipeline_options=kwargs.get("pipeline_options"))
        raise ValueError(f"Unknown PDF loader type: {loader_type}")

    @staticmethod
    def create_website_strategy(loader_type: WebsiteLoaderType = WebsiteLoaderType.BEAUTIFULSOUP, **kwargs) -> LoadingStrategy:
        if loader_type == WebsiteLoaderType.BEAUTIFULSOUP:
            return BeautifulSoupWebsiteLoadingStrategy()
        elif loader_type == WebsiteLoaderType.DOCLING:
            return DoclingWebsiteLoadingStrategy(pipeline_options=kwargs.get("pipeline_options"))
        raise ValueError(f"Unknown Website loader type: {loader_type}")

    @staticmethod
    def create_strategy(
        document_type: DocumentType,
        loader_config: Optional[Dict[str, Any]] = None
    ) -> LoadingStrategy:
        """CORRECTED: Create strategy based on document type and config."""
        config = loader_config or {}
        if document_type == DocumentType.PDF:
            loader_type = config.get('pdf_loader_type', PDFLoaderType.PYMUPDF)
            return LoadingStrategyFactory.create_pdf_strategy(loader_type, **config)
        elif document_type == DocumentType.HTML:
            loader_type = config.get("website_loader_type", WebsiteLoaderType.BEAUTIFULSOUP)
            return LoadingStrategyFactory.create_website_strategy(loader_type, **config)
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
        """CORRECTED: Auto-detect and set appropriate strategy based on source."""
        document_type = self._detect_document_type(source)
        # Pass the kwargs from load/lazy_load as loader_config
        strategy = LoadingStrategyFactory.create_strategy(document_type, loader_config=config)
        self.set_strategy(strategy)
    
    def _detect_document_type(self, source: str) -> DocumentType:
        """Detect document type (PDF or Website/HTML)."""
        if self._is_url(source):
            # URL: decide based on extension
            path = urlparse(source).path
            ext = Path(path).suffix.lower().lstrip('.')
            if ext == "pdf":
                return DocumentType.PDF
            # All other URLs are treated as website content
            return DocumentType.HTML

        # Local file: only support PDF
        file_path = Path(source)
        extension = file_path.suffix.lower().lstrip('.')

        if extension == "pdf":
            return DocumentType.PDF
        else:
            raise ValueError(f"Unsupported local file type: {extension}. Only PDF is supported locally.")

    
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
        """CORRECTED: Factory method to create DocumentLoader with specific configuration."""
        strategy = LoadingStrategyFactory.create_strategy(document_type, loader_config)
        return cls(strategy)

# if __name__ == "__main__":
#     document_loader = DocumentLoader.create_with_config(DocumentType.PDF)
def main():
    """Main function to run all tests."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # --- Test Setup ---
    # 📝 IMPORTANT: Replace this with the actual path to your PDF file.
    YOUR_PDF_PATH = "lec06-slides.pdf"

    PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    HTML_URL = "https://en.wikipedia.org/wiki/Liverpool_F.C."

    # --- Pre-flight check for the local PDF file ---
    if not os.path.exists(YOUR_PDF_PATH):
        logger.error(f"File not found: {YOUR_PDF_PATH}")
        logger.error("Please update the 'YOUR_PDF_PATH' variable in the script with a valid path to a PDF file.")
        return

    print("\n" + "="*50)
    print("🚀 STARTING DOCUMENT LOADER TESTS 🚀")
    print("="*50 + "\n")

    try:
        # === TEST 1: Auto-detect and load your LOCAL PDF file ===
        print("\n--- TEST 1: Auto-detecting and loading a local PDF ---")
       
        auto_loader = DocumentLoader.create_with_config(
            document_type=DocumentType.PDF,
            loader_config={"pdf_loader_type": PDFLoaderType.DOCLING}
        )
        documents = auto_loader.load(YOUR_PDF_PATH)
        print(f"📄 Loaded {len(documents)} page(s) from your local PDF.")
        print(f"Content of first page (first 100 chars): '{documents[0].page_content[:100].strip()}...'")

        # # === TEST 2: Auto-detect and load an HTML file from a URL ===
        # print("\n--- TEST 2: Auto-detecting and loading a PDF from URL ---")
       
        # documents = auto_loader.load(PDF_URL)
        # print(f"📄 Loaded {len(documents)} page(s) from PDF URL.")
        # print(documents[0].page_content)
        #print(f"Metadata: {documents[0].metadata}")
        

        # # === TEST 3: Load a PDF from a URL ===
        # print("\n--- TEST 3: Auto-detecting and loading an HTML webpage ---")
        # documents = auto_loader.load(HTML_URL)
        # print(f"📄 Loaded {len(documents)} document(s) from HTML URL.")
        # print(documents[0].page_content)
        
        
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}", exc_info=True)
            
    print("\n" + "="*50)
    print("🎉 DOCUMENT LOADER TESTS COMPLETED 🎉")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
