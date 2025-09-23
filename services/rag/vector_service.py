from langchain_chroma.vectorstores import Chroma
from langchain.schema.document import Document
from typing import List, Optional
import logging

from config.constants import DatabaseConstants
from .embedding_service import EmbeddingService, create_google_embedding_service, create_nvidia_embedding_service
from .embedding_service import EmbeddingType

# Configure logging
logger = logging.getLogger(__name__)

class VectorService:
    """Quản lý vector store dùng LangChain với embedding service"""
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None, 
                 embedding_type: EmbeddingType = EmbeddingType.GOOGLE_GEN_AI, 
                 persist_directory: Optional[str] = None):
        """
        Initialize VectorService.
        
        Args:
            embedding_service (Optional[EmbeddingService]): Pre-configured embedding service.
            embedding_type (str): Type of embedding if embedding_service is None.
            persist_directory (Optional[str]): Directory to persist vector store.
        """
        self.persist_directory = persist_directory or DatabaseConstants.VECTOR_STORE_CONFIGS["chroma"]["persist_directory"]
        
        # Use provided embedding service or create one
        if embedding_service:
            self.embedding_service = embedding_service
            logger.info(f"Using provided embedding service: {embedding_service.current_strategy_type}")
        else:
            self.embedding_service = EmbeddingService.create_with_type(embedding_type)
            logger.info(f"Created new embedding service: {embedding_type}")
        
        self.vectorstore = None
    
    @property
    def embedding(self):
        """Lấy embedding instance từ embedding service"""
        return self.embedding_service.embedding_instance

    def init_vectorstore(self, documents: Optional[List[Document]] = None):
        """Khởi tạo hoặc load vector store"""
        if documents:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]):
        """Thêm tài liệu vào vector store"""
        if not self.vectorstore:
            return self.init_vectorstore(documents)
        self.vectorstore.add_documents(documents)
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 4):
        """Tìm kiếm tài liệu tương tự"""
        if not self.vectorstore:
            raise ValueError("Vector store chưa được khởi tạo")
        return self.vectorstore.similarity_search(query, k=k)
    
if __name__ == "__main__":
    # Example 1: Using default embedding service
    vector_store = VectorService(embedding_type=EmbeddingType.GOOGLE_GEN_AI, persist_directory="./test_vector_service")
    vector_store.add_documents(documents=[Document(page_content="hello", metadata={"author": "khiem"})])
    
    # Example 2: Using custom embedding service
    custom_embedding_service = create_nvidia_embedding_service()
    vector_store_nvidia = VectorService(embedding_service=custom_embedding_service)
    
    print(f"Created vector service with embedding: {vector_store.embedding_service.current_strategy_type}")
    print(f"NVIDIA vector service with embedding: {vector_store_nvidia.embedding_service.current_strategy_type}")