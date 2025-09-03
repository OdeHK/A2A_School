from langchain_chroma.vectorstores import Chroma
from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.schema.document import Document
from typing import List, Optional
from pydantic import SecretStr

from config.constants import ModelConstants, DatabaseConstants
from config.settings import get_settings

class VectorService:
    """Quản lý vector store dùng LangChain"""
    
    def __init__(self, embedding_type: str = "google_gen_ai", persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or DatabaseConstants.VECTOR_STORE_CONFIGS["chroma"]["persist_directory"]
        self.embedding = self._get_embedding(embedding_type)
        self.vectorstore = None
    
    def _get_embedding(self, embedding_type: str):
        """Khởi tạo embedding model phù hợp"""
        settings = get_settings()

        if embedding_type.lower() == "google_gen_ai":
            from pydantic import SecretStr
            return GoogleGenerativeAIEmbeddings(
                model=ModelConstants.EMBEDDING_MODELS["google_gen_ai"],
                task_type='retrieval_document',
                google_api_key=SecretStr(settings.google_api_key) if settings.google_api_key is not None else None
            )
        elif embedding_type.lower() == "nvidia":
            return NVIDIAEmbeddings(
                model=ModelConstants.EMBEDDING_MODELS["nvidia"],
                nvidia_api_key=SecretStr(settings.nvidia_api_key) if settings.nvidia_api_key is not None else None
            )
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
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
    vector_store = VectorService(embedding_type="google_gen_ai", persist_directory="./test_vector_service")
    vector_store.add_documents(documents=[Document(page_content="hello", meta_data={"author": "khiem"})])