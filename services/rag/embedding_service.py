from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import logging

from pydantic import SecretStr
from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from config.constants import ModelConstants
from config.settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingType(str, Enum):
    """Enum defining available embedding types."""
    GOOGLE_GEN_AI = "google_gen_ai"
    NVIDIA = "nvidia"
    HUGGINGFACE = "huggingface"  


# === STRATEGY PATTERN IMPLEMENTATION ===

class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies."""
    
    @property
    @abstractmethod
    def embedding_type(self) -> str:
        """Return the type name of this embedding."""
        pass
    
    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate if the strategy is properly configured."""
        pass

    @property
    @abstractmethod
    def embedding_instance(self) -> Embeddings:
        """Return the embedding instance."""
        pass


class GoogleGenAIEmbeddingStrategy(EmbeddingStrategy):
    """Strategy for Google Generative AI embeddings."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs):
        """
        Initialize Google GenAI embedding strategy.
        
        Args:
            api_key (Optional[str]): Google API key. If None, uses from settings.
            model (Optional[str]): Model name. If None, uses from constants.
            **kwargs: Additional parameters for the embedding model.
        """
        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        self.model = model or ModelConstants.EMBEDDING_MODELS["google_gen_ai"]
        self.task_type = kwargs.get("task_type", "retrieval_document")

        self.google_embedding = self.create_embedding()
        logger.info(f"Initialized GoogleGenAI embedding strategy with model: {self.model}")
    
    @property
    def embedding_type(self) -> str:
        return EmbeddingType.GOOGLE_GEN_AI
    
    def validate_configuration(self) -> bool:
        """Validate Google GenAI configuration."""
        if not self.api_key:
            logger.warning("Google API key not provided")
            return False
        if not self.model:
            logger.error("Google GenAI model not specified")
            return False
        return True
    
    def create_embedding(self) -> Embeddings:
        """Create Google Generative AI embedding instance."""
        if not self.validate_configuration():
            raise ValueError("Google GenAI embedding configuration is invalid")
        
        try:
            return GoogleGenerativeAIEmbeddings(
                model=self.model,
                task_type=self.task_type,
                google_api_key=SecretStr(self.api_key) if self.api_key else None
            )
        except Exception as e:
            logger.error(f"Failed to create Google GenAI embedding: {e}")
            raise ValueError(f"Failed to initialize Google GenAI embedding: {e}")
    
    @property
    def embedding_instance(self) -> Embeddings:
        """Return the embedding instance."""
        return self.google_embedding

class NvidiaEmbeddingStrategy(EmbeddingStrategy):
    """Strategy for NVIDIA embeddings."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs):
        """
        Initialize NVIDIA embedding strategy.
        
        Args:
            api_key (Optional[str]): NVIDIA API key. If None, uses from settings.
            model (Optional[str]): Model name. If None, uses from constants.
            **kwargs: Additional parameters for the embedding model.
        """
        settings = get_settings()
        self.api_key = api_key or settings.nvidia_api_key
        self.model = model or ModelConstants.EMBEDDING_MODELS["nvidia"]
        self.dimension = kwargs.get("dimension", 1024)
        self.nvidia_embedding = self.create_embedding()
        logger.info(f"Initialized NVIDIA embedding strategy with model: {self.model}")
    
    @property
    def embedding_type(self) -> str:
        return EmbeddingType.NVIDIA
    
    def validate_configuration(self) -> bool:
        """Validate NVIDIA configuration."""
        if not self.api_key:
            logger.warning("NVIDIA API key not provided")
            return False
        if not self.model:
            logger.error("NVIDIA model not specified")
            return False
        return True
    
    def create_embedding(self) -> Embeddings:
        """Create NVIDIA embedding instance."""
        if not self.validate_configuration():
            raise ValueError("NVIDIA embedding configuration is invalid")
        
        try:
            return NVIDIAEmbeddings(
                model=self.model,
                nvidia_api_key=SecretStr(self.api_key) if self.api_key else None,
                dimension=self.dimension
            )
        except Exception as e:
            logger.error(f"Failed to create NVIDIA embedding: {e}")
            raise ValueError(f"Failed to initialize NVIDIA embedding: {e}")
        
    @property
    def embedding_instance(self) -> Embeddings:
        """Return the embedding instance."""
        return self.nvidia_embedding

class HuggingFaceStrategy(EmbeddingStrategy):
    def __init__(self, model: Optional[str] = None, **kwargs):
        """
        Initialize HuggingFace embedding strategy.
        
        Args:
            model (Optional[str]): Model name. If None, uses default multilingual model.
            **kwargs: Additional parameters for the embedding model.
        """
        self.model = model or ModelConstants.EMBEDDING_MODELS["huggingface"]
        self.cache_folder = kwargs.get("cache_folder", ModelConstants.HUGGINGFACE_CACHE_DIR)
        self.huggingface_embedding = self.create_embedding()
        logger.info(f"Initialized HuggingFace embedding strategy with model: {self.model}")

    @property
    def embedding_instance(self) -> Embeddings:
        return self.huggingface_embedding

    def validate_configuration(self) -> bool:
        """Validate HuggingFace configuration."""
        if not self.model:
            logger.error("HuggingFace model not specified")
            return False
        return True
    
    @property
    def embedding_type(self) -> str:
        return EmbeddingType.HUGGINGFACE
    
    def create_embedding(self) -> Embeddings:
        """Create embedding instance with model from HuggingFace Hub"""
        if not self.validate_configuration():
            raise ValueError("HuggingFace embedding configuration is invalid")
        
        try:
            model_kwargs = {
                "trust_remote_code": True,
                "device": "auto"
            }
            return HuggingFaceEmbeddings(model_name=self.model,
                                         cache_folder=self.cache_folder,
                                         model_kwargs=model_kwargs)

        except Exception as e:
            logger.error(f"Failed to create HuggingFace embedding: {e}")
            raise ValueError(f"Failed to initialize HuggingFace embedding: {e}")




# === FACTORY PATTERN IMPLEMENTATION ===

class EmbeddingStrategyFactory:
    """Factory for creating embedding strategies based on configuration."""
    
    @classmethod
    def create_strategy(
        cls,
        embedding_type: EmbeddingType, 
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingStrategy:
        """
        Create an embedding strategy based on the specified type.
        
        Args:
            embedding_type (EmbeddingType): Type of embedding to create.
            api_key (Optional[str]): API key for the embedding service.
            model (Optional[str]): Model name to use.
            **kwargs: Additional parameters for the embedding strategy.
            
        Returns:
            EmbeddingStrategy: The created embedding strategy.
            
        Raises:
            ValueError: If the embedding type is not supported.
        """
        
        if embedding_type == EmbeddingType.GOOGLE_GEN_AI:
            return GoogleGenAIEmbeddingStrategy(api_key=api_key, model=model, **kwargs)
        elif embedding_type == EmbeddingType.NVIDIA:
            return NvidiaEmbeddingStrategy(api_key=api_key, model=model, **kwargs)
        elif embedding_type == EmbeddingType.HUGGINGFACE:
            return HuggingFaceStrategy(model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available embedding types."""
        return list(EmbeddingType)


# === MAIN EMBEDDING SERVICE CLASS ===

class EmbeddingService:
    """
    Main embedding service class that manages embedding strategies.
    
    This class provides a clean interface for embedding creation while delegating
    the actual embedding logic to specific strategy implementations.
    """

    def __init__(self, embedding_strategy: EmbeddingStrategy, **kwargs):
        """
        Initialize the embedding service with a specific strategy and create Langchain Embedding.
        
        Args:
            embedding_strategy (EmbeddingStrategy): The embedding strategy to use.
            **kwargs: Additional parameters for the embedding strategy.
        """
        self.embedding_strategy = embedding_strategy
        self.kwargs = kwargs

        logger.info(f"Initialized EmbeddingService with strategy: {embedding_strategy} and created Langchain embedding")


    def set_strategy(self, embedding_strategy: EmbeddingType, **kwargs) -> None:
        """
        Change the embedding strategy and recreate the Langchain embedding.
        
        Args:
            embedding_strategy (EmbeddingStrategy): The new embedding strategy to use.
            **kwargs: Additional parameters for the embedding strategy.
        """
        self.embedding_strategy = EmbeddingStrategyFactory.create_strategy(
            embedding_type=embedding_strategy,
            **kwargs
        )

        logger.info(f"Changed embedding strategy to: {embedding_strategy} and recreated Langchain embedding")
    
    @property
    def current_strategy_type(self) -> str:
        """Get the current embedding strategy type."""
        return self.embedding_strategy.embedding_type
    
    @property
    def embedding_instance(self) -> Embeddings:
        """Get the Langchain Embedding instance from the current strategy."""
        return self.embedding_strategy.embedding_instance
    
    @classmethod
    def create_with_type(
        cls, 
        embedding_type: EmbeddingType, 
        **kwargs
    ) -> 'EmbeddingService':
        """
        Create an embedding service with the specified type.
        
        Args:
            embedding_type (EmbeddingType): Type of embedding to create.
            **kwargs: Additional parameters for the embedding strategy.
            
        Returns:
            EmbeddingService: The configured embedding service.
        """
        
        strategy = EmbeddingStrategyFactory.create_strategy(
            embedding_type=embedding_type,
            **kwargs
        )
        return cls(strategy)
    

# === CONVENIENCE FUNCTIONS ===

def create_google_embedding_service(api_key: Optional[str] = None, model: Optional[str] = None, **kwargs) -> EmbeddingService:
    """
    Create an embedding service with Google GenAI strategy.
    
    Args:
        api_key (Optional[str]): Google API key.
        model (Optional[str]): Model name.
        **kwargs: Additional parameters.
        
    Returns:
        EmbeddingService: Configured embedding service.
    """
    return EmbeddingService.create_with_type(
        embedding_type=EmbeddingType.GOOGLE_GEN_AI,
        **kwargs
    )


def create_nvidia_embedding_service(**kwargs) -> EmbeddingService:
    """
    Create an embedding service with NVIDIA strategy.
    
    Args:
        api_key (Optional[str]): NVIDIA API key.
        model (Optional[str]): Model name.
        **kwargs: Additional parameters.
        
    Returns:
        EmbeddingService: Configured embedding service.
    """
    return EmbeddingService.create_with_type(
        embedding_type=EmbeddingType.NVIDIA,
        **kwargs
    )


# === EXAMPLE USAGE ===

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # # Example 1: Using factory to create different embedding services
    # try:
    #     google_service = create_google_embedding_service()
    #     print(f"Created Google embedding service")
        
    #     nvidia_service = create_nvidia_embedding_service()
    #     print(f"Created NVIDIA embedding service")

    # except ValueError as e:
    #     print(f"Error creating embedding services: {e}")
    
    # # Example 2: Creating service with specific configuration
    # try:
    #     custom_service = EmbeddingService.create_with_type(
    #         embedding_type=EmbeddingType.GOOGLE_GEN_AI
    #     )
    #     print(f"Created custom service")
    # except ValueError as e:
    #     print(f"Error creating custom service: {e}")
    
    # # Example 3: Switching strategies
    # service = create_google_embedding_service()
    # print(f"Initial strategy: {service.current_strategy_type}")
    
    # # Switch to NVIDIA strategy
    # service.set_strategy(
    #     embedding_strategy=EmbeddingType.NVIDIA
    # )
    # print(f"After switch: {service.current_strategy_type}")
    
    # print("Embedding service implementation completed successfully!")

    #Example 4: Creating HuggingFace embedding service
    try:
        huggingface_service = EmbeddingService.create_with_type(
            embedding_type=EmbeddingType.HUGGINGFACE
        )
        print(f"Created HuggingFace embedding service with model: {huggingface_service.embedding_instance}")
    except ValueError as e:
        print(f"Error creating HuggingFace embedding service: {e}")
