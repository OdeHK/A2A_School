from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from typing import Optional

from config import ModelConstants
from config.settings import get_settings

class LLMService:
    """Quản lý việc khởi tạo và sử dụng LLM thông qua LangChain"""
    
    def __init__(self, llm_type: Optional[str] = None):
        """
        Initialize LLM Service
        
        Args:
            llm_type: Provider type (nvidia, google_gen_ai), nếu None dùng default
        """
        self.llm_type = llm_type or ModelConstants.DEFAULT_LLM_PROVIDER
        
        # Create llm service
        self.llm = self.get_llm(llm_type=llm_type)
        

    
    def get_llm(self, llm_type: Optional[str], model_name: Optional[str] = None) -> BaseChatModel:
        """Factory method để tạo LLM phù hợp"""
        settings = get_settings()

        if llm_type is None:
            llm_type = ModelConstants.DEFAULT_LLM_PROVIDER

        if llm_type.lower() == "nvidia":
            if settings.nvidia_api_key is None:
                raise ValueError("NVIDA API KEY is not set")
            else:
                return get_nvidia_llm(api_key=settings.nvidia_api_key, 
                                             model_name=model_name or ModelConstants.DEFAULT_MODELS['nvidia'])
        elif llm_type.lower() == "google_gen_ai":
            if settings.google_api_key is None:
                raise ValueError("GOOGLE GENAI API KEY is not set")
            else:
                return get_google_genai_llm(api_key=settings.google_api_key, 
                                                   model_name=model_name or ModelConstants.DEFAULT_MODELS['google_gen_ai'])
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        
    def invoke(self, prompt: PromptValue) -> BaseMessage:
        """
        Invoke LLM với current provider
        
        Args:
            prompt: Prompt để gửi đến LLM
            
        Returns:
            BaseMessage response từ LLM
        """
        return self.llm.invoke(prompt)


def get_nvidia_llm(api_key: str, model_name: str = ModelConstants.DEFAULT_MODELS['nvidia']) -> BaseChatModel:
    """Khởi tạo NVIDIA LLM thông qua LangChain"""
    return ChatNVIDIA(
        model=model_name,
        nvidia_api_key=api_key,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        max_completion_tokens=100000
    )
    

def get_google_genai_llm(api_key: str, model_name: str = ModelConstants.DEFAULT_MODELS['google_gen_ai']) -> BaseChatModel:
    """Khởi tạo Google LLM thông qua LangChain"""
    return ChatGoogleGenerativeAI(
        model_name=model_name,
        google_api_key=api_key,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )