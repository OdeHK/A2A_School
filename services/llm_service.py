from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from typing import Optional

from config.constants import ModelConstants
from config.settings import get_settings

class LLMService:
    """Quản lý việc khởi tạo và sử dụng LLM thông qua LangChain"""
    
    def __init__(
        self, 
        llm_type: str = ModelConstants.DEFAULT_LLM_PROVIDER,
        model_name: str = ModelConstants.DEFAULT_MODELS[ModelConstants.DEFAULT_LLM_PROVIDER],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 100000
    ):
        """
        Initialize LLM Service
        
        Args:
            llm_type: Provider type (nvidia, google_gen_ai), nếu None dùng default
            model_name: Model name to use, nếu None dùng default cho provider
            temperature: Temperature parameter for generation
            top_p: Top P parameter for generation
            max_completion_tokens: Maximum completion tokens
        """        
        self.llm_type = llm_type
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        
        # Create llm service
        self.llm = self.get_llm(
            llm_type=llm_type,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens
        )
        

    
    def get_llm(
        self, 
        llm_type: str, 
        model_name: str,
        temperature: float,
        top_p: float,
        max_completion_tokens: int
    ) -> BaseChatModel:
        """Factory method để tạo LLM phù hợp"""
        settings = get_settings()

        if llm_type.lower() == "nvidia":
            if settings.nvidia_api_key is None:
                raise ValueError("NVIDIA API KEY is not set")
            return get_nvidia_llm(
                api_key=settings.nvidia_api_key,
                model_name=model_name or ModelConstants.DEFAULT_MODELS['nvidia'],
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_completion_tokens
            )

        elif llm_type.lower() == "google_gen_ai":
            if settings.google_api_key is None:
                raise ValueError("GOOGLE GENAI API KEY is not set")
            return get_google_genai_llm(
                api_key=settings.google_api_key,
                model_name=model_name or ModelConstants.DEFAULT_MODELS['google_gen_ai'],
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_completion_tokens
            )

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


def get_nvidia_llm(
    api_key: str, 
    model_name: str = ModelConstants.DEFAULT_MODELS['nvidia'],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_completion_tokens: Optional[int] = None
) -> BaseChatModel:
    """Khởi tạo NVIDIA LLM thông qua LangChain"""
    return ChatNVIDIA(
        model=model_name,
        nvidia_api_key=api_key,
        temperature=temperature if temperature is not None else 1.0,
        top_p=top_p if top_p is not None else 1.0,
        streaming=False,
        callbacks=[StreamingStdOutCallbackHandler()],
        max_completion_tokens=max_completion_tokens if max_completion_tokens is not None else 100000
    )
    

def get_google_genai_llm(
    api_key: str,
    model_name: str = ModelConstants.DEFAULT_MODELS['google_gen_ai'],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_completion_tokens: Optional[int] = None
) -> BaseChatModel:
    """Khởi tạo Google LLM thông qua LangChain"""
    params = {
        "model": model_name,
        "api_key": api_key,
        "disable_streaming": False,
        "callbacks": [StreamingStdOutCallbackHandler()],
    }
    
    # Chỉ thêm các tham số nếu chúng được cung cấp
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if max_completion_tokens is not None:
        params["max_output_tokens"] = max_completion_tokens
    
    return ChatGoogleGenerativeAI(**params)
