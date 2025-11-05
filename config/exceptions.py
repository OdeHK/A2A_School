"""
Custom Exception Hierarchy for A2A_School
Provides structured error handling across the application
"""


class A2ASchoolException(Exception):
    """Base exception for all A2A_School errors"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AuthenticationError(A2ASchoolException):
    """Raised when authentication fails"""
    pass


class RateLimitError(A2ASchoolException):
    """Raised when rate limit is exceeded"""
    pass


class DocumentProcessingError(A2ASchoolException):
    """Raised when document processing fails"""
    pass


class RAGError(A2ASchoolException):
    """Raised when RAG operations fail"""
    pass


class QuizGenerationError(A2ASchoolException):
    """Raised when quiz generation fails"""
    pass


class ValidationError(A2ASchoolException):
    """Raised when input validation fails"""
    pass


class DatabaseError(A2ASchoolException):
    """Raised when database operations fail"""
    pass


class LLMError(A2ASchoolException):
    """Raised when LLM service fails"""
    pass
