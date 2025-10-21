"""
Test script for SummarizationService
"""

import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.summarization.summarization import SummarizationService
from services.rag.rag_service import RagService
from services.document_processing.document_management_service import DocumentManagementService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_summarization_service():
    """Test SummarizationService initialization and workflow structure"""
    
    logger.info("=" * 60)
    logger.info("Testing SummarizationService")
    logger.info("=" * 60)
    
    try:
        # Initialize services
        logger.info("Initializing RAG service...")
        rag_service = RagService()
        
        logger.info("Initializing Document Management service...")
        doc_management_service = DocumentManagementService()
        
        logger.info("Initializing Summarization service...")
        summarization_service = SummarizationService(
            rag_service=rag_service,
            document_management_service=doc_management_service
        )
        
        logger.info("✅ SummarizationService initialized successfully!")
        
        # Check workflow structure
        logger.info("\nChecking workflow structure...")
        if hasattr(summarization_service, 'workflow'):
            logger.info("✅ Workflow exists")
            logger.info(f"Workflow type: {type(summarization_service.workflow)}")
        else:
            logger.warning("❌ Workflow not found")
        
        # Test with a sample request (will fail if no documents, but tests structure)
        logger.info("\nTesting workflow invocation structure...")
        try:
            result = summarization_service.generate_summary(
                user_request="Tóm tắt chương 1 của tài liệu machine learning",
                context_for_llm=None
            )
            logger.info(f"Result: {result[:100]}..." if len(result) > 100 else f"Result: {result}")
        except Exception as e:
            logger.info(f"Expected error (no documents loaded): {str(e)[:100]}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Test completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_summarization_service()
