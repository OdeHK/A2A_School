"""
Test script for DocumentManagementService with focus on TOC extractor integration.
Tests the complete document processing workflow including ToC extraction.
"""

import unittest
import logging
import tempfile
import os
import json
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import classes to test
from services.document_processing.document_management_service import DocumentManagementService
from services.document_processing.document_repository import DocumentRepository
from services.document_processing.toc_extractor import TOCExtractor, TOCExtractionResult, TOCSection, ContentItem, TOCStructure, ContentData
from services.document_processing.document_loader import DocumentLoader, DocumentType
from services.document_processing.document_chunker import DocumentChunker, ChunkingStrategyType
from services.models import (
    DocumentMetadata, 
    TableOfContents, 
    TocSection,
    ProcessingResult, 
    ProcessingStatus
)
from langchain.schema.document import Document


def run_manual_tests():
    """Manual tests for debugging and development."""
    print("\n=== Manual TOC Extractor Integration Tests ===")
    
    # Test with real file (if available)
    test_file = "D:\\Project\\A2A_School\\example_data\\python_rat_la_co_ban.pdf"
    if os.path.exists(test_file):
        print(f"\nTesting with real file: {test_file}")
        
        try:
            # Create service with real dependencies
            service = DocumentManagementService()
            
            # Process document with TOC extraction
            result = service.process_uploaded_document(
                file_path=test_file,
                extract_toc=True
            )
            print(f"Processing result: {result.status}")
            print(f"Document ID: {result.document_id}")

            # Get content data using service method
            content_data = service.get_content_data(result.document_id)
            print(f"Content data retrieved: {content_data['content']} items")


            
        except Exception as e:
            print(f"Error in manual test: {str(e)}")
    else:
        print(f"Test file not found: {test_file}")


if __name__ == "__main__":
    # Run unit tests
    print("=== Running Unit Tests ===")
    unittest.main(verbosity=2, exit=False)
    
    # Run manual tests for debugging
    run_manual_tests()
