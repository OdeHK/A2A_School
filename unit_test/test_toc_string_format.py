"""
Test script for the new get_table_of_contents_as_string method
"""

from datetime import datetime
from services.models import TableOfContents, TocSection
from services.document_processing.document_management_service import DocumentManagementService

def test_toc_string_formatting():
    """Test the ToC string formatting functionality"""
    
    # Create sample ToC data
    sections = [
        TocSection(
            section_id="1",
            section_title="Introduction",
            level=1,
            page_number=1,
            children=[
                TocSection(
                    section_id="2",
                    section_title="Overview",
                    parent_section_id="1",
                    level=2,
                    page_number=2
                ),
                TocSection(
                    section_id="3",
                    section_title="Objectives",
                    parent_section_id="1",
                    level=2,
                    page_number=3
                )
            ]
        ),
        TocSection(
            section_id="4",
            section_title="Methodology",
            level=1,
            page_number=5,
            children=[
                TocSection(
                    section_id="5",
                    section_title="Data Collection",
                    parent_section_id="4",
                    level=2,
                    page_number=6
                )
            ]
        )
    ]
    
    toc = TableOfContents(
        document_id="test_doc_123",
        extraction_method="library",
        extraction_date=datetime.now(),
        sections=sections,
        raw_text="Sample raw text..."
    )
    
    # Create service instance
    service = DocumentManagementService()
    
    # Test the formatting method
    formatted_string = service._format_toc_as_string(toc)
    
    print("Formatted Table of Contents:")
    print("=" * 60)
    print(formatted_string)
    print("=" * 60)

if __name__ == "__main__":
    test_toc_string_formatting()