"""
Session-based Document Repository for managing documents, metadata, and ToC storage.
Uses JSON files for simple, session-scoped storage.
"""

import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from services.models import (
    DocumentMetadata, 
    TableOfContents, 
    SessionMetadata, 
    ProcessingStatus
)

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Session-based document repository using JSON storage.
    Each session gets its own folder with organized subdirectories.
    """
    
    def __init__(self, base_documents_dir: str = "documents"):
        """
        Initialize repository with base documents directory.
        
        Args:
            base_documents_dir: Base directory for all document storage
        """
        self.base_dir = Path(base_documents_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.temp_dir = self.base_dir / "temp"
        
        # Create base directories if they don't exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session_id: Optional[str] = None
        self.current_session_dir: Optional[Path] = None
    
    def create_new_session(self) -> str:
        """
        Create a new session with unique ID and directory structure.
        
        Returns:
            Session ID
        """
        logger.info("Creating new session")
        
        # Generate session ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Create session directory structure
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (session_dir / "raw_files").mkdir(exist_ok=True)
        (session_dir / "metadata").mkdir(exist_ok=True)
        (session_dir / "toc").mkdir(exist_ok=True)
        
        # Create session metadata
        session_metadata = SessionMetadata(
            session_id=session_id,
            created_date=datetime.now(),
            last_accessed=datetime.now(),
            documents=[],
            vector_store_path=str(session_dir / "vector_store")
        )
        
        # Save session metadata
        session_file = session_dir / "session.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_metadata.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
        
        # Set current session
        self.current_session_id = session_id
        self.current_session_dir = session_dir
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def load_session(self, session_id: str) -> bool:
        """
        Load an existing session.
        
        Args:
            session_id: ID of session to load
            
        Returns:
            True if session loaded successfully, False otherwise
        """
        session_dir = self.sessions_dir / session_id
        session_file = session_dir / "session.json"
        
        if not session_file.exists():
            logger.warning(f"Session {session_id} not found")
            return False
        
        try:
            # Load and update session metadata
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session_metadata = SessionMetadata(**session_data)
            session_metadata.last_accessed = datetime.now()
            
            # Save updated metadata
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_metadata.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
            
            self.current_session_id = session_id
            self.current_session_dir = session_dir
            
            logger.info(f"Loaded session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}")
            return False
    
    def get_current_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self.current_session_id
    
    def _ensure_session(self) -> Path:
        """Ensure current session exists, create new one if needed."""
        if self.current_session_dir is None:
            self.create_new_session()
        return self.current_session_dir
    
    def store_uploaded_file(self, source_file_path: str, document_id: Optional[str] = None) -> str:
        """
        Store uploaded file in current session's raw_files directory.
        
        Args:
            source_file_path: Path to the source file
            document_id: Optional document ID, will generate if not provided
            
        Returns:
            Document ID
        """
        session_dir = self._ensure_session()
        source_path = Path(source_file_path)
        
        if document_id is None:
            document_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        # Determine destination file name
        file_extension = source_path.suffix
        dest_filename = f"{document_id}{file_extension}"
        dest_path = session_dir / "raw_files" / dest_filename
        
        # Copy file to session directory
        shutil.copy2(source_path, dest_path)
        
        logger.info(f"Stored file {source_path.name} as {dest_filename} in session {self.current_session_id}")
        return document_id
    
    def save_document_metadata(self, metadata: DocumentMetadata) -> None:
        """
        Save document metadata to JSON file.
        
        Args:
            metadata: Document metadata to save
        """
        session_dir = self._ensure_session()
        metadata_file = session_dir / "metadata" / f"{metadata.document_id}.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
        
        # Update session document list
        self._add_document_to_session(metadata.document_id)
        
        logger.info(f"Saved metadata for document {metadata.document_id}")
    
    def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Get document metadata by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        if self.current_session_dir is None:
            return None
        
        metadata_file = self.current_session_dir / "metadata" / f"{document_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)
            
            return DocumentMetadata(**metadata_data)
        
        except Exception as e:
            logger.error(f"Error loading metadata for {document_id}: {str(e)}")
            return None
    
    def save_table_of_contents(self, document_id: str, toc: TableOfContents) -> None:
        """
        Save table of contents to JSON file.
        
        Args:
            document_id: Document ID
            toc: Table of contents to save
        """
        session_dir = self._ensure_session()
        toc_file = session_dir / "toc" / f"{document_id}_toc.json"
        
        with open(toc_file, 'w', encoding='utf-8') as f:
            json.dump(toc.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved ToC for document {document_id}")
    
    def get_table_of_contents(self, document_id: str) -> Optional[TableOfContents]:
        """
        Get table of contents by document ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Table of contents or None if not found
        """
        if self.current_session_dir is None:
            return None
        
        toc_file = self.current_session_dir / "toc" / f"{document_id}_toc.json"
        
        if not toc_file.exists():
            return None
        
        try:
            with open(toc_file, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)
            
            return TableOfContents(**toc_data)
        
        except Exception as e:
            logger.error(f"Error loading ToC for {document_id}: {str(e)}")
            return None
    
    def get_document_file_path(self, document_id: str) -> Optional[Path]:
        """
        Get file path for document in current session.
        
        Args:
            document_id: Document ID
            
        Returns:
            Path to document file or None if not found
        """
        if self.current_session_dir is None:
            return None
        
        raw_files_dir = self.current_session_dir / "raw_files"
        
        # Look for file with document_id prefix
        for file_path in raw_files_dir.glob(f"{document_id}.*"):
            return file_path
        
        return None
    
    def list_session_documents(self) -> List[DocumentMetadata]:
        """
        List all documents in current session.
        
        Returns:
            List of document metadata
        """
        if self.current_session_dir is None:
            return []
        
        documents = []
        metadata_dir = self.current_session_dir / "metadata"
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_data = json.load(f)
                
                documents.append(DocumentMetadata(**metadata_data))
                
            except Exception as e:
                logger.error(f"Error loading metadata from {metadata_file}: {str(e)}")
        
        return documents
    
    def get_vector_store_path(self) -> Optional[str]:
        """Get vector store path for current session."""
        if self.current_session_dir is None:
            return None
        
        return str(self.current_session_dir / "vector_store")
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir.exists():
            for item in self.temp_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    logger.warning(f"Error cleaning up {item}: {str(e)}")
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its data.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        session_dir = self.sessions_dir / session_id
        
        if not session_dir.exists():
            logger.warning(f"Session {session_id} not found")
            return False
        
        try:
            shutil.rmtree(session_dir)
            
            # Clear current session if it was deleted
            if self.current_session_id == session_id:
                self.current_session_id = None
                self.current_session_dir = None
            
            logger.info(f"Deleted session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {str(e)}")
            return False
    
    def _add_document_to_session(self, document_id: str) -> None:
        """Add document ID to session metadata."""
        if self.current_session_dir is None:
            return
        
        session_file = self.current_session_dir / "session.json"
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session_metadata = SessionMetadata(**session_data)
            
            if document_id not in session_metadata.documents:
                session_metadata.documents.append(document_id)
                session_metadata.last_accessed = datetime.now()
                
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(session_metadata.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.error(f"Error updating session with document {document_id}: {str(e)}")