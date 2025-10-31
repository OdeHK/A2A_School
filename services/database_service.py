from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.server_api import ServerApi

import logging
import dns.resolver
from typing import Dict, Any, Optional, List
from datetime import datetime
from services.models import (
        DocumentMetadata,
        QuizQuestionOutput
)

from config.settings import get_settings
from services.models import TableOfContents, TableOfContentsSection

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Database service for managing user documents and related data.
    Handles MongoDB operations for document storage, metadata, and user libraries.
    """
    
    def __init__(self) -> None:
        """
        Initialize database service and establish connection.
        
        Raises:
            ConnectionFailure: If database connection fails
        """
        # Load settings from environment
        self.settings = get_settings()
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.users_collection: Optional[Collection] = None
        self.documents_collection: Optional[Collection] = None
        
        # Establish initial connection
        self._connect()
    
    def _connect(self) -> None:
        """
        Establish connection to MongoDB database.
        
        Raises:
            ConnectionFailure: If database connection fails
        """
        try:
            # Configure DNS resolver
            dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
            dns.resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']  

            # Get MongoDB URI from settings
            uri = self.settings.mongodb_uri
            database_name = self.settings.mongodb_database_name
            
            # Create a new client and connect to the server
            self.client = MongoClient(uri, server_api=ServerApi('1'))
            
            # Test connection
            self.client.admin.command('ismaster')

            self.db = self.client.get_database(database_name)
            self.users_collection = self.db.get_collection("users")
            self.documents_collection = self.db.get_collection("documents")
            self.quizset_collection = self.db.get_collection("quizsets")

            logger.info("Database connection established successfully")
            
        except ConnectionFailure as e:
            logger.error(f"Database connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while connecting to the database: {e}")
            raise
    
    def _ensure_connection(self) -> None:
        """
        Ensure database connection is active. Reconnect if necessary.
        
        Raises:
            ConnectionFailure: If reconnection fails
        """
        try:
            if self.client is None:
                logger.warning("No database client found, attempting to reconnect...")
                self._connect()
                return
            
            # Test if connection is alive
            self.client.admin.command('ismaster')
            
        except Exception as e:
            logger.warning(f"Database connection lost: {e}. Attempting to reconnect...")
            try:
                self._connect()
                logger.info("Database reconnection successful")
            except Exception as reconnect_error:
                logger.error(f"Database reconnection failed: {reconnect_error}")
                raise ConnectionFailure(f"Failed to reconnect to database: {reconnect_error}")
        
        # Ensure collections are initialized
        if self.documents_collection is None or self.users_collection is None:
            raise ConnectionFailure("Collections not initialized properly")

    def health_check(self) -> bool:
        """
        Check if database connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            self._ensure_connection()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


    def save_document_metadata(self, username: str, document_id: str, metadata: DocumentMetadata) -> None:
        """
        Save document metadata to database.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            metadata: DocumentMetadata object to save
        """
        try:
            self._ensure_connection()
            assert self.documents_collection is not None
            
            # Convert metadata to dict and add username
            metadata_dict = metadata.model_dump(mode='json') #TODO: check DocumentMetadata
            metadata_dict['username'] = username
            metadata_dict['last_updated'] = datetime.now()
            
            result = self.documents_collection.update_one(
                filter={"document_id": document_id, "username": username},
                update={"$set": metadata_dict},
                upsert=True
            )
            logger.info(f"Document data saved/updated for document_id: {document_id}, username: {username}")
            logger.debug(f"Update result: {result.raw_result}")
            
        except Exception as e:
            logger.error(f"Error saving document data: {e}")
            raise
        

    def get_document_metadata(self, username: str, document_id: str) -> Optional[DocumentMetadata]:
        """
        Retrieve document metadata from database.
        
        Args:
            username: User identifier
            document_id: Document identifier
            P
        Returns:
            DocumentMetadata object or None if not found
        """
        try:
            self._ensure_connection()
            assert self.documents_collection is not None
            
            # Fetch only fields defined in DocumentMetadata
            result = self.documents_collection.find_one(
                filter={"document_id": document_id, "username": username},
                projection={"_id": 0, **{field: 1 for field in DocumentMetadata.model_fields.keys()}}
            )
            
            if result:
                # Remove MongoDB _id field and convert to DocumentMetadata
                result.pop('username', None)
                return DocumentMetadata(**result)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document metadata: {e}")
            return None

    def save_content_data(self, username: str, document_id: str, content_data: Dict[str, Any]) -> None:
        """
        Save content data from TOC extractor to database.
        
        Args:
            username: User identifier
            document_id: Document identifier
            content_data: Content data dict from TOCExtractionResult
        """
        try:
            self._ensure_connection()
            assert self.documents_collection is not None
            
            result = self.documents_collection.update_one(
                filter={"document_id": document_id, "username": username},
                update={"$set": {"short_content": content_data}}, #TODO: check content_data structure
                upsert=True
            )
            logger.info(f"Content data saved for document_id: {document_id}, username: {username}")
            logger.debug(f"Update result: {result.raw_result}")
            
        except Exception as e:
            logger.error(f"Error saving content data: {e}")

    def get_content_data(self, username: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve content data by document ID.
        
        Args:
            username: User identifier
            document_id: Document identifier
            
        Returns:
            Content data dict or None if not found
        """
        try:
            self._ensure_connection()
            assert self.documents_collection is not None
            
            result = self.documents_collection.find_one(
                filter={"document_id": document_id, "username": username},
                projection={"short_content": 1, "_id": 0}
            )

            if result and "short_content" in result:
                return result["short_content"]
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving content data: {e}")
            return None

    def save_toc_structure_data(self, username: str, document_id: str, toc_structure: Dict[str, Any]) -> None:
        """
        Save TOC structure data to database.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            toc_structure: TOC structure dict from TOCExtractionResult
        """
        try:
            self._ensure_connection()
            assert self.documents_collection is not None
            
            result = self.documents_collection.update_one(
                filter={"document_id": document_id, "username": username},
                update={"$set": {"table_of_contents": toc_structure}},
                upsert=True
            )
            logger.info(f"TOC structure data saved for document_id: {document_id}, username: {username}")
            logger.debug(f"Update result: {result.raw_result}")
            
        except Exception as e:
            logger.error(f"Error saving TOC structure data: {e}")

    def get_toc_structure_data(self, username: str, document_id: str, repeat_toc: bool = True) -> Optional[List[TableOfContentsSection]]:
        """
        Retrieve TOC structure data by document ID.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            repeat_toc: If True, return a section name "full_document" representing the entire document at the end of the TOC list.
            
        Returns:
            TOC structure list or None if not found
        """
        try:
            self._ensure_connection()
            assert self.documents_collection is not None
            
            result = self.documents_collection.find_one(
                filter={"document_id": document_id, "username": username},
                projection={"table_of_contents": 1, "_id": 0}
            )

            if result and "table_of_contents" in result:
                toc_data = result["table_of_contents"]["sections"]

                if not repeat_toc:
                    # Remove "full_document" section if exists
                    toc_data = [section for section in toc_data if section.get("section_id") != "full_document"]
                return [TableOfContentsSection(**section) for section in toc_data]
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving TOC structure data: {e}")
            return None

    def list_user_documents(self, username: str) -> List[DocumentMetadata]:
        """
        List all documents for a specific user.
        
        Args:
            username: User identifier
            
        Returns:
            List of DocumentMetadata objects
        """
        try:
            self._ensure_connection()
            assert self.documents_collection is not None
            
            cursor = self.documents_collection.find(
                filter={"username": username},
                projection={"_id": 0, "username": 0}
            )
            
            documents = []
            for doc_data in cursor:
                try:
                    documents.append(DocumentMetadata(**doc_data))
                except Exception as e:
                    logger.warning(f"Error parsing document metadata: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing user documents: {e}")
            return []

    def check_document_exists(self, username: str, document_id: str) -> bool:
        """
        Check if a document exists for a specific user.
        
        Args:
            username: User identifier
            document_id: Document identifier
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            self._ensure_connection()
            assert self.documents_collection is not None
            
            result = self.documents_collection.find_one(
                filter={"document_id": document_id, "username": username},
                projection={"_id": 1}
            )
            return result is not None
            
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False
        
    def save_quizset(self, username: str, quizset: QuizQuestionOutput) -> bool: 
        """
        Save quiz set to database after generation by Agent.
        
        Args:
            username: User identifier
            quizset: QuizQuestionOutput object containing generated quiz questions

        Returns:
            None
        """
        try:
            # Write data to collection "quizsets", if document for username exists, update it
            self._ensure_connection()
            self.quizset_collection.update_one(
                filter={"username": username},
                update={"$set": quizset.model_dump(mode='json')},
                upsert=True
            )
            logger.info(f"Quiz set saved for username: {username}")
            # Return the result of the write operation
            return True
        except Exception as e:
            logger.error(f"Error saving quiz set: {e}")
            return False

    def get_quizset(self, username: str) -> Optional[QuizQuestionOutput]:
        """
        Retrieve quiz set for a specific user.
        
        Args:
            username: User identifier

        Returns:
            QuizQuestionOutput object or None if not found
        """

        try:
            self._ensure_connection()
            result = self.quizset_collection.find_one(
                filter={"username": username},
                projection={"_id": 0, "username": 0}
            )
            if result:
                return QuizQuestionOutput(**result)
            return None
        except Exception as e:
            logger.error(f"Error retrieving quiz set: {e}")
            return None

    def authenticate_user(self, username: str, password: str) -> bool:
        """
        Authenticate user credentials against the database.
        
        Args:
            username: Username to authenticate
            password: Password to verify
            
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            self._ensure_connection()
            assert self.users_collection is not None
            
            user = self.users_collection.find_one({
                "username": username,
                "password": password
            })
            
            if user:
                logger.info(f"User authentication successful: {username}")
                return True
            else:
                logger.warning(f"User authentication failed: {username}")
                return False
                
        except Exception as e:
            logger.error(f"Error during user authentication: {e}")
            return False
    
    def close_connection(self) -> None:
        """Close the database connection."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                logger.info("Database connection closed.")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
            
    def remove_quizsets(self) -> None:
        try:
            self._ensure_connection()
            self.quizset_collection.delete_many({})
            logger.info("All quizsets have been removed from the database.")
        except Exception as e:
            logger.error(f"Error removing quizsets: {e}")
