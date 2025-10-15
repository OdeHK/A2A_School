from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import Dict, Any, Optional, List
from datetime import datetime
from services.models import (
        DocumentMetadata,
        TableOfContents,
        TocSection
)
import dns.resolver
from pymongo.server_api import ServerApi
import logging
from config.settings import get_settings

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Database service for managing user documents and related data.
    Handles MongoDB operations for document storage, metadata, and user libraries.
    """
    
    def __init__(self) -> None:
        """
        Initialize database connection and collections.
        
        Raises:
            ConnectionFailure: If database connection fails
        """
        try:
            # Load settings from environment
            settings = get_settings()
            
            # Configure DNS resolver
            dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
            dns.resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']  

            # Get MongoDB URI from settings
            uri = settings.mongodb_uri
            database_name = settings.mongodb_database_name
            
            # Create a new client and connect to the server
            self.client = MongoClient(uri, server_api=ServerApi('1'))
            
            # Test connection
            self.client.admin.command('ismaster')

            self.db = self.client.get_database(database_name)
            self.users_collection = self.db.get_collection("users")
            self.documents_collection = self.db.get_collection("documents")
            
            
            logger.info("Database connection established successfully")
            
        except ConnectionFailure as e:
            logger.error(f"Database connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while connecting to the database: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if database connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            self.client.admin.command('ismaster')
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

    # def save_table_of_content(self, username: str, document_id: str, toc: TableOfContents) -> None:
    #     """
    #     Save table of contents to database.
        
    #     Args:
    #         username: User identifier
    #         document_id: Document identifier  
    #         toc: TableOfContents object to save
    #     """
    #     try:
    #         # Convert TableOfContents to dict for storage
    #         toc_dict = toc.model_dump(mode='json')
    #         result = self.documents_collection.update_one(
    #             filter={"document_id": document_id, "username": username},
    #             update={"$set": {"table_of_contents": toc_dict}},
    #             upsert=True
    #         )
    #         logger.info(f"Table of contents saved for document_id: {document_id}, user_id: {user_id}")
    #         logger.debug(f"Update result: {result.raw_result}")
            
    #     except Exception as e:
    #         logger.error(f"Error saving table of contents: {e}")

    # def get_table_of_content(self, user_id: str, document_id: str) -> Optional[TableOfContents]:
    #     """
    #     Retrieve table of contents from database.
        
    #     Args:
    #         user_id: User identifier
    #         document_id: Document identifier
            
    #     Returns:
    #         TableOfContents object or None if not found
    #     """
    #     try:
    #         result = self.documents_collection.find_one(
    #             filter={"document_id": document_id, "user_id": user_id},
    #             projection={"table_of_contents": 1, "_id": 0}
    #         )
            
    #         if result and "table_of_contents" in result:
    #             return TableOfContents(**result["table_of_contents"])
    #         return None
            
    #     except Exception as e:
    #         logger.error(f"Error retrieving table of contents: {e}")
    #         return None


    def save_content_data(self, username: str, document_id: str, content_data: Dict[str, Any]) -> None:
        """
        Save content data from TOC extractor to database.
        
        Args:
            username: User identifier
            document_id: Document identifier
            content_data: Content data dict from TOCExtractionResult
        """
        try:
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
            result = self.documents_collection.update_one(
                filter={"document_id": document_id, "username": username},
                update={"$set": {"table_of_contents": toc_structure}},
                upsert=True
            )
            logger.info(f"TOC structure data saved for document_id: {document_id}, username: {username}")
            logger.debug(f"Update result: {result.raw_result}")
            
        except Exception as e:
            logger.error(f"Error saving TOC structure data: {e}")

    def get_toc_structure_data(self, username: str, document_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve TOC structure data by document ID.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            TOC structure list or None if not found
        """
        try:
            result = self.documents_collection.find_one(
                filter={"document_id": document_id, "username": username},
                projection={"table_of_contents": 1, "_id": 0}
            )

            if result and "table_of_contents" in result:
                toc_data = result["table_of_contents"]["sections"]
                return toc_data
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving TOC structure data: {e}")
            return None

    def save_document_library(self, username: str, document_library: Dict[str, Dict[str, Any]]) -> None:
        """
        Save complete document library to database.
        
        Args:
            username: User identifier
            document_library: Dictionary with document_name as key and document info as value
        """
        try:
            result = self.documents_collection.update_one(
                filter={"username": username},
                update={
                    "$set": {
                        "document_library": document_library,
                        "last_updated": datetime.now()
                    }
                },
                upsert=True
            )
            logger.info(f"Document library saved for username: {username} with {len(document_library)} documents")
            logger.debug(f"Update result: {result.raw_result}")
            
        except Exception as e:
            logger.error(f"Error saving document library: {e}")

    def get_document_library(self, username: str) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve complete document library for user.
        
        Args:
            username: User identifier
            
        Returns:
            Dictionary with document_name as key and document info as value, or empty dict if not found
        """
        try:
            result = self.documents_collection.find_one(
                filter={"username": username},
                projection={"document_library": 1, "_id": 0}
            )
            
            if result and "document_library" in result:
                return result["document_library"]
            return {}
            
        except Exception as e:
            logger.error(f"Error retrieving document library: {e}")
            return {}

    def add_document_to_library(self, username: str, document_id: str, name: str, title: List[str]) -> None:
        """
        Add or update a document in the user's library.
        
        Args:
            username: User identifier
            document_id: Unique document identifier
            name: Document name
            title: List of document titles/bookmarks
        """
        try:
            # Get existing library
            document_library = self.get_document_library(username)
            
            # Add/update document with name as key
            document_library[name] = {
                'document_id': document_id,
                'name': name,
                'title': title,
                'added_date': datetime.now().isoformat()
            }
            
            # Save updated library
            self.save_document_library(username, document_library)
            logger.info(f"Added document {name} to library for user {username}")
            
        except Exception as e:
            logger.error(f"Error adding document to library: {e}")

    def remove_document_from_library(self, username: str, name: str) -> bool:
        """
        Remove a document from the user's library.
        
        Args:
            username: User identifier
            name: Document name to remove
            
        Returns:
            True if removed, False if not found
        """
        try:
            # Get existing library
            document_library = self.get_document_library(username)
            
            if name in document_library:
                del document_library[name]
                self.save_document_library(username, document_library)
                logger.info(f"Removed document {name} from library for user {username}")
                return True

            logger.warning(f"Document {name} not found in library for user {username}")
            return False
            
        except Exception as e:
            logger.error(f"Error removing document from library: {e}")
            return False

    def get_document_from_library(self, username: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Get specific document from user's library.
        
        Args:
            username: User identifier
            name: Document name
            
        Returns:
            Document information or None if not found
        """
        try:
            document_library = self.get_document_library(username)
            return document_library.get(name)
            
        except Exception as e:
            logger.error(f"Error getting document from library: {e}")
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
            result = self.documents_collection.find_one(
                filter={"document_id": document_id, "username": username},
                projection={"_id": 1}
            )
            return result is not None
            
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False

    # def delete_document(self, user_id: str, document_id: str) -> bool:
    #     """
    #     Delete a document and all its associated data.
        
    #     Args:
    #         user_id: User identifier
    #         document_id: Document identifier
            
    #     Returns:
    #         True if deleted successfully, False otherwise
    #     """
    #     try:
    #         result = self.documents_collection.delete_one(
    #             filter={"document_id": document_id, "user_id": user_id}
    #         )
            
    #         if result.deleted_count > 0:
    #             logger.info(f"Deleted document {document_id} for user {user_id}")
    #             return True
    #         else:
    #             logger.warning(f"Document {document_id} not found for user {user_id}")
    #             return False
                
    #     except Exception as e:
    #         logger.error(f"Error deleting document: {e}")
    #         return False

    # def update_document_processing_status(self, user_id: str, document_id: str, status: str, error_message: Optional[str] = None) -> None:
    #     """
    #     Update document processing status.
        
    #     Args:
    #         user_id: User identifier
    #         document_id: Document identifier
    #         status: New processing status
    #         error_message: Optional error message if status is 'failed'
    #     """
    #     try:
    #         update_data = {
    #             "processing_status": status,
    #             "last_updated": datetime.now()
    #         }
            
    #         if error_message:
    #             update_data["error_message"] = error_message
            
    #         result = self.documents_collection.update_one(
    #             filter={"document_id": document_id, "user_id": user_id},
    #             update={"$set": update_data}
    #         )
            
    #         logger.info(f"Updated processing status for document {document_id} to {status}")
    #         logger.debug(f"Update result: {result.raw_result}")
            
    #     except Exception as e:
    #         logger.error(f"Error updating document processing status: {e}")

    # def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
    #     """
    #     Get statistics for a specific user.
        
    #     Args:
    #         user_id: User identifier
            
    #     Returns:
    #         Dictionary containing user statistics
    #     """
    #     try:
    #         # Count documents by status
    #         pipeline = [
    #             {"$match": {"user_id": user_id}},
    #             {"$group": {
    #                 "_id": "$processing_status",
    #                 "count": {"$sum": 1}
    #             }}
    #         ]
            
    #         status_counts = {}
    #         for result in self.documents_collection.aggregate(pipeline):
    #             status_counts[result["_id"]] = result["count"]
            
    #         # Get total documents
    #         total_documents = sum(status_counts.values())
            
    #         # Get library size
    #         library = self.get_document_library(user_id)
    #         library_size = len(library)
            
    #         return {
    #             "total_documents": total_documents,
    #             "library_size": library_size,
    #             "status_counts": status_counts,
    #             "last_updated": datetime.now().isoformat()
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Error getting user statistics: {e}")
    #         return {}

    # def cleanup_user_data(self, user_id: str) -> bool:
    #     """
    #     Clean up all data for a specific user.
        
    #     Args:
    #         user_id: User identifier
            
    #     Returns:
    #         True if cleanup successful, False otherwise
    #     """
    #     try:
    #         # Delete all documents for user
    #         documents_result = self.documents_collection.delete_many(
    #             filter={"user_id": user_id}
    #         )
            
    #         # Delete user library
    #         library_result = self.users_collection.delete_one(
    #             filter={"user_id": user_id}
    #         )
            
    #         logger.info(f"Cleaned up data for user {user_id}: "
    #                    f"{documents_result.deleted_count} documents, "
    #                    f"{library_result.deleted_count} library entries")
            
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Error cleaning up user data: {e}")
    #         return False
    
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

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close_connection()