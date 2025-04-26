"""
Vector database module for the RAG System.
Uses ChromaDB to store and retrieve document embeddings.
"""

import os
import logging
from typing import List, Dict, Optional, Any, Union
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database for storing and retrieving document embeddings."""
    
    def __init__(self, 
                 collection_name: str = "rag_documents",
                 persist_directory: str = "./vector_db",
                 embedding_function: Optional[Any] = None):
        """
        Initialize the vector store with ChromaDB.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory where ChromaDB will store data
            embedding_function: Optional custom embedding function
        """
        logger.info(f"Initializing vector store with collection: {collection_name}")
        
        # Ensure the persistence directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        try:
            # Initialize ChromaDB client with persistence - using new API format
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
            
            # Use default embedding function if none provided
            self.embedding_function = embedding_function or embedding_functions.DefaultEmbeddingFunction()
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            logger.info(f"Vector store initialized successfully with collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def add_documents(self, 
                      documents: List[str], 
                      metadatas: Optional[List[Dict[str, Any]]] = None,
                      ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries for each document
            ids: Optional list of IDs for each document
            
        Returns:
            List of document IDs
        """
        try:
            if ids is None:
                # Generate unique IDs if none provided
                from uuid import uuid4
                ids = [str(uuid4()) for _ in range(len(documents))]
            
            if metadatas is None:
                # Create empty metadata if none provided
                metadatas = [{} for _ in range(len(documents))]
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, 
               query: str, 
               n_results: int = 3,
               where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar documents based on a query.
        
        Args:
            query: Query text
            n_results: Number of results to return
            where: Optional filter criteria
            
        Returns:
            Dictionary containing ids, documents, metadatas and distances
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            logger.info(f"Search for '{query}' returned {len(results['ids'][0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "count": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            raise
            
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            None
        """
        try:
            self.collection.delete(
                ids=ids
            )
            logger.info(f"Deleted {len(ids)} documents from vector store")
        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {e}")
            raise
            
    def update_document(self, doc_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update an existing document in the vector store.
        
        Args:
            doc_id: ID of the document to update
            document: New document text
            metadata: Optional new metadata for the document
            
        Returns:
            None
        """
        try:
            # Create empty metadata if none provided
            if metadata is None:
                metadata = {}
                
            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated document {doc_id} in vector store")
        except Exception as e:
            logger.error(f"Error updating document in vector store: {e}")
            raise
            
    def get_all_documents(self) -> Dict[str, Any]:
        """
        Get all documents from the vector store.
        
        Returns:
            Dictionary containing ids, documents, and metadatas
        """
        try:
            # Get collection count
            count = self.collection.count()
            
            if count == 0:
                return {"ids": [], "documents": [], "metadatas": []}
            
            # Get all documents (up to 10000)
            results = self.collection.get(
                limit=10000
            )
            
            logger.info(f"Retrieved {len(results['ids'])} documents from vector store")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving all documents from vector store: {e}")
            raise

# Create a singleton instance for application-wide use
default_vector_store = None

def get_vector_store(collection_name: str = "rag_documents",
                     persist_directory: str = "./vector_db",
                     embedding_function: Optional[Any] = None):
    """
    Get or create the default vector store.
    
    Args:
        collection_name: Optional collection name override
        persist_directory: Optional persistence directory override
        embedding_function: Optional embedding function override
        
    Returns:
        VectorStore instance
    """
    global default_vector_store
    
    if default_vector_store is None:
        logger.info("Creating new vector store instance")
        default_vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
    
    return default_vector_store
