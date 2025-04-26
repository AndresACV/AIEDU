"""
RAG (Retrieval-Augmented Generation) pipeline for the AIEDU system.
Combines embedding generation, vector retrieval, and LLM inference.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import time

# Import local modules
from web_app.embeddings import get_embedding_generator
from web_app.vector_store import get_vector_store
from web_app.llm import get_llm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Main RAG pipeline that connects embedding generation, 
    vector database retrieval, and LLM inference.
    """
    
    def __init__(self,
                embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                collection_name: str = "rag_documents",
                persist_directory: str = "./vector_db",
                llm_model_path: str = "web_app/models/vicuna-7b-v1.3.ggmlv3.q4_K_S.bin"):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Name of the embedding model
            collection_name: Name of the vector store collection
            persist_directory: Directory for vector store persistence
            llm_model_path: Path to the quantized LLM model
        """
        logger.info("Initializing RAG pipeline")
        
        try:
            # Initialize components
            self.embedding_generator = get_embedding_generator(embedding_model)
            self.vector_store = get_vector_store(
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            self.llm = get_llm(llm_model_path)
            
            logger.info("RAG pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def add_documents(self, 
                     texts: Union[str, List[str]], 
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the RAG system's vector store.
        
        Args:
            texts: Document text(s) to add
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
            
        Returns:
            List of document IDs
        """
        try:
            # Convert single document to list
            if isinstance(texts, str):
                texts = [texts]
                
            # Handle metadata conversion for single document
            if metadatas is not None:
                # If metadatas is a dict (single document metadata), convert to list of dicts
                if isinstance(metadatas, dict):
                    metadatas = [metadatas]
                # Ensure it's a list of dictionaries
                elif not isinstance(metadatas, list):
                    raise ValueError(f"Expected metadatas to be a dict, list of dicts, or None, got {type(metadatas).__name__}")
            
            # Add documents to vector store
            doc_ids = self.vector_store.add_documents(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to RAG system: {e}")
            raise
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete documents from the RAG system's vector store.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            None
        """
        try:
            # Delete documents from vector store
            self.vector_store.delete_documents(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from RAG system")
        except Exception as e:
            logger.error(f"Error deleting documents from RAG system: {e}")
            raise
            
    def update_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update an existing document in the RAG system's vector store.
        
        Args:
            doc_id: ID of the document to update
            text: New document text
            metadata: Optional new metadata for the document
            
        Returns:
            None
        """
        try:
            # Handle metadata conversion
            if metadata is not None:
                if not isinstance(metadata, dict):
                    raise ValueError(f"Expected metadata to be a dict or None, got {type(metadata).__name__}")
            
            # Update the document in the vector store
            self.vector_store.update_document(doc_id, text, metadata)
            logger.info(f"Updated document {doc_id} in RAG system")
        except Exception as e:
            logger.error(f"Error updating document in RAG system: {e}")
            raise
    
    def get_all_documents(self) -> Dict[str, Any]:
        """
        Get all documents from the RAG system's vector store.
        
        Returns:
            Dictionary containing ids, documents, and metadatas
        """
        try:
            return self.vector_store.get_all_documents()
        except Exception as e:
            logger.error(f"Error retrieving all documents from RAG system: {e}")
            raise
    
    def process_query(self,
                     query: str,
                     n_results: int = 3,
                     max_tokens: int = 512,
                     temperature: float = 0.7) -> Dict[str, Any]:
        """
        Process a user query through the full RAG pipeline.
        
        Args:
            query: User query text
            n_results: Number of documents to retrieve
            max_tokens: Maximum tokens to generate
            temperature: Temperature for LLM sampling
            
        Returns:
            Dictionary with results and metrics
        """
        try:
            start_time = time.time()
            logger.info(f"Processing RAG query: '{query}'")
            
            # Step 1: Retrieve relevant documents
            retrieval_results = self.vector_store.search(
                query=query,
                n_results=n_results
            )
            
            # Extract document texts from results
            context_docs = []
            if 'documents' in retrieval_results and len(retrieval_results['documents']) > 0:
                context_docs = retrieval_results['documents'][0]  # First query's results
            
            # If no documents found, provide a fallback
            if not context_docs:
                logger.warning("No context documents found for query")
                context_docs = ["No relevant information found in the knowledge base."]
            
            # Step 2: Create RAG prompt with retrieved context
            rag_prompt = self.llm.create_rag_prompt(
                query=query,
                context_docs=context_docs
            )
            
            # Step 3: Generate response with LLM
            llm_response = self.llm.generate_response(
                prompt=rag_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Prepare final response
            response = {
                "query": query,
                "answer": llm_response["text"],
                "retrieved_documents": context_docs,
                "document_ids": retrieval_results.get("ids", [[]]) if 'ids' in retrieval_results else [[]],
                "metrics": {
                    "total_time": total_time,
                    "llm_time": llm_response["time_taken"],
                    "tokens_used": llm_response.get("tokens_used", 0),
                    "documents_retrieved": len(context_docs)
                }
            }
            
            logger.info(f"RAG processing completed in {total_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            raise

# Create a singleton instance for application-wide use
default_rag_pipeline = None

def get_rag_pipeline(embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                    collection_name: str = "rag_documents",
                    persist_directory: str = "./vector_db",
                    llm_model_path: str = "web_app/models/vicuna-7b-v1.3.ggmlv3.q4_K_S.bin"):
    """
    Get or create the default RAG pipeline instance.
    
    Args:
        embedding_model: Optional embedding model override
        collection_name: Optional vector store collection name override
        persist_directory: Optional vector store directory override
        llm_model_path: Optional LLM model path override
        
    Returns:
        RAGPipeline instance
    """
    global default_rag_pipeline
    
    if default_rag_pipeline is None:
        logger.info("Creating new RAG pipeline instance")
        default_rag_pipeline = RAGPipeline(
            embedding_model=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
            llm_model_path=llm_model_path
        )
    
    return default_rag_pipeline


# Conversation memory for managing context
class ConversationMemory:
    """
    Manages conversation history and context for continuous interactions.
    """
    
    def __init__(self, max_history: int = 5):
        """
        Initialize the conversation memory.
        
        Args:
            max_history: Maximum number of conversation turns to remember
        """
        self.history = []
        self.max_history = max_history
    
    def add_interaction(self, query: str, response: str):
        """
        Add a user query and system response to the conversation history.
        
        Args:
            query: User's query text
            response: System's response text
        """
        self.history.append({
            "query": query,
            "response": response,
            "timestamp": time.time()
        })
        
        # Trim history if it exceeds the maximum
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context_string(self) -> str:
        """
        Generate a string representation of the conversation context.
        
        Returns:
            Formatted conversation history string
        """
        if not self.history:
            return ""
        
        context_parts = []
        for i, interaction in enumerate(self.history):
            context_parts.append(f"User: {interaction['query']}")
            context_parts.append(f"Assistant: {interaction['response']}")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear the conversation history."""
        self.history = []
