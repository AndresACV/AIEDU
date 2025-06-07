"""
RAG Service Module
Provides Retrieval-Augmented Generation functionality for the FastAPI backend.
Migrated from Flask RAG pipeline with enhanced FastAPI integration.
"""

import os
import sys
import logging
import time
import tempfile
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Add project paths for accessing original modules
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RAG_MODULES_PATH = PROJECT_ROOT / "local_deployment" / "web_app"
sys.path.insert(0, str(RAG_MODULES_PATH))

# Import original modules
try:
    from embeddings import get_embedding_generator
    from vector_store import get_vector_store
    # Use local enhanced LLM module with better timeout handling
    import sys
    from pathlib import Path
    backend_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(backend_root))
    from llm import get_llm
    MODULES_AVAILABLE = True
    logger.info("RAG modules imported successfully (using enhanced LLM module)")
except ImportError as e:
    logger.error(f"Could not import RAG modules: {e}")
    MODULES_AVAILABLE = False

class RAGService:
    """
    FastAPI RAG Service providing document management and query processing.
    
    Features:
    - Document upload and processing with embeddings
    - Vector-based document retrieval  
    - Hybrid LLM-powered response generation (Local Ollama + Cloud Gemini)
    - Knowledge base management with CRUD operations
    - Performance tracking and statistics
    """
    
    def __init__(self, 
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 collection_name: str = "rag_documents",
                 persist_directory: str = None,
                 llm_model_name: str = "mistral:7b",
                 ollama_api_url: str = "http://localhost:11434",
                 provider_service = None):
        """
        Initialize the RAG Service.
        
        Args:
            embedding_model: Name of the embedding model
            collection_name: Name of the vector store collection
            persist_directory: Directory for vector store persistence
            llm_model_name: Ollama model name
            ollama_api_url: Ollama API base URL
            provider_service: Provider service for determining current LLM provider
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(PROJECT_ROOT / "vector_db")
        self.llm_model_name = llm_model_name
        self.ollama_api_url = ollama_api_url
        self.provider_service = provider_service
        
        # Initialize components
        self.embedding_generator = None
        self.vector_store = None
        self.local_llm = None  # Ollama LLM
        self.cloud_llm = None  # Gemini LLM
        self.initialized = False
        
        # Performance statistics
        self.stats = {
            'documents_added': 0,
            'documents_deleted': 0,
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_query_time': 0.0,
            'avg_retrieval_time': 0.0,
            'avg_generation_time': 0.0,
            'total_documents': 0
        }
        
        # Initialize RAG components
        self._initialize_components()
        
        logger.info(f"RAG Service initialized")
        logger.info(f"Vector DB: {self.persist_directory}")
        logger.info(f"LLM Model: {self.llm_model_name}")
        logger.info(f"Embedding Model: {self.embedding_model}")
    
    def _initialize_components(self):
        """Initialize RAG pipeline components with hybrid LLM support."""
        if not MODULES_AVAILABLE:
            logger.error("RAG modules not available - service will operate in limited mode")
            return
        
        try:
            # Initialize embedding generator
            self.embedding_generator = get_embedding_generator(self.embedding_model)
            logger.info("Embedding generator initialized")
            
            # Initialize vector store
            self.vector_store = get_vector_store(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            logger.info("Vector store initialized")
            
            # Initialize Local LLM (Ollama)
            try:
                self.local_llm = get_llm(
                    model_name=self.llm_model_name,
                    api_url=self.ollama_api_url
                )
                logger.info("Local LLM (Ollama) initialized")
            except Exception as e:
                logger.warning(f"Local LLM initialization failed: {e}")
                self.local_llm = None
            
            # Initialize Cloud LLM (Gemini)
            try:
                import sys
                from pathlib import Path
                
                # Add cloud_deployment to path
                cloud_path = Path(__file__).parent.parent.parent.parent / "cloud_deployment"
                if str(cloud_path) not in sys.path:
                    sys.path.append(str(cloud_path))
                
                from api.providers.gemini_provider import GeminiLLMProvider
                
                # Check if Gemini API key is available
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key and api_key.strip():
                    self.cloud_llm = GeminiLLMProvider(api_key)
                    if self.cloud_llm.is_available():
                        logger.info("Cloud LLM (Gemini) initialized successfully")
                    else:
                        logger.warning("Cloud LLM (Gemini) not available")
                        self.cloud_llm = None
                else:
                    logger.info("Cloud LLM (Gemini) not configured - no API key")
                    self.cloud_llm = None
            except Exception as e:
                logger.warning(f"Cloud LLM initialization failed: {e}")
                self.cloud_llm = None
            
            # Update document count
            try:
                vector_stats = self.vector_store.get_stats()
                self.stats['total_documents'] = vector_stats.get('count', 0)
            except Exception as e:
                logger.warning(f"Could not get initial document count: {e}")
            
            # Service is available if we have at least one LLM
            self.initialized = self.local_llm is not None or self.cloud_llm is not None
            
            if self.initialized:
                available_llms = []
                if self.local_llm: available_llms.append("Ollama")
                if self.cloud_llm: available_llms.append("Gemini")
                logger.info(f"RAG Service initialized successfully with LLMs: {', '.join(available_llms)}")
            else:
                logger.error("RAG Service initialization failed - no LLMs available")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            self.initialized = False
    
    def is_available(self) -> bool:
        """Check if RAG service is available."""
        return MODULES_AVAILABLE and self.initialized
    
    def _get_current_llm(self):
        """Get the current LLM based on provider selection."""
        # Check if provider service is available and get current provider
        if self.provider_service:
            current_provider = self.provider_service.current_provider
            
            if current_provider == "cloud" and self.cloud_llm:
                logger.info("Using Cloud LLM (Gemini) for RAG processing")
                return self.cloud_llm, "cloud"
            elif current_provider == "local" and self.local_llm:
                logger.info("Using Local LLM (Ollama) for RAG processing")
                return self.local_llm, "local"
        
        # Fallback logic: prioritize available LLMs
        if self.local_llm:
            logger.info("Fallback to Local LLM (Ollama) for RAG processing")
            return self.local_llm, "local"
        elif self.cloud_llm:
            logger.info("Fallback to Cloud LLM (Gemini) for RAG processing")
            return self.cloud_llm, "cloud"
        
        logger.error("No LLM available for RAG processing")
        return None, None
    
    async def add_document(self, 
                          title: str, 
                          content: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a document to the knowledge base.
        
        Args:
            title: Document title
            content: Document content
            metadata: Optional metadata for the document
            
        Returns:
            Dict with document ID and status
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'RAG service not available',
                'doc_id': None
            }
        
        try:
            start_time = time.time()
            
            # Generate unique document ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            
            # Prepare metadata
            doc_metadata = {
                'title': title,
                'added_at': datetime.now().isoformat(),
                'doc_type': 'user_uploaded',
                'content_length': len(content)
            }
            if metadata:
                doc_metadata.update(metadata)
            
            # Add document to vector store
            doc_ids = self.vector_store.add_documents(
                documents=[content],
                metadatas=[doc_metadata],
                ids=[doc_id]
            )
            
            duration = time.time() - start_time
            
            # Update statistics
            self.stats['documents_added'] += 1
            self.stats['total_documents'] += 1
            
            logger.info(f"Added document '{title}' with ID {doc_id} in {duration:.2f}s")
            
            return {
                'success': True,
                'doc_id': doc_id,
                'title': title,
                'content_length': len(content),
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return {
                'success': False,
                'error': str(e),
                'doc_id': None
            }
    
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document from the knowledge base.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Dict with deletion status
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'RAG service not available'
            }
        
        try:
            # Delete document from vector store
            self.vector_store.delete_documents([doc_id])
            
            # Update statistics
            self.stats['documents_deleted'] += 1
            self.stats['total_documents'] = max(0, self.stats['total_documents'] - 1)
            
            logger.info(f"Deleted document {doc_id}")
            
            return {
                'success': True,
                'doc_id': doc_id,
                'message': f'Document {doc_id} deleted successfully'
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def list_documents(self) -> Dict[str, Any]:
        """
        List all documents in the knowledge base.
        
        Returns:
            Dict with document list and metadata
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'RAG service not available',
                'documents': []
            }
        
        try:
            # Get all documents from vector store
            all_docs = self.vector_store.get_all_documents()
            
            # Format document information
            documents = []
            if 'ids' in all_docs and all_docs['ids']:
                for i, doc_id in enumerate(all_docs['ids']):
                    doc_info = {
                        'id': doc_id,
                        'title': 'Unknown',
                        'content_preview': '',
                        'added_at': None,
                        'content_length': 0
                    }
                    
                    # Extract metadata if available
                    if 'metadatas' in all_docs and i < len(all_docs['metadatas']):
                        metadata = all_docs['metadatas'][i] or {}
                        doc_info.update({
                            'title': metadata.get('title', 'Unknown'),
                            'added_at': metadata.get('added_at'),
                            'content_length': metadata.get('content_length', 0)
                        })
                    
                    # Extract content preview if available
                    if 'documents' in all_docs and i < len(all_docs['documents']):
                        content = all_docs['documents'][i] or ''
                        doc_info['content_preview'] = content[:200] + '...' if len(content) > 200 else content
                    
                    documents.append(doc_info)
            
            return {
                'success': True,
                'documents': documents,
                'total_count': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return {
                'success': False,
                'error': str(e),
                'documents': []
            }
    
    async def process_query(self, 
                           query: str,
                           n_results: int = 3,
                           max_tokens: int = 512,
                           temperature: float = 0.7) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User query text
            n_results: Number of documents to retrieve
            max_tokens: Maximum tokens to generate
            temperature: Temperature for LLM sampling
            
        Returns:
            Dict with response and metadata
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'RAG service not available',
                'response': None
            }
        
        overall_start_time = time.time()
        
        try:
            self.stats['queries_processed'] += 1
            logger.info(f"Processing RAG query: '{query[:100]}...'")
            
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            retrieval_results = self.vector_store.search(
                query=query,
                n_results=n_results
            )
            retrieval_duration = time.time() - retrieval_start
            
            # Extract document texts from results
            context_docs = []
            retrieved_docs_info = []
            
            if 'documents' in retrieval_results and len(retrieval_results['documents']) > 0:
                context_docs = retrieval_results['documents'][0]  # First query's results
                
                # Collect document information for response
                for i in range(len(context_docs)):
                    doc_info = {'content_preview': context_docs[i][:200] + '...'}
                    
                    # Add metadata if available
                    if 'metadatas' in retrieval_results and i < len(retrieval_results['metadatas'][0]):
                        metadata = retrieval_results['metadatas'][0][i] or {}
                        doc_info.update({
                            'title': metadata.get('title', 'Unknown'),
                            'doc_id': retrieval_results['ids'][0][i] if 'ids' in retrieval_results else None
                        })
                    
                    # Add distance/similarity score if available
                    if 'distances' in retrieval_results and i < len(retrieval_results['distances'][0]):
                        doc_info['similarity_score'] = 1.0 - retrieval_results['distances'][0][i]  # Convert distance to similarity
                    
                    retrieved_docs_info.append(doc_info)
            
            # If no documents found, provide a fallback
            if not context_docs:
                logger.warning("No context documents found for query")
                context_docs = ["No relevant information found in the knowledge base."]
            
            # Step 2: Determine which LLM to use based on provider selection
            current_llm, llm_type = self._get_current_llm()
            if not current_llm:
                raise Exception("No LLM provider available")
            
            # Step 3: Generate response with appropriate LLM
            generation_start = time.time()
            
            if llm_type == "cloud":
                # Use Gemini for cloud provider
                context_text = " ".join(context_docs)
                llm_response = current_llm.generate_rag_response(
                    query=query,
                    context=context_text,
                    conversation_history=None  # Could be extended to support conversation history
                )
                
                # Convert Gemini response format to match expected format
                if llm_response.get('success'):
                    llm_response = {
                        'text': llm_response['response'],
                        'tokens_used': llm_response.get('token_count', 0),
                        'provider': 'gemini'
                    }
                else:
                    raise Exception(f"Cloud LLM error: {llm_response.get('error', 'Unknown error')}")
            else:
                # Use Ollama for local provider
                rag_prompt = current_llm.create_rag_prompt(
                    query=query,
                    context_docs=context_docs
                )
                
                llm_response = current_llm.generate_response(
                    prompt=rag_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Ensure response has the expected format
                if llm_response and 'text' not in llm_response:
                    llm_response['provider'] = 'ollama'
            
            generation_duration = time.time() - generation_start
            
            overall_duration = time.time() - overall_start_time
            
            # Check if generation was successful
            if llm_response and 'text' in llm_response:
                self.stats['successful_queries'] += 1
                self._update_avg_time('query', overall_duration)
                self._update_avg_time('retrieval', retrieval_duration)
                self._update_avg_time('generation', generation_duration)
                
                response = {
                    'success': True,
                    'response': llm_response['text'],
                    'query': query,
                    'retrieved_documents': retrieved_docs_info,
                    'timing': {
                        'total_time': overall_duration,
                        'retrieval_time': retrieval_duration,
                        'generation_time': generation_duration
                    },
                    'metadata': {
                        'n_documents_retrieved': len(context_docs),
                        'tokens_generated': llm_response.get('tokens_used', 0),
                        'temperature': temperature
                    }
                }
                
                logger.info(f"Query processed successfully in {overall_duration:.2f}s")
                return response
            else:
                raise Exception("LLM failed to generate response")
                
        except Exception as e:
            overall_duration = time.time() - overall_start_time
            self.stats['failed_queries'] += 1
            logger.error(f"Error processing query: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'query': query,
                'timing': {
                    'total_time': overall_duration
                }
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get RAG service performance statistics."""
        try:
            # Get current vector store stats
            if self.vector_store:
                vector_stats = self.vector_store.get_stats()
                self.stats['total_documents'] = vector_stats.get('count', 0)
        except Exception as e:
            logger.warning(f"Could not update document count: {e}")
        
        # Calculate success rates
        total_queries = self.stats['queries_processed']
        success_rate = (self.stats['successful_queries'] / total_queries * 100) if total_queries > 0 else 0.0
        
        # Determine current LLM status
        current_llm_info = "None"
        if self.provider_service:
            if self.provider_service.current_provider == "cloud" and self.cloud_llm:
                current_llm_info = "Gemini 2.5 Flash (Cloud)"
            elif self.provider_service.current_provider == "local" and self.local_llm:
                current_llm_info = f"{self.llm_model_name} (Local)"
            else:
                current_llm_info = "Provider mismatch"
        else:
            # Fallback status
            if self.local_llm and self.cloud_llm:
                current_llm_info = f"Hybrid: {self.llm_model_name} + Gemini"
            elif self.local_llm:
                current_llm_info = f"{self.llm_model_name} (Local only)"
            elif self.cloud_llm:
                current_llm_info = "Gemini (Cloud only)"
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'service_status': 'available' if self.is_available() else 'unavailable',
            'components': {
                'embedding_model': self.embedding_model,
                'llm_model': current_llm_info,
                'vector_store': self.collection_name,
                'local_llm_available': self.local_llm is not None,
                'cloud_llm_available': self.cloud_llm is not None,
                'current_provider': self.provider_service.current_provider if self.provider_service else "unknown"
            }
        }
    
    def _update_avg_time(self, operation_type: str, duration: float):
        """Update average timing statistics."""
        key = f'avg_{operation_type}_time'
        if key in self.stats:
            # Simple moving average update
            current_avg = self.stats[key]
            if operation_type == 'query':
                call_count = self.stats['successful_queries']
            elif operation_type == 'retrieval':
                call_count = max(self.stats['successful_queries'], 1)
            else:  # generation
                call_count = max(self.stats['successful_queries'], 1)
            
            new_avg = ((current_avg * (call_count - 1)) + duration) / call_count
            self.stats[key] = new_avg
    
    async def upload_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process uploaded file and add to knowledge base.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Dict with processing results
        """
        try:
            # For now, assume text files
            content = file_content.decode('utf-8')
            
            # Add to knowledge base
            result = await self.add_document(
                title=filename,
                content=content,
                metadata={'source': 'file_upload', 'filename': filename}
            )
            
            return result
            
        except UnicodeDecodeError:
            return {
                'success': False,
                'error': 'File must be a text file (UTF-8 encoded)',
                'doc_id': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'doc_id': None
            }


# Singleton instance
_rag_service = None

def get_rag_service() -> RAGService:
    """Get or create the singleton RAGService instance."""
    global _rag_service
    if _rag_service is None:
        # Import here to avoid circular imports
        try:
            from ..core.dependencies import get_provider_service
            provider_service = get_provider_service()
            _rag_service = RAGService(provider_service=provider_service)
        except:
            # Fallback without provider service
            _rag_service = RAGService()
    return _rag_service
