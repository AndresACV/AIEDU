"""
Lazy-loading embedding generation module for the RAG System.
Uses SentenceTransformer to create embeddings but only loads when needed.
"""

import os
from typing import List, Union, Optional
import logging
import threading
import torch
from sentence_transformers import SentenceTransformer

# Set up logging
logger = logging.getLogger(__name__)

# Global state for lazy loading
_model_instance = None
_model_loading = False
_model_init_lock = threading.Lock()

class LazyEmbeddingGenerator:
    """Lazy-loading embedding generator that only loads the model when first used."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the lazy embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim: Optional[int] = None
        self._initialized = False
        
        logger.info(f"Lazy embedding generator created for model: {model_name}")
        logger.info("🕒 Model will be loaded on first use")
    
    def _load_model(self):
        """Load the model if not already loaded."""
        global _model_instance, _model_loading
        
        if self._initialized and self.model is not None:
            return
            
        with _model_init_lock:
            # Double-check after acquiring lock
            if self._initialized and self.model is not None:
                return
                
            if _model_loading:
                logger.info("Model is being loaded by another thread, waiting...")
                # Wait for other thread to finish loading
                while _model_loading:
                    threading.Event().wait(0.1)
                if _model_instance is not None:
                    self.model = _model_instance
                    self.embedding_dim = self.model.get_sentence_embedding_dimension()
                    self._initialized = True
                    return
            
            logger.info(f"🚀 Loading embedding model: {self.model_name}")
            _model_loading = True
            
            try:
                # Clear CUDA cache before loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("🧹 Cleared CUDA cache")
                
                # Determine device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Load the model
                self.model = SentenceTransformer(self.model_name, device=device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                
                # Store globally to share between instances
                _model_instance = self.model
                
                logger.info(f"✅ Embedding model loaded successfully on {device}")
                logger.info(f"📐 Embedding dimension: {self.embedding_dim}")
                
                if device == 'cuda':
                    logger.info("🚀 GPU acceleration enabled for embeddings!")
                    
                self._initialized = True
                
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise
            finally:
                _model_loading = False
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for the provided text or list of texts.
        Loads the model on first call.
        
        Args:
            texts: A single text or list of texts to embed
            
        Returns:
            List of embeddings (as lists of floats)
        """
        # Load model if not already loaded
        if not self._initialized:
            logger.info("🔄 First embedding request - loading model...")
            self._load_model()
        
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            logger.debug(f"Generating embeddings for {len(texts)} text(s)")
            embeddings = self.model.encode(texts)
            
            # Convert numpy arrays to Python lists for compatibility
            return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._initialized and self.model is not None
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'is_loaded': self.is_loaded(),
            'embedding_dim': self.embedding_dim,
            'device': str(self.model.device) if self.model else None
        }

# Global lazy embedding generator instance
_lazy_embedding_generator = None
_generator_init_lock = threading.Lock()

def get_embedding_generator(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """
    Get or create the lazy embedding generator.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        LazyEmbeddingGenerator instance
    """
    global _lazy_embedding_generator
    
    with _generator_init_lock:
        if _lazy_embedding_generator is None:
            logger.info("Creating lazy embedding generator instance")
            _lazy_embedding_generator = LazyEmbeddingGenerator(model_name)
    
    return _lazy_embedding_generator

 