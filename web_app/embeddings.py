"""
Embedding generation module for the RAG System.
Uses SentenceTransformer to create embeddings for documents and queries.
"""

import os
from typing import List, Union
import logging
import threading
import torch
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lock for thread safety during model initialization
model_init_lock = threading.Lock()

class EmbeddingGenerator:
    """Class for generating embeddings using SentenceTransformer."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the embedding generator with the specified model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
                       Default: 'sentence-transformers/all-MiniLM-L6-v2'
        """
        logger.info(f"Initializing embedding model: {model_name}")
        try:
            # Use a lock to prevent concurrent initializations
            with model_init_lock:
                # Determine the device - use CPU for stability
                device = 'cpu'
                # Create the model with explicit device assignment
                self.model = SentenceTransformer(model_name, device=device)
                # Get the embedding dimension for reference
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model loaded successfully. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for the provided text or list of texts.
        
        Args:
            texts: A single text or list of texts to embed
            
        Returns:
            List of embeddings (as lists of floats)
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            logger.debug(f"Generating embeddings for {len(texts)} text(s)")
            embeddings = self.model.encode(texts)
            
            # Convert numpy arrays to Python lists for better compatibility
            return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

# Create a singleton instance for application-wide use
default_embedding_generator = None
embedding_init_lock = threading.Lock()

def get_embedding_generator(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """
    Get or create the default embedding generator.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        EmbeddingGenerator instance
    """
    global default_embedding_generator
    
    # Use a lock to prevent race conditions when multiple threads try to initialize
    with embedding_init_lock:
        if default_embedding_generator is None:
            logger.info("Creating new embedding generator instance")
            try:
                # Clear any CUDA memory if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Create the embedding generator
                default_embedding_generator = EmbeddingGenerator(model_name)
            except Exception as e:
                logger.error(f"Error initializing embedding generator: {e}")
                # Return a dummy embedding generator that won't crash the application
                raise
    
    return default_embedding_generator
