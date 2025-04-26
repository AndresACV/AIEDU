"""
LLM integration module for the RAG System.
Uses llama-cpp-python to interact with the quantized vicuna model.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMInference:
    """Class for running inference with a quantized LLM model."""
    
    def __init__(self, model_path: str = "web_app/models/vicuna-7b-v1.3.ggmlv3.q4_K_S.bin"):
        """
        Initialize the LLM inference engine.
        
        Args:
            model_path: Path to the quantized model file (.bin)
        """
        logger.info(f"Initializing LLM with model: {model_path}")
        self.model_path = model_path
        self.is_mock_mode = False
        
        try:
            # Import here to avoid early loading of large libraries
            from llama_cpp import Llama
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}. Using mock responses instead.")
                self.is_mock_mode = True
                return
            
            # Load the model (with reasonable defaults for CPU inference)
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,            # Context window size
                n_batch=512,           # Batch size for prompt processing
                n_gpu_layers=0,        # CPU only by default
                verbose=False          # Reduce console output
            )
            
            logger.info("LLM model loaded successfully")
            
        except ImportError:
            logger.warning("Failed to import llama_cpp. Using mock responses instead.")
            self.is_mock_mode = True
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}. Using mock responses instead.")
            self.is_mock_mode = True
    
    def generate_response(self, 
                         prompt: str,
                         max_tokens: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.95,
                         top_k: int = 40,
                         stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: Optional list of stop sequences
            
        Returns:
            Dictionary with generated text and metadata
        """
        try:
            logger.info(f"Generating response for prompt of length {len(prompt)} chars")
            start_time = time.time()
            
            # Handle mock mode
            if self.is_mock_mode:
                return self._generate_mock_response(prompt)
            
            # Set default stop words if not provided
            if stop is None:
                stop = ["\n\n", "###", "User:", "Human:"]
            
            # Generate completion
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                echo=False  # Don't include prompt in output
            )
            
            # Calculate time taken
            time_taken = time.time() - start_time
            logger.info(f"Response generated in {time_taken:.2f} seconds")
            
            # Extract and return relevant information
            result = {
                "text": output["choices"][0]["text"].strip(),
                "tokens_used": output["usage"]["total_tokens"],
                "time_taken": time_taken
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Fallback to mock response in case of error
            return self._generate_mock_response(prompt)
    
    def create_rag_prompt(self, 
                         query: str, 
                         context_docs: List[str],
                         system_prompt: str = None) -> str:
        """
        Create a RAG-style prompt with retrieved context documents.
        
        Args:
            query: User query
            context_docs: List of retrieved context documents
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are a helpful, accurate and concise AI assistant. "
                "Answer questions based on the provided context information. "
                "If you don't know the answer or can't find it in the context, say 'I don't have enough information to answer this question.'"
            )
        
        # Combine context documents
        context_text = "\n\n".join([f"Document: {doc}" for doc in context_docs])
        
        # Format the full prompt
        prompt = f"""[System] {system_prompt}

[Context]
{context_text}

[User] {query}

[Assistant]"""

        return prompt

    def _generate_mock_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a mock response when the LLM is not available.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Dictionary with mock response and metadata
        """
        # Extract the user query from the prompt
        query = ""
        if "[User]" in prompt:
            query = prompt.split("[User]")[-1].strip()
        
        # Generate a simple response based on the query
        mock_responses = [
            f"I'm sorry, but I cannot provide a detailed answer as the LLM model is not available. "
            f"Please download the model file '{os.path.basename(self.model_path)}' and place it in the "
            f"correct directory.",
            
            f"To use the full RAG capabilities, you'll need to download the LLM model "
            f"('{os.path.basename(self.model_path)}') and place it in the '{os.path.dirname(self.model_path)}' directory.",
            
            f"I'm operating in limited mode because the LLM model is missing. Your query was: '{query}'. "
            f"For complete functionality, please install the required model file."
        ]
        
        import random
        response_text = random.choice(mock_responses)
        
        # Simulate processing time for a more realistic experience
        time.sleep(0.5)
        time_taken = 0.5
        
        logger.info(f"Generated mock response in {time_taken:.2f} seconds")
        
        return {
            "text": response_text,
            "tokens_used": len(response_text.split()),
            "time_taken": time_taken,
            "is_mock": True
        }

# Create a singleton instance for application-wide use
default_llm = None

def get_llm(model_path: str = "web_app/models/vicuna-7b-v1.3.ggmlv3.q4_K_S.bin"):
    """
    Get or create the default LLM instance.
    
    Args:
        model_path: Optional model path override
        
    Returns:
        LLMInference instance
    """
    global default_llm
    
    if default_llm is None:
        logger.info("Creating new LLM instance")
        default_llm = LLMInference(model_path)
    
    return default_llm
