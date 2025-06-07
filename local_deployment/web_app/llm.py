"""
LLM integration module for the RAG System.
Uses Ollama API to interact with local LLM models.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import time
import requests
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaLLMInference:
    """Class for running inference with Ollama API."""
    
    def __init__(self, 
                 model_name: str = "mistral:7b",
                 api_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama LLM inference engine.
        
        Args:
            model_name: Name of the Ollama model (e.g., "mistral:7b")
            api_url: Ollama API base URL
        """
        logger.info(f"Initializing Ollama LLM with model: {model_name}")
        self.model_name = model_name
        self.api_url = api_url
        self.generate_url = f"{api_url}/api/generate"
        self.is_mock_mode = False
        
        # Test Ollama connection
        if not self._test_ollama_connection():
            logger.warning("Ollama not available. Using mock responses instead.")
            self.is_mock_mode = True
        elif not self._check_model_available():
            logger.warning(f"Model '{model_name}' not available in Ollama. Using mock responses instead.")
            self.is_mock_mode = True
        else:
            logger.info("Ollama LLM connection established successfully")
    
    def _test_ollama_connection(self) -> bool:
        """Test if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _check_model_available(self) -> bool:
        """Check if the specified model is available in Ollama."""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return self.model_name in available_models
            return False
        except requests.exceptions.RequestException:
            return False
    
    def generate_response(self, 
                         prompt: str,
                         max_tokens: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.95,
                         top_k: int = 40,
                         stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM via Ollama.
        
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
            
            # Prepare Ollama API request
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                }
            }
            
            # Add stop sequences if provided
            if stop:
                payload["options"]["stop"] = stop
            
            # Make API call to Ollama
            response = requests.post(
                self.generate_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # Reduced to 30 second timeout to prevent hanging
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self._generate_mock_response(prompt)
            
            # Parse response
            result_data = response.json()
            generated_text = result_data.get("response", "").strip()
            
            # Calculate time taken
            time_taken = time.time() - start_time
            logger.info(f"Response generated in {time_taken:.2f} seconds")
            
            # Return result in expected format
            result = {
                "text": generated_text,
                "tokens_used": len(generated_text.split()),  # Approximate token count
                "time_taken": time_taken
            }
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error("Ollama API timeout")
            return self._generate_mock_response(prompt)
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return self._generate_mock_response(prompt)
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
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
        
        # Format the full prompt for Mistral-style instruction following
        prompt = f"""<s>[INST] {system_prompt}

Context:
{context_text}

Question: {query} [/INST]"""

        return prompt

    def _generate_mock_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a mock response when Ollama is not available.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Dictionary with mock response and metadata
        """
        # Extract the user query from the prompt
        query = ""
        if "Question:" in prompt:
            query = prompt.split("Question:")[-1].strip()
            if "[/INST]" in query:
                query = query.replace("[/INST]", "").strip()
        
        # Generate a simple response based on the query
        mock_responses = [
            f"I'm sorry, but I cannot provide a detailed answer as Ollama is not running. "
            f"Please start Ollama and ensure the '{self.model_name}' model is available.",
            
            f"To use the full RAG capabilities, you'll need to start Ollama and pull the model: "
            f"'ollama pull {self.model_name}'",
            
            f"I'm operating in limited mode because Ollama is not available. Your query was: '{query}'. "
            f"For complete functionality, please start Ollama with: 'ollama serve'"
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

    def get_available_models(self) -> List[str]:
        """Get list of available models in Ollama."""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except requests.exceptions.RequestException:
            return []

def get_llm(model_name: str = "mistral:7b", api_url: str = "http://localhost:11434"):
    """
    Get an instance of the Ollama LLM inference engine.
    
    Args:
        model_name: Ollama model name (e.g., "mistral:7b")
        api_url: Ollama API base URL
        
    Returns:
        OllamaLLMInference instance
    """
    return OllamaLLMInference(model_name=model_name, api_url=api_url)


