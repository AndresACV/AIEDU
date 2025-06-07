"""
AI Provider Factory for AIEDU RAG System
Enables seamless switching between local and cloud providers
"""

import os
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional

from ..utils.config_manager import ConfigManager, DeploymentMode

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    LLM = "llm"
    STT = "stt" 
    TTS = "tts"
    VECTOR_STORE = "vector_store"
    EMBEDDINGS = "embeddings"

class AIProviderInterface(ABC):
    """Base interface for all AI providers"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and functional"""
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information and capabilities"""
        pass

class AIProviderFactory:
    """Factory for creating appropriate AI providers based on deployment mode"""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.deployment_mode = self.config_manager.deployment_mode
        self._cached_providers = {}
        
        logger.info(f"AIProviderFactory initialized for {self.deployment_mode.value} deployment")
    
    def create_llm_provider(self) -> AIProviderInterface:
        """Create LLM provider based on deployment mode"""
        
        if ProviderType.LLM.value in self._cached_providers:
            return self._cached_providers[ProviderType.LLM.value]
        
        if self.deployment_mode == DeploymentMode.LOCAL:
            from local_deployment.web_app.llm import OllamaLLMInference
            provider = OllamaLLMInference()
            logger.info("Created Ollama LLM provider for local deployment")
            
        elif self.deployment_mode == DeploymentMode.CLOUD:
            from cloud_deployment.api.providers.gemini_provider import GeminiLLMProvider
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable required for cloud deployment")
            provider = GeminiLLMProvider(api_key)
            logger.info("Created Gemini LLM provider for cloud deployment")
            
        else:
            raise ValueError(f"Unsupported deployment mode: {self.deployment_mode}")
        
        self._cached_providers[ProviderType.LLM.value] = provider
        return provider
    
    def create_stt_provider(self) -> AIProviderInterface:
        """Create Speech-to-Text provider based on deployment mode"""
        
        if ProviderType.STT.value in self._cached_providers:
            return self._cached_providers[ProviderType.STT.value]
        
        if self.deployment_mode == DeploymentMode.LOCAL:
            from local_deployment.web_app.speech_recognition import VoskSTTProvider
            provider = VoskSTTProvider()
            logger.info("Created Vosk STT provider for local deployment")
            
        elif self.deployment_mode == DeploymentMode.CLOUD:
            from cloud_deployment.api.providers.gcp_stt_provider import GCPSTTProvider
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not credentials_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS required for cloud deployment")
            provider = GCPSTTProvider(credentials_path)
            logger.info("Created Google Cloud STT provider for cloud deployment")
            
        else:
            raise ValueError(f"Unsupported deployment mode: {self.deployment_mode}")
        
        self._cached_providers[ProviderType.STT.value] = provider
        return provider
    
    def create_tts_provider(self) -> AIProviderInterface:
        """Create Text-to-Speech provider based on deployment mode"""
        
        if ProviderType.TTS.value in self._cached_providers:
            return self._cached_providers[ProviderType.TTS.value]
        
        if self.deployment_mode == DeploymentMode.LOCAL:
            from local_deployment.web_app.tts import LocalTTSProvider
            provider = LocalTTSProvider()
            logger.info("Created Local TTS provider for local deployment")
            
        elif self.deployment_mode == DeploymentMode.CLOUD:
            from cloud_deployment.api.providers.gcp_tts_provider import GCPTTSProvider
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not credentials_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS required for cloud deployment")
            provider = GCPTTSProvider(credentials_path)
            logger.info("Created Google Cloud TTS provider for cloud deployment")
            
        else:
            raise ValueError(f"Unsupported deployment mode: {self.deployment_mode}")
        
        self._cached_providers[ProviderType.TTS.value] = provider
        return provider
    
    def create_vector_store_provider(self) -> AIProviderInterface:
        """Create Vector Store provider based on deployment mode"""
        
        if ProviderType.VECTOR_STORE.value in self._cached_providers:
            return self._cached_providers[ProviderType.VECTOR_STORE.value]
        
        if self.deployment_mode == DeploymentMode.LOCAL:
            from local_deployment.web_app.vector_store import ChromaDBVectorStore
            provider = ChromaDBVectorStore()
            logger.info("Created ChromaDB vector store for local deployment")
            
        elif self.deployment_mode == DeploymentMode.CLOUD:
            from cloud_deployment.api.providers.pinecone_provider import PineconeVectorStore
            api_key = os.getenv('PINECONE_API_KEY')
            environment = os.getenv('PINECONE_ENVIRONMENT')
            if not api_key or not environment:
                raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT required for cloud deployment")
            provider = PineconeVectorStore(api_key, environment)
            logger.info("Created Pinecone vector store for cloud deployment")
            
        else:
            raise ValueError(f"Unsupported deployment mode: {self.deployment_mode}")
        
        self._cached_providers[ProviderType.VECTOR_STORE.value] = provider
        return provider
    
    def create_embeddings_provider(self) -> AIProviderInterface:
        """Create Embeddings provider based on deployment mode"""
        
        if ProviderType.EMBEDDINGS.value in self._cached_providers:
            return self._cached_providers[ProviderType.EMBEDDINGS.value]
        
        if self.deployment_mode == DeploymentMode.LOCAL:
            from local_deployment.web_app.embeddings import SentenceTransformersEmbeddings
            provider = SentenceTransformersEmbeddings()
            logger.info("Created SentenceTransformers embeddings for local deployment")
            
        elif self.deployment_mode == DeploymentMode.CLOUD:
            from cloud_deployment.api.providers.openai_embeddings_provider import OpenAIEmbeddingsProvider
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY required for cloud deployment")
            provider = OpenAIEmbeddingsProvider(api_key)
            logger.info("Created OpenAI embeddings for cloud deployment")
            
        else:
            raise ValueError(f"Unsupported deployment mode: {self.deployment_mode}")
        
        self._cached_providers[ProviderType.EMBEDDINGS.value] = provider
        return provider
    
    def create_all_providers(self) -> Dict[str, AIProviderInterface]:
        """Create all providers for current deployment mode"""
        
        providers = {}
        
        try:
            providers['llm'] = self.create_llm_provider()
            providers['stt'] = self.create_stt_provider()
            providers['tts'] = self.create_tts_provider()
            providers['vector_store'] = self.create_vector_store_provider()
            providers['embeddings'] = self.create_embeddings_provider()
            
            logger.info(f"Successfully created all providers for {self.deployment_mode.value} deployment")
            
        except Exception as e:
            logger.error(f"Failed to create providers: {str(e)}")
            raise
        
        return providers
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        
        status = {}
        
        for provider_type in ProviderType:
            try:
                if provider_type == ProviderType.LLM:
                    provider = self.create_llm_provider()
                elif provider_type == ProviderType.STT:
                    provider = self.create_stt_provider()
                elif provider_type == ProviderType.TTS:
                    provider = self.create_tts_provider()
                elif provider_type == ProviderType.VECTOR_STORE:
                    provider = self.create_vector_store_provider()
                elif provider_type == ProviderType.EMBEDDINGS:
                    provider = self.create_embeddings_provider()
                
                status[provider_type.value] = {
                    'available': provider.is_available(),
                    'info': provider.get_provider_info(),
                    'deployment_mode': self.deployment_mode.value
                }
                
            except Exception as e:
                status[provider_type.value] = {
                    'available': False,
                    'error': str(e),
                    'deployment_mode': self.deployment_mode.value
                }
        
        return status
    
    def switch_deployment_mode(self, new_mode: DeploymentMode):
        """Switch to a different deployment mode (clears cache)"""
        
        if new_mode != self.deployment_mode:
            logger.info(f"Switching deployment mode from {self.deployment_mode.value} to {new_mode.value}")
            
            # Clear cached providers
            self._cached_providers.clear()
            
            # Update configuration
            self.deployment_mode = new_mode
            self.config_manager.deployment_mode = new_mode
            
            logger.info(f"Successfully switched to {new_mode.value} deployment mode")


# Convenience function for easy access
def get_provider_factory(deployment_mode: DeploymentMode = None) -> AIProviderFactory:
    """Get AIProviderFactory instance with optional deployment mode override"""
    return AIProviderFactory(ConfigManager(deployment_mode=deployment_mode)) 