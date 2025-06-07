from typing import Dict, Any
import os
import logging
import requests
import json
from ..models.providers import ProviderType, ProviderResponse, ProviderInfo, ProvidersDict

logger = logging.getLogger(__name__)

class ProviderService:
    """Service for managing speech and AI providers."""
    
    def __init__(self):
        self.current_provider: ProviderType = "local"
        
    def get_current_providers(self) -> ProviderResponse:
        """Get current provider status and configuration."""
        # Check actual provider availability
        local_status = self._check_local_providers()
        cloud_status = self._check_cloud_providers()
        
        providers_dict = ProvidersDict(local=local_status, cloud=cloud_status)
        
        # Determine current provider names based on selection
        if self.current_provider == "local":
            stt_provider = "Vosk (Local)" if local_status.stt == "available" else "Vosk (Error)"
            tts_provider = "espeak (Local)" if local_status.tts == "available" else "espeak (Error)"
            llm_provider = "Ollama (Local)" if local_status.llm == "available" else "Ollama (Error)"
        else:  # cloud
            stt_provider = "Google Cloud STT" if cloud_status.stt == "available" else "Google Cloud STT (Error)"
            tts_provider = "Google Cloud TTS" if cloud_status.tts == "available" else "Google Cloud TTS (Error)"
            llm_provider = "Gemini 2.5 Flash" if cloud_status.llm == "available" else "Gemini 2.5 Flash (Error)"
        
        return ProviderResponse(
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            llm_provider=llm_provider,
            status="FastAPI Backend is running",
            providers=providers_dict
        )
    
    def force_provider(self, provider: ProviderType) -> Dict[str, Any]:
        """Force switch to a specific provider."""
        old_provider = self.current_provider
        self.current_provider = provider
        
        return {
            "success": True,
            "message": f"Switched from {old_provider} to {provider}",
            "new_provider": provider
        }
    
    def get_available_voices(self) -> list:
        """Get available voices for current provider."""
        if self.current_provider == "cloud":
            # Cloud voices (Google Cloud TTS)
            return [
                {
                    "id": "es-ES-Neural2-C",
                    "name": "Spanish Neural (Female)",
                    "language_type": "Spanish",
                    "provider": "Google Cloud"
                },
                {
                    "id": "en-US-Neural2-F",
                    "name": "English Neural (Female)",
                    "language_type": "English",  
                    "provider": "Google Cloud"
                },
                {
                    "id": "en-US-Neural2-A",
                    "name": "English Neural (Male)",
                    "language_type": "English",
                    "provider": "Google Cloud"
                }
            ]
        else:
            # Local voices (espeak)
            return [
                {
                    "id": "local_es",
                    "name": "Spanish (Local)",
                    "language_type": "Spanish",
                    "provider": "espeak"
                },
                {
                    "id": "local_en",
                    "name": "English (Local)",
                    "language_type": "English",
                    "provider": "espeak"
                }
            ]
    
    def _check_local_providers(self) -> ProviderInfo:
        """Check availability of local providers."""
        stt_status = self._check_vosk_availability()
        tts_status = self._check_espeak_availability()
        llm_status = self._check_ollama_availability()
        
        return ProviderInfo(stt=stt_status, tts=tts_status, llm=llm_status)
    
    def _check_cloud_providers(self) -> ProviderInfo:
        """Check availability of cloud providers."""
        stt_status = self._check_google_cloud_stt()
        tts_status = self._check_google_cloud_tts()
        llm_status = self._check_gemini_availability()
        
        return ProviderInfo(stt=stt_status, tts=tts_status, llm=llm_status)
    
    def _check_vosk_availability(self) -> str:
        """Check if Vosk STT is available."""
        try:
            import vosk
            # Check for model files
            from pathlib import Path
            
            # Try different possible paths
            possible_paths = [
                Path("local_deployment/web_app/models"),
                Path("../local_deployment/web_app/models"),
                Path("../../local_deployment/web_app/models")
            ]
            
            for models_path in possible_paths:
                if models_path.exists():
                    # Check for at least one model
                    model_dirs = [d for d in models_path.iterdir() if d.is_dir() and "vosk-model" in d.name]
                    if model_dirs:
                        return "available"
            
            return "error"
            
        except ImportError:
            return "error"
        except Exception:
            return "error"
    
    def _check_espeak_availability(self) -> str:
        """Check if espeak TTS is available."""
        try:
            import subprocess
            import platform
            
            if platform.system() == "Linux":
                result = subprocess.run(["which", "espeak"], capture_output=True, timeout=5)
                return "available" if result.returncode == 0 else "error"
            else:
                # For other platforms, check pyttsx3
                import pyttsx3
                return "available"
                
        except Exception:
            return "error"
    
    def _check_ollama_availability(self) -> str:
        """Check if Ollama LLM is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return "available" if models else "error"
            return "error"
            
        except Exception:
            return "error"
    
    def _check_google_cloud_stt(self) -> str:
        """Check if Google Cloud STT is available."""
        try:
            # Import the existing cloud provider
            import sys
            from pathlib import Path
            
            # Add cloud_deployment to path
            cloud_path = Path(__file__).parent.parent.parent.parent / "cloud_deployment"
            if str(cloud_path) not in sys.path:
                sys.path.append(str(cloud_path))
            
            from api.providers.gcp_stt_provider import GCPSTTProvider
            
            # Try to initialize the provider
            provider = GCPSTTProvider()
            
            # Test if the provider is available
            if provider.is_available():
                logger.info("Google Cloud STT provider initialized successfully")
                return "available"
            else:
                logger.warning("Google Cloud STT provider failed availability check")
                return "error"
                
        except ImportError as e:
            logger.debug(f"Google Cloud STT libraries not installed: {e}")
            return "error"
        except Exception as e:
            logger.debug(f"Google Cloud STT error: {e}")
            return "error"
    
    def _check_google_cloud_tts(self) -> str:
        """Check if Google Cloud TTS is available."""
        try:
            # Import the existing cloud provider
            import sys
            from pathlib import Path
            
            # Add cloud_deployment to path
            cloud_path = Path(__file__).parent.parent.parent.parent / "cloud_deployment"
            if str(cloud_path) not in sys.path:
                sys.path.append(str(cloud_path))
            
            from api.providers.gcp_tts_provider import GCPTTSProvider
            
            # Try to initialize the provider
            provider = GCPTTSProvider()
            
            # Test if the provider is available
            if provider.is_available():
                logger.info("Google Cloud TTS provider initialized successfully")
                return "available"
            else:
                logger.warning("Google Cloud TTS provider failed availability check")
                return "error"
                
        except ImportError as e:
            logger.debug(f"Google Cloud TTS libraries not installed: {e}")
            return "error"
        except Exception as e:
            logger.debug(f"Google Cloud TTS error: {e}")
            return "error"
    
    def _check_gemini_availability(self) -> str:
        """Check if Gemini LLM is available."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            return "error"
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Try to create a model instance
            model = genai.GenerativeModel('gemini-2.0-flash')
            return "available"
        except Exception:
            return "error"
