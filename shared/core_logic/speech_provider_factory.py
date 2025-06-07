"""
Speech Provider Factory
Manages creation and selection of Speech-to-Text (STT) and Text-to-Speech (TTS) providers.
Supports both local (privacy-focused) and cloud (performance-focused) providers.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)

class SpeechMode(Enum):
    """Speech processing mode preferences."""
    PRIVACY = "privacy"          # Prefer local providers
    PERFORMANCE = "performance"  # Prefer cloud providers  
    COST_CONSCIOUS = "cost"      # Prefer free/local providers
    AUTO = "auto"               # Intelligent selection

class SpeechProviderFactory:
    """
    Factory for creating and managing speech providers (STT and TTS).
    
    Supports:
    - Local providers: Vosk (STT), espeak/pyttsx3 (TTS)
    - Cloud providers: Google Cloud Speech services
    - Intelligent provider selection based on mode and availability
    - Automatic fallback between local and cloud
    """
    
    def __init__(self, 
                 mode: SpeechMode = SpeechMode.AUTO,
                 credentials_path: Optional[str] = None):
        """
        Initialize Speech Provider Factory.
        
        Args:
            mode: Default speech processing mode
            credentials_path: Path to Google Cloud service account JSON
        """
        self.mode = mode
        self.credentials_path = credentials_path
        self._stt_providers_cache = {}
        self._tts_providers_cache = {}
        self._availability_cache = {}
        
        logger.info(f"Speech Provider Factory initialized with mode: {mode.value}")
    
    def get_stt_provider(self, 
                        mode: Optional[SpeechMode] = None,
                        language: str = "en-US") -> Optional[Any]:
        """
        Get Speech-to-Text provider based on mode and availability.
        
        Args:
            mode: Override default mode for this request
            language: Target language for STT
            
        Returns:
            STT provider instance or None if unavailable
        """
        effective_mode = mode or self.mode
        provider_key = f"stt_{effective_mode.value}_{language}"
        
        # Check cache first
        if provider_key in self._stt_providers_cache:
            return self._stt_providers_cache[provider_key]
        
        # Determine provider preference order
        if effective_mode == SpeechMode.PRIVACY:
            providers = ["vosk", "gcp_stt"]
        elif effective_mode == SpeechMode.PERFORMANCE:
            providers = ["gcp_stt", "vosk"]
        elif effective_mode == SpeechMode.COST_CONSCIOUS:
            providers = ["vosk", "gcp_stt"]  # Prefer free local
        else:  # AUTO mode
            providers = self._auto_select_stt_providers(language)
        
        # Try providers in order
        for provider_type in providers:
            try:
                provider = self._create_stt_provider(provider_type, language)
                if provider and self._is_stt_available(provider):
                    self._stt_providers_cache[provider_key] = provider
                    logger.info(f"Selected STT provider: {provider_type} for language: {language}")
                    return provider
            except Exception as e:
                logger.warning(f"Failed to create {provider_type} STT provider: {e}")
                continue
        
        logger.error(f"No available STT providers for language: {language}")
        return None
    
    def get_tts_provider(self, 
                        mode: Optional[SpeechMode] = None,
                        language: str = "en-US") -> Optional[Any]:
        """
        Get Text-to-Speech provider based on mode and availability.
        
        Args:
            mode: Override default mode for this request
            language: Target language for TTS
            
        Returns:
            TTS provider instance or None if unavailable
        """
        effective_mode = mode or self.mode
        provider_key = f"tts_{effective_mode.value}_{language}"
        
        # Check cache first
        if provider_key in self._tts_providers_cache:
            return self._tts_providers_cache[provider_key]
        
        # Determine provider preference order
        if effective_mode == SpeechMode.PRIVACY:
            providers = ["local_tts", "gcp_tts"]
        elif effective_mode == SpeechMode.PERFORMANCE:
            providers = ["gcp_tts", "local_tts"]
        elif effective_mode == SpeechMode.COST_CONSCIOUS:
            providers = ["local_tts", "gcp_tts"]  # Prefer free local
        else:  # AUTO mode
            providers = self._auto_select_tts_providers(language)
        
        # Try providers in order
        for provider_type in providers:
            try:
                provider = self._create_tts_provider(provider_type, language)
                if provider and self._is_tts_available(provider):
                    self._tts_providers_cache[provider_key] = provider
                    logger.info(f"Selected TTS provider: {provider_type} for language: {language}")
                    return provider
            except Exception as e:
                logger.warning(f"Failed to create {provider_type} TTS provider: {e}")
                continue
        
        logger.error(f"No available TTS providers for language: {language}")
        return None
    
    def _create_stt_provider(self, provider_type: str, language: str) -> Optional[Any]:
        """
        Create STT provider instance.
        
        Args:
            provider_type: Type of provider ("vosk", "gcp_stt")
            language: Target language
            
        Returns:
            Provider instance or None
        """
        if provider_type == "vosk":
            return self._create_vosk_provider(language)
        elif provider_type == "gcp_stt":
            return self._create_gcp_stt_provider(language)
        else:
            logger.error(f"Unknown STT provider type: {provider_type}")
            return None
    
    def _create_tts_provider(self, provider_type: str, language: str) -> Optional[Any]:
        """
        Create TTS provider instance.
        
        Args:
            provider_type: Type of provider ("local_tts", "gcp_tts")
            language: Target language
            
        Returns:
            Provider instance or None
        """
        if provider_type == "local_tts":
            return self._create_local_tts_provider(language)
        elif provider_type == "gcp_tts":
            return self._create_gcp_tts_provider(language)
        else:
            logger.error(f"Unknown TTS provider type: {provider_type}")
            return None
    
    def _create_vosk_provider(self, language: str) -> Optional[Any]:
        """Create Vosk STT provider."""
        try:
            # Import here to avoid dependency issues
            import vosk
            import json
            
            # Get absolute path to models directory from Flask app location
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_file_dir, '..', '..')
            models_base_path = os.path.join(project_root, 'local_deployment', 'web_app', 'models')
            
            # Map language codes to Vosk model paths
            model_paths = {
                "en-US": os.path.join(models_base_path, "vosk-model-small-en-us-0.15"),
                "en": os.path.join(models_base_path, "vosk-model-small-en-us-0.15"),
                "es-ES": os.path.join(models_base_path, "vosk-model-small-es-0.42"),
                "es": os.path.join(models_base_path, "vosk-model-small-es-0.42")
            }
            
            model_path = model_paths.get(language, model_paths.get("en-US"))
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"❌ Model {os.path.basename(model_path)} not found at {model_path}")
                logger.info(f"🔍 Checking available models in: {models_base_path}")
                if os.path.exists(models_base_path):
                    available_models = [d for d in os.listdir(models_base_path) 
                                      if os.path.isdir(os.path.join(models_base_path, d))]
                    logger.info(f"📋 Available models: {available_models}")
                else:
                    logger.warning(f"❌ Models directory not found: {models_base_path}")
                return None
            
            # Create simple wrapper for consistent interface
            class VoskSTTProvider:
                def __init__(self, model_path: str, language: str):
                    self.model = vosk.Model(model_path)
                    self.language = language
                    
                def transcribe_audio(self, audio_data: bytes, **kwargs) -> Dict[str, Any]:
                    try:
                        recognizer = vosk.KaldiRecognizer(self.model, 16000)
                        recognizer.AcceptWaveform(audio_data)
                        result = json.loads(recognizer.FinalResult())
                        
                        return {
                            "transcript": result.get("text", ""),
                            "confidence": 0.8,  # Vosk doesn't provide confidence
                            "language_detected": self.language,
                            "status": "success"
                        }
                    except Exception as e:
                        return {
                            "transcript": "",
                            "confidence": 0.0,
                            "language_detected": self.language,
                            "status": "error",
                            "error": str(e)
                        }
                
                def is_available(self) -> bool:
                    return True
            
            return VoskSTTProvider(model_path, language)
            
        except Exception as e:
            logger.error(f"Failed to create Vosk provider: {e}")
            return None
    
    def _create_gcp_stt_provider(self, language: str) -> Optional[Any]:
        """Create Google Cloud STT provider."""
        try:
            # Import the GCP STT provider
            from cloud_deployment.api.providers.gcp_stt_provider import GCPSTTProvider
            
            return GCPSTTProvider(
                credentials_path=self.credentials_path,
                default_language=language
            )
            
        except ImportError as e:
            logger.warning(f"Google Cloud STT not available (missing dependencies): {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create Google Cloud STT provider: {e}")
            return None
    
    def _create_local_tts_provider(self, language: str) -> Optional[Any]:
        """Create local TTS provider (espeak/pyttsx3)."""
        try:
            import platform
            
            # Create wrapper for local TTS
            class LocalTTSProvider:
                def __init__(self, language: str):
                    self.language = language
                    self.platform = platform.system()
                    
                def synthesize_speech(self, text: str, **kwargs) -> Dict[str, Any]:
                    try:
                        if self.platform == "Linux":
                            # Use espeak on Linux/WSL
                            return self._espeak_synthesis(text)
                        else:
                            # Use pyttsx3 on Windows/Mac
                            return self._pyttsx3_synthesis(text)
                    except Exception as e:
                        return {
                            "audio_content": b"",
                            "status": "error",
                            "error": str(e)
                        }
                
                def _espeak_synthesis(self, text: str) -> Dict[str, Any]:
                    import subprocess
                    import tempfile
                    
                    # Create temporary file for output
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Run espeak
                    lang_map = {"en-US": "en", "es-ES": "es", "es": "es", "en": "en"}
                    espeak_lang = lang_map.get(self.language, "en")
                    
                    subprocess.run([
                        "espeak", "-v", espeak_lang, "-s", "150", "-w", temp_path, text
                    ], check=True, capture_output=True)
                    
                    # Read audio data
                    with open(temp_path, 'rb') as audio_file:
                        audio_content = audio_file.read()
                    
                    os.unlink(temp_path)
                    
                    return {
                        "audio_content": audio_content,
                        "audio_format": "wav",
                        "language_used": self.language,
                        "status": "success"
                    }
                
                def _pyttsx3_synthesis(self, text: str) -> Dict[str, Any]:
                    import pyttsx3
                    import tempfile
                    
                    engine = pyttsx3.init()
                    
                    # Configure voice if available
                    voices = engine.getProperty('voices')
                    if voices:
                        # Simple language selection
                        if self.language.startswith('es'):
                            for voice in voices:
                                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                                    engine.setProperty('voice', voice.id)
                                    break
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    engine.save_to_file(text, temp_path)
                    engine.runAndWait()
                    
                    # Read audio data
                    with open(temp_path, 'rb') as audio_file:
                        audio_content = audio_file.read()
                    
                    os.unlink(temp_path)
                    
                    return {
                        "audio_content": audio_content,
                        "audio_format": "wav",
                        "language_used": self.language,
                        "status": "success"
                    }
                
                def is_available(self) -> bool:
                    try:
                        if self.platform == "Linux":
                            import subprocess
                            result = subprocess.run(["which", "espeak"], capture_output=True)
                            return result.returncode == 0
                        else:
                            import pyttsx3
                            return True
                    except:
                        return False
            
            return LocalTTSProvider(language)
            
        except Exception as e:
            logger.error(f"Failed to create local TTS provider: {e}")
            return None
    
    def _create_gcp_tts_provider(self, language: str) -> Optional[Any]:
        """Create Google Cloud TTS provider."""
        try:
            # Import the GCP TTS provider
            from cloud_deployment.api.providers.gcp_tts_provider import GCPTTSProvider
            
            return GCPTTSProvider(
                credentials_path=self.credentials_path,
                default_language=language
            )
            
        except ImportError as e:
            logger.warning(f"Google Cloud TTS not available (missing dependencies): {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create Google Cloud TTS provider: {e}")
            return None
    
    def _auto_select_stt_providers(self, language: str) -> list[str]:
        """Automatically select STT provider order based on intelligent criteria."""
        # For educational content, prioritize accuracy (cloud) but maintain privacy option
        if self._is_cloud_environment():
            return ["gcp_stt", "vosk"]  # Cloud environment - prefer performance
        else:
            return ["vosk", "gcp_stt"]  # Local environment - prefer privacy
    
    def _auto_select_tts_providers(self, language: str) -> list[str]:
        """Automatically select TTS provider order based on intelligent criteria."""
        # For TTS, neural voices provide much better quality for educational content
        if self._is_cloud_environment():
            return ["gcp_tts", "local_tts"]  # Cloud environment - prefer quality
        else:
            return ["local_tts", "gcp_tts"]  # Local environment - prefer privacy
    
    def _is_cloud_environment(self) -> bool:
        """Detect if running in cloud environment."""
        # Check for cloud-specific environment variables or deployment indicators
        cloud_indicators = [
            "VERCEL", "NETLIFY", "HEROKU", "GOOGLE_CLOUD_PROJECT",
            "AWS_LAMBDA_FUNCTION_NAME", "AZURE_FUNCTIONS_ENVIRONMENT"
        ]
        return any(os.getenv(indicator) for indicator in cloud_indicators)
    
    def _is_stt_available(self, provider: Any) -> bool:
        """Check if STT provider is available and working."""
        try:
            return hasattr(provider, 'is_available') and provider.is_available()
        except Exception as e:
            logger.error(f"Error checking STT provider availability: {e}")
            return False
    
    def _is_tts_available(self, provider: Any) -> bool:
        """Check if TTS provider is available and working."""
        try:
            return hasattr(provider, 'is_available') and provider.is_available()
        except Exception as e:
            logger.error(f"Error checking TTS provider availability: {e}")
            return False
    
    def get_provider_status(self) -> Dict[str, Any]:
        """
        Get status of all speech providers.
        
        Returns:
            Dict with provider availability and performance info
        """
        status = {
            "stt_providers": {},
            "tts_providers": {},
            "current_mode": self.mode.value
        }
        
        # Check STT providers
        for provider_type in ["vosk", "gcp_stt"]:
            try:
                provider = self._create_stt_provider(provider_type, "en-US")
                status["stt_providers"][provider_type] = {
                    "available": provider is not None and self._is_stt_available(provider),
                    "languages": self._get_provider_languages(provider_type, "stt")
                }
            except Exception as e:
                status["stt_providers"][provider_type] = {
                    "available": False,
                    "error": str(e)
                }
        
        # Check TTS providers
        for provider_type in ["local_tts", "gcp_tts"]:
            try:
                provider = self._create_tts_provider(provider_type, "en-US")
                status["tts_providers"][provider_type] = {
                    "available": provider is not None and self._is_tts_available(provider),
                    "languages": self._get_provider_languages(provider_type, "tts")
                }
            except Exception as e:
                status["tts_providers"][provider_type] = {
                    "available": False,
                    "error": str(e)
                }
        
        return status
    
    def _get_provider_languages(self, provider_type: str, service_type: str) -> list[str]:
        """Get supported languages for a provider."""
        # Basic language support mapping
        if provider_type in ["vosk"]:
            return ["en-US", "es-ES"]
        elif provider_type in ["gcp_stt", "gcp_tts"]:
            return ["en-US", "en-GB", "es-ES", "es-MX", "fr-FR", "de-DE", "it-IT"]
        elif provider_type == "local_tts":
            return ["en-US", "es-ES"]
        else:
            return []
    
    def clear_cache(self):
        """Clear all provider caches."""
        self._stt_providers_cache.clear()
        self._tts_providers_cache.clear()
        self._availability_cache.clear()
        logger.info("Speech provider caches cleared") 