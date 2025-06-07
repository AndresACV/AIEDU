"""
Google Cloud Text-to-Speech Provider
Provides cloud-based text-to-speech with neural voices and enhanced quality.
"""

import os
import logging
import hashlib
import tempfile
import json
from typing import Optional, Dict, Any, List
from google.cloud import texttospeech
from google.oauth2 import service_account
import io

logger = logging.getLogger(__name__)

class GCPTTSProvider:
    """
    Google Cloud Text-to-Speech provider for high-quality speech synthesis.
    
    Features:
    - Neural voice models with natural speech
    - Multi-language support (English, Spanish)
    - Voice parameter fine-tuning
    - Audio output optimization
    - Response caching for efficiency
    """
    
    def __init__(self, 
                 credentials_path: Optional[str] = None,
                 default_language: str = "en-US",
                 default_voice_name: Optional[str] = None,
                 audio_format: str = "mp3",
                 speaking_rate: float = 1.0,
                 pitch: float = 0.0,
                 cache_enabled: bool = True):
        """
        Initialize Google Cloud Text-to-Speech provider.
        
        Args:
            credentials_path: Path to Google Cloud service account JSON
            default_language: Default language code (e.g., "en-US", "es-ES")
            default_voice_name: Default voice name (None for auto-selection)
            audio_format: Audio output format ("mp3", "wav", "ogg")
            speaking_rate: Speech rate (0.25 to 4.0, 1.0 = normal)
            pitch: Voice pitch (-20.0 to 20.0, 0.0 = normal)
            cache_enabled: Enable audio caching for repeated requests
        """
        self.default_language = default_language
        self.default_voice_name = default_voice_name
        self.audio_format = audio_format
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self.cache_enabled = cache_enabled
        self.audio_cache = {} if cache_enabled else None
        
        # Set up credentials with intelligent handling
        credentials = self._setup_credentials(credentials_path)
        
        try:
            if credentials:
                self.client = texttospeech.TextToSpeechClient(credentials=credentials)
            else:
                self.client = texttospeech.TextToSpeechClient()
            logger.info("Google Cloud Text-to-Speech client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud TTS client: {e}")
            raise
    
    def synthesize_speech(self, 
                         text: str, 
                         language_code: Optional[str] = None,
                         voice_name: Optional[str] = None,
                         speaking_rate: Optional[float] = None,
                         pitch: Optional[float] = None,
                         volume_gain_db: Optional[float] = None) -> Dict[str, Any]:
        """
        Convert text to speech using Google Cloud Text-to-Speech.
        
        Args:
            text: Text to convert to speech
            language_code: Language code (overrides default if provided)
            voice_name: Voice name (overrides default if provided)
            speaking_rate: Speech rate (overrides default if provided)
            pitch: Voice pitch (overrides default if provided)
            volume_gain_db: Volume adjustment in dB
            
        Returns:
            Dict containing:
            - audio_content: Generated audio data as bytes
            - audio_format: Audio format used
            - language_used: Language code used
            - voice_used: Voice name used
            - cached: Whether result was retrieved from cache
        """
        try:
            # Use provided parameters or defaults
            lang_code = language_code or self.default_language
            voice = voice_name or self._get_best_voice(lang_code)
            rate = speaking_rate if speaking_rate is not None else self.speaking_rate
            voice_pitch = pitch if pitch is not None else self.pitch
            
            # Check cache first
            cache_key = self._generate_cache_key(text, lang_code, voice, rate, voice_pitch)
            if self.cache_enabled and cache_key in self.audio_cache:
                logger.debug(f"Retrieved TTS from cache for text: {text[:50]}...")
                cached_result = self.audio_cache[cache_key]
                cached_result["cached"] = True
                return cached_result
            
            # Create synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Configure voice
            voice_selection = texttospeech.VoiceSelectionParams(
                language_code=lang_code,
                name=voice
            )
            
            # Configure audio
            audio_config = texttospeech.AudioConfig(
                audio_encoding=self._get_audio_encoding(),
                speaking_rate=rate,
                pitch=voice_pitch
            )
            
            # Add volume adjustment if specified
            if volume_gain_db is not None:
                audio_config.volume_gain_db = volume_gain_db
            
            # Perform synthesis
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_selection,
                audio_config=audio_config
            )
            
            result = {
                "audio_content": response.audio_content,
                "audio_format": self.audio_format,
                "language_used": lang_code,
                "voice_used": voice,
                "cached": False,
                "status": "success"
            }
            
            # Cache the result
            if self.cache_enabled:
                self.audio_cache[cache_key] = result.copy()
                logger.debug(f"Cached TTS result for text: {text[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during Google Cloud TTS synthesis: {e}")
            return {
                "audio_content": b"",
                "audio_format": self.audio_format,
                "language_used": lang_code,
                "voice_used": voice or "unknown",
                "cached": False,
                "status": "error",
                "error": str(e)
            }
    
    def save_audio_to_file(self, 
                          text: str, 
                          output_path: str,
                          language_code: Optional[str] = None,
                          voice_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize speech and save directly to file.
        
        Args:
            text: Text to convert to speech
            output_path: Path where to save the audio file
            language_code: Language code
            voice_name: Voice name
            
        Returns:
            Dict containing synthesis results and file path
        """
        try:
            result = self.synthesize_speech(
                text=text,
                language_code=language_code,
                voice_name=voice_name
            )
            
            if result["status"] == "success" and result["audio_content"]:
                # Save audio to file
                with open(output_path, 'wb') as audio_file:
                    audio_file.write(result["audio_content"])
                
                result["file_path"] = output_path
                result["file_size"] = len(result["audio_content"])
                logger.info(f"Audio saved to {output_path} ({result['file_size']} bytes)")
                
            return result
            
        except Exception as e:
            logger.error(f"Error saving TTS audio to file {output_path}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "file_path": output_path
            }
    
    def get_available_voices(self, language_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available voices for a language.
        
        Args:
            language_code: Language code (None for all languages)
            
        Returns:
            List of voice information dictionaries
        """
        try:
            response = self.client.list_voices()
            voices = []
            
            for voice in response.voices:
                # Filter by language if specified
                if language_code and language_code not in voice.language_codes:
                    continue
                
                voices.append({
                    "name": voice.name,
                    "language_codes": list(voice.language_codes),
                    "gender": voice.ssml_gender.name,
                    "natural_sample_rate": voice.natural_sample_rate_hertz
                })
            
            return voices
            
        except Exception as e:
            logger.error(f"Error retrieving available voices: {e}")
            return []
    
    def get_best_voices_for_language(self, language_code: str) -> List[str]:
        """
        Get recommended voice names for a specific language.
        
        Args:
            language_code: Language code (e.g., "en-US", "es-ES")
            
        Returns:
            List of recommended voice names
        """
        # Recommended neural voices for better quality
        voice_recommendations = {
            "en-US": [
                "en-US-Neural2-A",  # Female, natural
                "en-US-Neural2-C",  # Female, natural  
                "en-US-Neural2-D",  # Male, natural
                "en-US-Neural2-F",  # Female, natural
                "en-US-Neural2-H",  # Female, natural
                "en-US-Neural2-I",  # Male, natural
                "en-US-Neural2-J",  # Male, natural
            ],
            "en-GB": [
                "en-GB-Neural2-A",  # Female
                "en-GB-Neural2-B",  # Male
                "en-GB-Neural2-C",  # Female
                "en-GB-Neural2-D",  # Male
            ],
            "es-ES": [
                "es-ES-Neural2-A",  # Female
                "es-ES-Neural2-B",  # Male
                "es-ES-Neural2-C",  # Female
                "es-ES-Neural2-D",  # Female
                "es-ES-Neural2-E",  # Female
                "es-ES-Neural2-F",  # Male
            ],
            "es-MX": [
                "es-US-Neural2-A",  # Female
                "es-US-Neural2-B",  # Male
                "es-US-Neural2-C",  # Male
            ],
            "fr-FR": [
                "fr-FR-Neural2-A",  # Female
                "fr-FR-Neural2-B",  # Male
                "fr-FR-Neural2-C",  # Female
                "fr-FR-Neural2-D",  # Male
            ]
        }
        
        return voice_recommendations.get(language_code, [])
    
    def _get_best_voice(self, language_code: str) -> str:
        """
        Get the best recommended voice for a language.
        
        Args:
            language_code: Language code
            
        Returns:
            Best voice name for the language
        """
        if self.default_voice_name:
            return self.default_voice_name
        
        recommended_voices = self.get_best_voices_for_language(language_code)
        if recommended_voices:
            return recommended_voices[0]  # Return the first (best) recommendation
        
        # Fallback to standard voices
        standard_voices = {
            "en-US": "en-US-Standard-A",
            "en-GB": "en-GB-Standard-A", 
            "es-ES": "es-ES-Standard-A",
            "es-MX": "es-US-Standard-A",
            "fr-FR": "fr-FR-Standard-A"
        }
        
        return standard_voices.get(language_code, "en-US-Standard-A")
    
    def _get_audio_encoding(self) -> texttospeech.AudioEncoding:
        """
        Get Google Cloud TTS audio encoding based on format.
        
        Returns:
            AudioEncoding enum value
        """
        encoding_mapping = {
            "mp3": texttospeech.AudioEncoding.MP3,
            "wav": texttospeech.AudioEncoding.LINEAR16,
            "ogg": texttospeech.AudioEncoding.OGG_OPUS
        }
        
        return encoding_mapping.get(
            self.audio_format.lower(),
            texttospeech.AudioEncoding.MP3
        )
    
    def _generate_cache_key(self, 
                           text: str, 
                           language_code: str, 
                           voice_name: str,
                           speaking_rate: float,
                           pitch: float) -> str:
        """
        Generate a cache key for TTS requests.
        
        Args:
            text: Text content
            language_code: Language code
            voice_name: Voice name
            speaking_rate: Speaking rate
            pitch: Voice pitch
            
        Returns:
            MD5 hash as cache key
        """
        cache_string = f"{text}|{language_code}|{voice_name}|{speaking_rate}|{pitch}|{self.audio_format}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def clear_cache(self) -> int:
        """
        Clear the audio cache.
        
        Returns:
            Number of cached items cleared
        """
        if not self.cache_enabled or not self.audio_cache:
            return 0
        
        cache_size = len(self.audio_cache)
        self.audio_cache.clear()
        logger.info(f"Cleared TTS cache ({cache_size} items)")
        return cache_size
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.
        
        Returns:
            Dict with cache statistics
        """
        if not self.cache_enabled:
            return {"enabled": False}
        
        total_size = sum(
            len(item.get("audio_content", b"")) 
            for item in self.audio_cache.values()
        )
        
        return {
            "enabled": True,
            "items": len(self.audio_cache),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }
    
    def is_available(self) -> bool:
        """
        Check if Google Cloud Text-to-Speech service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            # Test with minimal synthesis request
            synthesis_input = texttospeech.SynthesisInput(text="test")
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.default_language,
                name=self._get_best_voice(self.default_language)
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=self._get_audio_encoding()
            )
            
            # This should succeed if service is available
            self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Google Cloud TTS service unavailable: {e}")
            return False
    
    def _setup_credentials(self, credentials_path: Optional[str]) -> Optional[service_account.Credentials]:
        """
        Set up Google Cloud credentials with intelligent handling of file paths and JSON content.
        
        Args:
            credentials_path: Path to service account file or None
            
        Returns:
            Credentials object or None for default authentication
        """
        try:
            # Option 1: Explicit credentials path provided
            if credentials_path:
                if os.path.isfile(credentials_path):
                    logger.info(f"Using service account file: {credentials_path}")
                    return service_account.Credentials.from_service_account_file(credentials_path)
                else:
                    logger.warning(f"Credentials file not found: {credentials_path}")
            
            # Option 2: Check environment variable for JSON content
            creds_env = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if creds_env:
                logger.debug(f"Found GOOGLE_APPLICATION_CREDENTIALS (length: {len(creds_env)})")
                
                # Check if it's JSON content (starts with '{')
                if creds_env.strip().startswith('{') and creds_env.strip().endswith('}'):
                    logger.info("Detected service account JSON in environment variable")
                    try:
                        creds_info = json.loads(creds_env.strip())
                        logger.info("Successfully parsed service account JSON")
                        return service_account.Credentials.from_service_account_info(creds_info)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS: {e}")
                        logger.debug(f"JSON content preview: {creds_env[:100]}...")
                        return None
                
                # Check if it's a file path
                elif os.path.isfile(creds_env):
                    logger.info(f"Using service account file from environment: {creds_env}")
                    return service_account.Credentials.from_service_account_file(creds_env)
                else:
                    logger.warning(f"GOOGLE_APPLICATION_CREDENTIALS is not valid JSON or file path")
                    logger.debug(f"Content preview: {creds_env[:50]}...")
            
            # Option 3: Fall back to Application Default Credentials
            logger.info("Using Application Default Credentials (ADC)")
            return None  # Let the client use ADC
            
        except Exception as e:
            logger.error(f"Error setting up credentials: {e}")
            return None 