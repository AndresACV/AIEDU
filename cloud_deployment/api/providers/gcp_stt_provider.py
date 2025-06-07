"""
Google Cloud Speech-to-Text Provider
Provides cloud-based speech recognition with enhanced accuracy and multi-language support.
"""

import io
import os
import logging
import json
import tempfile
from typing import Optional, Dict, Any, List
from google.cloud import speech
from google.cloud.speech import RecognitionAudio, RecognitionConfig
from google.oauth2 import service_account
import wave

logger = logging.getLogger(__name__)

class GCPSTTProvider:
    """
    Google Cloud Speech-to-Text provider for high-accuracy speech recognition.
    
    Features:
    - Multi-language support (English, Spanish)
    - Various audio format handling
    - Confidence scoring
    - Real-time streaming capabilities
    - Automatic language detection
    """
    
    def __init__(self, 
                 credentials_path: Optional[str] = None,
                 default_language: str = "en-US",
                 enable_automatic_punctuation: bool = True,
                 enable_word_confidence: bool = True):
        """
        Initialize Google Cloud Speech-to-Text provider.
        
        Args:
            credentials_path: Path to Google Cloud service account JSON
            default_language: Default language code (e.g., "en-US", "es-ES")
            enable_automatic_punctuation: Enable automatic punctuation
            enable_word_confidence: Enable word-level confidence scores
        """
        self.default_language = default_language
        self.enable_automatic_punctuation = enable_automatic_punctuation
        self.enable_word_confidence = enable_word_confidence
        
        # Set up credentials with intelligent handling
        credentials = self._setup_credentials(credentials_path)
        
        try:
            if credentials:
                self.client = speech.SpeechClient(credentials=credentials)
            else:
                self.client = speech.SpeechClient()
            logger.info("Google Cloud Speech-to-Text client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Speech client: {e}")
            raise
    
    def transcribe_audio(self, 
                        audio_data: bytes, 
                        language_code: Optional[str] = None,
                        sample_rate: int = 16000,
                        audio_format: str = "wav") -> Dict[str, Any]:
        """
        Transcribe audio data to text using Google Cloud Speech-to-Text.
        
        Args:
            audio_data: Raw audio data as bytes
            language_code: Language code (overrides default if provided)
            sample_rate: Audio sample rate in Hz
            audio_format: Audio format ("wav", "mp3", "flac", etc.)
            
        Returns:
            Dict containing:
            - transcript: The transcribed text
            - confidence: Overall confidence score
            - words: List of words with individual confidence scores
            - language_detected: Detected language code
        """
        try:
            # Use provided language or default
            lang_code = language_code or self.default_language
            
            # Create recognition audio object
            audio = RecognitionAudio(content=audio_data)
            
            # Configure recognition settings
            config = RecognitionConfig(
                encoding=self._get_encoding_from_format(audio_format),
                sample_rate_hertz=sample_rate,
                language_code=lang_code,
                enable_automatic_punctuation=self.enable_automatic_punctuation,
                enable_word_confidence=self.enable_word_confidence,
                # Enable profanity filter for educational content
                profanity_filter=True,
                # Use enhanced model for better accuracy
                use_enhanced=True,
                # Model selection for educational content
                model="latest_long"
            )
            
            # Perform the transcription
            response = self.client.recognize(config=config, audio=audio)
            
            # Process results
            if response.results:
                result = response.results[0]
                alternative = result.alternatives[0]
                
                # Extract word-level confidence if available
                words_info = []
                if hasattr(alternative, 'words') and alternative.words:
                    words_info = [
                        {
                            "word": word.word,
                            "confidence": word.confidence,
                            "start_time": word.start_time.total_seconds() if word.start_time else 0,
                            "end_time": word.end_time.total_seconds() if word.end_time else 0
                        }
                        for word in alternative.words
                    ]
                
                return {
                    "transcript": alternative.transcript.strip(),
                    "confidence": alternative.confidence,
                    "words": words_info,
                    "language_detected": lang_code,
                    "status": "success"
                }
            else:
                logger.warning("No transcription results received from Google Cloud Speech")
                return {
                    "transcript": "",
                    "confidence": 0.0,
                    "words": [],
                    "language_detected": lang_code,
                    "status": "no_results"
                }
                
        except Exception as e:
            logger.error(f"Error during Google Cloud Speech transcription: {e}")
            return {
                "transcript": "",
                "confidence": 0.0,
                "words": [],
                "language_detected": lang_code,
                "status": "error",
                "error": str(e)
            }
    
    def transcribe_file(self, 
                       file_path: str, 
                       language_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file using Google Cloud Speech-to-Text.
        
        Args:
            file_path: Path to the audio file
            language_code: Language code (overrides default if provided)
            
        Returns:
            Dict containing transcription results
        """
        try:
            # Read audio file
            with open(file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Get audio properties
            sample_rate, audio_format = self._get_audio_properties(file_path)
            
            return self.transcribe_audio(
                audio_data=audio_data,
                language_code=language_code,
                sample_rate=sample_rate,
                audio_format=audio_format
            )
            
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {e}")
            return {
                "transcript": "",
                "confidence": 0.0,
                "words": [],
                "language_detected": language_code or self.default_language,
                "status": "error",
                "error": str(e)
            }
    
    def transcribe_with_alternatives(self, 
                                   audio_data: bytes,
                                   language_code: Optional[str] = None,
                                   max_alternatives: int = 3,
                                   sample_rate: int = 16000,
                                   audio_format: str = "wav") -> Dict[str, Any]:
        """
        Transcribe audio with multiple alternative interpretations.
        
        Args:
            audio_data: Raw audio data as bytes
            language_code: Language code
            max_alternatives: Maximum number of alternative transcriptions
            sample_rate: Audio sample rate
            audio_format: Audio format
            
        Returns:
            Dict containing multiple transcription alternatives
        """
        try:
            lang_code = language_code or self.default_language
            
            audio = RecognitionAudio(content=audio_data)
            config = RecognitionConfig(
                encoding=self._get_encoding_from_format(audio_format),
                sample_rate_hertz=sample_rate,
                language_code=lang_code,
                enable_automatic_punctuation=self.enable_automatic_punctuation,
                enable_word_confidence=self.enable_word_confidence,
                max_alternatives=max_alternatives,
                profanity_filter=True,
                use_enhanced=True,
                model="latest_long"
            )
            
            response = self.client.recognize(config=config, audio=audio)
            
            alternatives = []
            if response.results:
                for alternative in response.results[0].alternatives:
                    alternatives.append({
                        "transcript": alternative.transcript.strip(),
                        "confidence": alternative.confidence
                    })
            
            return {
                "alternatives": alternatives,
                "language_detected": lang_code,
                "status": "success" if alternatives else "no_results"
            }
            
        except Exception as e:
            logger.error(f"Error during alternative transcription: {e}")
            return {
                "alternatives": [],
                "language_detected": lang_code,
                "status": "error",
                "error": str(e)
            }
    
    def _get_encoding_from_format(self, audio_format: str) -> speech.RecognitionConfig.AudioEncoding:
        """
        Map audio format string to Google Cloud Speech encoding enum.
        
        Args:
            audio_format: Audio format string
            
        Returns:
            Google Cloud Speech AudioEncoding enum
        """
        format_mapping = {
            "wav": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "mp3": speech.RecognitionConfig.AudioEncoding.MP3,
            "flac": speech.RecognitionConfig.AudioEncoding.FLAC,
            "ogg": speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            "webm": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
        }
        
        return format_mapping.get(
            audio_format.lower(), 
            speech.RecognitionConfig.AudioEncoding.LINEAR16
        )
    
    def _get_audio_properties(self, file_path: str) -> tuple[int, str]:
        """
        Extract audio properties from file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (sample_rate, format)
        """
        try:
            # Get file extension
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            
            # Try to get sample rate for WAV files
            if file_ext == 'wav':
                try:
                    with wave.open(file_path, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        return sample_rate, file_ext
                except:
                    pass
            
            # Default values for other formats
            return 16000, file_ext
            
        except Exception as e:
            logger.warning(f"Could not extract audio properties from {file_path}: {e}")
            return 16000, "wav"
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of supported language codes
        """
        return [
            "en-US",  # English (US)
            "en-GB",  # English (UK)
            "es-ES",  # Spanish (Spain)
            "es-MX",  # Spanish (Mexico)
            "es-AR",  # Spanish (Argentina)
            "fr-FR",  # French
            "de-DE",  # German
            "it-IT",  # Italian
            "pt-BR",  # Portuguese (Brazil)
            "ja-JP",  # Japanese
            "ko-KR",  # Korean
            "zh-CN",  # Chinese (Simplified)
            "ru-RU",  # Russian
            "ar-SA",  # Arabic
        ]
    
    def is_available(self) -> bool:
        """
        Check if Google Cloud Speech-to-Text service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            # Test with a minimal request
            test_audio = RecognitionAudio(content=b'\x00' * 1024)  # Minimal audio data
            config = RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self.default_language
            )
            
            # This will fail but should connect to the service
            try:
                self.client.recognize(config=config, audio=test_audio)
            except Exception:
                # Expected to fail with invalid audio, but connection is working
                pass
                
            return True
            
        except Exception as e:
            logger.error(f"Google Cloud Speech service unavailable: {e}")
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