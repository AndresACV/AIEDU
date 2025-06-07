"""
Speech Service Module
Provides text-to-speech and speech-to-text functionality for the FastAPI backend.
Migrated from Flask hybrid_speech.py with enhanced FastAPI integration.
Supports both local (Vosk, espeak) and cloud (Google Cloud) providers.
"""

import os
import sys
import logging
import tempfile
import uuid
import subprocess
import platform
import json
import wave
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Add project paths for Vosk model access
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
VOSK_MODELS_PATH = PROJECT_ROOT / "local_deployment" / "web_app" / "models"

# Add cloud_deployment to path for Google Cloud providers
CLOUD_PATH = PROJECT_ROOT / "cloud_deployment"
if str(CLOUD_PATH) not in sys.path:
    sys.path.append(str(CLOUD_PATH))

class SpeechService:
    """
    FastAPI Speech Service providing TTS and STT functionality.
    
    Features:
    - Text-to-Speech: Local (espeak/pyttsx3) and Cloud (Google Cloud TTS)
    - Speech-to-Text: Local (Vosk models) and Cloud (Google Cloud STT)
    - Multi-platform support with automatic detection
    - Provider switching capability
    - Audio file management and cleanup
    """
    
    def __init__(self, audio_folder: str = None, provider: str = "local"):
        """
        Initialize the Speech Service.
        
        Args:
            audio_folder: Directory to store generated audio files
            provider: Provider type ("local" or "cloud")
        """
        if audio_folder:
            self.audio_folder = audio_folder
        else:
            # Use a folder inside the repo for easy access
            repo_root = Path(__file__).parent.parent.parent.parent  # Go up to AIEDU root
            self.audio_folder = str(repo_root / "audio_files")
        
        self.platform = platform.system()
        self.provider = provider
        self.vosk_models = {}
        self.cloud_stt_provider = None
        self.cloud_tts_provider = None
        self.performance_stats = {
            'stt_calls': 0,
            'tts_calls': 0,
            'stt_success': 0,
            'tts_success': 0,
            'avg_stt_time': 0.0,
            'avg_tts_time': 0.0
        }
        
        # Ensure audio folder exists
        os.makedirs(self.audio_folder, exist_ok=True)
        
        # Initialize providers based on selection
        if provider == "local":
            self._initialize_local_providers()
        else:
            self._initialize_cloud_providers()
        
        logger.info(f"Speech Service initialized for {self.platform} with {provider} provider")
        logger.info(f"Audio folder: {self.audio_folder}")
        if provider == "local":
            logger.info(f"Vosk models available: {list(self.vosk_models.keys())}")
    
    def set_provider(self, provider: str):
        """Switch between local and cloud providers."""
        old_provider = self.provider
        self.provider = provider
        
        if provider == "local":
            self._initialize_local_providers()
        else:
            self._initialize_cloud_providers()
            
        logger.info(f"Switched speech provider from {old_provider} to {provider}")
    
    def _initialize_local_providers(self):
        """Initialize local speech providers (Vosk, espeak)."""
        # Initialize Vosk models
        self._initialize_vosk_models()
        # Check TTS dependencies
        self._check_tts_dependencies()
        
    def _initialize_cloud_providers(self):
        """Initialize cloud speech providers (Google Cloud)."""
        try:
            from api.providers.gcp_stt_provider import GCPSTTProvider
            from api.providers.gcp_tts_provider import GCPTTSProvider
            
            # Test Google Cloud STT provider
            try:
                self.cloud_stt_provider = GCPSTTProvider()
                # Test the provider is working
                if not self.cloud_stt_provider.is_available():
                    logger.warning("Google Cloud STT provider is not available - credentials may be invalid")
                    self.cloud_stt_provider = None
                else:
                    logger.info("Google Cloud STT provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud STT provider: {e}")
                self.cloud_stt_provider = None
            
            # Test Google Cloud TTS provider
            try:
                self.cloud_tts_provider = GCPTTSProvider(
                    audio_format="wav",  # Use WAV format to match our system
                    speaking_rate=1.0,
                    pitch=0.0
                )
                logger.info("Google Cloud TTS provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud TTS provider: {e}")
                self.cloud_tts_provider = None
            
            # If both providers failed, log warning
            if not self.cloud_stt_provider and not self.cloud_tts_provider:
                logger.warning("Both Google Cloud speech providers failed to initialize - falling back to local providers")
                self._initialize_local_providers()
            elif not self.cloud_stt_provider:
                logger.warning("Google Cloud STT provider failed - STT will use local Vosk")
            elif not self.cloud_tts_provider:
                logger.warning("Google Cloud TTS provider failed - TTS will use local espeak")
            
        except ImportError as e:
            logger.error(f"Google Cloud libraries not available: {e}")
            self.cloud_stt_provider = None
            self.cloud_tts_provider = None
            # Fall back to local providers
            self._initialize_local_providers()
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud providers: {e}")
            self.cloud_stt_provider = None
            self.cloud_tts_provider = None
            # Fall back to local providers
            self._initialize_local_providers()
    
    def _initialize_vosk_models(self):
        """Initialize Vosk speech recognition models."""
        try:
            import vosk
            
            # Define available models
            models_config = {
                "en-US": {
                    "path": VOSK_MODELS_PATH / "vosk-model-small-en-us-0.15",
                    "name": "English (US)"
                },
                "es-ES": {
                    "path": VOSK_MODELS_PATH / "vosk-model-small-es-0.42", 
                    "name": "Spanish (ES)"
                }
            }
            
            # Load available models
            for language, config in models_config.items():
                model_path = config["path"]
                if model_path.exists():
                    try:
                        model = vosk.Model(str(model_path))
                        self.vosk_models[language] = {
                            "model": model,
                            "name": config["name"],
                            "path": str(model_path)
                        }
                        logger.info(f"Loaded Vosk model: {language} from {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to load Vosk model {language}: {e}")
                else:
                    logger.warning(f"Vosk model not found: {model_path}")
                    
        except ImportError:
            logger.error("Vosk not available - install with: pip install vosk")
    
    def _check_tts_dependencies(self):
        """Check and log TTS dependencies availability."""
        if self.platform == "Linux":
            # Check espeak availability
            try:
                result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("espeak available for TTS")
                else:
                    logger.warning("espeak not found - install with: sudo apt-get install espeak")
            except Exception as e:
                logger.error(f"Error checking espeak: {e}")
        else:
            # Check pyttsx3 availability
            try:
                import pyttsx3
                logger.info("pyttsx3 available for TTS")
            except ImportError:
                logger.warning("pyttsx3 not available - install with: pip install pyttsx3")
    
    async def synthesize_speech(self, text: str, voice_id: str = None, language: str = "en-US") -> Dict[str, Any]:
        """
        Convert text to speech using selected provider.
        
        Args:
            text: Text to synthesize
            voice_id: Voice identifier (optional)
            language: Language code (e.g., "en-US", "es-ES")
            
        Returns:
            Dict with synthesis results and audio file path
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"🎯 SpeechService.synthesize_speech called")
            logger.info(f"📝 Text length: {len(text)} characters")
            logger.info(f"🌍 Language: {language}")
            logger.info(f"🎤 Voice ID: {voice_id}")
            
            self.performance_stats['tts_calls'] += 1
            
            # Get current provider from provider service
            from ..core.dependencies import get_provider_service
            provider_service = get_provider_service()
            current_provider = provider_service.current_provider
            
            # Update our provider if it has changed
            if self.provider != current_provider:
                logger.info(f"🔄 Switching TTS provider from {self.provider} to {current_provider}")
                self.set_provider(current_provider)
            
            # Try cloud provider first if requested and available
            if self.provider == "cloud" and self.cloud_tts_provider:
                logger.info("🌩️ Using Google Cloud TTS")
                return await self._synthesize_with_cloud(text, voice_id, language, start_time)
            elif self.provider == "cloud" and not self.cloud_tts_provider:
                logger.warning("☁️ Cloud TTS requested but not available - falling back to local espeak")
                return await self._synthesize_with_local(text, voice_id, language, start_time)
            else:
                logger.info("🏠 Using local espeak TTS")
                return await self._synthesize_with_local(text, voice_id, language, start_time)
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in speech synthesis: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    async def _synthesize_with_cloud(self, text: str, voice_id: str, language: str, start_time: float) -> Dict[str, Any]:
        """Synthesize speech using Google Cloud TTS."""
        try:
            # Determine voice name
            if voice_id and voice_id.startswith(('es-ES', 'en-US')):
                voice_name = voice_id
            else:
                # Default voices
                if language.startswith("es"):
                    voice_name = "es-ES-Neural2-C"
                else:
                    voice_name = "en-US-Neural2-F"
            
            # Generate unique filename
            audio_filename = f"tts_cloud_{uuid.uuid4().hex[:8]}.wav"
            audio_path = os.path.join(self.audio_folder, audio_filename)
            
            # Use Google Cloud TTS
            result = self.cloud_tts_provider.save_audio_to_file(
                text=text,
                output_path=audio_path,
                language_code=language,
                voice_name=voice_name
            )
            
            duration = time.time() - start_time
            
            if result["status"] == "success":
                self.performance_stats['tts_success'] += 1
                self._update_avg_time('tts', duration)
                
                return {
                    'success': True,
                    'audio_path': audio_path,
                    'audio_filename': audio_filename,
                    'voice_used': f"Google Cloud {voice_name}",
                    'duration': duration,
                    'file_size': result.get('file_size', 0),
                    'provider': 'Google Cloud TTS'
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Cloud TTS failed'),
                    'duration': duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Cloud TTS error: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    async def _synthesize_with_local(self, text: str, voice_id: str, language: str, start_time: float) -> Dict[str, Any]:
        """Synthesize speech using local TTS (espeak/pyttsx3)."""
        try:
            # Generate unique filename
            audio_filename = f"tts_local_{uuid.uuid4().hex[:8]}.wav"
            audio_path = os.path.join(self.audio_folder, audio_filename)
            
            # Determine voice based on language and voice_id
            if language.startswith("es"):
                espeak_voice = "es-la"
                voice_name = "Spanish (Latin America)"
            else:
                espeak_voice = "en+f3"  # Softer English female variant
                voice_name = "English (US)"
            
            success = False
            
            if self.platform == "Linux":
                success = await self._synthesize_with_espeak(text, audio_path, espeak_voice)
            else:
                success = await self._synthesize_with_pyttsx3(text, audio_path, language)
            
            duration = time.time() - start_time
            
            if success:
                self.performance_stats['tts_success'] += 1
                self._update_avg_time('tts', duration)
                
                return {
                    'success': True,
                    'audio_path': audio_path,
                    'audio_filename': audio_filename,
                    'voice_used': voice_name,
                    'duration': duration,
                    'file_size': os.path.getsize(audio_path) if os.path.exists(audio_path) else 0,
                    'provider': 'Local TTS'
                }
            else:
                return {
                    'success': False,
                    'error': 'Local TTS synthesis failed',
                    'duration': duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Local TTS error: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    async def transcribe_audio(self, audio_path: str, language: str = "en-US") -> Dict[str, Any]:
        """
        Transcribe audio file using selected provider.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., "en-US", "es-ES")
            
        Returns:
            Dict with transcription results
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"🎯 SpeechService.transcribe_audio called with: {audio_path}")
            logger.info(f"📏 Input file size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'FILE NOT FOUND'} bytes")
            
            self.performance_stats['stt_calls'] += 1
            
            # Get current provider from provider service
            from ..core.dependencies import get_provider_service
            provider_service = get_provider_service()
            current_provider = provider_service.current_provider
            
            # Update our provider if it has changed
            if self.provider != current_provider:
                logger.info(f"🔄 Switching STT provider from {self.provider} to {current_provider}")
                self.set_provider(current_provider)
            
            # Try cloud provider first if requested and available
            if self.provider == "cloud" and self.cloud_stt_provider:
                logger.info("🌩️ Using Google Cloud STT")
                return await self._transcribe_with_cloud(audio_path, language, start_time)
            elif self.provider == "cloud" and not self.cloud_stt_provider:
                logger.warning("☁️ Cloud STT requested but not available - falling back to local Vosk")
                return await self._transcribe_with_local(audio_path, language, start_time)
            else:
                logger.info("🏠 Using local Vosk STT")
                return await self._transcribe_with_local(audio_path, language, start_time)
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in speech transcription: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    async def _transcribe_with_cloud(self, audio_path: str, language: str, start_time: float) -> Dict[str, Any]:
        """Transcribe audio using Google Cloud STT."""
        try:
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Detect audio format and configure accordingly
            file_ext = os.path.splitext(audio_path)[1].lower().lstrip('.')
            
            if file_ext == 'wav':
                # For WAV files, detect exact sample rate
                try:
                    import wave
                    with wave.open(audio_path, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        audio_format = "wav"
                        logger.info(f"🎵 Detected WAV sample rate: {sample_rate} Hz")
                except Exception as e:
                    logger.warning(f"Could not read WAV properties: {e}, using default 16000 Hz")
                    sample_rate = 16000
                    audio_format = "wav"
            elif file_ext == 'webm':
                # For WebM files, use OPUS encoding with standard sample rates
                # WebM typically uses 48000 Hz, but let Google Cloud auto-detect
                sample_rate = 48000  # Standard for WebM OPUS
                audio_format = "webm"
                logger.info(f"🎥 Detected WebM file, using OPUS encoding with {sample_rate} Hz")
            elif file_ext in ['mp3', 'flac', 'ogg']:
                # For other formats, use appropriate defaults
                sample_rate = 16000  # Common default
                audio_format = file_ext
                logger.info(f"🎵 Detected {file_ext.upper()} file, using {sample_rate} Hz")
            else:
                # Unknown format, treat as WAV with default sample rate
                sample_rate = 16000
                audio_format = "wav"
                logger.warning(f"Unknown audio format '{file_ext}', treating as WAV with {sample_rate} Hz")
            
            # Use Google Cloud STT with detected configuration
            logger.info(f"🌩️ Transcribing with Google Cloud STT: format={audio_format}, sample_rate={sample_rate}")
            result = self.cloud_stt_provider.transcribe_audio(
                audio_data=audio_data,
                language_code=language,
                sample_rate=sample_rate,
                audio_format=audio_format
            )
            
            duration = time.time() - start_time
            
            if result["status"] == "success":
                self.performance_stats['stt_success'] += 1
                self._update_avg_time('stt', duration)
                
                transcript_text = result["transcript"]
                logger.info(f"✅ Cloud STT success: '{transcript_text[:50]}...' (confidence: {result.get('confidence', 0):.3f})")
                
                # Convert Google Cloud format to our format
                return {
                    'success': True,
                    'text': transcript_text,  # Use 'text' field to match frontend expectations
                    'confidence': result.get("confidence", 0.0),
                    'language': result.get("language_detected", language),
                    'model_used': "Google Cloud STT",
                    'duration': duration,
                    'words': result.get("words", []),
                    'provider': 'Google Cloud STT'
                }
            else:
                logger.error(f"❌ Cloud STT failed: {result.get('error', 'No results returned')}")
                return {
                    'success': False,
                    'error': result.get('error', 'Cloud STT failed'),
                    'duration': duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Cloud STT error: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    async def _transcribe_with_local(self, audio_path: str, language: str, start_time: float) -> Dict[str, Any]:
        """Transcribe audio using local Vosk models."""
        try:
            # Check if we have the model for this language
            if language not in self.vosk_models:
                available_languages = list(self.vosk_models.keys())
                logger.warning(f"Language {language} not available. Available: {available_languages}")
                # Fallback to English if available
                if "en-US" in self.vosk_models:
                    language = "en-US"
                    logger.info("Falling back to English (en-US)")
                else:
                    return {
                        'success': False,
                        'error': f'No Vosk model available for {language}',
                        'available_languages': available_languages
                    }
            
            # Get the model
            vosk_model = self.vosk_models[language]["model"]
            logger.info(f"📈 Using Vosk model for {language}")
            
            # Copy file to permanent storage for debugging
            import uuid
            debug_filename = f"debug_recording_{uuid.uuid4().hex[:8]}{os.path.splitext(audio_path)[1]}"
            debug_path = os.path.join(self.audio_folder, debug_filename)
            import shutil
            shutil.copy2(audio_path, debug_path)
            logger.info(f"💾 Copied input file to permanent storage: {debug_path}")
            
            # Perform transcription
            logger.info(f"🔍 Starting Vosk transcription...")
            result = await self._transcribe_with_vosk(audio_path, vosk_model)
            
            duration = time.time() - start_time
            
            if result['success']:
                self.performance_stats['stt_success'] += 1
                self._update_avg_time('stt', duration)
                
                return {
                    'success': True,
                    'text': result['transcript'],  # Use 'text' field to match frontend expectations
                    'confidence': result.get('confidence', 0.0),
                    'language': language,
                    'model_used': self.vosk_models[language]["name"],
                    'duration': duration,
                    'words': result.get('words', []),
                    'provider': 'Local Vosk'
                }
            else:
                return {
                    'success': False,
                    'error': result['error'],
                    'duration': duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Local STT error: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    async def _synthesize_with_espeak(self, text: str, output_path: str, voice: str) -> bool:
        """Synthesize speech using espeak (Linux/WSL2)."""
        try:
            cmd = [
                'espeak',
                '-v', voice,           # Voice: es-la or en+f3
                '-w', output_path,     # Write to WAV file
                '-s', '175',           # Speed: 175 wpm
                '-a', '120',           # Amplitude: 120
                '-p', '48',            # Pitch: 48
                '-g', '2',             # Gaps: 2
                '-z',                  # No final pause
                text
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.debug(f"espeak synthesis successful: {output_path}")
                return True
            else:
                logger.error(f"espeak failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("espeak synthesis timed out")
            return False
        except Exception as e:
            logger.error(f"espeak synthesis error: {e}")
            return False
    
    async def _synthesize_with_pyttsx3(self, text: str, output_path: str, language: str) -> bool:
        """Synthesize speech using pyttsx3 (Windows)."""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            # Set voice based on language
            voices = engine.getProperty('voices')
            if language.startswith('es') and len(voices) > 1:
                engine.setProperty('voice', voices[1].id)  # Try second voice for Spanish
            elif len(voices) > 0:
                engine.setProperty('voice', voices[0].id)
            
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            if os.path.exists(output_path):
                logger.debug(f"pyttsx3 synthesis successful: {output_path}")
                return True
            else:
                logger.error("pyttsx3 synthesis failed - no output file")
                return False
                
        except ImportError:
            logger.error("pyttsx3 not available")
            return False
        except Exception as e:
            logger.error(f"pyttsx3 synthesis error: {e}")
            return False
    
    async def _transcribe_with_vosk(self, audio_path: str, model) -> Dict[str, Any]:
        """Transcribe audio using Vosk model."""
        try:
            import vosk
            import json
            import subprocess
            import tempfile
            
            # Convert audio to WAV format if needed
            logger.info(f"🔧 Processing audio file: {audio_path}")
            logger.info(f"📦 Original file size: {os.path.getsize(audio_path)} bytes")
            
            wav_path = audio_path
            temp_wav_path = None
            
            # Check if file is already in WAV format
            if not audio_path.lower().endswith('.wav'):
                logger.info(f"🔄 Converting {audio_path} to WAV format...")
                # Convert to WAV using FFmpeg
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_wav_path = temp_file.name
                
                try:
                    # Use FFmpeg to convert to 16kHz mono WAV with proper RIFF header
                    cmd = [
                        'ffmpeg', '-i', audio_path,
                        '-ar', '16000',       # Sample rate 16kHz (Vosk requirement)
                        '-ac', '1',           # Mono channel
                        '-sample_fmt', 's16', # 16-bit signed PCM
                        '-acodec', 'pcm_s16le', # PCM 16-bit little-endian codec
                        '-f', 'wav',          # Force WAV format with RIFF header
                        '-avoid_negative_ts', 'make_zero', # Avoid timestamp issues
                        '-y',                 # Overwrite output
                        temp_wav_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # Verify the WAV file has proper RIFF header
                        try:
                            with open(temp_wav_path, 'rb') as f:
                                header = f.read(4)
                                if header != b'RIFF':
                                    logger.error(f"Generated WAV file does not have RIFF header: {header}")
                                    return {
                                        'success': False,
                                        'error': 'Generated audio file is not a valid WAV format'
                                    }
                            wav_path = temp_wav_path
                            converted_size = os.path.getsize(wav_path)
                            logger.info(f"✅ Audio converted to WAV with valid RIFF header: {audio_path} -> {wav_path} ({converted_size} bytes)")
                        except Exception as e:
                            logger.error(f"Failed to verify WAV file: {e}")
                            return {
                                'success': False,
                                'error': f'Failed to verify converted audio file: {e}'
                            }
                    else:
                        # Extract just the relevant error from FFmpeg output
                        error_lines = result.stderr.strip().split('\n')
                        relevant_error = error_lines[-1] if error_lines else result.stderr
                        logger.error(f"FFmpeg conversion failed: {relevant_error}")
                        return {
                            'success': False,
                            'error': f'Audio conversion failed: {relevant_error}'
                        }
                except FileNotFoundError:
                    logger.error("FFmpeg not found - cannot convert audio format")
                    return {
                        'success': False,
                        'error': 'FFmpeg not available for audio conversion'
                    }
            
            # Open audio file
            logger.info(f"📂 Opening WAV file for processing: {wav_path}")
            wf = wave.open(wav_path, 'rb')
            
            # Check audio format
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / frame_rate
            
            logger.info(f"🎵 Audio format: {channels} channels, {sample_width} bytes/sample, {frame_rate} Hz, {frames} frames, {duration:.2f}s duration")
            
            if channels != 1 or sample_width != 2 or wf.getcomptype() != 'NONE':
                logger.warning(f"⚠️ Audio format may not be optimal for Vosk: {channels}ch, {sample_width}B/sample")
            
            # Create recognizer
            logger.info(f"🧠 Creating Vosk recognizer with sample rate: {frame_rate}")
            rec = vosk.KaldiRecognizer(model, frame_rate)
            rec.SetWords(True)  # Enable word-level timestamps
            
            transcript_parts = []
            words = []
            
            # Process audio in chunks
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                    
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get('text'):
                        transcript_parts.append(result['text'])
                        if 'result' in result:
                            words.extend(result['result'])
            
            # Get final result
            final_result = json.loads(rec.FinalResult())
            if final_result.get('text'):
                transcript_parts.append(final_result['text'])
                if 'result' in final_result:
                    words.extend(final_result['result'])
            
            wf.close()
            
            # Combine transcript
            full_transcript = ' '.join(transcript_parts).strip()
            
            if full_transcript:
                # Calculate average confidence
                confidences = [word.get('conf', 0.0) for word in words if 'conf' in word]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                return {
                    'success': True,
                    'transcript': full_transcript,
                    'confidence': avg_confidence,
                    'words': words
                }
            else:
                return {
                    'success': False,
                    'error': 'No speech detected in audio'
                }
                
        except Exception as e:
            logger.error(f"Vosk transcription error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # Clean up temporary WAV file if created
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                    logger.debug(f"Cleaned up temporary WAV file: {temp_wav_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary WAV file {temp_wav_path}: {e}")
    
    def get_available_voices(self) -> List[Dict[str, str]]:
        """
        Get list of available voices for TTS.
        
        Returns:
            List of voice dictionaries with id, name, and language_type
        """
        voices = [
            {
                'id': 'spanish_voice',
                'name': 'Spanish (Latin America)',
                'language_type': 'Spanish',
                'platform': self.platform,
                'engine': 'espeak' if self.platform == 'Linux' else 'pyttsx3'
            },
            {
                'id': 'english_voice',
                'name': 'English (US)',
                'language_type': 'English', 
                'platform': self.platform,
                'engine': 'espeak' if self.platform == 'Linux' else 'pyttsx3'
            }
        ]
        return voices
    
    def get_available_languages(self) -> List[Dict[str, str]]:
        """
        Get list of available languages for STT.
        
        Returns:
            List of language dictionaries
        """
        languages = []
        for lang_code, model_info in self.vosk_models.items():
            languages.append({
                'code': lang_code,
                'name': model_info['name'],
                'model_path': model_info['path']
            })
        return languages
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_calls = self.performance_stats['stt_calls'] + self.performance_stats['tts_calls']
        total_success = self.performance_stats['stt_success'] + self.performance_stats['tts_success']
        
        return {
            **self.performance_stats,
            'total_calls': total_calls,
            'total_success': total_success,
            'overall_success_rate': (total_success / total_calls * 100) if total_calls > 0 else 0.0,
            'stt_success_rate': (self.performance_stats['stt_success'] / self.performance_stats['stt_calls'] * 100) 
                               if self.performance_stats['stt_calls'] > 0 else 0.0,
            'tts_success_rate': (self.performance_stats['tts_success'] / self.performance_stats['tts_calls'] * 100)
                               if self.performance_stats['tts_calls'] > 0 else 0.0
        }
    
    def _update_avg_time(self, service_type: str, duration: float):
        """Update average timing statistics."""
        calls_key = f'{service_type}_calls'
        avg_key = f'avg_{service_type}_time'
        
        total_calls = self.performance_stats[calls_key]
        current_avg = self.performance_stats[avg_key]
        
        # Calculate new average
        new_avg = ((current_avg * (total_calls - 1)) + duration) / total_calls
        self.performance_stats[avg_key] = new_avg
    
    def cleanup_audio_files(self, max_age_minutes: int = 60):
        """
        Clean up old audio files from the audio folder.
        
        Args:
            max_age_minutes: Maximum age of files to keep in minutes
        """
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
            for filename in os.listdir(self.audio_folder):
                file_path = os.path.join(self.audio_folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - datetime.fromtimestamp(os.path.getctime(file_path))
                    if file_age.total_seconds() > (max_age_minutes * 60):
                        os.remove(file_path)
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old audio files")
                
        except Exception as e:
            logger.error(f"Error cleaning up audio files: {e}")


# Singleton instance
_speech_service = None

def get_speech_service() -> SpeechService:
    """Get or create the singleton SpeechService instance."""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService()
    return _speech_service
