"""
Speech Service Module
Provides text-to-speech and speech-to-text functionality for the FastAPI backend.
Migrated from Flask hybrid_speech.py with enhanced FastAPI integration.
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
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Add project paths for Vosk model access
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
VOSK_MODELS_PATH = PROJECT_ROOT / "local_deployment" / "web_app" / "models"

class SpeechService:
    """
    FastAPI Speech Service providing TTS and STT functionality.
    
    Features:
    - Text-to-Speech: espeak (Linux/WSL2) and pyttsx3 (Windows)
    - Speech-to-Text: Vosk models for English and Spanish
    - Multi-platform support with automatic detection
    - Audio file management and cleanup
    """
    
    def __init__(self, audio_folder: str = None):
        """
        Initialize the Speech Service.
        
        Args:
            audio_folder: Directory to store generated audio files
        """
        if audio_folder:
            self.audio_folder = audio_folder
        else:
            # Use a folder inside the repo for easy access
            repo_root = Path(__file__).parent.parent.parent.parent  # Go up to AIEDU root
            self.audio_folder = str(repo_root / "audio_files")
        
        self.platform = platform.system()
        self.vosk_models = {}
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
        
        # Initialize Vosk models
        self._initialize_vosk_models()
        
        # Check TTS dependencies
        self._check_tts_dependencies()
        
        logger.info(f"Speech Service initialized for {self.platform}")
        logger.info(f"Audio folder: {self.audio_folder}")
        logger.info(f"Vosk models available: {list(self.vosk_models.keys())}")
    
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
        Convert text to speech using platform-appropriate TTS engine.
        
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
            self.performance_stats['tts_calls'] += 1
            
            # Generate unique filename
            audio_filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
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
                    'file_size': os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
                }
            else:
                return {
                    'success': False,
                    'error': 'TTS synthesis failed',
                    'duration': duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in speech synthesis: {e}")
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
    
    async def transcribe_audio(self, audio_path: str, language: str = "en-US") -> Dict[str, Any]:
        """
        Transcribe audio file using Vosk speech recognition.
        
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
                    'transcript': result['transcript'],
                    'confidence': result.get('confidence', 0.0),
                    'language': language,
                    'model_used': self.vosk_models[language]["name"],
                    'duration': duration,
                    'words': result.get('words', [])
                }
            else:
                return {
                    'success': False,
                    'error': result['error'],
                    'duration': duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in speech transcription: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
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
