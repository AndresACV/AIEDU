"""
Hybrid Speech Integration Module
Integrates Speech Provider Factory with Flask web application.
Provides seamless switching between local and cloud speech providers.
"""

import os
import sys
import logging
import tempfile
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# Add project paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Configure logging
logger = logging.getLogger(__name__)

# Speech Provider Factory and modes
try:
    from shared.core_logic.speech_provider_factory import SpeechProviderFactory, SpeechMode
    SPEECH_FACTORY_AVAILABLE = True
    logger.info("Speech Provider Factory imported successfully")
except ImportError as e:
    logger.error(f"Could not import Speech Provider Factory: {e}")
    SPEECH_FACTORY_AVAILABLE = False

class HybridSpeechManager:
    """
    Manages hybrid speech functionality for the Flask web app.
    
    Provides:
    - Intelligent provider selection (local vs cloud)
    - Speech mode management (Privacy/Performance/Cost/Auto)
    - Seamless fallback between providers
    - Performance monitoring and comparison
    """
    
    def __init__(self, audio_folder: str):
        """
        Initialize the Hybrid Speech Manager.
        
        Args:
            audio_folder: Directory to store generated audio files
        """
        self.audio_folder = audio_folder
        self.current_mode = SpeechMode.AUTO
        self.factory = None
        self.provider_status = {}
        self.performance_stats = {
            'stt_calls': 0,
            'tts_calls': 0,
            'local_stt_success': 0,
            'cloud_stt_success': 0,
            'local_tts_success': 0,
            'cloud_tts_success': 0,
            'avg_stt_time': 0.0,
            'avg_tts_time': 0.0
        }
        
        # Initialize speech factory if available
        if SPEECH_FACTORY_AVAILABLE:
            try:
                self.factory = SpeechProviderFactory(mode=self.current_mode)
                self.update_provider_status()
                logger.info(f"Hybrid Speech Manager initialized with mode: {self.current_mode.value}")
            except Exception as e:
                logger.error(f"Failed to initialize Speech Provider Factory: {e}")
                self.factory = None
        else:
            logger.warning("Speech Provider Factory not available - using fallback mode")
    
    def is_available(self) -> bool:
        """Check if hybrid speech functionality is available."""
        return SPEECH_FACTORY_AVAILABLE and self.factory is not None
    
    def set_speech_mode(self, mode: str) -> bool:
        """
        Set the speech processing mode.
        
        Args:
            mode: One of 'privacy', 'performance', 'cost', 'auto'
            
        Returns:
            True if mode was set successfully
        """
        try:
            mode_mapping = {
                'privacy': SpeechMode.PRIVACY,
                'performance': SpeechMode.PERFORMANCE,
                'cost': SpeechMode.COST_CONSCIOUS,
                'auto': SpeechMode.AUTO
            }
            
            if mode.lower() not in mode_mapping:
                logger.error(f"Invalid speech mode: {mode}")
                return False
            
            self.current_mode = mode_mapping[mode.lower()]
            
            # Reinitialize factory with new mode
            if self.factory:
                self.factory = SpeechProviderFactory(mode=self.current_mode)
                self.update_provider_status()
                logger.info(f"Speech mode changed to: {self.current_mode.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error setting speech mode: {e}")
            return False
    
    def get_current_mode(self) -> str:
        """Get the current speech processing mode."""
        return self.current_mode.value if self.current_mode else 'auto'
    
    def update_provider_status(self):
        """Update the status of all speech providers."""
        if not self.factory:
            return
        
        try:
            self.provider_status = self.factory.get_provider_status()
            logger.debug(f"Provider status updated: {self.provider_status}")
        except Exception as e:
            logger.error(f"Error updating provider status: {e}")
            self.provider_status = {}
    
    def transcribe_audio(self, file_path: str, language: str = "en-US") -> Dict[str, Any]:
        """
        Transcribe audio using hybrid speech providers.
        
        Args:
            file_path: Path to audio file
            language: Language code (e.g., "en-US", "es-ES")
            
        Returns:
            Dict with transcription results
        """
        import time
        start_time = time.time()
        
        try:
            self.performance_stats['stt_calls'] += 1
            
            if not self.factory:
                # Fallback to local Vosk implementation
                return self._fallback_transcribe(file_path, language)
            
            # Get STT provider
            stt_provider = self.factory.get_stt_provider(language=language)
            if not stt_provider:
                logger.warning("No STT provider available, using fallback")
                return self._fallback_transcribe(file_path, language)
            
            # Transcribe using provider
            provider_type = type(stt_provider).__name__
            logger.info(f"Using STT provider: {provider_type} for language: {language}")
            
            if hasattr(stt_provider, 'transcribe_file'):
                result = stt_provider.transcribe_file(file_path, language_code=language)
            elif hasattr(stt_provider, 'transcribe_audio'):
                # Read audio file and transcribe
                with open(file_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                result = stt_provider.transcribe_audio(audio_data, language_code=language)
            else:
                logger.error(f"STT provider {provider_type} has no transcription method")
                return self._fallback_transcribe(file_path, language)
            
            # Track performance
            duration = time.time() - start_time
            self._update_performance_stats('stt', provider_type, True, duration)
            
            # Standardize result format
            if result.get('status') == 'success':
                return {
                    'success': True,
                    'transcript': result.get('transcript', ''),
                    'confidence': result.get('confidence', 0.0),
                    'provider': provider_type,
                    'language': result.get('language_detected', language),
                    'duration': duration,
                    'words': result.get('words', [])
                }
            else:
                logger.warning(f"STT provider failed: {result.get('error', 'Unknown error')}")
                return self._fallback_transcribe(file_path, language)
                
        except Exception as e:
            logger.error(f"Error in hybrid STT: {e}")
            duration = time.time() - start_time
            self._update_performance_stats('stt', 'error', False, duration)
            return self._fallback_transcribe(file_path, language)
    
    def synthesize_speech(self, text: str, language: str = "en-US", voice_id: str = None) -> Dict[str, Any]:
        """
        Synthesize speech using hybrid speech providers.
        
        Args:
            text: Text to convert to speech
            language: Language code (e.g., "en-US", "es-ES")
            voice_id: Optional voice identifier
            
        Returns:
            Dict with synthesis results
        """
        import time
        start_time = time.time()
        
        try:
            self.performance_stats['tts_calls'] += 1
            
            if not self.factory:
                # Fallback to local TTS
                return self._fallback_synthesize(text, language, voice_id)
            
            # Get TTS provider
            tts_provider = self.factory.get_tts_provider(language=language)
            if not tts_provider:
                logger.warning("No TTS provider available, using fallback")
                return self._fallback_synthesize(text, language, voice_id)
            
            # Synthesize using provider
            provider_type = type(tts_provider).__name__
            logger.info(f"Using TTS provider: {provider_type} for language: {language}")
            
            if hasattr(tts_provider, 'synthesize_speech'):
                result = tts_provider.synthesize_speech(text, language_code=language, voice_name=voice_id)
            else:
                logger.error(f"TTS provider {provider_type} has no synthesis method")
                return self._fallback_synthesize(text, language, voice_id)
            
            # Save audio to file
            if result.get('status') == 'success' and result.get('audio_content'):
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                extension = 'mp3' if result.get('audio_format') == 'mp3' else 'wav'
                filename = f"hybrid_speech_{timestamp}_{unique_id}.{extension}"
                output_path = os.path.join(self.audio_folder, filename)
                
                # Save audio content
                with open(output_path, 'wb') as audio_file:
                    audio_file.write(result['audio_content'])
                
                # Track performance
                duration = time.time() - start_time
                self._update_performance_stats('tts', provider_type, True, duration)
                
                return {
                    'success': True,
                    'audio_url': f"/static/audio/{filename}",
                    'file_path': output_path,
                    'file_size': len(result['audio_content']),
                    'provider': provider_type,
                    'voice_used': result.get('voice_used', 'default'),
                    'language': result.get('language_used', language),
                    'duration': duration,
                    'cached': result.get('cached', False),
                    'audio_format': result.get('audio_format', extension)
                }
            else:
                logger.warning(f"TTS provider failed: {result.get('error', 'Unknown error')}")
                return self._fallback_synthesize(text, language, voice_id)
                
        except Exception as e:
            logger.error(f"Error in hybrid TTS: {e}")
            duration = time.time() - start_time
            self._update_performance_stats('tts', 'error', False, duration)
            return self._fallback_synthesize(text, language, voice_id)
    
    def get_available_voices(self, language: str = "en-US") -> list:
        """
        Get available voices for the current TTS provider.
        
        Args:
            language: Language code
            
        Returns:
            List of available voices
        """
        try:
            if not self.factory:
                return self._get_fallback_voices()
            
            tts_provider = self.factory.get_tts_provider(language=language)
            if not tts_provider:
                return self._get_fallback_voices()
            
            provider_type = type(tts_provider).__name__
            
            if provider_type == 'GCPTTSProvider':
                # Get Google Cloud voices
                voices = tts_provider.get_available_voices(language)
                return [
                    {
                        'id': voice['name'],
                        'name': f"{voice['name']} ({voice['gender']})",
                        'language': ', '.join(voice['language_codes']),
                        'provider': 'Google Cloud'
                    }
                    for voice in voices[:10]  # Limit to first 10 for UI
                ]
            elif provider_type == 'LocalTTSProvider':
                # Get local voices
                return [
                    {
                        'id': 'local_en',
                        'name': 'Local English Voice',
                        'language': 'en-US',
                        'provider': 'Local TTS'
                    },
                    {
                        'id': 'local_es',
                        'name': 'Local Spanish Voice', 
                        'language': 'es-ES',
                        'provider': 'Local TTS'
                    }
                ]
            else:
                return self._get_fallback_voices()
                
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return self._get_fallback_voices()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for speech operations."""
        stats = self.performance_stats.copy()
        
        # Add provider status
        stats['provider_status'] = self.provider_status
        stats['current_mode'] = self.get_current_mode()
        stats['factory_available'] = self.is_available()
        
        return stats
    
    def _fallback_transcribe(self, file_path: str, language: str) -> Dict[str, Any]:
        """Fallback to original Vosk transcription."""
        try:
            # Import the original transcription function
            from app import transcribe_audio_file
            
            result = transcribe_audio_file(file_path, language)
            
            if result.get('success'):
                return {
                    'success': True,
                    'transcript': result.get('transcript', ''),
                    'confidence': 0.8,  # Default confidence for Vosk
                    'provider': 'Vosk (fallback)',
                    'language': language,
                    'duration': 0.0,
                    'words': []
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Fallback transcription failed'),
                    'provider': 'Vosk (fallback)'
                }
                
        except Exception as e:
            logger.error(f"Fallback transcription failed: {e}")
            return {
                'success': False,
                'error': f"All transcription methods failed: {e}",
                'provider': 'error'
            }
    
    def _fallback_synthesize(self, text: str, language: str, voice_id: str) -> Dict[str, Any]:
        """Fallback to original local TTS."""
        try:
            # Use a simplified version of the original synthesis logic
            import pyttsx3
            import platform
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"fallback_speech_{timestamp}_{unique_id}.wav"
            output_path = os.path.join(self.audio_folder, filename)
            
            # Create TTS engine
            engine = pyttsx3.init()
            
            # Configure engine
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            # Set voice if available
            voices = engine.getProperty('voices')
            if voices and voice_id:
                for voice in voices:
                    if voice.id == voice_id:
                        engine.setProperty('voice', voice_id)
                        break
            
            # Generate speech
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            # Check if file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                return {
                    'success': True,
                    'audio_url': f"/static/audio/{filename}",
                    'file_path': output_path,
                    'file_size': os.path.getsize(output_path),
                    'provider': 'Local TTS (fallback)',
                    'voice_used': voice_id or 'default',
                    'language': language,
                    'duration': 0.0,
                    'cached': False,
                    'audio_format': 'wav'
                }
            else:
                raise Exception("Audio file not generated or empty")
                
        except Exception as e:
            logger.error(f"Fallback synthesis failed: {e}")
            return {
                'success': False,
                'error': f"All synthesis methods failed: {e}",
                'provider': 'error'
            }
    
    def _get_fallback_voices(self) -> list:
        """Get fallback voice list."""
        return [
            {
                'id': 'default',
                'name': 'Default Voice',
                'language': 'en-US, es-ES',
                'provider': 'Local TTS'
            }
        ]
    
    def _update_performance_stats(self, service_type: str, provider: str, success: bool, duration: float):
        """Update performance statistics."""
        try:
            if service_type == 'stt':
                if success:
                    if 'gcp' in provider.lower() or 'google' in provider.lower():
                        self.performance_stats['cloud_stt_success'] += 1
                    else:
                        self.performance_stats['local_stt_success'] += 1
                
                # Update average STT time
                current_avg = self.performance_stats['avg_stt_time']
                calls = self.performance_stats['stt_calls']
                self.performance_stats['avg_stt_time'] = (current_avg * (calls - 1) + duration) / calls
                
            elif service_type == 'tts':
                if success:
                    if 'gcp' in provider.lower() or 'google' in provider.lower():
                        self.performance_stats['cloud_tts_success'] += 1
                    else:
                        self.performance_stats['local_tts_success'] += 1
                
                # Update average TTS time
                current_avg = self.performance_stats['avg_tts_time']
                calls = self.performance_stats['tts_calls']
                self.performance_stats['avg_tts_time'] = (current_avg * (calls - 1) + duration) / calls
                
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")

# Global hybrid speech manager instance
_hybrid_speech_manager = None

def get_hybrid_speech_manager(audio_folder: str) -> HybridSpeechManager:
    """Get or create the global hybrid speech manager instance."""
    global _hybrid_speech_manager
    
    if _hybrid_speech_manager is None:
        _hybrid_speech_manager = HybridSpeechManager(audio_folder)
    
    return _hybrid_speech_manager