"""
Flask application for the RAG System with Speech Interaction.
Provides a web interface for converting text to speech and speech to text
with local RAG (Retrieval-Augmented Generation) capabilities.
"""

import os
import sys
import json
import uuid
import pyttsx3
import tempfile
import threading
import pyaudio
import speech_recognition as sr
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from datetime import datetime

# Configure logging for optimal performance
logging.basicConfig(
    level=logging.WARNING,  # Reduced from INFO for speed
    format='%(asctime)s - %(levelname)s - %(message)s',  # Shorter format
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import RAG system components
from web_app.embeddings import get_embedding_generator
from web_app.vector_store import get_vector_store
from web_app.llm import get_llm
from web_app.rag_pipeline import get_rag_pipeline, ConversationMemory

# Add the project root to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# Configure the static folder for audio files
AUDIO_FOLDER = os.path.join(app.static_folder, 'audio')
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Configure upload folder for audio recordings
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize TTS engine
engine = None

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize RAG components
rag_pipeline = None
conversation_memory = ConversationMemory(max_history=5)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4000
RECORD_SECONDS = 5

def init_rag_components():
    """Initialize RAG pipeline components."""
    global rag_pipeline
    if rag_pipeline is None:
        try:
            rag_pipeline = get_rag_pipeline()
            print("✅ RAG pipeline initialized")
        except Exception as e:
            print(f"❌ RAG pipeline initialization failed: {e}")
            rag_pipeline = None

def get_engine():
    """Get or initialize the TTS engine with proper driver selection for WSL."""
    global engine
    if engine is None:
        import platform
        system = platform.system()
        
        print(f"Initializing TTS on {system}...")
        
        # Check for espeak on Linux/WSL first
        if system == 'Linux':
            try:
                import subprocess
                result = subprocess.run(['which', 'espeak'], capture_output=True)
                if result.returncode != 0:
                    print("❌ espeak not found! Install with:")
                    print("sudo apt update && sudo apt install -y espeak espeak-data libespeak1")
            except:
                pass
        
        # Try drivers in order
        drivers = ['espeak', 'sapi5', 'nsss'] if system == 'Linux' else ['sapi5', 'espeak', 'nsss']
        
        for driver in drivers:
            try:
                print(f"Trying TTS driver: {driver}")
                test_engine = pyttsx3.init(driver)
                if test_engine:
                    voices = test_engine.getProperty('voices')
                    if voices and len(voices) > 0:
                        print(f"✅ {driver} working with {len(voices)} voices")
                        engine = test_engine
                        return engine
                    else:
                        print(f"❌ {driver} has no voices")
            except Exception as e:
                print(f"❌ {driver} failed: {e}")
        
        # Last resort: try default
        try:
            print("Trying default TTS...")
            engine = pyttsx3.init()
            if engine and engine.getProperty('voices'):
                print("✅ Default TTS working")
                return engine
        except Exception as e:
            print(f"❌ Default TTS failed: {e}")
        
        # All failed
        error_msg = "No TTS engine working!"
        if system == 'Linux':
            error_msg += "\nInstall espeak: sudo apt install -y espeak espeak-data libespeak1"
        raise Exception(error_msg)
    
    return engine



def convert_webm_to_wav(webm_path):
    """Convert WebM to WAV using ffmpeg - simplified and reliable."""
    import subprocess
    import os
    
    # Verify the input file exists and has content
    if not os.path.exists(webm_path):
        print(f"ERROR: Input WebM file does not exist: {webm_path}")
        return None
        
    file_size = os.path.getsize(webm_path)
    if file_size == 0:
        print(f"ERROR: Input WebM file is empty: {webm_path}")
        return None
    
    print(f"Converting WebM to WAV: {webm_path} (size: {file_size} bytes)")
    wav_path = webm_path.replace('.webm', '.wav')
    
    try:
        # Use ffmpeg for reliable conversion
        cmd = [
            "ffmpeg", "-y",  # -y to overwrite output file
            "-i", webm_path,
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",          # 16kHz sample rate
            "-ac", "1",              # Mono
            wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
            print(f"FFmpeg conversion successful: {wav_path} ({os.path.getsize(wav_path)} bytes)")
            return wav_path
        else:
            print(f"FFmpeg conversion failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("FFmpeg conversion timed out")
        return None
    except FileNotFoundError:
        print("FFmpeg not found. Please install ffmpeg.")
        return None
    except Exception as e:
        print(f"Error during WebM conversion: {e}")
        return None

def install_vosk_if_needed():
    """Install vosk package if not already installed."""
    try:
        import vosk
        return True
    except ImportError:
        print("Installing vosk package for offline speech recognition...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vosk"])
            return True
        except Exception as e:
            print(f"Failed to install vosk: {e}")
            return False

def get_model_path(language='en'):
    """Get the path to the local Vosk model. Models must be pre-installed."""
    from pathlib import Path
    
    models_dir = Path("web_app/models")
    
    # Define model names based on what's expected to be pre-installed
    if language == 'en':
        model_name = "vosk-model-small-en-us-0.15"
    elif language == 'es':
        model_name = "vosk-model-small-es-0.42"
    else:
        model_name = "vosk-model-small-en-us-0.15"  # Default to English
    
    model_path = models_dir / model_name
    
    # Check if model exists locally
    if model_path.exists() and model_path.is_dir():
        print(f"✅ Using local model: {model_name}")
        return str(model_path)
    
    # Model not found - this is a configuration error
    print(f"❌ Model {model_name} not found at {model_path}")
    print(f"📋 Please ensure speech recognition models are installed in {models_dir}")
    print("   Required models:")
    print("   - vosk-model-small-en-us-0.15/ (English)")
    print("   - vosk-model-small-es-0.42/ (Spanish)")
    print("   See README.md for model installation instructions")
    return None

def transcribe_audio_file(file_path, language='en-US'):
    """Transcribe audio using Vosk for 100% offline speech recognition."""
    import os
    from pathlib import Path
    import json
    import time
    import shutil
    import subprocess
    import wave
    import numpy as np
    
    print(f"Transcribing {file_path} in {language}")
    
    # Language detection
    if language == 'es-ES':
        lang_code = 'es'
    else:
        lang_code = 'en'
    
    print(f"Using {lang_code.upper()} recognition")
    

    
    # If file is WebM format, convert it to WAV
    if file_path.lower().endswith('.webm'):
        wav_file = convert_webm_to_wav(file_path)
        if wav_file:
            file_path = wav_file
            print(f"Using converted WAV file: {file_path}")
        else:
            return {
                'success': False,
                'error': "Could not convert WebM audio format to WAV"
            }
    
    # Check if the file exists
    if not os.path.exists(file_path):
        return {
            'success': False,
            'error': "Audio file not found"
        }
        
    try:
        # Install vosk if needed
        if not install_vosk_if_needed():
            return {
                'success': False,
                'error': "Could not install required speech recognition libraries"
            }
            
        import vosk
        import json
        import wave
        import subprocess
        
        # Get model path (must be pre-installed)
        model_path = get_model_path(lang_code)
        if not model_path:
            return {
                'success': False,
                'error': f"Could not load speech recognition model for language: {lang_code}"
            }
        
        print(f"Using Vosk model at: {model_path}")
        
        # Load the model (works for both English and Spanish)
        try:
            model = vosk.Model(model_path)
            print(f"{lang_code.upper()} model loaded successfully")
        except Exception as model_error:
            print(f"Error loading {lang_code} model: {model_error}")
            return {
                'success': False,
                'error': f"Could not load {lang_code} speech model: {str(model_error)}"
            }
        
        try:
            wf = wave.open(file_path, "rb")
        except Exception as wav_error:
            print(f"Error opening WAV file: {wav_error}")
            return {
                'success': False,
                'error': "Could not open audio file for processing"
            }
        
        # Check if format is correct
        if wf.getnchannels() != 1:
            wf.close()
            return {
                'success': False,
                'error': "Audio must be mono (single channel)"
            }
            
        print(f"Audio file details: rate={wf.getframerate()}, channels={wf.getnchannels()}, width={wf.getsampwidth()}")
        
        # Create recognizer - simplified approach for both languages
        # We're using the exact same approach for both languages now
        # This avoids any configuration issues that might be causing problems
        print(f"Creating {lang_code} recognizer with standard settings")
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        
        # Enable word timestamps for both languages
        # This is helpful for timing information but doesn't affect recognition
        rec.SetWords(True)
        
        print(f"{lang_code} recognizer ready for processing")
            
        # Process audio with careful handling
        result_text = ""
        try:
            print("Processing audio data...")
            # First verify the WAV file is valid
            try:
                data_size = os.path.getsize(file_path)
                print(f"Audio file size: {data_size} bytes")
                if data_size < 100:  # Suspiciously small
                    print("WARNING: Audio file is suspiciously small!")
                    return {
                        'success': False,
                        'error': "Invalid audio file detected"
                    }
            except Exception as size_error:
                print(f"Could not check file size: {size_error}")
            
            # First try processing in smaller chunks
            print("Processing audio in chunks for better recognition...")
            chunk_size = 4000  # Process 0.25 seconds at a time at 16kHz
            total_frames = wf.getnframes()
            frame_count = 0
            
            # Process in chunks for better recognition
            while frame_count < total_frames:
                frames_to_read = min(chunk_size, total_frames - frame_count)
                data = wf.readframes(frames_to_read)
                frame_count += frames_to_read
                
                if len(data) > 0:
                    if rec.AcceptWaveform(data):
                        partial_result = json.loads(rec.Result())
                        if partial_result.get("text", ""):
                            print(f"Recognized partial text: {partial_result['text']}")
                
            # Final result
            final_result = json.loads(rec.FinalResult())
            print(f"Final recognition result: {final_result}")
            
            # If no text was recognized, try processing the entire file at once
            if not final_result.get("text", ""):
                print("No text recognized in chunks, trying entire file at once...")
                # Reset the file and recognizer
                wf.rewind()
                rec = vosk.KaldiRecognizer(model, wf.getframerate())
                rec.SetWords(True)
                
                # Read the entire file
                all_data = wf.readframes(wf.getnframes())
                if len(all_data) > 0:
                    print(f"Processing entire file ({len(all_data)} bytes) at once")
                    rec.AcceptWaveform(all_data)
                    final_result = json.loads(rec.FinalResult())
                    print(f"Full-file processing result: {final_result}")
                else:
                    print("WARNING: No audio data read from file!")
            
            print("Finished processing audio data")
            # Use the final result for further processing
            result_json = final_result
        except Exception as proc_error:
            print(f"Error during audio processing: {proc_error}")
            return {
                'success': False,
                'error': f"Error processing audio: {proc_error}"
            }
            
        # Get the text from the result JSON
        result_text = result_json.get("text", "")
        
        print(f"Recognition result: '{result_text}'")
        
        # Check language-specific handling of results
        if lang_code == 'es':
            # Special handling for Spanish results
            if not result_text or len(result_text.strip()) == 0:
                print("No Spanish text detected in the audio")
                return {
                    'success': False,
                    'error': "No Spanish speech detected. Try speaking louder or more clearly."
                }
            
            # Look for signs that English was incorrectly detected
            english_words = ['the', 'and', 'or', 'but', 'for', 'with', 'your', 'this', 'that']
            word_count = len(result_text.split())
            english_word_count = sum(1 for word in result_text.lower().split() if word in english_words)
            
            # If more than 30% of words are common English words, something's wrong
            if word_count > 3 and english_word_count / word_count > 0.3:
                print(f"WARNING: Detected {english_word_count}/{word_count} English words in Spanish audio")
                return {
                    'success': False,
                    'error': "La detección de voz en español no funcionó correctamente. Por favor, intente de nuevo."
                }
        else:
            # Standard handling for English
            if not result_text or len(result_text.strip()) == 0:
                return {
                    'success': False,
                    'error': "No speech detected. Try speaking louder or more clearly."
                }
            
        return {
            'success': True,
            'text': result_text
        }
        
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
        return {
            'success': False,
            'error': "Speech could not be understood. Try speaking more clearly."
        }
    except sr.RequestError as e:
        print(f"Could not request results from speech recognition service; {e}")
        return {
            'success': False,
            'error': f"Speech recognition service error: {e}"
        }
    except Exception as audio_error:
        import traceback
        traceback.print_exc()
        print(f"Error processing audio file: {audio_error}")
        return {
            'success': False,
            'error': f"Audio file could not be processed - {str(audio_error)}"
        }




def get_available_voices():
    """Get available TTS voices - one English (US) and one Spanish (Latin America) only."""
    e = get_engine()
    voices = e.getProperty('voices')
    
    english_voice = None
    spanish_voice = None
    
    for voice in voices:
        voice_name = voice.name if hasattr(voice, 'name') else voice.id.split('\\')[-1]
        voice_id = voice.id
        languages = voice.languages if hasattr(voice, 'languages') else []
        
        # Convert languages to strings for easier checking
        lang_strings = [str(lang).lower() for lang in languages]
        voice_name_lower = voice_name.lower()
        
        # Look for Spanish voice (prioritize Latin American Spanish)
        if not spanish_voice:
            is_spanish = any('es' in lang or 'spanish' in lang for lang in lang_strings) or \
                        'spanish' in voice_name_lower or 'español' in voice_name_lower or \
                        'mexico' in voice_name_lower or 'latin' in voice_name_lower or \
                        'sabina' in voice_name_lower or 'es_' in voice_name_lower
            if is_spanish:
                spanish_voice = {
                    'id': voice_id,
                    'name': 'Spanish (Latin America)',
                    'language_type': 'Spanish',
                    'languages': languages
                }
        
        # Look for English US voice
        if not english_voice:
            is_english_us = any('en' in lang and ('us' in lang or 'united' in lang) for lang in lang_strings) or \
                           ('english' in voice_name_lower and ('us' in voice_name_lower or 'united' in voice_name_lower)) or \
                           'david' in voice_name_lower or 'zira' in voice_name_lower
            if is_english_us:
                english_voice = {
                    'id': voice_id,
                    'name': 'English (US)',
                    'language_type': 'English',
                    'languages': languages
                }
        
        # Stop searching if we found both
        if spanish_voice and english_voice:
            break
    
    # Build the final list with Spanish first (as default)
    available_voices = []
    if spanish_voice:
        available_voices.append(spanish_voice)
    if english_voice:
        available_voices.append(english_voice)
    
    print(f"Available voices: {[v['name'] for v in available_voices]}")
    return available_voices

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    """Handle audio file upload for speech recognition."""
    try:
        print("\n\n============= NEW AUDIO UPLOAD REQUEST =============\n\n")
        
        # Check if language was specified - handle with more robust language detection
        language = request.form.get('language', 'en-US').strip()
        # Make sure language code is properly formatted using a unified approach
        if language.lower() in ['spanish', 'español', 'es', 'es-es', 'es-mx', 'es-ar', 'es-co']:
            language = 'es-ES'
            print("SPANISH LANGUAGE REQUESTED")
        else:
            language = 'en-US'
            print("ENGLISH LANGUAGE REQUESTED OR DEFAULTED")
            
        print(f"Final language parameter: {language}")
        
        # Check if the post request has the file part
        if 'audio' not in request.files:
            print("No audio file part in request")
            return jsonify({'success': False, 'error': 'No audio file part'})
        
        audio_file = request.files['audio']
        print(f"Received file: {audio_file.filename}, Content-Type: {audio_file.content_type}")
        
        # If user does not select file, browser submits an empty file
        if audio_file.filename == '':
            print("Empty filename")
            return jsonify({'success': False, 'error': 'No audio file selected'})
        
        # Directly use WAV files if possible (thanks to our client-side conversion)
        if 'wav' in audio_file.content_type or audio_file.filename.lower().endswith('.wav'):
            print("Received WAV file directly from client - excellent!")
            file_ext = '.wav'
        else:
            # Fall back to WebM handling if needed
            file_ext = '.webm'
            print("Received WebM file, will need conversion")
        
        # Save the file with a temporary name
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_path = temp_file.name
            audio_file.save(temp_path)
            print(f"Saved audio to temporary file: {temp_path}")
            
            # Check if file was actually saved and has content
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                print(f"Confirmed file was saved with size: {os.path.getsize(temp_path)} bytes")
            else:
                print("Warning: File may not have been saved properly")
        
        # Show a message that we're using offline recognition only
        print("Using 100% offline speech recognition (no APIs)")
        
        # Transcribe the audio file
        print("Starting transcription")
        transcript_result = transcribe_audio_file(temp_path, language)
        print(f"Transcription result: {transcript_result}")
        
        # Clean up - remove temporary files
        try:
            os.remove(temp_path)
            
            # If we converted from WebM to WAV, clean up the WAV file too
            if file_ext == '.webm':
                wav_path = temp_path.replace('.webm', '.wav')
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                
                # Also check for models directory
                model_dir = os.path.join('web_app', 'models')
                if os.path.exists(model_dir):
                    print(f"Models directory exists at {model_dir}")
        except Exception as e:
            print(f"Warning: Could not remove temporary files: {str(e)}")
        
        # Check transcription success
        if not transcript_result['success']:
            print(f"Returning error: {transcript_result['error']}")
            return jsonify({'success': False, 'error': transcript_result['error']})
        
        # Return successful transcription result
        transcript_text = transcript_result['text']
        print(f"Returning transcript: {transcript_text}")
        return jsonify({'success': True, 'transcript': transcript_text})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Exception in upload_audio: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/voices')
def voices():
    """Get all available voices."""
    try:
        available_voices = get_available_voices()
        return jsonify({'success': True, 'voices': available_voices})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Convert text to speech with enhanced error handling for WSL."""
    try:
        data = request.json
        text = data.get('text', '')
        voice_id = data.get('voice_id', '')
        
        print(f"TTS Request - Text: '{text}', Voice ID: '{voice_id}'")
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        # Create a fresh TTS engine for this specific request
        try:
            print("Creating fresh TTS engine...")
            # Use direct pyttsx3.init() to avoid the global engine cache
            import pyttsx3
            e = pyttsx3.init()
            
            if not e:
                return jsonify({'success': False, 'error': 'Could not create TTS engine'})
            
            # Get available voices
            voices = e.getProperty('voices')
            print(f"TTS engine has {len(voices) if voices else 0} voices")
            
            # Set voice if specified
            if voice_id and voices:
                voice_found = False
                for voice in voices:
                    if voice.id == voice_id:
                        print(f"Setting voice: {voice.name}")
                        e.setProperty('voice', voice_id)
                        voice_found = True
                        break
                
                if not voice_found:
                    print(f"Voice {voice_id} not found, using default")
            
            # Set properties for clear speech
            e.setProperty('rate', 150)  # Speech rate
            e.setProperty('volume', 1.0)  # Max volume
            
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"speech_{timestamp}_{unique_id}.wav"
            output_path = os.path.join(AUDIO_FOLDER, filename)
            
            print(f"Generating speech file: {output_path}")
            print(f"Text length: {len(text)} characters")
            
            # Generate speech with error handling
            try:
                # Detect environment and choose TTS engine
                import platform
                import subprocess
                
                system_info = platform.uname()
                
                if platform.system() == 'Linux' and 'microsoft' in system_info.release.lower():
                    print("Using Windows TTS from WSL")
                    
                    # Determine Windows voice based on selected voice_id
                    windows_voice = 'Microsoft Sabina Desktop'  # Default to Spanish voice
                    
                    if voice_id and voices:
                        for voice in voices:
                            if voice.id == voice_id:
                                voice_name = voice.name.lower() if hasattr(voice, 'name') else ''
                                print(f"Selected voice: {voice.name}")
                                
                                # Map to Windows voices
                                if 'english' in voice_name:
                                    windows_voice = 'Microsoft Zira Desktop'  # English US female
                                    print("Using Windows English voice")
                                elif 'spanish' in voice_name or 'español' in voice_name:
                                    windows_voice = 'Microsoft Sabina Desktop'  # Spanish female
                                    print("Using Windows Spanish voice")
                                break
                    
                    # Convert WSL path to Windows path for PowerShell
                    windows_path = output_path.replace('/mnt/', '').replace('/', '\\')
                    if windows_path.startswith('d\\'):
                        windows_path = 'D:\\' + windows_path[2:]
                    
                    # Use Windows TTS via PowerShell to generate WAV file
                    ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("{windows_voice}")
$synth.Rate = 0
$synth.SetOutputToWaveFile("{windows_path}")
$synth.Speak("{text.replace('"', '`"')}")
$synth.Dispose()
'''
                    
                    try:
                        result = subprocess.run([
                            'powershell.exe', '-Command', ps_script
                        ], capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            print("Windows TTS generation successful")
                        else:
                            print(f"Windows TTS error: {result.stderr}")
                            raise Exception(f"Windows TTS failed: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        raise Exception("Windows TTS timed out")
                        
                elif platform.system() == 'Linux':
                    print("Using espeak-ng for pure Linux")
                    
                    # Use espeak-ng (better than espeak) for pure Linux
                    cmd = [
                        'espeak-ng',
                        '-v', 'es-la' if 'spanish' in (voice_id or '').lower() else 'en-us',
                        '-w', output_path,
                        '-s', '150',  # Natural speed
                        text
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"espeak-ng failed: {result.stderr}")
                        
                else:
                    # Use pyttsx3 for Windows
                    e.save_to_file(text, output_path)
                    print("save_to_file completed")
                    e.runAndWait()
                    print("runAndWait completed")
                    
            except Exception as tts_error:
                print(f"TTS generation error: {tts_error}")
                return jsonify({'success': False, 'error': f'TTS generation failed: {str(tts_error)}'})
            
            # Wait for file to be fully written
            import time
            time.sleep(1.0)  # Increased wait time
            
            # Check if file was created successfully
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"Speech file exists with size: {file_size} bytes")
                
                if file_size > 100:  # Minimum reasonable file size
                    # Try to get audio duration for debugging
                    try:
                        import wave
                        with wave.open(output_path, 'rb') as wav_file:
                            frames = wav_file.getnframes()
                            rate = wav_file.getframerate()
                            duration = frames / float(rate)
                            print(f"Audio duration: {duration:.2f} seconds")
                            
                            if duration < 0.1:  # Less than 100ms is suspicious
                                print("WARNING: Audio duration is very short!")
                    except Exception as wave_error:
                        print(f"Could not analyze audio file: {wave_error}")
                    
                    # Return the URL to the audio file
                    audio_url = f"/static/audio/{filename}"
                    print(f"Success! Audio URL: {audio_url}")
                    return jsonify({
                        'success': True, 
                        'audio_url': audio_url, 
                        'message': 'Audio generated successfully',
                        'file_size': file_size
                    })
                else:
                    print(f"Speech file too small ({file_size} bytes), likely empty")
                    return jsonify({'success': False, 'error': 'Generated audio file is empty or corrupted'})
            else:
                print("Speech file was not created")
                return jsonify({'success': False, 'error': 'Failed to generate speech file'})
                
        except Exception as engine_error:
            print(f"TTS Engine Error: {engine_error}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'TTS engine error: {str(engine_error)}'})
        
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

# Configure basic logging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# API endpoint for RAG queries
@app.route('/rag_query', methods=['POST'])
def rag_query():
    """Process a query through the RAG pipeline."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        
        query = data['query']
        use_memory = data.get('use_memory', True)
        
        # Check if RAG pipeline is initialized
        global rag_pipeline
        if rag_pipeline is None:
            # Try initializing again
            init_rag_components()
            if rag_pipeline is None:
                return jsonify({'success': False, 'error': 'RAG system not initialized'}), 500
        
        # Get conversation context if using memory
        context = None
        if use_memory and conversation_memory.history:
            context = conversation_memory.get_context_string()
        
        # Process query through RAG pipeline
        response = rag_pipeline.process_query(query)
        
        # Add interaction to conversation memory
        if use_memory:
            conversation_memory.add_interaction(query, response['answer'])
        
        return jsonify({
            'success': True,
            'answer': response['answer'],
            'retrieved_documents': response.get('retrieved_documents', []),
            'metrics': response.get('metrics', {})
        })
        
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/add_document', methods=['POST'])
def add_document():
    """Add a document to the RAG knowledge base."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No document text provided'}), 400
        
        text = data['text']
        metadata = data.get('metadata', {})
        
        # Ensure metadata is a dictionary, not a list or other type
        if metadata and not isinstance(metadata, dict):
            return jsonify({'success': False, 'error': 'Metadata must be a dictionary'}), 400
        
        # Check if RAG pipeline is initialized
        global rag_pipeline
        if rag_pipeline is None:
            # Try initializing again
            init_rag_components()
            if rag_pipeline is None:
                return jsonify({'success': False, 'error': 'RAG system not initialized'}), 500
        
        # Add document to knowledge base - pass metadata directly, not as a list
        doc_id = rag_pipeline.add_documents(text, metadatas=metadata)
        
        return jsonify({
            'success': True,
            'document_id': doc_id[0] if doc_id else None
        })
        
    except Exception as e:
        logger.error(f"Error adding document to RAG system: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear the conversation history."""
    try:
        conversation_memory.clear()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Speech-to-RAG endpoint - combines speech recognition with RAG
@app.route('/speech_to_rag', methods=['POST'])
def speech_to_rag():
    """Process speech input through speech recognition and RAG pipeline."""
    try:
        logger.info("Processing speech-to-RAG request")
        
        # Check if file is in the request
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({'success': False, 'error': 'No audio file in request'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'en-US')
        use_memory = request.form.get('use_memory', 'true').lower() == 'true'
        
        logger.info(f"Received audio with language: {language}, use_memory: {use_memory}")
        
        # Log the content type of the received file
        content_type = audio_file.content_type if hasattr(audio_file, 'content_type') else 'unknown'
        logger.info(f"Received audio with content type: {content_type}")
        
        # Verify audio file is not empty
        audio_file.seek(0, os.SEEK_END)
        file_size = audio_file.tell()
        audio_file.seek(0)
        
        if file_size == 0:
            logger.error("Received empty audio file")
            return jsonify({'success': False, 'error': 'Empty audio file received'}), 400
            
        # Create a dedicated folder for this request
        req_id = str(uuid.uuid4())
        req_folder = os.path.join(UPLOAD_FOLDER, req_id)
        os.makedirs(req_folder, exist_ok=True)
        
        # Save audio file temporarily with an appropriate extension based on content type
        if 'webm' in content_type.lower():
            audio_ext = '.webm'
        elif 'wav' in content_type.lower():
            audio_ext = '.wav'
        elif 'ogg' in content_type.lower() or 'opus' in content_type.lower():
            audio_ext = '.ogg'
        else:
            # Default to webm as it's most common from browsers
            audio_ext = '.webm'
            
        audio_filename = secure_filename("recording" + audio_ext)
        audio_path = os.path.join(req_folder, audio_filename)
        audio_file.save(audio_path)
        
        logger.info(f"Saved audio file to {audio_path}, file size: {file_size} bytes, format: {audio_ext}")
        
        # Process speech to text
        logger.info(f"Starting transcription for {audio_filename}")
        transcription_result = transcribe_audio_file(audio_path, language)
        logger.info(f"Transcription complete: {transcription_result}")
        
        if not transcription_result['success']:
            logger.error(f"Transcription failed: {transcription_result['error']}")
            return jsonify(transcription_result), 400
        
        text = transcription_result['text']
        
        # If text is empty, return early
        if not text.strip():
            logger.error("Transcription produced empty text")
            return jsonify({
                'success': False,
                'error': 'Could not transcribe audio or audio was silent'
            }), 400
        
        logger.info(f"Successfully transcribed: '{text}'")
        
        # Process through RAG pipeline
        global rag_pipeline
        if rag_pipeline is None:
            logger.warning("RAG pipeline not initialized, attempting to initialize")
            # Try initializing again
            init_rag_components()
            if rag_pipeline is None:
                logger.error("Failed to initialize RAG pipeline")
                return jsonify({'success': False, 'error': 'RAG system not initialized'}), 500
        
        # Process query through RAG pipeline
        logger.info(f"Processing query through RAG pipeline: '{text}'")
        try:
            rag_response = rag_pipeline.process_query(text)
            logger.info("RAG processing complete")
        except Exception as rag_error:
            logger.error(f"Error in RAG processing: {rag_error}")
            return jsonify({
                'success': False, 
                'error': f'RAG processing error: {str(rag_error)}',
                'transcription': text  # At least return the transcription
            }), 500
        
        # Add to conversation memory if requested
        if use_memory:
            conversation_memory.add_interaction(text, rag_response['answer'])
            logger.info("Added interaction to conversation memory")
        
        # Cleanup temporary file
        try:
            os.remove(audio_path)
            logger.info(f"Removed temporary audio file: {audio_path}")
        except Exception as cleanup_error:
            logger.warning(f"Could not remove temporary file {audio_path}: {cleanup_error}")
        
        # Return combined result
        logger.info("Returning speech-to-RAG response")
        return jsonify({
            'success': True,
            'transcription': text,
            'answer': rag_response['answer'],
            'retrieved_documents': rag_response.get('retrieved_documents', []),
            'metrics': rag_response.get('metrics', {})
        })
        
    except Exception as e:
        logger.error(f"Unhandled error in speech to RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

# Knowledge base statistics endpoint
@app.route('/kb_stats', methods=['GET'])
def kb_stats():
    """Get statistics about the knowledge base."""
    try:
        # Check if RAG pipeline is initialized
        global rag_pipeline
        if rag_pipeline is None:
            # Try initializing again
            init_rag_components()
            if rag_pipeline is None:
                # Return empty stats if still not initialized
                return jsonify({
                    'success': True,
                    'count': 0,
                    'collection_name': 'rag_documents',
                    'status': 'No vector store available'
                })
        
        # Get vector store stats if available
        try:
            stats = rag_pipeline.vector_store.get_stats()
            status = 'Ready' if not hasattr(rag_pipeline.llm, 'is_mock_mode') or not rag_pipeline.llm.is_mock_mode else 'Limited (LLM model not available)'
            
            return jsonify({
                'success': True,
                'count': stats.get('count', 0),
                'collection_name': stats.get('collection_name', 'rag_documents'),
                'status': status
            })
        except Exception as store_error:
            logger.warning(f"Error getting vector store stats: {store_error}")
            return jsonify({
                'success': True,
                'count': 0,
                'collection_name': 'rag_documents',
                'status': 'Vector store error: ' + str(store_error)
            })
        
    except Exception as e:
        logger.error(f"Error getting knowledge base stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': True,  # Return success with empty data rather than error
            'count': 0,
            'collection_name': 'unknown',
            'status': 'Error: ' + str(e)
        })

# List all documents in the knowledge base
@app.route('/list_documents', methods=['GET'])
def list_documents():
    """List all documents in the knowledge base."""
    try:
        # Check if RAG pipeline is initialized
        global rag_pipeline
        if rag_pipeline is None:
            # Try initializing again
            init_rag_components()
            if rag_pipeline is None:
                return jsonify({
                    'success': False,
                    'error': 'RAG system not initialized'
                }), 500
        
        # Get all documents
        documents = rag_pipeline.get_all_documents()
        
        return jsonify({
            'success': True,
            'documents': {
                'ids': documents.get('ids', []),
                'documents': documents.get('documents', []),
                'metadatas': documents.get('metadatas', [])
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing documents from knowledge base: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Delete documents from the knowledge base
@app.route('/delete_documents', methods=['POST'])
def delete_documents():
    """Delete documents from the knowledge base."""
    try:
        data = request.get_json()
        if not data or 'doc_ids' not in data:
            return jsonify({
                'success': False,
                'error': 'No document IDs provided'
            }), 400
        
        doc_ids = data['doc_ids']
        
        # Ensure doc_ids is a list
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        
        if not isinstance(doc_ids, list):
            return jsonify({
                'success': False,
                'error': 'Document IDs must be a string or list of strings'
            }), 400
        
        # Check if RAG pipeline is initialized
        global rag_pipeline
        if rag_pipeline is None:
            # Try initializing again
            init_rag_components()
            if rag_pipeline is None:
                return jsonify({
                    'success': False,
                    'error': 'RAG system not initialized'
                }), 500
        
        # Delete documents
        rag_pipeline.delete_documents(doc_ids)
        
        return jsonify({
            'success': True,
            'deleted_count': len(doc_ids),
            'deleted_ids': doc_ids
        })
        
    except Exception as e:
        logger.error(f"Error deleting documents from knowledge base: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Update document in the knowledge base
@app.route('/update_document', methods=['POST'])
def update_document():
    """Update a document in the knowledge base."""
    try:
        data = request.get_json()
        if not data or 'doc_id' not in data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Document ID and text are required'
            }), 400
        
        doc_id = data['doc_id']
        text = data['text']
        metadata = data.get('metadata', {})
        
        # Ensure metadata is a dictionary
        if metadata and not isinstance(metadata, dict):
            return jsonify({
                'success': False,
                'error': 'Metadata must be a dictionary'
            }), 400
        
        # Check if RAG pipeline is initialized
        global rag_pipeline
        if rag_pipeline is None:
            # Try initializing again
            init_rag_components()
            if rag_pipeline is None:
                return jsonify({
                    'success': False,
                    'error': 'RAG system not initialized'
                }), 500
        
        # Update document
        rag_pipeline.update_document(doc_id, text, metadata)
        
        return jsonify({
            'success': True,
            'doc_id': doc_id
        })
        
    except Exception as e:
        logger.error(f"Error updating document in knowledge base: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# RAG-to-speech endpoint - generates speech from RAG response
@app.route('/rag_to_speech', methods=['POST'])
def rag_to_speech():
    """Process a query through RAG and convert response to speech."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        
        query = data['query']
        voice_id = data.get('voice_id', None)
        use_memory = data.get('use_memory', True)
        
        # Check if RAG pipeline is initialized
        global rag_pipeline
        if rag_pipeline is None:
            # Try initializing again
            init_rag_components()
            if rag_pipeline is None:
                return jsonify({'success': False, 'error': 'RAG system not initialized'}), 500
        
        # Process query through RAG pipeline
        rag_response = rag_pipeline.process_query(query)
        answer_text = rag_response['answer']
        
        # Add to conversation memory if requested
        if use_memory:
            conversation_memory.add_interaction(query, answer_text)
        
        # Generate speech from answer using enhanced TTS
        try:
            print("Creating fresh TTS engine for RAG response...")
            engine = get_engine()
            
            # Get available voices for debugging
            voices = engine.getProperty('voices')
            print(f"Available voices for RAG: {len(voices) if voices else 0}")
            
            # Set voice if provided and valid
            if voice_id and voices:
                voice_found = False
                for voice in voices:
                    if voice.id == voice_id:
                        print(f"Setting RAG voice to: {voice.name} ({voice_id})")
                        engine.setProperty('voice', voice_id)
                        voice_found = True
                        break
                
                if not voice_found:
                    print(f"RAG Voice ID {voice_id} not found, using default")
                    if voices and len(voices) > 0:
                        fallback_voice = voices[0].id
                        print(f"Using fallback RAG voice: {voices[0].name} ({fallback_voice})")
                        engine.setProperty('voice', fallback_voice)
            else:
                print("No voice specified for RAG response, using default")
            
            # Set speech properties for better quality
            engine.setProperty('rate', 140)  # Slower for clarity
            engine.setProperty('volume', 1.0)  # Maximum volume
            
            # Generate unique filename (use WAV for better compatibility)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            audio_filename = f"rag_{timestamp}_{unique_id}.wav"
            audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
            
            print(f"Generating RAG speech file: {audio_path}")
            print(f"Text to speak ({len(answer_text)} chars): '{answer_text[:100]}{'...' if len(answer_text) > 100 else ''}'")
            
            # Generate speech file with error handling using same logic as /synthesize
            try:
                # Detect environment and choose TTS engine (same as /synthesize endpoint)
                import platform
                import subprocess
                
                system_info = platform.uname()
                
                if platform.system() == 'Linux' and 'microsoft' in system_info.release.lower():
                    print("Using Windows TTS from WSL for RAG response")
                    
                    # Determine Windows voice based on selected voice_id (same mapping as /synthesize)
                    windows_voice = 'Microsoft Sabina Desktop'  # Default to Spanish voice
                    
                    if voice_id and voices:
                        for voice in voices:
                            if voice.id == voice_id:
                                voice_name = voice.name.lower() if hasattr(voice, 'name') else ''
                                print(f"RAG using voice: {voice.name}")
                                
                                # Map to Windows voices (same as /synthesize)
                                if 'english' in voice_name:
                                    windows_voice = 'Microsoft Zira Desktop'  # English US female
                                    print("Using Windows English voice for RAG")
                                elif 'spanish' in voice_name or 'español' in voice_name:
                                    windows_voice = 'Microsoft Sabina Desktop'  # Spanish female
                                    print("Using Windows Spanish voice for RAG")
                                break
                    
                    # Convert WSL path to Windows path for PowerShell (same as /synthesize)
                    windows_path = audio_path.replace('/mnt/', '').replace('/', '\\')
                    if windows_path.startswith('d\\'):
                        windows_path = 'D:\\' + windows_path[2:]
                    
                    # Use Windows TTS via PowerShell to generate WAV file (same as /synthesize)
                    ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("{windows_voice}")
$synth.Rate = 0
$synth.SetOutputToWaveFile("{windows_path}")
$synth.Speak("{answer_text.replace('"', '`"')}")
$synth.Dispose()
'''
                    
                    try:
                        result = subprocess.run([
                            'powershell.exe', '-Command', ps_script
                        ], capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            print("Windows TTS RAG generation successful")
                        else:
                            print(f"Windows TTS RAG error: {result.stderr}")
                            raise Exception(f"Windows TTS failed: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        raise Exception("Windows TTS RAG timed out")
                        
                elif platform.system() == 'Linux':
                    print("Using espeak-ng for pure Linux RAG response")
                    
                    # Use espeak-ng (better than espeak) for pure Linux (same as /synthesize)
                    cmd = [
                        'espeak-ng',
                        '-v', 'es-la' if 'spanish' in (voice_id or '').lower() else 'en-us',
                        '-w', audio_path,
                        '-s', '150',  # Natural speed
                        answer_text
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"espeak-ng failed: {result.stderr}")
                        
                else:
                    # Use pyttsx3 for Windows (same as /synthesize)
                    engine.save_to_file(answer_text, audio_path)
                    print("RAG save_to_file completed")
                    engine.runAndWait()
                    print("RAG runAndWait completed")
                    
            except Exception as tts_error:
                print(f"RAG TTS generation error: {tts_error}")
                # Continue without audio but return the text response
                return jsonify({
                    'success': True,
                    'query': query,
                    'answer': answer_text,
                    'audio_url': None,
                    'error': f'TTS generation failed: {str(tts_error)}',
                    'retrieved_documents': rag_response.get('retrieved_documents', []),
                    'metrics': rag_response.get('metrics', {})
                })
            
            # Wait for file to be written
            import time
            time.sleep(1.0)  # Increased wait time
            
            # Check if file was created
            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path)
                print(f"RAG speech file exists with size: {file_size} bytes")
                
                if file_size > 100:  # Minimum reasonable file size
                    # Try to get audio duration for debugging
                    try:
                        import wave
                        with wave.open(audio_path, 'rb') as wav_file:
                            frames = wav_file.getnframes()
                            rate = wav_file.getframerate()
                            duration = frames / float(rate)
                            print(f"RAG Audio duration: {duration:.2f} seconds")
                            
                            if duration < 0.1:
                                print("WARNING: RAG Audio duration is very short!")
                    except Exception as wave_error:
                        print(f"Could not analyze RAG audio file: {wave_error}")
                    
                    audio_url = f"/static/audio/{audio_filename}"
                    print(f"Success! RAG Audio URL: {audio_url}")
                    
                    return jsonify({
                        'success': True,
                        'query': query,
                        'answer': answer_text,
                        'audio_url': audio_url,
                        'file_size': file_size,
                        'retrieved_documents': rag_response.get('retrieved_documents', []),
                        'metrics': rag_response.get('metrics', {})
                    })
                else:
                    print(f"RAG speech file too small ({file_size} bytes), likely empty")
                    return jsonify({
                        'success': True,
                        'query': query,
                        'answer': answer_text,
                        'audio_url': None,
                        'error': 'Generated audio file is empty or corrupted',
                        'retrieved_documents': rag_response.get('retrieved_documents', []),
                        'metrics': rag_response.get('metrics', {})
                    })
            else:
                print("RAG speech file was not created")
                return jsonify({
                    'success': True,
                    'query': query,
                    'answer': answer_text,
                    'audio_url': None,
                    'error': 'Failed to generate speech file',
                    'retrieved_documents': rag_response.get('retrieved_documents', []),
                    'metrics': rag_response.get('metrics', {})
                })
                
        except Exception as tts_engine_error:
            print(f"RAG TTS Engine Error: {tts_engine_error}")
            import traceback
            traceback.print_exc()
            # Still return the text response even if TTS fails
            return jsonify({
                'success': True,
                'query': query,
                'answer': answer_text,
                'audio_url': None,
                'error': f'TTS engine error: {str(tts_engine_error)}',
                'retrieved_documents': rag_response.get('retrieved_documents', []),
                'metrics': rag_response.get('metrics', {})
            })
        
    except Exception as e:
        logger.error(f"Error in RAG to speech pipeline: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Start preloading speech models and RAG components at application startup
# Make this safer by not running in parallel to avoid threading conflicts
def safe_initialization():
    """Safely initialize components sequentially to avoid memory conflicts."""
    try:
        # First, speech models (in main thread to avoid conflicts)
        print("🎤 Verifying pre-installed speech models...")
        install_vosk_if_needed()
        en_model = get_model_path('en')
        es_model = get_model_path('es')
        print(f"Speech models: EN({'✅' if en_model else '❌'}) ES({'✅' if es_model else '❌'})")
        
        # Load embedding model NOW (not lazy) to avoid wait on first question
        print("🧠 Loading embedding model (this takes ~30s)...")
        try:
            embedding_gen = get_embedding_generator()
            # Force load the model by generating a test embedding
            test_embedding = embedding_gen.generate_embeddings("test")
            print("✅ Embedding model loaded and ready")
        except Exception as embed_error:
            print(f"❌ Embedding model error: {embed_error}")
            print("   App will still work but embeddings will load on first use")
        
        # Warm up TTS engine to avoid first-call delay
        print("🎤 Warming up TTS engine...")
        try:
            tts_engine = get_engine()
            if tts_engine:
                print("✅ TTS engine ready")
            else:
                print("⚠️ TTS engine initialization deferred")
        except Exception as tts_error:
            print(f"⚠️ TTS warmup failed: {tts_error}")
            print("   First TTS request may have slight delay")
        
        # Check if Ollama is running before initializing RAG
        print("🤖 Checking Ollama connection...")
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama is running")
                # Now initialize RAG pipeline
                print("🔗 Initializing RAG pipeline...")
                global rag_pipeline
                rag_pipeline = get_rag_pipeline()
                print("✅ RAG pipeline ready")
                
                # Warm up the LLM with a test query to avoid first-call delay
                print("🔥 Warming up LLM model...")
                try:
                    test_response = rag_pipeline.llm.generate_response("Hello", max_tokens=10)
                    if test_response and not rag_pipeline.llm.is_mock_mode:
                        print("✅ LLM model warmed up successfully")
                    else:
                        print("⚠️ LLM in mock mode - queries will use fallback responses")
                except Exception as warmup_error:
                    print(f"⚠️ LLM warmup failed: {warmup_error}")
                    print("   First query may have slight delay")
            else:
                print("❌ Ollama responded but not ready")
                print("   Start Ollama with: ollama serve")
        except requests.exceptions.ConnectionError:
            print("❌ Ollama not running")
            print("   Start Ollama with: ollama serve")
            print("   App will work but RAG queries will fail")
        except Exception as ollama_error:
            print(f"❌ Ollama check failed: {ollama_error}")
            print("   App will work but RAG queries will fail")
        
    except Exception as e:
        print(f"⚠️ Initialization warning: {e}")
        # Continue anyway - components will initialize on first use

# Run safe initialization
safe_initialization()

def generate_self_signed_cert():
    """
    Generate a self-signed certificate for HTTPS
    """
    import os
    from OpenSSL import crypto
    
    # Check if certificate already exists
    cert_file = 'web_app/ssl/cert.pem'
    key_file = 'web_app/ssl/key.pem'
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("SSL certificates already exist, using existing files")
        return cert_file, key_file
    
    # Create directory if it doesn't exist
    os.makedirs('web_app/ssl', exist_ok=True)
    
    # Create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)
    
    # Create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = "US"
    cert.get_subject().ST = "California"
    cert.get_subject().L = "Silicon Valley"
    cert.get_subject().O = "AIEDU"
    cert.get_subject().OU = "RAG Speech System"
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(10*365*24*60*60)  # 10 years validity
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')
    
    # Write certificate
    with open(cert_file, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    # Write private key
    with open(key_file, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    print(f"Self-signed certificate generated and saved to {cert_file} and {key_file}")
    return cert_file, key_file

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # Default to fast production mode (user can override with environment variables)
    PRODUCTION_MODE = os.environ.get('AIEDU_PRODUCTION', 'true').lower() == 'true'
    DEBUG_MODE = os.environ.get('AIEDU_DEBUG', 'false').lower() == 'true'
    
    # Apply speed optimizations by default
    print("🚀 Starting AIEDU with speed optimizations...")
    
    # Memory safety and GPU optimizations (apply early to prevent conflicts)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['OMP_NUM_THREADS'] = '1'  # Prevent threading conflicts
    os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL threading
    
    # Reduce logging noise for faster startup (always apply for speed)
    logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
    logging.getLogger('chromadb').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    # Install required package for SSL certificates (silent)
    try:
        import OpenSSL
    except ImportError:
        print("Installing pyOpenSSL...")
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "pip", "install", "pyOpenSSL"], 
                      capture_output=True, check=True)

    # Generate SSL certificate for HTTPS
    try:
        cert_file, key_file = generate_self_signed_cert()
        
        startup_time = time.time() - start_time
        print(f"🌐 Server ready in {startup_time:.1f}s - https://127.0.0.1:5000")
        print("💡 Models pre-loaded for instant responses")
        
        if DEBUG_MODE:
            print("🔧 Debug mode enabled")
        
        # Add memory cleanup before starting Flask
        import gc
        gc.collect()
        
        # Start with optimized settings (always fast)
        app.run(
            debug=DEBUG_MODE,                    # Debug off by default
            host='0.0.0.0', 
            port=5000, 
            ssl_context=(cert_file, key_file),
            threaded=True,                       # Better performance
            use_reloader=False,                  # Disable reloader for speed
            processes=1                          # Single process to avoid memory conflicts
        )
        
    except Exception as e:
        print(f"⚠️ HTTPS failed, using HTTP: {e}")
        
        # Add memory cleanup before starting Flask
        import gc
        gc.collect()
        
        app.run(
            debug=DEBUG_MODE,
            host='0.0.0.0', 
            port=5000,
            threaded=True,
            use_reloader=False,                   # Always disable for speed
            processes=1                          # Single process to avoid memory conflicts
        )
