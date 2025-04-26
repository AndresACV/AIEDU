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
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from datetime import datetime

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

# Start a background thread to pre-download speech models for all supported languages
def preload_speech_models():
    """Preload all speech recognition models in a background thread."""
    print("Starting background thread to preload speech models...")
    
    def download_all_models():
        try:
            # Install Vosk if needed
            install_vosk_if_needed()
            
            # Check if English model exists, download only if needed
            print("Checking English speech model...")
            en_model = download_model_if_needed('en')
            print(f"English model status: {'Ready' if en_model else 'Failed to download'}")
            
            # Check if Spanish model exists, download only if needed
            print("Checking Spanish speech model...")
            es_model = download_model_if_needed('es')
            print(f"Spanish model status: {'Ready' if es_model else 'Failed to download'}")
            
            # Print model locations for clarity
            print("\nSPEECH MODELS IN USE:")
            print(f"- English voice-to-text: {en_model}")
            print(f"- Spanish voice-to-text: {es_model}")
            print(f"- Text-to-speech: Windows SAPI5 voices via pyttsx3\n")
        except Exception as e:
            print(f"Error in preloading models: {str(e)}")
    
    # Start thread
    download_thread = threading.Thread(target=download_all_models)
    download_thread.daemon = True
    download_thread.start()

# Initialize TTS engine
engine = None

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize RAG components
rag_pipeline = None
conversation_memory = ConversationMemory(max_history=5)

def init_rag_components():
    """Initialize RAG pipeline components in a background thread."""
    global rag_pipeline
    
    try:
        logger.info("Initializing RAG pipeline components")
        # Initialize RAG pipeline with default settings
        rag_pipeline = get_rag_pipeline()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        import traceback
        traceback.print_exc()

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4000
RECORD_SECONDS = 5

def get_engine():
    """Get or initialize the TTS engine."""
    global engine
    if engine is None:
        engine = pyttsx3.init()
    return engine



def convert_webm_to_wav(webm_path):
    """Convert WebM to WAV using the audio data from SpeechRecognition."""
    try:
        import subprocess
        import sys
        import os
        import wave
        import numpy as np
        from pathlib import Path
        
        # Verify the input file exists and has content
        if not os.path.exists(webm_path):
            print(f"ERROR: Input WebM file does not exist: {webm_path}")
            return None
            
        file_size = os.path.getsize(webm_path)
        if file_size == 0:
            print(f"ERROR: Input WebM file is empty: {webm_path}")
            return None
            
        print(f"Processing WebM file: {webm_path} (size: {file_size} bytes)")
        
        wav_path = webm_path.replace('.webm', '.wav')
        
        # First, attempt to decode the audio using SpeechRecognition
        try:
            print("Attempting to use SpeechRecognition for WebM decoding")
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            try:
                # Try using AudioFile directly first - sometimes it works with webm
                with sr.AudioFile(webm_path) as source:
                    audio_data = recognizer.record(source)
                    
                print(f"Success reading WebM directly with SpeechRecognition, writing to WAV")
                with open(wav_path, 'wb') as wav_file:
                    wav_file.write(audio_data.get_wav_data())
                    
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    print(f"SpeechRecognition direct conversion successful with size {os.path.getsize(wav_path)} bytes")
                    return wav_path
            except Exception as direct_error:
                print(f"Direct reading failed: {direct_error}, trying fallback method")
                
                # Raw conversion approach as fallback
                with open(webm_path, 'rb') as source_file:
                    audio_binary = source_file.read()
                    
                # Try to create a WAV file directly
                with open(wav_path, 'wb') as wav_file:
                    # First, try to use the SR conversion function
                    try:
                        # We need to create a fake AudioData object since we don't know sample rate yet
                        # 16000 Hz, 16-bit mono are standard parameters for speech recognition
                        temp_audio_data = sr.AudioData(audio_binary, 16000, 2)
                        wav_file.write(temp_audio_data.get_wav_data())
                        print(f"SpeechRecognition AudioData conversion succeeded, created {wav_path}")
                    except Exception as conversion_error:
                        print(f"SR AudioData conversion failed: {conversion_error}")
                        return None
                        
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    print(f"Created WAV file with size {os.path.getsize(wav_path)} bytes")
                    return wav_path
        except Exception as sr_error:
            print(f"SpeechRecognition method failed: {sr_error}")
        
        # Install pydub if needed and try that route
        try:
            try:
                from pydub import AudioSegment
            except ImportError:
                print("Installing pydub package")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub"])
                from pydub import AudioSegment
                
            # Try to install ffmpeg for Windows if not available
            try:
                print("Installing ffmpeg-python package (if not already installed)")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
            except Exception as ffmpeg_install_error:
                print(f"Note: ffmpeg-python installation failed: {ffmpeg_install_error}")
                
            # Try pydub conversion
            try:
                audio = AudioSegment.from_file(webm_path, format="webm")
                audio = audio.set_channels(1)  # Mono
                audio = audio.set_frame_rate(16000)  # 16kHz
                audio = audio.set_sample_width(2)  # 16-bit
                
                # Export as WAV with explicit parameters
                print(f"Exporting to WAV format with pydub: {wav_path}")
                audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"])
                
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    print(f"Pydub conversion successful. Created {wav_path} with size {os.path.getsize(wav_path)} bytes")
                    return wav_path
            except Exception as pydub_error:
                print(f"Pydub processing error: {pydub_error}")
        except Exception as pydub_install_error:
            print(f"Pydub route failed: {pydub_install_error}")
        
        # Direct ffmpeg approach as a last resort
        try:
            print("Attempting direct ffmpeg command")
            result = subprocess.run(["ffmpeg", "-i", webm_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path], 
                       check=True, capture_output=True)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                print(f"Direct FFmpeg conversion successful. Created {wav_path}")
                return wav_path
        except Exception as direct_ffmpeg_error:
            print(f"Direct FFmpeg conversion failed: {direct_ffmpeg_error}")
        
        # If all conversion methods fail
        print("All conversion methods failed. Cannot process audio.")
        return None
    except Exception as e:
        print(f"Error converting WebM to WAV: {e}")
        import traceback
        traceback.print_exc()
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

def download_model_if_needed(language='en'):
    """Download Vosk model for the specified language if not already available."""
    import os
    from pathlib import Path
    import zipfile
    import requests
    
    # Create models directory if it doesn't exist
    models_dir = Path("web_app/models")
    models_dir.mkdir(exist_ok=True)
    
    # Define model names and URLs
    if language == 'en':
        model_name = "vosk-model-small-en-us-0.15"
        model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    elif language == 'es':
        # Using the full Spanish model for better recognition
        model_name = "vosk-model-small-es-0.42"
        model_url = "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
    else:
        # Default to English
        model_name = "vosk-model-small-en-us-0.15"
        model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    
    model_path = models_dir / model_name
    
    print(f"Checking for Vosk model for language: {language}")
    
    # model_name and model_url are already set above based on language
    print(f"Using model name: {model_name}")
    print(f"Using model URL: {model_url}")
    
    # Check if model already exists - NEVER force redownload
    if model_path.exists():
        print(f"Model {model_name} already exists at {model_path}")
        return str(model_path)
    
    # Download and extract model
    zip_path = models_dir / f"{model_name}.zip"
    print(f"Downloading model {model_name} from {model_url}...")
    
    try:
        # Download with progress reporting
        response = requests.get(model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        wrote = 0
        
        with open(zip_path, 'wb') as f:
            for data in response.iter_content(block_size):
                wrote = wrote + len(data)
                f.write(data)
                print(f"Downloaded {wrote} of {total_size} bytes", end='\r')
        
        print(f"\nExtracting model to {model_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the name of the folder inside the zip
            extracted_dir = zipfile.Path(zip_ref, zip_ref.namelist()[0].split('/')[0])
            # Extract to models directory
            zip_ref.extractall(models_dir)
        
        # Rename extracted directory to our standard name
        extracted_path = next(models_dir.glob("vosk-model-*"))
        extracted_path.rename(model_path)
        
        # Remove zip file
        zip_path.unlink()
        
        print(f"Model {model_name} ready")
        return str(model_path)
    except Exception as e:
        print(f"Error downloading model: {e}")
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
    
    print(f"Attempting to transcribe file: {file_path}")
    print(f"Using language: {language}")
    
    # Strict language detection
    # This is critical for proper model selection
    if language == 'es-ES':
        lang_code = 'es'
        print("============================================")
        print("SPANISH LANGUAGE SELECTED")
        print("============================================")
    else:
        lang_code = 'en'
        print("============================================")
        print("ENGLISH LANGUAGE SELECTED")
        print("============================================")
    
    print(f"Selected language code: {lang_code}")
    
    # Keep the original file for debugging
    original_file = file_path
    original_dir = os.path.dirname(file_path)
    temp_dir = os.path.join(original_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # If file is WebM format, try to convert it to WAV using ffmpeg directly
    if file_path.lower().endswith('.webm'):
        print("Converting WebM to WAV format using multiple methods")
        
        # Method 1: Try using ffmpeg directly (most reliable)
        wav_file = os.path.join(temp_dir, os.path.basename(file_path).replace('.webm', '.wav'))
        try:
            print(f"ATTEMPT 1: Direct ffmpeg conversion: {file_path} -> {wav_file}")
            # First try to use ffmpeg directly
            try:
                cmd = ["ffmpeg", "-y", "-i", file_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_file]
                print(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True)
                stdout = result.stdout.decode() if result.stdout else ''
                stderr = result.stderr.decode() if result.stderr else ''
                print(f"FFmpeg output: {stderr}")
                
                if os.path.exists(wav_file) and os.path.getsize(wav_file) > 100:
                    print(f"FFmpeg direct conversion successful: {wav_file} ({os.path.getsize(wav_file)} bytes)")
                    file_path = wav_file
                    print(f"Using converted WAV file: {file_path}")
                else:
                    print("FFmpeg conversion failed or produced invalid output")
                    raise Exception("Invalid FFmpeg output")
            except Exception as ffmpeg_error:
                print(f"Direct FFmpeg conversion failed: {ffmpeg_error}, trying fallback")
                
                # Method 2: Try using pydub if available
                try:
                    from pydub import AudioSegment
                    print("ATTEMPT 2: Using pydub for conversion")
                    audio = AudioSegment.from_file(file_path)
                    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                    audio.export(wav_file, format="wav")
                    
                    if os.path.exists(wav_file) and os.path.getsize(wav_file) > 100:
                        print(f"Pydub conversion successful: {wav_file} ({os.path.getsize(wav_file)} bytes)")
                        file_path = wav_file
                    else:
                        raise Exception("Invalid pydub output")
                except Exception as pydub_error:
                    print(f"Pydub conversion failed: {pydub_error}, trying another fallback")
                    
                    # Method 3: Use our original complex conversion function
                    wav_file = convert_webm_to_wav(file_path)
                    if wav_file:
                        file_path = wav_file
                        print(f"Fallback conversion successful: {file_path}")
                    else:
                        return {
                            'success': False,
                            'error': "Could not convert audio format to WAV after multiple attempts"
                        }
        except Exception as conversion_error:
            print(f"All conversion methods failed: {conversion_error}")
            return {
                'success': False,
                'error': "Could not convert audio format to WAV"
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
        
        # Download model if needed
        model_path = download_model_if_needed(lang_code)
        if not model_path:
            return {
                'success': False,
                'error': f"Could not load speech recognition model for language: {lang_code}"
            }
        
        print(f"Using Vosk model at: {model_path}")
        
        # For Spanish recognition
        if lang_code == 'es':
            print("*** USING SPANISH RECOGNITION WORKFLOW ***")
            
            # Use the existing Spanish model without forcing redownload
            # This avoids unnecessary downloads and speeds up startup
            print("Using existing Spanish model if available...")
            model_path = download_model_if_needed('es')  # Will NOT force redownload
            if not model_path or not os.path.exists(model_path):
                return {
                    'success': False,
                    'error': "Could not find or download Spanish speech recognition model"
                }
                
            print(f"Using Spanish Vosk model at: {model_path}")
            
            # Load the Spanish model with simple configuration
            try:
                model = vosk.Model(model_path)
                print("Spanish model loaded successfully")
            except Exception as model_error:
                print(f"Error loading Spanish model: {model_error}")
                return {
                    'success': False,
                    'error': f"Could not load Spanish speech model: {str(model_error)}"
                }
        else:
            # Standard English model loading
            print(f"Using English Vosk model at: {model_path}")
            model = vosk.Model(model_path)
            print("English model loaded successfully")
        
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
        
        # Record language used for debugging
        print(f"Vosk recognition result ({lang_code}): '{result_text}'")
        print(f"Full recognition result: {result_json}")
        
        # Verify the transcription made sense for the language
        if lang_code == 'es' and not result_text:
            print("WARNING: Empty result from Spanish model - this likely indicates a model loading issue.")
        elif lang_code == 'es':
            print("Spanish transcription completed. If you see English-like words, there might be an issue with language detection.")
        
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


# Removed API-based direct_streaming_recognition function

def get_available_voices():
    """Get all available TTS voices."""
    e = get_engine()
    voices = e.getProperty('voices')
    
    available_voices = []
    for voice in voices:
        # Extract information about the voice
        voice_info = {
            'id': voice.id,
            'name': voice.name if hasattr(voice, 'name') else voice.id.split('\\')[-1],
            'languages': voice.languages if hasattr(voice, 'languages') else [],
            'gender': voice.gender if hasattr(voice, 'gender') else None,
            'age': voice.age if hasattr(voice, 'age') else None,
        }
        available_voices.append(voice_info)
    
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
    """Convert text to speech."""
    try:
        data = request.json
        text = data.get('text', '')
        voice_id = data.get('voice_id', '')
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        # Initialize the TTS engine
        e = get_engine()
        
        # Set voice if specified
        if voice_id:
            e.setProperty('voice', voice_id)
        
        # Set properties
        e.setProperty('rate', 170)  # Speed
        e.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"speech_{timestamp}_{unique_id}.wav"
        output_path = os.path.join(AUDIO_FOLDER, filename)
        
        # Generate speech
        e.save_to_file(text, output_path)
        e.runAndWait()
        
        # Return the URL to the audio file
        audio_url = f"/static/audio/{filename}"
        return jsonify({'success': True, 'audio_url': audio_url})
        
    except Exception as e:
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
        
        # Generate speech from answer
        engine = get_engine()
        
        # Set voice if provided
        if voice_id:
            engine.setProperty('voice', voice_id)
        
        # Generate unique filename
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        
        # Generate speech file
        engine.save_to_file(answer_text, audio_path)
        engine.runAndWait()
        
        # Return results
        audio_url = f"/static/audio/{audio_filename}"
        return jsonify({
            'success': True,
            'query': query,
            'answer': answer_text,
            'audio_url': audio_url,
            'retrieved_documents': rag_response.get('retrieved_documents', []),
            'metrics': rag_response.get('metrics', {})
        })
        
    except Exception as e:
        logger.error(f"Error in RAG to speech pipeline: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Start preloading speech models and RAG components at application startup
preload_speech_models()
threading.Thread(target=init_rag_components, daemon=True).start()

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
    # Install required package for SSL certificates
    try:
        import OpenSSL
    except ImportError:
        print("Installing pyOpenSSL for HTTPS support...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyOpenSSL"])
        print("pyOpenSSL installed successfully")

    # Generate SSL certificate for HTTPS
    try:
        cert_file, key_file = generate_self_signed_cert()
        print("\nStarting server with HTTPS support for secure microphone access")
        print("Note: Your browser will show a security warning because we're using a self-signed certificate.")
        print("This is normal and you can safely proceed.\n")
        app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=(cert_file, key_file))
    except Exception as e:
        print(f"Failed to setup HTTPS: {e}. Falling back to HTTP (microphone access may not work)")
        app.run(debug=True, host='0.0.0.0', port=5000)
