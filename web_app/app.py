"""
Flask application for the RAG System Text-to-Voice component.
Provides a web interface for converting text to speech in real-time.
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
        from pathlib import Path
        
        wav_path = webm_path.replace('.webm', '.wav')
        
        # Try to use ffmpeg if available
        try:
            print("Attempting to use ffmpeg for conversion")
            subprocess.run(['ffmpeg', '-i', webm_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path], 
                           check=True, capture_output=True)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                print(f"FFmpeg conversion successful. Created {wav_path} with size {os.path.getsize(wav_path)} bytes")
                return wav_path
            else:
                print("FFmpeg did not produce a valid WAV file")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"FFmpeg not available or failed: {e}")
        
        # Try to install and use pydub for better audio processing
        try:
            print("Attempting conversion with pydub")
            # Try to import pydub, install if not available
            try:
                from pydub import AudioSegment
            except ImportError:
                print("Installing pydub package")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub"])
                from pydub import AudioSegment
            
            # Use pydub to convert
            audio = AudioSegment.from_file(webm_path, format="webm")
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(16000)  # 16kHz
            audio = audio.set_sample_width(2)  # 16-bit
            
            # Export as WAV
            audio.export(wav_path, format="wav")
            
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                print(f"Pydub conversion successful. Created {wav_path} with size {os.path.getsize(wav_path)} bytes")
                return wav_path
            else:
                print("Pydub did not produce a valid WAV file")
        except Exception as e:
            print(f"Pydub conversion failed: {e}")
        
        # No fallback to online services - we want 100% offline for all languages
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
    
    # If file is WebM format, try to convert it to WAV first
    if file_path.lower().endswith('.webm'):
        print("Converting WebM to WAV format")
        wav_file = convert_webm_to_wav(file_path)
        if wav_file:
            file_path = wav_file
        else:
            return "Error: Could not convert audio format to WAV"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        return "Error: Audio file not found"
        
    try:
        # Install vosk if needed
        if not install_vosk_if_needed():
            return "Error: Could not install required speech recognition libraries"
            
        import vosk
        import json
        import wave
        
        # Download model if needed
        model_path = download_model_if_needed(lang_code)
        if not model_path:
            return "Error: Could not download speech recognition model"
        
        print(f"Using Vosk model at: {model_path}")
        
        # For Spanish recognition
        if lang_code == 'es':
            print("*** USING SPANISH RECOGNITION WORKFLOW ***")
            
            # Use the existing Spanish model without forcing redownload
            # This avoids unnecessary downloads and speeds up startup
            print("Using existing Spanish model if available...")
            model_path = download_model_if_needed('es')  # Will NOT force redownload
            if not model_path or not os.path.exists(model_path):
                return "Error: Could not find or download Spanish speech recognition model"
                
            print(f"Using Spanish Vosk model at: {model_path}")
            
            # Load the Spanish model with simple configuration
            try:
                model = vosk.Model(model_path)
                print("Spanish model loaded successfully")
            except Exception as model_error:
                print(f"Error loading Spanish model: {model_error}")
                return f"Error: Could not load Spanish speech model: {str(model_error)}"
        else:
            # Standard English model loading
            print(f"Using English Vosk model at: {model_path}")
            model = vosk.Model(model_path)
            print("English model loaded successfully")
        
        try:
            wf = wave.open(file_path, "rb")
        except Exception as wav_error:
            print(f"Error opening WAV file: {wav_error}")
            return "Error: Could not open audio file for processing"
        
        # Check if format is correct
        if wf.getnchannels() != 1:
            print("Audio must be mono for Vosk")
            return "Error: Audio must be mono (single channel)"
            
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
            while True:
                data = wf.readframes(4000)  # Read 4000 frames at a time
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
            print("Finished processing audio data")
        except Exception as proc_error:
            print(f"Error during audio processing: {proc_error}")
            return f"Error processing audio: {proc_error}"
            
        # Get the final result
        result_json = json.loads(rec.FinalResult())
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
                return "Error: No Spanish speech detected. Try speaking louder or more clearly."
            
            # Look for signs that English was incorrectly detected
            english_words = ['the', 'and', 'or', 'but', 'for', 'with', 'your', 'this', 'that']
            word_count = len(result_text.split())
            english_word_count = sum(1 for word in result_text.lower().split() if word in english_words)
            
            # If more than 30% of words are common English words, something's wrong
            if word_count > 3 and english_word_count / word_count > 0.3:
                print(f"WARNING: Detected {english_word_count}/{word_count} English words in Spanish audio")
                return "Error: La detección de voz en español no funcionó correctamente. Por favor, intente de nuevo."
        else:
            # Standard handling for English
            if not result_text or len(result_text.strip()) == 0:
                return "Error: No speech detected. Try speaking louder or more clearly."
            
        return result_text
        
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
        return "Error: Speech could not be understood. Try speaking more clearly."
    except sr.RequestError as e:
        print(f"Could not request results from speech recognition service; {e}")
        return f"Error: {e}"
    except Exception as audio_error:
        import traceback
        traceback.print_exc()
        print(f"Error processing audio file: {audio_error}")
        return f"Error: Audio file could not be processed - {str(audio_error)}"


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
        transcript = transcribe_audio_file(temp_path, language)
        print(f"Transcription result: {transcript}")
        
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
        
        if transcript.startswith('Error:'):
            print(f"Returning error: {transcript}")
            return jsonify({'success': False, 'error': transcript})
            
        return jsonify({'success': True, 'transcript': transcript})
        
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

# Start preloading speech models at application startup
preload_speech_models()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
