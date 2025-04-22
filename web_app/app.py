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
        
        # Last resort: direct Google Streaming Recognition
        print("All conversion methods failed. Using direct streaming recognition instead.")
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
    
    # Define model URLs for different languages
    model_urls = {
        'en': "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        'es': "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
    }
    
    # Get appropriate URL
    model_url = model_urls.get(language, model_urls['en'])
    model_name = f"vosk-model-small-{language}"
    model_path = models_dir / model_name
    
    # Check if model already exists
    if model_path.exists():
        print(f"Model {model_name} already exists")
        return str(model_path)
    
    # Download and extract model
    zip_path = models_dir / f"{model_name}.zip"
    print(f"Downloading model {model_name}...")
    
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
    
    print(f"Attempting to transcribe file: {file_path}")
    print(f"Using language: {language}")
    
    # Determine language code for model
    lang_code = 'en' if language.startswith('en') else 'es'
    
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
        
        # Load the model
        model = vosk.Model(model_path)
        
        # Open the WAV file
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
        
        # Create recognizer
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        
        # Process audio
        result_text = ""
        while True:
            data = wf.readframes(4000)  # Read 4000 frames at a time
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)
            
        # Get the final result
        result_json = json.loads(rec.FinalResult())
        result_text = result_json.get("text", "")
        
        print(f"Vosk recognition result: '{result_text}'")
        
        # Check if result is empty
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
        print("Audio upload request received")
        
        # Check if language was specified
        language = request.form.get('language', 'en-US')
        print(f"Selected language: {language}")
        
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
