import sys
import os
import tempfile
import glob
sys.path.append('.')

from app.services.speech_service import SpeechService

def debug_audio_folder():
    # Create speech service instance
    speech_service = SpeechService()
    
    print(f"Audio folder: {speech_service.audio_folder}")
    print(f"Audio folder exists: {os.path.exists(speech_service.audio_folder)}")
    
    # List files in audio folder
    if os.path.exists(speech_service.audio_folder):
        files = os.listdir(speech_service.audio_folder)
        print(f"Files in audio folder: {files}")
        
        # Check file sizes
        for file in files:
            file_path = os.path.join(speech_service.audio_folder, file)
            size = os.path.getsize(file_path)
            print(f"  {file}: {size} bytes")
    
    # Also check /tmp for any uploaded files
    print(f"\nChecking /tmp for audio files:")
    tmp_files = glob.glob("/tmp/*.webm") + glob.glob("/tmp/*.wav") + glob.glob("/tmp/*audio*")
    for file in tmp_files:
        size = os.path.getsize(file)
        print(f"  {file}: {size} bytes")
    
    # Check system temp directory
    temp_dir = tempfile.gettempdir()
    print(f"\nSystem temp directory: {temp_dir}")
    temp_audio_files = glob.glob(os.path.join(temp_dir, "*.webm")) + glob.glob(os.path.join(temp_dir, "*.wav"))
    for file in temp_audio_files:
        size = os.path.getsize(file)
        print(f"  {file}: {size} bytes")

if __name__ == "__main__":
    debug_audio_folder() 