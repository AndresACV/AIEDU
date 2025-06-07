import asyncio
import sys
import os
sys.path.append('.')

from app.services.speech_service import SpeechService

async def test_transcribe():
    # Create speech service
    speech_service = SpeechService()
    
    # Test with a dummy file to see what happens
    result = await speech_service.transcribe_audio(
        audio_path="/tmp/nonexistent.webm",
        language="en-US"
    )
    
    print("Transcription result:")
    print(f"Success: {result.get('success')}")
    print(f"Error: {result.get('error')}")
    print(f"Full result: {result}")

if __name__ == "__main__":
    asyncio.run(test_transcribe()) 