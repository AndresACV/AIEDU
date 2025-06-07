from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import tempfile
import logging

from ...core.dependencies import get_speech_service
from ...services.speech_service import SpeechService
from ...models.speech import Voice, SynthesizeRequest, SynthesizeResponse, TranscribeResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/speech", tags=["speech"])

@router.get("/voices", response_model=List[Voice])
async def get_voices(
    speech_service: SpeechService = Depends(get_speech_service)
) -> List[Voice]:
    """Get available voices for text-to-speech."""
    try:
        voices_data = speech_service.get_available_voices()
        return [Voice(**voice) for voice in voices_data]
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/languages")
async def get_languages(
    speech_service: SpeechService = Depends(get_speech_service)
):
    """Get available languages for speech-to-text."""
    try:
        languages = speech_service.get_available_languages()
        return {"languages": languages}
    except Exception as e:
        logger.error(f"Error getting languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(
    request: SynthesizeRequest,
    speech_service: SpeechService = Depends(get_speech_service)
):
    """
    Convert text to speech and return audio file information.
    
    Args:
        request: Text synthesis request with text, voice_id, and language
        
    Returns:
        Response with audio file details or error information
    """
    try:
        logger.info(f"TTS request: '{request.text[:50]}...' voice={request.voice_id} lang={request.language}")
        
        # Perform speech synthesis
        result = await speech_service.synthesize_speech(
            text=request.text,
            voice_id=request.voice_id,
            language=request.language or "en-US"
        )
        
        if result['success']:
            # Return successful response with audio file info
            # Use absolute URL to ensure frontend can access audio from backend server
            return SynthesizeResponse(
                success=True,
                audio_url=f"http://127.0.0.1:8000/api/v1/speech/audio/{result['audio_filename']}",
                audio_filename=result['audio_filename'],
                voice_used=result['voice_used'],
                duration=result['duration'],
                file_size=result['file_size']
            )
        else:
            # Return error response
            return SynthesizeResponse(
                success=False,
                error=result['error']
            )
            
    except Exception as e:
        logger.error(f"Error in speech synthesis: {e}")
        return SynthesizeResponse(
            success=False,
            error=str(e)
        )

@router.get("/audio/{filename}")
async def get_audio_file(
    filename: str,
    speech_service: SpeechService = Depends(get_speech_service)
):
    """
    Serve generated audio files.
    
    Args:
        filename: Name of the audio file to serve
        
    Returns:
        Audio file response
    """
    try:
        audio_path = os.path.join(speech_service.audio_folder, filename)
        
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Security check - ensure filename doesn't contain path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = "en-US",
    speech_service: SpeechService = Depends(get_speech_service)
):
    """
    Transcribe uploaded audio file to text.
    
    Args:
        file: Audio file to transcribe (WAV format recommended)
        language: Language code for transcription (e.g., "en-US", "es-ES")
        
    Returns:
        Transcription results with text and confidence scores
    """
    temp_audio_path = None
    
    try:
        logger.info(f"🎤 STT request: file={file.filename} lang={language}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.webm')):
            logger.error(f"❌ Unsupported file type: {file.filename}")
            return TranscribeResponse(
                success=False,
                error="Unsupported audio format. Please use WAV, MP3, FLAC, OGG, or WebM."
            )
        
        # Save uploaded file to temporary location with original extension
        file_ext = os.path.splitext(file.filename)[1] if file.filename else '.webm'
        logger.info(f"📁 Creating temp file with extension: {file_ext}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_audio_path = temp_file.name
            content = await file.read()
            content_size = len(content)
            temp_file.write(content)
            logger.info(f"💾 Saved uploaded file to: {temp_audio_path} (size: {content_size} bytes)")
            
        # Verify file was actually written
        if os.path.exists(temp_audio_path):
            actual_size = os.path.getsize(temp_audio_path)
            logger.info(f"✅ Temp file verified: {actual_size} bytes on disk")
        else:
            logger.error(f"❌ Temp file not found after writing: {temp_audio_path}")
            return TranscribeResponse(
                success=False,
                error="Failed to save uploaded audio file"
            )
        
        # Perform transcription
        logger.info(f"🔄 Starting transcription for: {temp_audio_path}")
        result = await speech_service.transcribe_audio(
            audio_path=temp_audio_path,
            language=language or "en-US"
        )
        
        logger.info(f"📊 Transcription result: success={result['success']}")
        if not result['success']:
            logger.error(f"❌ Transcription error: {result.get('error', 'Unknown error')}")
        
        if result['success']:
            logger.info(f"✅ Transcription successful: '{result['transcript'][:50]}...'")
            return TranscribeResponse(
                success=True,
                text=result['transcript'],
                confidence=result['confidence'],
                language=result['language'],
                model_used=result['model_used'],
                duration=result['duration'],
                words=result.get('words', [])
            )
        else:
            return TranscribeResponse(
                success=False,
                error=result['error']
            )
            
    except Exception as e:
        logger.error(f"Error in speech transcription: {e}")
        return TranscribeResponse(
            success=False,
            error=str(e)
        )
    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_audio_path}: {e}")

@router.get("/stats")
async def get_speech_stats(
    speech_service: SpeechService = Depends(get_speech_service)
):
    """Get speech service performance statistics."""
    try:
        stats = speech_service.get_performance_stats()
        return {"stats": stats}
    except Exception as e:
        logger.error(f"Error getting speech stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup_audio_files(
    max_age_minutes: int = 60,
    speech_service: SpeechService = Depends(get_speech_service)
):
    """Clean up old audio files."""
    try:
        speech_service.cleanup_audio_files(max_age_minutes)
        return {"message": f"Audio files older than {max_age_minutes} minutes cleaned up"}
    except Exception as e:
        logger.error(f"Error cleaning up audio files: {e}")
        raise HTTPException(status_code=500, detail=str(e))
