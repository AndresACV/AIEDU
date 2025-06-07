from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Voice(BaseModel):
    id: str
    name: str
    language_type: str
    platform: Optional[str] = None
    engine: Optional[str] = None

class VoicesResponse(BaseModel):
    voices: List[Voice]

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    language: Optional[str] = "en-US"

class SynthesizeResponse(BaseModel):
    success: bool
    audio_url: Optional[str] = None
    audio_filename: Optional[str] = None
    voice_used: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    error: Optional[str] = None

class WordInfo(BaseModel):
    """Word-level transcription information."""
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    conf: Optional[float] = None  # Confidence score

class TranscribeResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    confidence: Optional[float] = None
    language: Optional[str] = None
    model_used: Optional[str] = None
    duration: Optional[float] = None
    words: Optional[List[WordInfo]] = None
    error: Optional[str] = None

class LanguageInfo(BaseModel):
    """Available language information for STT."""
    code: str
    name: str
    model_path: str

class LanguagesResponse(BaseModel):
    languages: List[LanguageInfo]

class SpeechStats(BaseModel):
    """Speech service performance statistics."""
    stt_calls: int
    tts_calls: int
    stt_success: int
    tts_success: int
    total_calls: int
    total_success: int
    overall_success_rate: float
    stt_success_rate: float
    tts_success_rate: float
    avg_stt_time: float
    avg_tts_time: float

class SpeechStatsResponse(BaseModel):
    stats: SpeechStats
