from pydantic import BaseModel
from typing import List, Optional

class Voice(BaseModel):
    id: str
    name: str
    language_type: str

class VoicesResponse(BaseModel):
    voices: List[Voice]

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None

class SynthesizeResponse(BaseModel):
    success: bool
    audio_url: Optional[str] = None
    error: Optional[str] = None

class TranscribeResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
