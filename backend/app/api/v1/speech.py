from fastapi import APIRouter, Depends, HTTPException
from typing import List

from ...core.dependencies import get_provider_service
from ...services.provider_service import ProviderService
from ...models.speech import Voice

router = APIRouter(prefix="/speech", tags=["speech"])

@router.get("/voices", response_model=List[Voice])
async def get_voices(
    provider_service: ProviderService = Depends(get_provider_service)
) -> List[Voice]:
    """Get available voices for text-to-speech."""
    try:
        voices_data = provider_service.get_available_voices()
        return [Voice(**voice) for voice in voices_data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
