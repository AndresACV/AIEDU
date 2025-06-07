from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from ...core.dependencies import get_provider_service
from ...services.provider_service import ProviderService
from ...models.providers import ProviderResponse, ForceProviderRequest, ForceProviderResponse

router = APIRouter(prefix="/providers", tags=["providers"])

@router.get("/current", response_model=ProviderResponse)
async def get_current_providers(
    provider_service: ProviderService = Depends(get_provider_service)
) -> ProviderResponse:
    """Get current provider status and configuration."""
    return provider_service.get_current_providers()

@router.post("/force", response_model=Dict[str, Any])
async def force_provider(
    request: ForceProviderRequest,
    provider_service: ProviderService = Depends(get_provider_service)
) -> Dict[str, Any]:
    """Force switch to a specific provider (local or cloud)."""
    try:
        result = provider_service.force_provider(request.provider)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
