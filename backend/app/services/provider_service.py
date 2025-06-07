from typing import Dict, Any
from ..models.providers import ProviderType, ProviderResponse, ProviderInfo, ProvidersDict

class ProviderService:
    """Service for managing speech and AI providers."""
    
    def __init__(self):
        self.current_provider: ProviderType = "local"
        
    def get_current_providers(self) -> ProviderResponse:
        """Get current provider status and configuration."""
        # Mock provider information for Phase 1
        # In later phases, this will check actual service availability
        local_info = ProviderInfo(stt="available", tts="available", llm="available")
        cloud_info = ProviderInfo(stt="available", tts="available", llm="available")
        
        providers_dict = ProvidersDict(local=local_info, cloud=cloud_info)
        
        return ProviderResponse(
            stt_provider=self.current_provider,
            tts_provider=self.current_provider,
            llm_provider=self.current_provider,
            status="FastAPI Backend is running",
            providers=providers_dict
        )
    
    def force_provider(self, provider: ProviderType) -> Dict[str, Any]:
        """Force switch to a specific provider."""
        old_provider = self.current_provider
        self.current_provider = provider
        
        return {
            "success": True,
            "message": f"Switched from {old_provider} to {provider}",
            "new_provider": provider
        }
    
    def get_available_voices(self) -> list:
        """Get available voices for current provider."""
        # Mock voice data for Phase 1
        # In later phases, this will query actual TTS services
        return [
            {
                "id": "spanish_voice",
                "name": "Spanish (Latin America)",
                "language_type": "Spanish"
            },
            {
                "id": "english_voice", 
                "name": "English (US)",
                "language_type": "English"
            }
        ]
