from typing import Generator
from ..services.provider_service import ProviderService

# Global service instances (will be initialized at startup)
_provider_service: ProviderService = None

def get_provider_service() -> ProviderService:
    """Dependency to get the provider service instance."""
    global _provider_service
    if _provider_service is None:
        _provider_service = ProviderService()
    return _provider_service

# Service initialization function (called at startup)
def initialize_services():
    """Initialize all services at application startup."""
    global _provider_service
    _provider_service = ProviderService()
    print("✅ Services initialized")

# Service cleanup function (called at shutdown)
def cleanup_services():
    """Clean up services at application shutdown."""
    # Add cleanup logic here if needed
    print("✅ Services cleaned up")
