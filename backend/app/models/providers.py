from pydantic import BaseModel
from typing import Dict, Literal

# Type definitions
ProviderType = Literal["local", "cloud"]
ServiceStatus = Literal["available", "unavailable", "connecting", "error"]

class ServiceStatusDetails(BaseModel):
    stt: ServiceStatus
    tts: ServiceStatus
    llm: ServiceStatus

class ProviderInfo(BaseModel):
    stt: ServiceStatus
    tts: ServiceStatus
    llm: ServiceStatus

class ProvidersDict(BaseModel):
    local: ProviderInfo
    cloud: ProviderInfo

class ProviderResponse(BaseModel):
    stt_provider: str
    tts_provider: str
    llm_provider: str
    status: str
    providers: ProvidersDict

class ForceProviderRequest(BaseModel):
    provider: ProviderType

class ForceProviderResponse(BaseModel):
    success: bool
    message: str
    new_provider: ProviderType
