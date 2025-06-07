export type ProviderType = 'local' | 'cloud'

export type ServiceStatus = 'unknown' | 'connecting' | 'working' | 'error'

export interface ProviderStatus {
  stt: ServiceStatus
  tts: ServiceStatus
  llm: ServiceStatus
}

export interface ProviderInfo {
  stt_provider: string
  tts_provider: string
  llm_provider: string
}

export interface ProviderResponse {
  stt_provider: string
  tts_provider: string
  llm_provider: string
  status: string
  providers: {
    local: {
      stt: string
      tts: string
      llm: string
    }
    cloud: {
      stt: string
      tts: string
      llm: string
    }
  }
}

export interface ForceProviderRequest {
  provider: ProviderType
}

export interface ForceProviderResponse {
  success: boolean
  provider_forced?: ProviderType
  error?: string
} 