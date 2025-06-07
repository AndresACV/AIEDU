export interface Voice {
  id: string
  name: string
  provider?: string
  language_type?: string
}

export interface VoicesResponse {
  success: boolean
  voices?: Voice[]
  error?: string
}

export interface SynthesizeRequest {
  text: string
  voice_id?: string
  language?: string
}

export interface SynthesizeResponse {
  success: boolean
  audio_url?: string
  provider?: string
  duration?: number
  cached?: boolean
  error?: string
}

export interface TranscribeResponse {
  success: boolean
  text?: string
  confidence?: number
  language?: string
  model_used?: string
  duration?: number
  words?: Array<{
    word: string
    start?: number
    end?: number
    conf?: number
  }>
  error?: string
}

export interface RecordingState {
  isRecording: boolean
  volume: number
  duration: number
} 