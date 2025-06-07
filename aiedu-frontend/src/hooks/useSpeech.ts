import { useState, useCallback, useRef, useEffect } from 'react'
import { apiClient } from '@/services/api'
import { Voice, SynthesizeRequest, TranscribeResponse } from '@/types/speech'

interface SpeechState {
  // Voice management
  voices: Voice[]
  selectedVoice: Voice | null
  isLoadingVoices: boolean
  
  // Recording state
  isRecording: boolean
  isProcessingRecording: boolean
  recordingDuration: number
  
  // TTS state
  isSynthesizing: boolean
  isPlaying: boolean
  audioUrl: string | null
  
  // General state
  error: string | null
  lastTranscription: string | null
  lastConfidence: number | null
}

interface UseSpeechOptions {
  defaultLanguage?: string
  autoSelectVoice?: boolean
  onTranscription?: (text: string, confidence?: number) => void
  onSynthesisComplete?: () => void
  onError?: (error: string) => void
}

export function useSpeech({
  defaultLanguage = 'en-US',
  autoSelectVoice = true,
  onTranscription,
  onSynthesisComplete,
  onError
}: UseSpeechOptions = {}) {
  
  const [state, setState] = useState<SpeechState>({
    voices: [],
    selectedVoice: null,
    isLoadingVoices: false,
    isRecording: false,
    isProcessingRecording: false,
    recordingDuration: 0,
    isSynthesizing: false,
    isPlaying: false,
    audioUrl: null,
    error: null,
    lastTranscription: null,
    lastConfidence: null
  })

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordingStartTime = useRef<number>(0)
  const durationInterval = useRef<NodeJS.Timeout>()
  const stateRef = useRef(state)
  
  // Keep state ref updated
  useEffect(() => {
    stateRef.current = state
  }, [state])

  // Load available voices  
  const loadVoices = useCallback(async () => {
    setState(prev => ({ ...prev, isLoadingVoices: true, error: null }))
    
    try {
      const response = await apiClient.getVoices()
      
      if (response.success && response.voices) {
        setState(prev => ({ ...prev, voices: response.voices!, isLoadingVoices: false }))
        
        // Auto-select default voice if enabled
        if (autoSelectVoice && response.voices.length > 0) {
          const defaultVoice = response.voices.find(v => 
            v.language_type === (defaultLanguage.includes('es') ? 'Spanish' : 'English')
          ) || response.voices[0]
          
          setState(prev => ({ ...prev, selectedVoice: defaultVoice }))
        }
      } else {
        const error = 'Failed to load voices'
        setState(prev => ({ ...prev, error, isLoadingVoices: false }))
        onError?.(error)
      }
    } catch (error) {
      const errorMessage = `Error loading voices: ${error instanceof Error ? error.message : 'Unknown error'}`
      setState(prev => ({ ...prev, error: errorMessage, isLoadingVoices: false }))
      onError?.(errorMessage)
    }
  }, []) // Removed dependencies to prevent infinite loop

  // Select a voice
  const selectVoice = useCallback((voice: Voice) => {
    setState(prev => ({ ...prev, selectedVoice: voice }))
  }, [])

  // Start audio recording
  const startRecording = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, error: null }))

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 48000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      })

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })

      const chunks: Blob[] = []
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop())
        
        if (chunks.length > 0) {
          setState(prev => ({ ...prev, isProcessingRecording: true }))
          
          try {
            const audioBlob = new Blob(chunks, { type: 'audio/webm' })
            const response: TranscribeResponse = await apiClient.transcribe(audioBlob, defaultLanguage)
            
            if (response.success && response.text) {
              setState(prev => ({ 
                ...prev, 
                lastTranscription: response.text!,
                lastConfidence: response.confidence || null,
                isProcessingRecording: false
              }))
              onTranscription?.(response.text, response.confidence)
            } else {
              const error = response.error || 'Transcription failed'
              setState(prev => ({ ...prev, error, isProcessingRecording: false }))
              onError?.(error)
            }
          } catch (error) {
            const errorMessage = `Transcription error: ${error instanceof Error ? error.message : 'Unknown error'}`
            setState(prev => ({ ...prev, error: errorMessage, isProcessingRecording: false }))
            onError?.(errorMessage)
          }
        }
      }

      mediaRecorderRef.current = mediaRecorder
      recordingStartTime.current = Date.now()
      
      // Start duration tracking
      durationInterval.current = setInterval(() => {
        if (recordingStartTime.current > 0) {
          const duration = (Date.now() - recordingStartTime.current) / 1000
          setState(prev => ({ ...prev, recordingDuration: duration }))
        }
      }, 100)

      mediaRecorder.start(100)
      setState(prev => ({ ...prev, isRecording: true, recordingDuration: 0 }))

    } catch (error) {
      const errorMessage = `Failed to start recording: ${error instanceof Error ? error.message : 'Unknown error'}`
      setState(prev => ({ ...prev, error: errorMessage }))
      onError?.(errorMessage)
    }
  }, [defaultLanguage, onTranscription, onError])

  // Stop audio recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      setState(prev => ({ ...prev, isRecording: false }))
      
      if (durationInterval.current) {
        clearInterval(durationInterval.current)
      }
    }
  }, [])

  // Synthesize text to speech
  const synthesizeText = useCallback(async (text: string, voiceId?: string) => {
    if (!text.trim()) {
      const error = 'No text provided for synthesis'
      setState(prev => ({ ...prev, error }))
      onError?.(error)
      return
    }

    // Get current state for voice selection
    const currentState = stateRef.current
    const voice = voiceId ? currentState.voices.find(v => v.id === voiceId) : currentState.selectedVoice
    if (!voice) {
      const error = 'No voice selected for synthesis'
      setState(prev => ({ ...prev, error }))
      onError?.(error)
      return
    }

    setState(prev => ({ ...prev, isSynthesizing: true, error: null }))

    try {
      const request: SynthesizeRequest = {
        text,
        voice_id: voice.id,
        language: voice.language_type === 'Spanish' ? 'es-ES' : 'en-US'
      }

      const response = await apiClient.synthesize(request)

      if (response.success && response.audio_url) {
        // Clean up previous audio
        if (audioRef.current) {
          audioRef.current.pause()
          const currentAudioUrl = stateRef.current.audioUrl
          if (currentAudioUrl) {
            URL.revokeObjectURL(currentAudioUrl)
          }
        }

        // Create new audio
        const audio = new Audio(response.audio_url)
        audioRef.current = audio

        audio.addEventListener('ended', () => {
          setState(prev => ({ ...prev, isPlaying: false }))
          onSynthesisComplete?.()
        })

        audio.addEventListener('error', () => {
          const error = 'Audio playback failed'
          setState(prev => ({ ...prev, error, isPlaying: false, isSynthesizing: false }))
          onError?.(error)
        })

        // Auto-play the audio
        await audio.play()
        setState(prev => ({ 
          ...prev, 
          audioUrl: response.audio_url!,
          isPlaying: true,
          isSynthesizing: false
        }))

      } else {
        const error = response.error || 'Speech synthesis failed'
        setState(prev => ({ ...prev, error, isSynthesizing: false }))
        onError?.(error)
      }
    } catch (error) {
      const errorMessage = `Synthesis error: ${error instanceof Error ? error.message : 'Unknown error'}`
      setState(prev => ({ ...prev, error: errorMessage, isSynthesizing: false }))
      onError?.(errorMessage)
    }
  }, [onSynthesisComplete, onError])

  // Play/pause audio
  const togglePlayback = useCallback(async () => {
    if (!audioRef.current) return

    try {
      const isCurrentlyPlaying = stateRef.current.isPlaying
      if (isCurrentlyPlaying) {
        audioRef.current.pause()
        setState(prev => ({ ...prev, isPlaying: false }))
      } else {
        await audioRef.current.play()
        setState(prev => ({ ...prev, isPlaying: true }))
      }
    } catch (error) {
      onError?.('Playback control failed')
    }
  }, [onError])

  // Stop audio playback
  const stopPlayback = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      setState(prev => ({ ...prev, isPlaying: false }))
    }
  }, [])

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }))
  }, [])

  // Clear transcription
  const clearTranscription = useCallback(() => {
    setState(prev => ({ 
      ...prev, 
      lastTranscription: null, 
      lastConfidence: null 
    }))
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (durationInterval.current) {
        clearInterval(durationInterval.current)
      }
      if (audioRef.current) {
        audioRef.current.pause()
      }
      const currentState = stateRef.current
      if (currentState.audioUrl) {
        URL.revokeObjectURL(currentState.audioUrl)
      }
      if (mediaRecorderRef.current && currentState.isRecording) {
        mediaRecorderRef.current.stop()
      }
    }
  }, [])

  // Load voices on mount
  useEffect(() => {
    loadVoices()
  }, []) // Remove loadVoices dependency to prevent infinite loop

  return {
    // State
    ...state,
    
    // Voice methods
    loadVoices,
    selectVoice,
    
    // Recording methods
    startRecording,
    stopRecording,
    
    // TTS methods
    synthesizeText,
    togglePlayback,
    stopPlayback,
    
    // Utility methods
    clearError,
    clearTranscription,
    
    // Helper getters
    canRecord: !state.isRecording && !state.isProcessingRecording,
    canSynthesize: !state.isSynthesizing && state.voices.length > 0,
    hasAudio: !!state.audioUrl,
    isLoading: state.isLoadingVoices || state.isSynthesizing || state.isProcessingRecording
  }
} 