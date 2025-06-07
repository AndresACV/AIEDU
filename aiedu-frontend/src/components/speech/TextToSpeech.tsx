'use client'

import React, { useState, useRef, useEffect } from 'react'
import { apiClient } from '@/services/api'
import { SynthesizeRequest, SynthesizeResponse, Voice } from '@/types/speech'

interface TextToSpeechProps {
  text?: string
  onError?: (error: string) => void
  onSynthesisStart?: () => void
  onSynthesisComplete?: () => void
  className?: string
  autoSpeak?: boolean
}

interface PlaybackState {
  isPlaying: boolean
  isPaused: boolean
  isLoading: boolean
  duration: number
  currentTime: number
  volume: number
  error?: string
}

export default function TextToSpeech({
  text = '',
  onError,
  onSynthesisStart,
  onSynthesisComplete,
  className = '',
  autoSpeak = false
}: TextToSpeechProps) {
  const [inputText, setInputText] = useState(text)
  const [voices, setVoices] = useState<Voice[]>([])
  const [selectedVoice, setSelectedVoice] = useState<string>('')
  const [state, setState] = useState<PlaybackState>({
    isPlaying: false,
    isPaused: false,
    isLoading: false,
    duration: 0,
    currentTime: 0,
    volume: 0.8
  })

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const currentAudioUrl = useRef<string>('')

  // Load available voices on mount
  useEffect(() => {
    loadVoices()
  }, [])

  // Auto-speak when text changes (if enabled)
  useEffect(() => {
    if (autoSpeak && text && text !== inputText) {
      setInputText(text)
      handleSpeak(text)
    }
  }, [text, autoSpeak])

  // Load available voices
  const loadVoices = async () => {
    try {
      const response = await apiClient.getVoices()
      if (response.success && response.voices) {
        setVoices(response.voices)
        // Set default voice (prefer Spanish, then English)
        const defaultVoice = response.voices.find(v => v.language_type === 'Spanish') || 
                            response.voices.find(v => v.language_type === 'English') ||
                            response.voices[0]
        if (defaultVoice) {
          setSelectedVoice(defaultVoice.id)
        }
      }
    } catch (error) {
      console.error('Failed to load voices:', error)
      onError?.('Failed to load voices')
    }
  }

  // Synthesize speech
  const handleSpeak = async (textToSpeak?: string) => {
    const speakText = textToSpeak || inputText.trim()
    
    if (!speakText) {
      onError?.('Please enter text to speak')
      return
    }

    if (!selectedVoice && voices.length > 0) {
      onError?.('Please select a voice')
      return
    }

    setState(prev => ({ ...prev, isLoading: true, error: undefined }))
    onSynthesisStart?.()

    try {
      const request: SynthesizeRequest = {
        text: speakText,
        voice_id: selectedVoice,
        language: voices.find(v => v.id === selectedVoice)?.language_type === 'Spanish' ? 'es-ES' : 'en-US'
      }

      const response: SynthesizeResponse = await apiClient.synthesize(request)

      if (response.success && response.audio_url) {
        // Clean up previous audio URL
        if (currentAudioUrl.current) {
          URL.revokeObjectURL(currentAudioUrl.current)
        }

        // Create new audio element
        const audio = new Audio(response.audio_url)
        audioRef.current = audio
        currentAudioUrl.current = response.audio_url

        // Setup audio event listeners
        audio.addEventListener('loadedmetadata', () => {
          setState(prev => ({ ...prev, duration: audio.duration, isLoading: false }))
        })

        audio.addEventListener('timeupdate', () => {
          setState(prev => ({ ...prev, currentTime: audio.currentTime }))
        })

        audio.addEventListener('ended', () => {
          setState(prev => ({ ...prev, isPlaying: false, isPaused: false, currentTime: 0 }))
          onSynthesisComplete?.()
        })

        audio.addEventListener('error', (e) => {
          const error = 'Audio playback failed'
          setState(prev => ({ ...prev, error, isLoading: false, isPlaying: false }))
          onError?.(error)
        })

        // Set volume and play
        audio.volume = state.volume
        await audio.play()
        setState(prev => ({ ...prev, isPlaying: true, isPaused: false }))

      } else {
        const error = response.error || 'Speech synthesis failed'
        setState(prev => ({ ...prev, error, isLoading: false }))
        onError?.(error)
      }
    } catch (error) {
      const errorMessage = `Speech synthesis error: ${error instanceof Error ? error.message : 'Unknown error'}`
      setState(prev => ({ ...prev, error: errorMessage, isLoading: false }))
      onError?.(errorMessage)
    }
  }

  // Play/pause control
  const togglePlayback = async () => {
    if (!audioRef.current) return

    try {
      if (state.isPlaying) {
        audioRef.current.pause()
        setState(prev => ({ ...prev, isPlaying: false, isPaused: true }))
      } else {
        await audioRef.current.play()
        setState(prev => ({ ...prev, isPlaying: true, isPaused: false }))
      }
    } catch (error) {
      onError?.('Playback control failed')
    }
  }

  // Stop playback
  const stopPlayback = () => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      setState(prev => ({ ...prev, isPlaying: false, isPaused: false, currentTime: 0 }))
    }
  }

  // Volume control
  const handleVolumeChange = (volume: number) => {
    setState(prev => ({ ...prev, volume }))
    if (audioRef.current) {
      audioRef.current.volume = volume
    }
  }

  // Seek control
  const handleSeek = (time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time
      setState(prev => ({ ...prev, currentTime: time }))
    }
  }

  // Format time for display
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Progress bar
  const progressPercentage = state.duration > 0 ? (state.currentTime / state.duration) * 100 : 0

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
      }
      if (currentAudioUrl.current) {
        URL.revokeObjectURL(currentAudioUrl.current)
      }
    }
  }, [])

  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Text-to-Speech</h3>
        <div className="text-sm text-gray-500">
          {voices.length} voice{voices.length !== 1 ? 's' : ''} available
        </div>
      </div>

      {/* Voice Selection */}
      {voices.length > 0 && (
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Voice
          </label>
          <select
            value={selectedVoice}
            onChange={(e) => setSelectedVoice(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {voices.map((voice) => (
              <option key={voice.id} value={voice.id}>
                {voice.name} ({voice.language_type})
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Text Input */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Text to Speak
        </label>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter text to convert to speech..."
          rows={4}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
        />
        <div className="text-xs text-gray-500 mt-1">
          {inputText.length} characters
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center space-x-3 mb-4">
        <button
          onClick={() => handleSpeak()}
          disabled={state.isLoading || !inputText.trim() || !selectedVoice}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {state.isLoading ? (
            <>
              <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
              <span>Generating...</span>
            </>
          ) : (
            <>
              <div className="w-4 h-4 bg-white rounded-sm" />
              <span>Speak</span>
            </>
          )}
        </button>

        {audioRef.current && (
          <>
            <button
              onClick={togglePlayback}
              className="flex items-center space-x-2 px-3 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-all"
            >
              {state.isPlaying ? (
                <>
                  <div className="w-3 h-3 bg-white rounded-sm" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <div className="w-0 h-0 border-l-[6px] border-l-white border-y-[4px] border-y-transparent" />
                  <span>Play</span>
                </>
              )}
            </button>

            <button
              onClick={stopPlayback}
              className="flex items-center space-x-2 px-3 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-all"
            >
              <div className="w-3 h-3 bg-white rounded-sm" />
              <span>Stop</span>
            </button>
          </>
        )}
      </div>

      {/* Audio Player */}
      {audioRef.current && (
        <div className="bg-gray-50 rounded-lg p-3 space-y-3">
          {/* Progress Bar */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-gray-500">
              <span>{formatTime(state.currentTime)}</span>
              <span>{formatTime(state.duration)}</span>
            </div>
            <div 
              className="w-full h-2 bg-gray-200 rounded-full cursor-pointer"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect()
                const percent = (e.clientX - rect.left) / rect.width
                const newTime = percent * state.duration
                handleSeek(newTime)
              }}
            >
              <div 
                className="h-full bg-blue-500 rounded-full transition-all duration-100"
                style={{ width: `${progressPercentage}%` }}
              />
            </div>
          </div>

          {/* Volume Control */}
          <div className="flex items-center space-x-3">
            <span className="text-xs text-gray-500 min-w-[3rem]">Volume:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={state.volume}
              onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
              className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <span className="text-xs text-gray-500 min-w-[3rem]">
              {Math.round(state.volume * 100)}%
            </span>
          </div>
        </div>
      )}

      {/* Error Display */}
      {state.error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mt-4">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-red-500 rounded-full flex-shrink-0" />
            <span className="text-sm text-red-700">{state.error}</span>
          </div>
        </div>
      )}

      {/* Tips */}
      <div className="text-xs text-gray-500 bg-gray-50 rounded-lg p-3 mt-4">
        <p><strong>Tips:</strong></p>
        <ul className="mt-1 space-y-1">
          <li>• Short sentences work best for natural speech</li>
          <li>• Punctuation affects speech rhythm and pauses</li>
          <li>• Different voices may work better for different languages</li>
        </ul>
      </div>
    </div>
  )
} 