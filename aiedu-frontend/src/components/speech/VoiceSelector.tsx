'use client'

import React, { useState, useEffect } from 'react'
import { apiClient } from '@/services/api'
import { Voice } from '@/types/speech'

interface VoiceSelectorProps {
  selectedVoice?: string
  onVoiceChange?: (voiceId: string, voice: Voice) => void
  onError?: (error: string) => void
  className?: string
  showPreview?: boolean
}

export default function VoiceSelector({
  selectedVoice,
  onVoiceChange,
  onError,
  className = '',
  showPreview = true
}: VoiceSelectorProps) {
  const [voices, setVoices] = useState<Voice[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [previewText, setPreviewText] = useState('')
  const [isPreviewPlaying, setIsPreviewPlaying] = useState<string | null>(null)

  // Default preview texts for different languages
  const defaultPreviewTexts = {
    English: "Hello! This is how I sound when speaking English.",
    Spanish: "¡Hola! Así es como sueno cuando hablo en español."
  }

  useEffect(() => {
    loadVoices()
  }, [])

  const loadVoices = async () => {
    try {
      setIsLoading(true)
      const response = await apiClient.getVoices()
      
      if (response.success && response.voices) {
        setVoices(response.voices)
        
        // Set default selection if none provided
        if (!selectedVoice && response.voices.length > 0) {
          const defaultVoice = response.voices.find(v => v.language_type === 'Spanish') || 
                              response.voices.find(v => v.language_type === 'English') ||
                              response.voices[0]
          if (defaultVoice) {
            onVoiceChange?.(defaultVoice.id, defaultVoice)
          }
        }
      } else {
        onError?.('Failed to load voices')
      }
    } catch (error) {
      console.error('Error loading voices:', error)
      onError?.('Error loading voices')
    } finally {
      setIsLoading(false)
    }
  }

  const handleVoiceSelect = (voice: Voice) => {
    onVoiceChange?.(voice.id, voice)
    
    // Update preview text based on voice language
    if (voice.language_type && defaultPreviewTexts[voice.language_type as keyof typeof defaultPreviewTexts]) {
      setPreviewText(defaultPreviewTexts[voice.language_type as keyof typeof defaultPreviewTexts])
    }
  }

  const playPreview = async (voice: Voice) => {
    if (isPreviewPlaying) return

    const textToSpeak = previewText || 
                      defaultPreviewTexts[voice.language_type as keyof typeof defaultPreviewTexts] ||
                      "This is a voice preview."

    try {
      setIsPreviewPlaying(voice.id)

      const response = await apiClient.synthesize({
        text: textToSpeak,
        voice_id: voice.id,
        language: voice.language_type === 'Spanish' ? 'es-ES' : 'en-US'
      })

      if (response.success && response.audio_url) {
        const audio = new Audio(response.audio_url)
        
        audio.addEventListener('ended', () => {
          setIsPreviewPlaying(null)
        })
        
        audio.addEventListener('error', () => {
          setIsPreviewPlaying(null)
          onError?.('Preview playback failed')
        })

        await audio.play()
      } else {
        setIsPreviewPlaying(null)
        onError?.('Preview generation failed')
      }
    } catch (error) {
      setIsPreviewPlaying(null)
      onError?.('Preview error: ' + (error instanceof Error ? error.message : 'Unknown error'))
    }
  }

  const groupedVoices = voices.reduce((groups, voice) => {
    const lang = voice.language_type || 'Other'
    if (!groups[lang]) {
      groups[lang] = []
    }
    groups[lang].push(voice)
    return groups
  }, {} as Record<string, Voice[]>)

  if (isLoading) {
    return (
      <div className={`bg-white rounded-lg border border-gray-200 p-4 ${className}`}>
        <div className="flex items-center justify-center space-x-2">
          <div className="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full" />
          <span className="text-gray-600">Loading voices...</span>
        </div>
      </div>
    )
  }

  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Voice Selection</h3>
        <div className="text-sm text-gray-500">
          {voices.length} voice{voices.length !== 1 ? 's' : ''} available
        </div>
      </div>

      {voices.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <div className="w-12 h-12 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
            <div className="w-6 h-6 bg-gray-400 rounded-full" />
          </div>
          <p>No voices available</p>
          <button 
            onClick={loadVoices}
            className="mt-2 px-3 py-1 bg-blue-500 text-white rounded-lg text-sm hover:bg-blue-600 transition-all"
          >
            Retry
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Custom Preview Text */}
          {showPreview && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Preview Text (Optional)
              </label>
              <input
                type="text"
                value={previewText}
                onChange={(e) => setPreviewText(e.target.value)}
                placeholder="Enter custom preview text..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
              />
            </div>
          )}

          {/* Voice Groups */}
          {Object.entries(groupedVoices).map(([language, languageVoices]) => (
            <div key={language} className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700 border-b border-gray-200 pb-1">
                {language} ({languageVoices.length})
              </h4>
              
              <div className="grid gap-2">
                {languageVoices.map((voice) => (
                  <div
                    key={voice.id}
                    className={`p-3 border rounded-lg cursor-pointer transition-all ${
                      selectedVoice === voice.id
                        ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                        : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                    }`}
                    onClick={() => handleVoiceSelect(voice)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <div className={`w-3 h-3 rounded-full ${
                            selectedVoice === voice.id ? 'bg-blue-500' : 'bg-gray-300'
                          }`} />
                          <span className="font-medium text-gray-800">{voice.name}</span>
                          {voice.provider && (
                            <span className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded-full">
                              {voice.provider}
                            </span>
                          )}
                        </div>
                        {voice.language_type && (
                          <div className="text-sm text-gray-500 mt-1 ml-5">
                            Language: {voice.language_type}
                          </div>
                        )}
                      </div>

                      {/* Preview Button */}
                      {showPreview && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            playPreview(voice)
                          }}
                          disabled={isPreviewPlaying !== null}
                          className={`ml-3 px-3 py-1 text-xs rounded-lg transition-all ${
                            isPreviewPlaying === voice.id
                              ? 'bg-green-500 text-white'
                              : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                          } ${isPreviewPlaying ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          {isPreviewPlaying === voice.id ? (
                            <div className="flex items-center space-x-1">
                              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                              <span>Playing</span>
                            </div>
                          ) : (
                            <div className="flex items-center space-x-1">
                              <div className="w-0 h-0 border-l-[4px] border-l-gray-600 border-y-[2px] border-y-transparent" />
                              <span>Preview</span>
                            </div>
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Voice Info */}
      {selectedVoice && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="text-sm text-blue-800">
            <strong>Selected Voice:</strong>{' '}
            {voices.find(v => v.id === selectedVoice)?.name || selectedVoice}
          </div>
          {voices.find(v => v.id === selectedVoice)?.language_type && (
            <div className="text-sm text-blue-600 mt-1">
              Language: {voices.find(v => v.id === selectedVoice)?.language_type}
            </div>
          )}
        </div>
      )}

      {/* Tips */}
      <div className="text-xs text-gray-500 bg-gray-50 rounded-lg p-3 mt-4">
        <p><strong>Voice Selection Tips:</strong></p>
        <ul className="mt-1 space-y-1">
          <li>• Click on a voice to select it</li>
          {showPreview && <li>• Use preview to hear how each voice sounds</li>}
          <li>• Different voices work better for different languages</li>
          <li>• Some voices may have different speaking speeds or styles</li>
        </ul>
      </div>
    </div>
  )
} 