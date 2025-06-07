'use client'

import React, { useState, useCallback } from 'react'
import AudioRecorder from './AudioRecorder'
import TextToSpeech from './TextToSpeech'
import VoiceSelector from './VoiceSelector'
import { Voice } from '@/types/speech'

interface SpeechControlsProps {
  onTranscription?: (text: string, confidence?: number) => void
  onSpeechComplete?: () => void
  onError?: (error: string) => void
  initialText?: string
  className?: string
  layout?: 'horizontal' | 'vertical' | 'tabs'
  showVoiceSelector?: boolean
  language?: string
}

type TabType = 'record' | 'speak' | 'voices'

export default function SpeechControls({
  onTranscription,
  onSpeechComplete,
  onError,
  initialText = '',
  className = '',
  layout = 'tabs',
  showVoiceSelector = true,
  language = 'en-US'
}: SpeechControlsProps) {
  const [activeTab, setActiveTab] = useState<TabType>('record')
  const [selectedVoice, setSelectedVoice] = useState<string>('')
  const [currentVoice, setCurrentVoice] = useState<Voice | null>(null)
  const [transcribedText, setTranscribedText] = useState('')
  const [errors, setErrors] = useState<string[]>([])

  // Handle transcription from audio recorder
  const handleTranscription = useCallback((text: string, confidence?: number) => {
    setTranscribedText(text)
    onTranscription?.(text, confidence)
    
    // Auto-switch to speak tab after successful transcription
    if (text.trim()) {
      setActiveTab('speak')
    }
  }, [onTranscription])

  // Handle voice selection
  const handleVoiceChange = useCallback((voiceId: string, voice: Voice) => {
    setSelectedVoice(voiceId)
    setCurrentVoice(voice)
  }, [])

  // Handle errors from components
  const handleError = useCallback((error: string) => {
    setErrors(prev => [...prev.slice(-4), error]) // Keep last 5 errors
    onError?.(error)
  }, [onError])

  // Clear errors
  const clearErrors = () => {
    setErrors([])
  }

  // Tab Navigation
  const TabButton = ({ tab, label, icon, disabled = false }: { 
    tab: TabType, 
    label: string, 
    icon: React.ReactNode,
    disabled?: boolean 
  }) => (
    <button
      onClick={() => setActiveTab(tab)}
      disabled={disabled}
      className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all ${
        activeTab === tab
          ? 'bg-blue-500 text-white shadow-md'
          : disabled
          ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
          : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  )

  // Render content based on layout
  const renderTabsLayout = () => (
    <div className="space-y-4">
      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2 p-1 bg-gray-50 rounded-lg">
        <TabButton
          tab="record"
          label="Record"
          icon={<div className="w-4 h-4 bg-current rounded-full" />}
        />
        <TabButton
          tab="speak"
          label="Text-to-Speech"
          icon={<div className="w-4 h-4 bg-current rounded-sm" />}
        />
        {showVoiceSelector && (
          <TabButton
            tab="voices"
            label="Voice Settings"
            icon={<div className="w-4 h-4 bg-current rounded-full border-2 border-white" />}
          />
        )}
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {activeTab === 'record' && (
          <AudioRecorder
            onTranscription={handleTranscription}
            onError={handleError}
            language={language}
            className="h-full"
          />
        )}
        
        {activeTab === 'speak' && (
          <TextToSpeech
            text={transcribedText || initialText}
            onError={handleError}
            onSynthesisComplete={onSpeechComplete}
            className="h-full"
          />
        )}
        
        {activeTab === 'voices' && showVoiceSelector && (
          <VoiceSelector
            selectedVoice={selectedVoice}
            onVoiceChange={handleVoiceChange}
            onError={handleError}
            className="h-full"
          />
        )}
      </div>
    </div>
  )

  const renderVerticalLayout = () => (
    <div className="space-y-6">
      {/* Audio Recorder */}
      <AudioRecorder
        onTranscription={handleTranscription}
        onError={handleError}
        language={language}
      />
      
      {/* Text-to-Speech */}
      <TextToSpeech
        text={transcribedText || initialText}
        onError={handleError}
        onSynthesisComplete={onSpeechComplete}
      />
      
      {/* Voice Selector */}
      {showVoiceSelector && (
        <VoiceSelector
          selectedVoice={selectedVoice}
          onVoiceChange={handleVoiceChange}
          onError={handleError}
        />
      )}
    </div>
  )

  const renderHorizontalLayout = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
      {/* Audio Recorder */}
      <AudioRecorder
        onTranscription={handleTranscription}
        onError={handleError}
        language={language}
        className="h-fit"
      />
      
      {/* Text-to-Speech */}
      <TextToSpeech
        text={transcribedText || initialText}
        onError={handleError}
        onSynthesisComplete={onSpeechComplete}
        className="h-fit"
      />
      
      {/* Voice Selector */}
      {showVoiceSelector && (
        <div className="lg:col-span-2 xl:col-span-1">
          <VoiceSelector
            selectedVoice={selectedVoice}
            onVoiceChange={handleVoiceChange}
            onError={handleError}
            className="h-fit"
          />
        </div>
      )}
    </div>
  )

  return (
    <div className={`${className}`}>
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-2xl font-bold text-gray-800">Speech Interface</h2>
          <div className="flex items-center space-x-2">
            {currentVoice && (
              <span className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded-full">
                Voice: {currentVoice.name}
              </span>
            )}
            <span className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded-full">
              Language: {language === 'en-US' ? 'English' : 'Spanish'}
            </span>
          </div>
        </div>
        
        {transcribedText && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="text-sm text-green-800">
              <strong>Last Transcription:</strong> {transcribedText}
            </div>
          </div>
        )}
      </div>

      {/* Error Display */}
      {errors.length > 0 && (
        <div className="mb-4 space-y-2">
          {errors.map((error, index) => (
            <div key={index} className="bg-red-50 border border-red-200 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-red-700">{error}</span>
                <button
                  onClick={() => setErrors(prev => prev.filter((_, i) => i !== index))}
                  className="text-red-500 hover:text-red-700 ml-2"
                >
                  ✕
                </button>
              </div>
            </div>
          ))}
          {errors.length > 1 && (
            <button
              onClick={clearErrors}
              className="text-xs text-red-600 hover:text-red-800 underline"
            >
              Clear all errors
            </button>
          )}
        </div>
      )}

      {/* Main Content */}
      {layout === 'tabs' && renderTabsLayout()}
      {layout === 'vertical' && renderVerticalLayout()}
      {layout === 'horizontal' && renderHorizontalLayout()}

      {/* Quick Actions */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Quick Actions</h3>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setActiveTab('record')}
            className="px-3 py-1 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg text-sm transition-all"
          >
            🎤 Record Audio
          </button>
          <button
            onClick={() => setActiveTab('speak')}
            disabled={!transcribedText && !initialText}
            className="px-3 py-1 bg-green-100 hover:bg-green-200 text-green-700 rounded-lg text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            🔊 Convert to Speech
          </button>
          {showVoiceSelector && (
            <button
              onClick={() => setActiveTab('voices')}
              className="px-3 py-1 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-lg text-sm transition-all"
            >
              🎭 Change Voice
            </button>
          )}
          <button
            onClick={() => {
              setTranscribedText('')
              clearErrors()
            }}
            className="px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-sm transition-all"
          >
            🗑️ Clear All
          </button>
        </div>
      </div>

      {/* Usage Tips */}
      <div className="mt-4 text-xs text-gray-500 bg-gray-50 rounded-lg p-3">
        <p><strong>Speech Interface Guide:</strong></p>
        <ul className="mt-1 space-y-1">
          <li>• <strong>Record:</strong> Click "Start Recording" and speak clearly into your microphone</li>
          <li>• <strong>Text-to-Speech:</strong> Enter text or use transcribed audio to generate speech</li>
          <li>• <strong>Voice Settings:</strong> Choose different voices and preview them</li>
          <li>• <strong>Language:</strong> Ensure your microphone language matches the selected interface language</li>
          <li>• <strong>Quality:</strong> Use a quiet environment for best speech recognition results</li>
        </ul>
      </div>
    </div>
  )
} 