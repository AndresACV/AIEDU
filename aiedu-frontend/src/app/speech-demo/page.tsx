'use client'

import React, { useState } from 'react'
import { SpeechControls } from '@/components/speech'

export default function SpeechDemoPage() {
  const [transcriptions, setTranscriptions] = useState<Array<{
    text: string
    confidence?: number
    timestamp: Date
  }>>([])
  const [errors, setErrors] = useState<Array<{
    message: string
    timestamp: Date
  }>>([])
  const [selectedLanguage, setSelectedLanguage] = useState('en-US')
  const [layout, setLayout] = useState<'tabs' | 'vertical' | 'horizontal'>('tabs')

  const handleTranscription = (text: string, confidence?: number) => {
    setTranscriptions(prev => [...prev, {
      text,
      confidence,
      timestamp: new Date()
    }])
  }

  const handleError = (error: string) => {
    setErrors(prev => [...prev, {
      message: error,
      timestamp: new Date()
    }])
  }

  const clearTranscriptions = () => {
    setTranscriptions([])
  }

  const clearErrors = () => {
    setErrors([])
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Speech Interface Demo
          </h1>
          <p className="text-gray-600">
            Test the complete speech functionality including audio recording, transcription, 
            text-to-speech synthesis, and voice selection.
          </p>
        </div>

        {/* Controls */}
        <div className="mb-6 bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Demo Settings</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Language Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Speech Language
              </label>
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="en-US">English (US)</option>
                <option value="es-ES">Spanish (ES)</option>
              </select>
            </div>

            {/* Layout Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Interface Layout
              </label>
              <select
                value={layout}
                onChange={(e) => setLayout(e.target.value as any)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="tabs">Tabs (Recommended)</option>
                <option value="vertical">Vertical Stack</option>
                <option value="horizontal">Horizontal Grid</option>
              </select>
            </div>

            {/* Clear Actions */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Clear Data
              </label>
              <div className="flex space-x-2">
                <button
                  onClick={clearTranscriptions}
                  className="px-3 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg text-sm transition-all"
                >
                  Clear Transcriptions
                </button>
                <button
                  onClick={clearErrors}
                  className="px-3 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg text-sm transition-all"
                >
                  Clear Errors
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Speech Interface */}
        <div className="mb-8">
          <SpeechControls
            onTranscription={handleTranscription}
            onError={handleError}
            language={selectedLanguage}
            layout={layout}
            showVoiceSelector={true}
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
          />
        </div>

        {/* Results Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Transcription History */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">
                Transcription History
              </h3>
              <span className="text-sm text-gray-500">
                {transcriptions.length} result{transcriptions.length !== 1 ? 's' : ''}
              </span>
            </div>

            {transcriptions.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <div className="w-12 h-12 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                  🎤
                </div>
                <p>No transcriptions yet</p>
                <p className="text-sm">Record some audio to see results here</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {transcriptions.slice().reverse().map((transcription, index) => (
                  <div key={index} className="p-3 bg-gray-50 rounded-lg">
                    <div className="text-sm text-gray-800 mb-1">
                      "{transcription.text}"
                    </div>
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>
                        {transcription.confidence && 
                          `Confidence: ${Math.round(transcription.confidence * 100)}%`
                        }
                      </span>
                      <span>{transcription.timestamp.toLocaleTimeString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Error Log */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">
                Error Log
              </h3>
              <span className="text-sm text-gray-500">
                {errors.length} error{errors.length !== 1 ? 's' : ''}
              </span>
            </div>

            {errors.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <div className="w-12 h-12 mx-auto mb-4 bg-green-100 rounded-full flex items-center justify-center">
                  ✅
                </div>
                <p>No errors</p>
                <p className="text-sm">All speech operations are working correctly</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {errors.slice().reverse().map((error, index) => (
                  <div key={index} className="p-3 bg-red-50 border border-red-200 rounded-lg">
                    <div className="text-sm text-red-800 mb-1">
                      {error.message}
                    </div>
                    <div className="text-xs text-red-500">
                      {error.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* System Status */}
        <div className="mt-8 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">System Status</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-700">FastAPI Backend Connected</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-700">Speech Services Available</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-700">Voice Models Loaded</span>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="text-sm text-blue-800">
              <strong>API Endpoint:</strong> {process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'}
            </div>
            <div className="text-sm text-blue-600 mt-1">
              <strong>Current Language:</strong> {selectedLanguage === 'en-US' ? 'English (US)' : 'Spanish (ES)'}
            </div>
          </div>
        </div>

        {/* Feature Documentation */}
        <div className="mt-8 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Features & Testing Guide</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-medium text-gray-800 mb-2">🎤 Audio Recording</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Click "Start Recording" to begin</li>
                <li>• Real-time volume visualization</li>
                <li>• Automatic speech-to-text processing</li>
                <li>• Supports English and Spanish</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-800 mb-2">🔊 Text-to-Speech</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Enter custom text or use transcriptions</li>
                <li>• Multiple voice options</li>
                <li>• Audio playback controls</li>
                <li>• Volume and progress controls</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-800 mb-2">🎭 Voice Selection</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Browse available voices by language</li>
                <li>• Preview voices before selection</li>
                <li>• Custom preview text support</li>
                <li>• Voice information display</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 