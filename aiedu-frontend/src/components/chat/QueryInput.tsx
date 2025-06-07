'use client'

import React, { useState, useRef } from 'react'
import { useSpeech } from '@/hooks/useSpeech'

interface QueryInputProps {
  onSubmit: (query: string) => void
  onVoiceInput?: (text: string) => void
  isLoading?: boolean
  placeholder?: string
  enableVoice?: boolean
  className?: string
}

export default function QueryInput({
  onSubmit,
  onVoiceInput,
  isLoading = false,
  placeholder = "Ask a question about your documents...",
  enableVoice = true,
  className = ''
}: QueryInputProps) {
  const [inputText, setInputText] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Speech integration
  const {
    startRecording,
    stopRecording,
    isRecording,
    isProcessingRecording,
    lastTranscription,
    clearTranscription,
    error: speechError
  } = useSpeech({
    onTranscription: (text) => {
      setInputText(text)
      onVoiceInput?.(text)
      // Focus textarea after voice input
      setTimeout(() => textareaRef.current?.focus(), 100)
    }
  })

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (inputText.trim() && !isLoading) {
      onSubmit(inputText.trim())
      setInputText('')
      clearTranscription()
    }
  }

  // Handle keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value)
    
    // Auto-resize
    const textarea = e.target
    textarea.style.height = 'auto'
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px'
  }

  // Toggle voice recording
  const toggleRecording = () => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  return (
    <div className={`bg-white border border-gray-200 rounded-lg p-4 ${className}`}>
      {/* Voice transcription feedback */}
      {lastTranscription && (
        <div className="mb-3 p-3 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-green-800">Voice Input:</span>
              <p className="text-sm text-green-700 mt-1">"{lastTranscription}"</p>
            </div>
            <button
              onClick={clearTranscription}
              className="text-green-600 hover:text-green-800 ml-2"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Speech error */}
      {speechError && (
        <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg">
          <span className="text-sm text-red-700">{speechError}</span>
        </div>
      )}

      {/* Main input form */}
      <form onSubmit={handleSubmit}>
        <div className="flex items-end space-x-3">
          {/* Text input */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={inputText}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              rows={1}
              disabled={isLoading || isRecording}
              className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all disabled:bg-gray-50 disabled:text-gray-500"
              style={{ 
                minHeight: '48px',
                maxHeight: '120px'
              }}
            />
            
            {/* Character count */}
            <div className="absolute bottom-2 right-2 text-xs text-gray-400">
              {inputText.length}
            </div>
          </div>

          {/* Voice button */}
          {enableVoice && (
            <button
              type="button"
              onClick={toggleRecording}
              disabled={isLoading || isProcessingRecording}
              className={`p-3 rounded-lg transition-all min-w-[48px] h-[48px] flex items-center justify-center ${
                isRecording
                  ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse'
                  : isProcessingRecording
                  ? 'bg-yellow-500 text-white cursor-not-allowed'
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-600'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
              title={isRecording ? 'Stop recording' : 'Start voice input'}
            >
              {isProcessingRecording ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : isRecording ? (
                <div className="w-4 h-4 bg-white rounded-sm" />
              ) : (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
                </svg>
              )}
            </button>
          )}

          {/* Submit button */}
          <button
            type="submit"
            disabled={!inputText.trim() || isLoading || isRecording}
            className="px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed min-w-[80px] h-[48px] flex items-center justify-center font-medium"
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <span>Send</span>
            )}
          </button>
        </div>

        {/* Helper text */}
        <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
          <div className="flex items-center space-x-4">
            <span>Press Enter to send, Shift+Enter for new line</span>
            {enableVoice && (
              <span>• Click microphone for voice input</span>
            )}
          </div>
          
          {isRecording && (
            <div className="flex items-center space-x-2 text-red-600">
              <div className="w-2 h-2 bg-red-600 rounded-full animate-pulse" />
              <span>Recording... Click to stop</span>
            </div>
          )}
        </div>
      </form>

      {/* Example queries */}
      {!inputText && !isLoading && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <div className="text-xs text-gray-500 mb-2">Try asking:</div>
          <div className="flex flex-wrap gap-2">
            {[
              "What documents do you have?",
              "Summarize the main topics",
              "What is artificial intelligence?",
              "Tell me about machine learning"
            ].map((suggestion, index) => (
              <button
                key={index}
                onClick={() => setInputText(suggestion)}
                className="px-3 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-full text-xs transition-all"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
} 