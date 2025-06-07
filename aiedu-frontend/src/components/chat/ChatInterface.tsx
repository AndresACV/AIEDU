'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { apiClient } from '@/services/api'
import { RagQueryRequest, RagQueryResponse } from '@/types/rag'
import { useSpeech } from '@/hooks/useSpeech'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  confidence?: number
  retrievedDocs?: Array<{
    title: string
    content_preview: string
    doc_id: string
    similarity_score: number
  }>
  metadata?: {
    response_time?: number
    retrieval_time?: number
    generation_time?: number
    tokens_generated?: number
  }
}

interface ChatInterfaceProps {
  onError?: (error: string) => void
  enableSpeech?: boolean
  enableVoiceInput?: boolean
  autoSpeak?: boolean
  className?: string
  maxMessages?: number
}

export default function ChatInterface({
  onError,
  enableSpeech = true,
  enableVoiceInput = true,
  autoSpeak = false,
  className = '',
  maxMessages = 50
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputText, setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showDocuments, setShowDocuments] = useState(true)
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Speech integration
  const {
    synthesizeText,
    isPlaying,
    lastTranscription,
    clearTranscription,
    voices,
    selectedVoice
  } = useSpeech({
    defaultLanguage: 'en-US',
    onTranscription: handleVoiceInput,
    onError: (error) => onError?.(error)
  })

  // Handle voice input transcription
  function handleVoiceInput(text: string, confidence?: number) {
    setInputText(text)
    if (inputRef.current) {
      inputRef.current.focus()
    }
  }

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Submit query
  const submitQuery = async (query: string) => {
    if (!query.trim() || isLoading) return

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: query.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputText('')
    setIsLoading(true)

    try {
      const request: RagQueryRequest = {
        query: query.trim(),
        n_results: 3,
        temperature: 0.7
      }

      const response: RagQueryResponse = await apiClient.ragQuery(request)

      if (response.success && response.response) {
        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          type: 'assistant',
          content: response.response,
          timestamp: new Date(),
          retrievedDocs: response.retrieved_documents,
          metadata: {
            response_time: response.timing?.total_time,
            retrieval_time: response.timing?.retrieval_time,
            generation_time: response.timing?.generation_time,
            tokens_generated: response.metadata?.tokens_generated
          }
        }

        setMessages(prev => {
          const newMessages = [...prev, assistantMessage]
          return newMessages.slice(-maxMessages) // Keep only last N messages
        })

        // Auto-speak response if enabled
        if (autoSpeak && enableSpeech && response.response) {
          synthesizeText(response.response)
        }
      } else {
        const errorMessage = response.error || 'Failed to get response'
        onError?.(errorMessage)
        
        const errorResponse: Message = {
          id: `error-${Date.now()}`,
          type: 'assistant',
          content: `Sorry, I encountered an error: ${errorMessage}`,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, errorResponse])
      }
    } catch (error) {
      const errorMessage = `Chat error: ${error instanceof Error ? error.message : 'Unknown error'}`
      onError?.(errorMessage)
      
      const errorResponse: Message = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: 'Sorry, I\'m having trouble connecting to the AI service. Please try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorResponse])
    } finally {
      setIsLoading(false)
    }
  }

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    submitQuery(inputText)
  }

  // Handle keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submitQuery(inputText)
    }
  }

  // Speak message
  const speakMessage = (content: string) => {
    if (enableSpeech) {
      synthesizeText(content)
    }
  }

  // Clear conversation
  const clearConversation = () => {
    setMessages([])
    setSelectedMessageId(null)
  }

  // Format time
  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className={`flex flex-col h-full bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
            <span className="text-white text-sm font-bold">AI</span>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-800">RAG Assistant</h3>
            <p className="text-sm text-gray-500">
              Ask questions about your documents
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {/* Document toggle */}
          <button
            onClick={() => setShowDocuments(!showDocuments)}
            className={`px-3 py-1 text-xs rounded-lg transition-all ${
              showDocuments 
                ? 'bg-blue-100 text-blue-700' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {showDocuments ? 'Hide Docs' : 'Show Docs'}
          </button>

          {/* Clear button */}
          <button
            onClick={clearConversation}
            disabled={messages.length === 0}
            className="px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Clear
          </button>

          {/* Status indicator */}
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-xs text-gray-500">Online</span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
              <span className="text-2xl">💬</span>
            </div>
            <h4 className="text-lg font-medium text-gray-800 mb-2">Start a conversation</h4>
            <p className="text-gray-500 max-w-md mx-auto">
              Ask questions about your documents and I'll provide answers based on your knowledge base.
            </p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              <button
                onClick={() => setInputText("What documents do you have?")}
                className="px-3 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg text-sm transition-all"
              >
                What documents do you have?
              </button>
              <button
                onClick={() => setInputText("Explain artificial intelligence")}
                className="px-3 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg text-sm transition-all"
              >
                Explain artificial intelligence
              </button>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] ${
                  message.type === 'user'
                    ? 'bg-blue-500 text-white rounded-l-lg rounded-tr-lg'
                    : 'bg-gray-100 text-gray-800 rounded-r-lg rounded-tl-lg'
                } p-3 shadow-sm`}
              >
                {/* Message content */}
                <div className="whitespace-pre-wrap break-words">
                  {message.content}
                </div>

                {/* Message metadata */}
                <div className={`flex items-center justify-between mt-2 text-xs ${
                  message.type === 'user' ? 'text-blue-100' : 'text-gray-500'
                }`}>
                  <span>{formatTime(message.timestamp)}</span>
                  
                  {message.type === 'assistant' && (
                    <div className="flex items-center space-x-2">
                      {message.metadata?.response_time && (
                        <span>{message.metadata.response_time.toFixed(1)}s</span>
                      )}
                      
                      {enableSpeech && (
                        <button
                          onClick={() => speakMessage(message.content)}
                          disabled={isPlaying}
                          className="hover:bg-gray-200 p-1 rounded transition-all"
                          title="Speak this message"
                        >
                          {isPlaying ? '🔊' : '🔉'}
                        </button>
                      )}

                      {message.retrievedDocs && message.retrievedDocs.length > 0 && (
                        <button
                          onClick={() => setSelectedMessageId(
                            selectedMessageId === message.id ? null : message.id
                          )}
                          className="hover:bg-gray-200 p-1 rounded transition-all"
                          title="Show retrieved documents"
                        >
                          📚 {message.retrievedDocs.length}
                        </button>
                      )}
                    </div>
                  )}
                </div>

                {/* Retrieved documents */}
                {message.type === 'assistant' && 
                 message.retrievedDocs && 
                 showDocuments && 
                 selectedMessageId === message.id && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <div className="text-xs font-medium text-gray-600 mb-2">
                      Retrieved Documents:
                    </div>
                    <div className="space-y-2">
                      {message.retrievedDocs.map((doc, index) => (
                        <div key={doc.doc_id} className="bg-white border border-gray-200 rounded p-2">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-gray-800">
                              {doc.title || `Document ${index + 1}`}
                            </span>
                            <span className="text-xs text-gray-500">
                              {Math.round((1 + doc.similarity_score) * 50)}% match
                            </span>
                          </div>
                          <p className="text-xs text-gray-600 line-clamp-2">
                            {doc.content_preview}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-r-lg rounded-tl-lg p-3 shadow-sm">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-xs text-gray-500">Thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4">
        {/* Voice transcription indicator */}
        {lastTranscription && (
          <div className="mb-3 p-2 bg-green-50 border border-green-200 rounded-lg flex items-center justify-between">
            <span className="text-sm text-green-800">
              <strong>Voice input:</strong> {lastTranscription}
            </span>
            <button
              onClick={clearTranscription}
              className="text-green-600 hover:text-green-800 text-sm"
            >
              ✕
            </button>
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex items-end space-x-3">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              rows={1}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              style={{ minHeight: '44px', maxHeight: '120px' }}
            />
          </div>

          <button
            type="submit"
            disabled={!inputText.trim() || isLoading}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed min-w-[80px] h-[44px] flex items-center justify-center"
          >
            {isLoading ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              'Send'
            )}
          </button>
        </form>

        {/* Quick actions */}
        <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
          <div className="flex items-center space-x-3">
            <span>Press Enter to send, Shift+Enter for new line</span>
            {enableVoiceInput && (
              <span>• Use voice input button for speech-to-text</span>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <span>{messages.length} messages</span>
            {voices.length > 0 && selectedVoice && (
              <span>• Voice: {selectedVoice.name}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
} 