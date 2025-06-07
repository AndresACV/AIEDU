'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { apiClient } from '@/services/api'
import { RagQueryRequest, RagQueryResponse } from '@/types/rag'
import { useSpeech } from '@/hooks/useSpeech'
import QueryInput from './QueryInput'
import DocumentViewer from './DocumentViewer'

interface Message {
  id: string
  type: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  query?: string
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
  error?: string
}

interface RagChatInterfaceProps {
  onError?: (error: string) => void
  enableSpeech?: boolean
  enableVoiceInput?: boolean
  autoSpeak?: boolean
  className?: string
  maxMessages?: number
  showDocuments?: boolean
  layout?: 'vertical' | 'horizontal' | 'sidebar'
}

export default function RagChatInterface({
  onError,
  enableSpeech = true,
  enableVoiceInput = true,
  autoSpeak = false,
  className = '',
  maxMessages = 50,
  showDocuments = true,
  layout = 'vertical'
}: RagChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null)
  const [ragStats, setRagStats] = useState<{
    totalQueries: number
    avgResponseTime: number
    totalDocuments: number
  }>({ totalQueries: 0, avgResponseTime: 0, totalDocuments: 0 })

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  // Speech integration
  const {
    synthesizeText,
    isPlaying,
    stopSpeech,
    selectedVoice,
    voices
  } = useSpeech({
    onError: (error) => onError?.(error)
  })

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Initialize with welcome message
  useEffect(() => {
    const welcomeMessage: Message = {
      id: 'welcome',
      type: 'system',
      content: "👋 Welcome to the RAG Assistant! I can help you find information from your documents. Ask me anything!",
      timestamp: new Date()
    }
    setMessages([welcomeMessage])
  }, [])

  // Submit RAG query
  const submitQuery = async (query: string) => {
    if (!query.trim() || isLoading) return

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: query.trim(),
      timestamp: new Date(),
      query: query.trim()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      const request: RagQueryRequest = {
        query: query.trim(),
        n_results: 5,
        temperature: 0.7
      }

      const startTime = performance.now()
      const response: RagQueryResponse = await apiClient.ragQuery(request)
      const endTime = performance.now()

      if (response.success && response.response) {
        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          type: 'assistant',
          content: response.response,
          timestamp: new Date(),
          query: query.trim(),
          retrievedDocs: response.retrieved_documents,
          metadata: {
            response_time: response.timing?.total_time || ((endTime - startTime) / 1000),
            retrieval_time: response.timing?.retrieval_time,
            generation_time: response.timing?.generation_time,
            tokens_generated: response.metadata?.tokens_generated
          }
        }

        setMessages(prev => {
          const newMessages = [...prev, assistantMessage]
          return newMessages.slice(-maxMessages)
        })

        // Update stats
        setRagStats(prev => ({
          totalQueries: prev.totalQueries + 1,
          avgResponseTime: (prev.avgResponseTime * prev.totalQueries + (assistantMessage.metadata?.response_time || 0)) / (prev.totalQueries + 1),
          totalDocuments: Math.max(prev.totalDocuments, response.retrieved_documents?.length || 0)
        }))

        // Auto-speak response if enabled
        if (autoSpeak && enableSpeech && response.response) {
          setTimeout(() => synthesizeText(response.response!), 500)
        }
      } else {
        const errorMessage = response.error || 'Failed to get response from RAG system'
        const errorResponse: Message = {
          id: `error-${Date.now()}`,
          type: 'assistant',
          content: `❌ Sorry, I encountered an error: ${errorMessage}`,
          timestamp: new Date(),
          error: errorMessage
        }
        setMessages(prev => [...prev, errorResponse])
        onError?.(errorMessage)
      }
    } catch (error) {
      const errorMessage = `Connection error: ${error instanceof Error ? error.message : 'Unknown error'}`
      const errorResponse: Message = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: `🔌 Sorry, I'm having trouble connecting to the AI service. Please check your connection and try again.`,
        timestamp: new Date(),
        error: errorMessage
      }
      setMessages(prev => [...prev, errorResponse])
      onError?.(errorMessage)
    } finally {
      setIsLoading(false)
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
    setMessages([{
      id: 'welcome-new',
      type: 'system',
      content: "💫 Chat cleared! Ready for new questions.",
      timestamp: new Date()
    }])
    setSelectedMessageId(null)
    setRagStats({ totalQueries: 0, avgResponseTime: 0, totalDocuments: 0 })
  }

  // Export conversation
  const exportConversation = () => {
    const exportData = {
      timestamp: new Date().toISOString(),
      messages: messages.filter(m => m.type !== 'system'),
      stats: ragStats
    }
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `rag-chat-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Format time
  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  // Get message icon
  const getMessageIcon = (message: Message): string => {
    switch (message.type) {
      case 'user': return '👤'
      case 'assistant': return message.error ? '❌' : '🤖'
      case 'system': return '🔮'
      default: return '💬'
    }
  }

  // Render chat messages
  const renderChatMessages = () => (
    <div className="flex-1 overflow-y-auto p-4 space-y-4" ref={chatContainerRef}>
      {messages.map((message) => (
        <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
          <div className={`max-w-[85%] ${
            message.type === 'user'
              ? 'bg-blue-500 text-white rounded-l-xl rounded-tr-xl'
              : message.type === 'system'
              ? 'bg-purple-100 text-purple-800 rounded-xl border border-purple-200'
              : message.error
              ? 'bg-red-50 text-red-800 rounded-r-xl rounded-tl-xl border border-red-200'
              : 'bg-gray-100 text-gray-800 rounded-r-xl rounded-tl-xl'
          } p-4 shadow-sm relative group`}>
            
            {/* Message header */}
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-sm">{getMessageIcon(message)}</span>
              <span className={`text-xs font-medium ${
                message.type === 'user' ? 'text-blue-100' : 
                message.type === 'system' ? 'text-purple-600' :
                'text-gray-600'
              }`}>
                {message.type === 'user' ? 'You' : 
                 message.type === 'system' ? 'System' : 'Assistant'}
              </span>
              <span className={`text-xs ${
                message.type === 'user' ? 'text-blue-200' : 
                message.type === 'system' ? 'text-purple-500' :
                'text-gray-500'
              }`}>
                {formatTime(message.timestamp)}
              </span>
            </div>

            {/* Message content */}
            <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {message.content}
            </div>

            {/* Message metadata and actions */}
            {message.type === 'assistant' && !message.error && (
              <div className="mt-3 pt-2 border-t border-gray-200 flex items-center justify-between text-xs text-gray-500">
                <div className="flex items-center space-x-3">
                  {message.metadata?.response_time && (
                    <span>⚡ {message.metadata.response_time.toFixed(1)}s</span>
                  )}
                  {message.retrievedDocs && (
                    <span>📚 {message.retrievedDocs.length} docs</span>
                  )}
                  {message.metadata?.tokens_generated && (
                    <span>🔤 {message.metadata.tokens_generated} tokens</span>
                  )}
                </div>

                <div className="flex items-center space-x-2">
                  {enableSpeech && (
                    <button
                      onClick={() => speakMessage(message.content)}
                      className="p-1 hover:bg-gray-200 rounded transition-all text-gray-400 hover:text-gray-600"
                      title="Read aloud"
                    >
                      {isPlaying ? '🔊' : '🔈'}
                    </button>
                  )}
                  
                  {message.retrievedDocs && message.retrievedDocs.length > 0 && (
                    <button
                      onClick={() => setSelectedMessageId(
                        selectedMessageId === message.id ? null : message.id
                      )}
                      className="p-1 hover:bg-gray-200 rounded transition-all text-gray-400 hover:text-gray-600"
                      title="Show source documents"
                    >
                      📋
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Document preview for selected message */}
            {message.type === 'assistant' && 
             message.retrievedDocs && 
             showDocuments && 
             selectedMessageId === message.id && (
              <div className="mt-4 pt-3 border-t border-gray-200">
                <DocumentViewer
                  documents={message.retrievedDocs}
                  query={message.query}
                  className="text-xs"
                  maxPreviewLength={200}
                />
              </div>
            )}
          </div>
        </div>
      ))}

      {/* Loading indicator */}
      {isLoading && (
        <div className="flex justify-start">
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-r-xl rounded-tl-xl p-4 shadow-sm border border-blue-200">
            <div className="flex items-center space-x-3">
              <div className="flex space-x-1">
                <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"></div>
                <div className="w-3 h-3 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
              <div className="text-sm">
                <div className="text-gray-700 font-medium">🧠 AI is thinking...</div>
                <div className="text-xs text-gray-500 mt-1">
                  Searching knowledge base • Loading AI model (first use may take 1-2 min)
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  )

  // Render stats panel
  const renderStatsPanel = () => (
    <div className="bg-gray-50 border-t border-gray-200 p-3">
      <div className="flex items-center justify-between text-xs text-gray-600">
        <div className="flex items-center space-x-4">
          <span>💬 {ragStats.totalQueries} queries</span>
          <span>⚡ {ragStats.avgResponseTime.toFixed(1)}s avg</span>
          <span>📚 {ragStats.totalDocuments} docs</span>
          {voices.length > 0 && selectedVoice && (
            <span>🔊 {selectedVoice.name}</span>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={exportConversation}
            disabled={messages.length <= 1}
            className="text-blue-600 hover:text-blue-800 disabled:text-gray-400 disabled:cursor-not-allowed"
            title="Export conversation"
          >
            📥
          </button>
          <button
            onClick={clearConversation}
            disabled={messages.length <= 1}
            className="text-red-600 hover:text-red-800 disabled:text-gray-400 disabled:cursor-not-allowed"
            title="Clear conversation"
          >
            🗑️
          </button>
        </div>
      </div>
    </div>
  )

  return (
    <div className={`flex flex-col h-full bg-white rounded-lg border border-gray-200 shadow-sm ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-purple-50">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white text-lg font-bold">
            🤖
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-800">RAG Assistant</h3>
            <p className="text-sm text-gray-600">
              Intelligent document search and Q&A
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 px-2 py-1 bg-green-100 rounded-full">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-xs text-green-700 font-medium">Online</span>
          </div>
        </div>
      </div>

      {/* Chat messages */}
      {renderChatMessages()}

      {/* Input area */}
      <div className="border-t border-gray-200">
        <QueryInput
          onSubmit={submitQuery}
          isLoading={isLoading}
          enableVoice={enableVoiceInput}
          className="border-0 rounded-none"
        />
      </div>

      {/* Stats panel */}
      {renderStatsPanel()}
    </div>
  )
} 