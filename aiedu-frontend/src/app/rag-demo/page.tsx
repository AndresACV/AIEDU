'use client'

import React, { useState, useEffect } from 'react'
import { RagChatInterface } from '@/components/chat'
import { apiClient } from '@/services/api'

export default function RagDemoPage() {
  const [settings, setSettings] = useState({
    enableSpeech: true,
    enableVoiceInput: true,
    autoSpeak: false,
    showDocuments: true
  })
  const [errors, setErrors] = useState<Array<{ timestamp: Date; message: string }>>([])
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await apiClient.ragHealth()
        setIsConnected(health.available)
      } catch (error) {
        setIsConnected(false)
        addError(`Health check failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
      }
    }
    checkHealth()
  }, [])

  const addError = (message: string) => {
    setErrors(prev => [{ timestamp: new Date(), message }, ...prev.slice(0, 9)])
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            🤖 RAG Chat Interface Demo
          </h1>
          <p className="text-gray-600">
            Test the Retrieval-Augmented Generation chat system
          </p>
          <div className="mt-4 flex items-center space-x-4">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className={isConnected ? 'text-green-700' : 'text-red-700'}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 h-[700px]">
          <RagChatInterface
            onError={addError}
            enableSpeech={settings.enableSpeech}
            enableVoiceInput={settings.enableVoiceInput}
            autoSpeak={settings.autoSpeak}
            showDocuments={settings.showDocuments}
            className="h-full"
          />
        </div>

        {errors.length > 0 && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 className="text-red-800 font-medium mb-2">Recent Errors:</h3>
            {errors.slice(0, 3).map((error, index) => (
              <div key={index} className="text-sm text-red-700">
                [{error.timestamp.toLocaleTimeString()}] {error.message}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
} 