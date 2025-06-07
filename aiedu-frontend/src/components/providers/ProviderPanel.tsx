'use client'

import React from 'react'
import { useProvider } from '@/hooks/useProvider'
import ProviderSelector from './ProviderSelector'
import StatusIndicator from './StatusIndicator'
import { Settings } from 'lucide-react'

interface ProviderPanelProps {
  className?: string
}

export const ProviderPanel: React.FC<ProviderPanelProps> = ({ className = '' }) => {
  const { 
    isLoading, 
    error
  } = useProvider()

  return (
    <div className={`bg-white rounded-lg border border-gray-200 shadow-sm ${className}`}>
      {/* Header */}
      <div className="flex items-center gap-2 bg-green-500 text-white px-4 py-3 rounded-t-lg">
        <Settings size={16} />
        <h6 className="text-sm font-semibold">🎛️ Proveedores</h6>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Connection Error Notice */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center">
              <span className="text-red-600 text-sm font-medium">⚠️ Connection Issue</span>
            </div>
            <p className="text-red-600 text-sm mt-1">{error}</p>
            {error.includes('CORS') || error.includes('not reachable') ? (
              <div className="mt-2 text-xs text-red-500">
                <p>🔧 <strong>Troubleshooting:</strong></p>
                <ol className="list-decimal list-inside mt-1 space-y-1">
                  <li>Visit <a href="https://127.0.0.1:5000/current-providers" target="_blank" rel="noopener noreferrer" className="underline text-blue-600">https://127.0.0.1:5000/current-providers</a></li>
                  <li>Accept the SSL certificate when prompted</li>
                  <li>Return here and refresh the page</li>
                </ol>
              </div>
            ) : null}
          </div>
        )}
        
        {/* Provider Selector */}
        <div className="mb-6">
          <ProviderSelector />
        </div>

        {/* Status Section */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Service Status</h3>
          
          <StatusIndicator service="stt" label="Speech-to-Text" />
          <StatusIndicator service="tts" label="Text-to-Speech" />
          <StatusIndicator service="llm" label="Language Model" />
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="mt-4 text-center">
            <div className="inline-flex items-center text-sm text-blue-600">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
              Switching providers...
            </div>
          </div>
        )}

        {/* Info Section */}
        <div className="mt-6 p-3 bg-blue-50 rounded-lg">
          <p className="text-xs text-blue-700">
            <strong>💡 Tip:</strong> Switch between Local (privacy-focused) and Cloud (performance-optimized) providers. 
            Status refreshes every 5 seconds.
          </p>
        </div>
      </div>
    </div>
  )
}

export default ProviderPanel 