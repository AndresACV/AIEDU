'use client'

import React, { useState } from 'react'
import { useProvider } from '@/hooks/useProvider'
import ProviderToggle from './ProviderToggle'
import ProviderStatus from './ProviderStatus'
import { Settings, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'

interface EnhancedProviderSelectorProps {
  className?: string
  defaultExpanded?: boolean
  showRefreshButton?: boolean
}

export const EnhancedProviderSelector: React.FC<EnhancedProviderSelectorProps> = ({ 
  className = '',
  defaultExpanded = true,
  showRefreshButton = true
}) => {
  const { error, refreshStatus, isLoading } = useProvider()
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleRefresh = async () => {
    setIsRefreshing(true)
    try {
      await refreshStatus()
    } finally {
      // Add a small delay to show the refresh animation
      setTimeout(() => setIsRefreshing(false), 500)
    }
  }

  return (
    <div className={`enhanced-provider-selector bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden ${className}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-emerald-500 text-white px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings size={18} />
            <h3 className="font-semibold">AI Provider Control</h3>
          </div>
          <div className="flex items-center gap-2">
            {showRefreshButton && (
              <button
                onClick={handleRefresh}
                disabled={isRefreshing || isLoading}
                className="p-1 rounded hover:bg-white/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title="Refresh provider status"
              >
                <RefreshCw 
                  size={16} 
                  className={isRefreshing ? 'animate-spin' : ''} 
                />
              </button>
            )}
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1 rounded hover:bg-white/20 transition-colors"
              title={isExpanded ? 'Collapse details' : 'Expand details'}
            >
              {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-4 space-y-4">
        {/* Connection Error Notice */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center mb-2">
              <span className="text-red-600 text-sm font-medium">⚠️ Connection Issue</span>
            </div>
            <p className="text-red-600 text-sm mb-2">{error}</p>
            {(error.includes('CORS') || error.includes('not reachable')) && (
              <div className="text-xs text-red-500">
                <p className="font-medium mb-1">🔧 Troubleshooting:</p>
                <ol className="list-decimal list-inside space-y-1">
                  <li>Ensure the backend is running on <code className="bg-red-100 px-1 rounded">http://127.0.0.1:8000</code></li>
                  <li>Check if the virtual environment is activated</li>
                  <li>Verify all services are initialized properly</li>
                </ol>
              </div>
            )}
          </div>
        )}

        {/* Provider Toggle */}
        <div className="space-y-4">
          <ProviderToggle showTooltips={isExpanded} />
        </div>

        {/* Expanded Details */}
        {isExpanded && (
          <div className="space-y-4 border-t pt-4">
            <ProviderStatus />
          </div>
        )}

        {/* Compact Status when collapsed */}
        {!isExpanded && (
          <div className="border-t pt-3">
            <ProviderStatus compact />
          </div>
        )}

        {/* Quick Info */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <span className="text-blue-600 text-lg">💡</span>
            <div className="text-sm text-blue-800">
              <p className="font-medium mb-1">Provider Selection Tips:</p>
              <ul className="space-y-1 text-xs">
                <li>• <strong>Local:</strong> Best for privacy, works offline, no costs</li>
                <li>• <strong>Cloud:</strong> Best for quality, requires internet, usage-based</li>
                <li>• Switch anytime based on your current needs</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Status Bar */}
      <div className="bg-gray-50 px-4 py-2 border-t">
        <div className="flex items-center justify-between text-xs text-gray-600">
          <span>Status updates every 5 seconds</span>
          <span className="text-blue-600">
            🚀 AIEDU Hybrid AI System
          </span>
        </div>
      </div>
    </div>
  )
}

export default EnhancedProviderSelector 