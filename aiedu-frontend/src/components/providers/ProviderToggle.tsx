'use client'

import React from 'react'
import { ProviderType } from '@/types/provider'
import { useProvider } from '@/hooks/useProvider'
import { Home, Cloud, Lock, Zap, Info } from 'lucide-react'

interface ProviderToggleProps {
  className?: string
  showTooltips?: boolean
}

export const ProviderToggle: React.FC<ProviderToggleProps> = ({ 
  className = '', 
  showTooltips = true 
}) => {
  const { currentProvider, switchProvider, isLoading } = useProvider()

  const handleProviderChange = (provider: ProviderType) => {
    if (!isLoading && provider !== currentProvider) {
      switchProvider(provider)
    }
  }

  return (
    <div className={`provider-toggle w-full ${className}`}>
      {/* Header with Info */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
          <span className="text-blue-600">🎛️</span>
          AI Provider Selection
        </h3>
        {showTooltips && (
          <div className="group relative">
            <Info size={16} className="text-gray-400 hover:text-blue-500 cursor-help" />
            <div className="invisible group-hover:visible absolute right-0 top-6 w-64 p-3 bg-gray-800 text-white text-xs rounded-lg shadow-lg z-10">
              <strong>Local:</strong> Privacy-focused, offline processing<br/>
              <strong>Cloud:</strong> High-performance, neural AI models
            </div>
          </div>
        )}
      </div>

      {/* Toggle Switch Container */}
      <div className="relative bg-gray-100 rounded-xl p-1 shadow-inner">
        {/* Background Slider */}
        <div 
          className={`
            absolute top-1 left-1 h-[calc(100%-8px)] w-[calc(50%-4px)] 
            rounded-lg shadow-sm transition-all duration-300 ease-in-out
            ${currentProvider === 'local' 
              ? 'transform translate-x-0 bg-blue-500' 
              : 'transform translate-x-full bg-emerald-500'
            }
          `}
        />

        {/* Toggle Buttons */}
        <div className="relative flex">
          {/* Local Provider Button */}
          <button
            onClick={() => handleProviderChange('local')}
            disabled={isLoading}
            className={`
              flex-1 relative z-10 flex items-center justify-center gap-2 px-4 py-3 
              rounded-lg font-medium text-sm transition-all duration-300
              ${currentProvider === 'local'
                ? 'text-white'
                : 'text-gray-600 hover:text-gray-800'
              }
              ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <Home size={18} />
            <div className="flex flex-col items-start">
              <span className="font-semibold">Local</span>
              <span className="text-xs opacity-90 flex items-center gap-1">
                <Lock size={10} />
                Privacy
              </span>
            </div>
          </button>

          {/* Cloud Provider Button */}
          <button
            onClick={() => handleProviderChange('cloud')}
            disabled={isLoading}
            className={`
              flex-1 relative z-10 flex items-center justify-center gap-2 px-4 py-3 
              rounded-lg font-medium text-sm transition-all duration-300
              ${currentProvider === 'cloud'
                ? 'text-white'
                : 'text-gray-600 hover:text-gray-800'
              }
              ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <Cloud size={18} />
            <div className="flex flex-col items-start">
              <span className="font-semibold">Cloud</span>
              <span className="text-xs opacity-90 flex items-center gap-1">
                <Zap size={10} />
                Performance
              </span>
            </div>
          </button>
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="mt-3 flex items-center justify-center">
          <div className="flex items-center gap-2 text-sm text-blue-600">
            <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
            <span>Switching providers...</span>
          </div>
        </div>
      )}

      {/* Provider Benefits Display */}
      <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
        {/* Local Benefits */}
        <div className={`
          p-3 rounded-lg border transition-all duration-300
          ${currentProvider === 'local' 
            ? 'bg-blue-50 border-blue-200 text-blue-800' 
            : 'bg-gray-50 border-gray-200 text-gray-600'
          }
        `}>
          <div className="font-semibold mb-1 flex items-center gap-1">
            <Lock size={12} />
            Local Processing
          </div>
          <ul className="space-y-1 text-xs">
            <li>• Complete privacy</li>
            <li>• Offline capability</li>
            <li>• No data transmission</li>
            <li>• Cost-effective</li>
          </ul>
        </div>

        {/* Cloud Benefits */}
        <div className={`
          p-3 rounded-lg border transition-all duration-300
          ${currentProvider === 'cloud' 
            ? 'bg-emerald-50 border-emerald-200 text-emerald-800' 
            : 'bg-gray-50 border-gray-200 text-gray-600'
          }
        `}>
          <div className="font-semibold mb-1 flex items-center gap-1">
            <Zap size={12} />
            Cloud Processing
          </div>
          <ul className="space-y-1 text-xs">
            <li>• Neural voice quality</li>
            <li>• Higher accuracy</li>
            <li>• Latest AI models</li>
            <li>• Faster responses</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default ProviderToggle 