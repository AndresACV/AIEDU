'use client'

import React from 'react'
import { ProviderType } from '@/types/provider'
import { useProvider } from '@/hooks/useProvider'
import { Home, Cloud } from 'lucide-react'

interface ProviderSelectorProps {
  className?: string
}

export const ProviderSelector: React.FC<ProviderSelectorProps> = ({ className = '' }) => {
  const { currentProvider, switchProvider, isLoading } = useProvider()

  const handleProviderChange = (provider: ProviderType) => {
    if (!isLoading) {
      switchProvider(provider)
    }
  }

  return (
    <div className={`provider-selector ${className}`}>
      <div className="flex w-full rounded-lg bg-gray-100 p-1">
        <button
          onClick={() => handleProviderChange('local')}
          disabled={isLoading}
          className={`
            flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors
            ${currentProvider === 'local'
              ? 'bg-blue-500 text-white shadow-sm'
              : 'text-gray-700 hover:bg-gray-200'
            }
            ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
        >
          <Home size={16} />
          Local
        </button>
        
        <button
          onClick={() => handleProviderChange('cloud')}
          disabled={isLoading}
          className={`
            flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors
            ${currentProvider === 'cloud'
              ? 'bg-green-500 text-white shadow-sm'
              : 'text-gray-700 hover:bg-gray-200'
            }
            ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
        >
          <Cloud size={16} />
          Cloud
        </button>
      </div>
      
      {isLoading && (
        <div className="mt-2 flex items-center justify-center">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-500 border-t-transparent"></div>
          <span className="ml-2 text-xs text-gray-600">Switching...</span>
        </div>
      )}
    </div>
  )
}

export default ProviderSelector 