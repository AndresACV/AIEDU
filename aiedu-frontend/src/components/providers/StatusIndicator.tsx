'use client'

import React from 'react'
import { ServiceStatus } from '@/types/provider'
import { useProvider } from '@/hooks/useProvider'

interface StatusIndicatorProps {
  service: 'stt' | 'tts' | 'llm'
  label: string
  className?: string
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ 
  service, 
  label, 
  className = '' 
}) => {
  const { status, providers, getStatusBadgeClass, getStatusText } = useProvider()

  const serviceStatus = status[service]
  const providerName = providers ? 
    (service === 'stt' ? providers.stt_provider :
     service === 'tts' ? providers.tts_provider :
     providers.llm_provider) : undefined

  const badgeClass = getStatusBadgeClass(serviceStatus)
  const statusText = getStatusText(serviceStatus, providerName)

  return (
    <div className={`flex items-center justify-between ${className}`}>
      <span className="text-sm text-gray-700">{label}:</span>
      <div 
        className={`h-3 w-3 rounded-full ${badgeClass} transition-colors duration-200`}
        title={statusText}
      />
    </div>
  )
}

export default StatusIndicator 