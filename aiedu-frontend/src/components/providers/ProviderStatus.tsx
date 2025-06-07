'use client'

import React from 'react'
import { ServiceStatus } from '@/types/provider'
import { useProvider } from '@/hooks/useProvider'
import { Mic, Volume2, Brain, CheckCircle, AlertCircle, Clock, XCircle } from 'lucide-react'

interface ServiceInfo {
  icon: React.ReactNode
  label: string
  service: 'stt' | 'tts' | 'llm'
  description: string
}

const serviceConfigs: ServiceInfo[] = [
  {
    icon: <Mic size={16} />,
    label: 'Speech-to-Text',
    service: 'stt',
    description: 'Audio transcription service'
  },
  {
    icon: <Volume2 size={16} />,
    label: 'Text-to-Speech',
    service: 'tts',
    description: 'Voice synthesis service'
  },
  {
    icon: <Brain size={16} />,
    label: 'Language Model',
    service: 'llm',
    description: 'AI reasoning and generation'
  }
]

interface ProviderStatusProps {
  className?: string
  compact?: boolean
}

export const ProviderStatus: React.FC<ProviderStatusProps> = ({ 
  className = '', 
  compact = false 
}) => {
  const { status, providers, currentProvider, getStatusBadgeClass } = useProvider()

  const getStatusIcon = (serviceStatus: ServiceStatus) => {
    switch (serviceStatus) {
      case 'working':
        return <CheckCircle size={16} className="text-green-500" />
      case 'connecting':
        return <Clock size={16} className="text-yellow-500" />
      case 'error':
        return <XCircle size={16} className="text-red-500" />
      default:
        return <AlertCircle size={16} className="text-gray-500" />
    }
  }

  const getProviderDetails = (service: 'stt' | 'tts' | 'llm') => {
    if (!providers) return { name: 'Unknown', type: 'unknown' }
    
    let providerName = ''
    switch (service) {
      case 'stt':
        providerName = providers.stt_provider
        break
      case 'tts':
        providerName = providers.tts_provider
        break
      case 'llm':
        providerName = providers.llm_provider
        break
    }

    // Determine provider type and clean name
    const isLocal = providerName.toLowerCase().includes('local') || 
                   providerName.toLowerCase().includes('vosk') || 
                   providerName.toLowerCase().includes('espeak') || 
                   providerName.toLowerCase().includes('ollama')
    
    const isCloud = providerName.toLowerCase().includes('cloud') || 
                   providerName.toLowerCase().includes('google') || 
                   providerName.toLowerCase().includes('gemini')

    let type = 'unknown'
    let cleanName = providerName

    if (isLocal) {
      type = 'local'
      // Clean up local provider names
      if (providerName.includes('Vosk')) cleanName = 'Vosk'
      else if (providerName.includes('espeak')) cleanName = 'espeak'
      else if (providerName.includes('Ollama')) cleanName = 'Ollama'
    } else if (isCloud) {
      type = 'cloud'
      // Clean up cloud provider names
      if (providerName.includes('Google Cloud')) {
        if (service === 'stt') cleanName = 'Google Cloud STT'
        else if (service === 'tts') cleanName = 'Google Cloud TTS'
      } else if (providerName.includes('Gemini')) {
        cleanName = 'Gemini 2.5 Flash'
      }
    }

    return { name: cleanName, type }
  }

  const getQualityInfo = (service: 'stt' | 'tts' | 'llm', providerType: string) => {
    if (providerType === 'local') {
      switch (service) {
        case 'stt': return { quality: 'Good', latency: '~2s', note: 'Offline capable' }
        case 'tts': return { quality: 'Synthetic', latency: '~0.1s', note: 'Fast generation' }
        case 'llm': return { quality: 'Good', latency: '~3s', note: '7B parameter model' }
      }
    } else if (providerType === 'cloud') {
      switch (service) {
        case 'stt': return { quality: 'Excellent', latency: '~1s', note: 'Neural processing' }
        case 'tts': return { quality: 'Neural', latency: '~0.6s', note: 'Human-like voice' }
        case 'llm': return { quality: 'Superior', latency: '~1.5s', note: 'Latest AI model' }
      }
    }
    return { quality: 'Unknown', latency: 'Unknown', note: '' }
  }

  if (compact) {
    return (
      <div className={`provider-status-compact ${className}`}>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Services:</span>
          <div className="flex items-center gap-2">
            {serviceConfigs.map(({ service, icon }) => {
              const serviceStatus = status[service]
              return (
                <div key={service} className="flex items-center gap-1">
                  {icon}
                  <div className={`w-2 h-2 rounded-full ${getStatusBadgeClass(serviceStatus)}`} />
                </div>
              )
            })}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`provider-status ${className}`}>
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h4 className="font-semibold text-gray-800 flex items-center gap-2">
            <span className="text-green-600">🟢</span>
            Service Status
          </h4>
          <div className={`
            px-2 py-1 rounded-md text-xs font-medium
            ${currentProvider === 'local' 
              ? 'bg-blue-100 text-blue-800' 
              : 'bg-emerald-100 text-emerald-800'
            }
          `}>
            {currentProvider === 'local' ? '🏠 Local Mode' : '☁️ Cloud Mode'}
          </div>
        </div>

        {/* Service List */}
        <div className="space-y-3">
          {serviceConfigs.map(({ service, icon, label, description }) => {
            const serviceStatus = status[service]
            const providerDetails = getProviderDetails(service)
            const qualityInfo = getQualityInfo(service, providerDetails.type)
            
            return (
              <div 
                key={service}
                className="bg-white border rounded-lg p-3 hover:shadow-sm transition-shadow"
              >
                {/* Service Header */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {icon}
                    <span className="font-medium text-gray-800">{label}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(serviceStatus)}
                    <span className="text-xs text-gray-500 capitalize">{serviceStatus}</span>
                  </div>
                </div>

                {/* Provider Info */}
                <div className="text-sm text-gray-600 mb-2">
                  <div className="font-medium">{providerDetails.name}</div>
                  <div className="text-xs text-gray-500">{description}</div>
                </div>

                {/* Quality Metrics */}
                {serviceStatus === 'working' && (
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="text-center p-1 bg-gray-50 rounded">
                      <div className="font-medium text-gray-800">{qualityInfo.quality}</div>
                      <div className="text-gray-500">Quality</div>
                    </div>
                    <div className="text-center p-1 bg-gray-50 rounded">
                      <div className="font-medium text-gray-800">{qualityInfo.latency}</div>
                      <div className="text-gray-500">Latency</div>
                    </div>
                    <div className="text-center p-1 bg-gray-50 rounded">
                      <div className="font-medium text-gray-800 text-[10px]">{qualityInfo.note}</div>
                      <div className="text-gray-500">Notes</div>
                    </div>
                  </div>
                )}

                {/* Error State */}
                {serviceStatus === 'error' && (
                  <div className="text-xs text-red-600 bg-red-50 p-2 rounded">
                    Service unavailable. Check provider configuration.
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Overall Status Summary */}
        <div className={`
          p-3 rounded-lg text-sm
          ${Object.values(status).every(s => s === 'working')
            ? 'bg-green-50 border border-green-200 text-green-800'
            : Object.values(status).some(s => s === 'error')
            ? 'bg-red-50 border border-red-200 text-red-800'
            : 'bg-yellow-50 border border-yellow-200 text-yellow-800'
          }
        `}>
          <div className="font-medium mb-1">
            {Object.values(status).every(s => s === 'working')
              ? '✅ All services operational'
              : Object.values(status).some(s => s === 'error')
              ? '⚠️ Some services have issues'
              : '🔄 Services connecting...'
            }
          </div>
          <div className="text-xs">
            Ready for {currentProvider === 'local' ? 'privacy-focused' : 'high-performance'} AI processing
          </div>
        </div>
      </div>
    </div>
  )
}

export default ProviderStatus 