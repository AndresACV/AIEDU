import { useState, useEffect, useCallback } from 'react'
import { ProviderType, ProviderStatus, ProviderInfo, ServiceStatus } from '@/types/provider'
import apiClient from '@/services/api'

const PROVIDER_STATUS_REFRESH_INTERVAL = 5000 // 5 seconds

// Helper function to determine service status from provider name
const getServiceStatus = (providerName: string): ServiceStatus => {
  if (!providerName) return 'error'
  
  const name = providerName.toLowerCase()
  
  // Error states
  if (name.includes('error') || name.includes('failed') || name.includes('not installed') ||
      name.includes('not available') || name.includes('missing') || name.includes('none') ||
      name.includes('unknown') || name.includes('mock')) {
    return 'error'
  }
  
  // Connecting states
  if (name.includes('connecting') || name.includes('initializing') ||
      name.includes('not responding') || name.includes('no models')) {
    return 'connecting'
  }
  
  // Working states
  if (name.includes('local') || name.includes('vosk') || name.includes('espeak') ||
      name.includes('ollama') || name.includes('gemini') || name.includes('cloud') ||
      name.includes('pyttsx3')) {
    return 'working'
  }
  
  return 'unknown'
}

export const useProvider = () => {
  const [currentProvider, setCurrentProvider] = useState<ProviderType>('local')
  const [status, setStatus] = useState<ProviderStatus>({
    stt: 'unknown',
    tts: 'unknown',
    llm: 'unknown'
  })
  const [providers, setProviders] = useState<ProviderInfo | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load current provider status
  const refreshStatus = useCallback(async () => {
    try {
      const response = await apiClient.getCurrentProviders()
      
      // FastAPI response structure
      if (response) {
        setProviders({
          stt_provider: response.stt_provider,
          tts_provider: response.tts_provider,
          llm_provider: response.llm_provider
        })
        
        // Update status based on provider names
        setStatus({
          stt: getServiceStatus(response.stt_provider),
          tts: getServiceStatus(response.tts_provider),
          llm: getServiceStatus(response.llm_provider)
        })
        
        // Determine current provider type from provider names
        const isCloud = response.stt_provider.toLowerCase().includes('cloud') ||
                        response.stt_provider.toLowerCase().includes('google') ||
                        response.llm_provider.toLowerCase().includes('gemini')
        setCurrentProvider(isCloud ? 'cloud' : 'local')
        
        setError(null)
      } else {
        setError('Failed to get provider status')
      }
    } catch (err: any) {
      // Handle different types of connection errors
      if (err.code === 'ERR_NETWORK' || err.message === 'Network Error') {
        setError('Backend server not reachable')
        console.warn('🔌 Backend connection issue - using fallback status')
      } else if (err.response?.status === 0) {
        setError('CORS/SSL configuration issue')
        console.warn('⚠️ CORS/SSL issue - check backend HTTPS configuration')
      } else {
        setError('Connection error')
        console.warn('⚠️ Provider status refresh failed:', err.message)
      }
      
      // Set error status for all services to indicate connection issue
      setStatus({
        stt: 'error',
        tts: 'error',
        llm: 'error'
      })
      setProviders({
        stt_provider: 'Connection Error',
        tts_provider: 'Connection Error',
        llm_provider: 'Connection Error'
      })
    }
  }, [])

  // Switch provider with loading states
  const switchProvider = useCallback(async (provider: ProviderType) => {
    if (provider === currentProvider) return

    setIsLoading(true)
    setError(null)
    
    // Show connecting state immediately
    setStatus({
      stt: 'connecting',
      tts: 'connecting',
      llm: 'connecting'
    })

    try {
      const response = await apiClient.forceProvider(provider)
      
      // FastAPI response structure
      if (response && response.success) {
        setCurrentProvider(provider)
        
        // Wait a moment then refresh status
        setTimeout(() => {
          refreshStatus()
        }, 1000)
        
        console.log(`✅ Successfully switched to ${provider} provider`)
      } else {
        setError('Failed to switch provider')
        console.error('Failed to switch provider:', response)
        
        // Revert to previous status
        await refreshStatus()
      }
    } catch (err) {
      console.error('Error switching provider:', err)
      setError('Connection error during provider switch')
      
      // Revert to previous status
      await refreshStatus()
    } finally {
      setIsLoading(false)
    }
  }, [currentProvider, refreshStatus])

  // Get status badge class for UI display
  const getStatusBadgeClass = useCallback((serviceStatus: ServiceStatus): string => {
    switch (serviceStatus) {
      case 'working':
        return 'bg-green-500'
      case 'connecting':
        return 'bg-yellow-500'
      case 'error':
        return 'bg-red-500'
      default:
        return 'bg-gray-500'
    }
  }, [])

  // Get status text for tooltips
  const getStatusText = useCallback((serviceStatus: ServiceStatus, providerName?: string): string => {
    switch (serviceStatus) {
      case 'working':
        return providerName || 'Service working'
      case 'connecting':
        return 'Connecting...'
      case 'error':
        return 'Service error'
      default:
        return 'Status unknown'
    }
  }, [])

  // Initial load and periodic refresh with delay for backend startup
  useEffect(() => {
    // Wait 2 seconds before first API call to allow backend to start
    const initialTimeout = setTimeout(() => {
      refreshStatus()
    }, 2000)
    
    // Then start periodic refresh
    const interval = setInterval(refreshStatus, PROVIDER_STATUS_REFRESH_INTERVAL)
    
    return () => {
      clearTimeout(initialTimeout)
      clearInterval(interval)
    }
  }, [refreshStatus])

  return {
    currentProvider,
    status,
    providers,
    isLoading,
    error,
    switchProvider,
    refreshStatus,
    getStatusBadgeClass,
    getStatusText
  }
} 