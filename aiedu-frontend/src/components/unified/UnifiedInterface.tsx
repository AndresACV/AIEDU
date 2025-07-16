'use client'

import React, { useState, useEffect, useCallback, useRef } from 'react'
import { RagChatInterface } from '@/components/chat'
import { SpeechControls, useSpeech } from '@/components/speech'
import { KnowledgeManager } from '@/components/unified'
import { apiClient } from '@/services/api'

interface SystemHealth {
  backend: boolean
  speech: boolean
  rag: boolean
  documents: number
  voices: number
}

interface UnifiedSettings {
  mode: 'chat' | 'speech' | 'unified' | 'knowledge'
  enableSpeech: boolean
  enableVoiceInput: boolean
  autoSpeak: boolean
  autoSendOnVoice: boolean
  showDocuments: boolean
  language: 'en-US' | 'es-ES'
}

interface ActivityLog {
  id: string
  type: 'speech' | 'rag' | 'system'
  action: string
  timestamp: Date
  details?: string
  success: boolean
}

export default function UnifiedInterface() {
  // Core state
  const [settings, setSettings] = useState<UnifiedSettings>({
    mode: 'unified',
    enableSpeech: true,
    enableVoiceInput: true,
    autoSpeak: false,
    autoSendOnVoice: false,
    showDocuments: true,
    language: 'en-US'
  })

  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    backend: false,
    speech: false,
    rag: false,
    documents: 0,
    voices: 0
  })

  const [activityLog, setActivityLog] = useState<ActivityLog[]>([])
  const [showSettings, setShowSettings] = useState(false)
  const [showSystemPanel, setShowSystemPanel] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  // Speech integration
  const speech = useSpeech({
    defaultLanguage: settings.language
  })

  // Activity logging - removed circular dependency
  const addActivity = useCallback((type: ActivityLog['type'], action: string, details?: string, success: boolean = true) => {
    const activity: ActivityLog = {
      id: `${type}-${Date.now()}`,
      type,
      action,
      timestamp: new Date(),
      details,
      success
    }
    setActivityLog(prev => [activity, ...prev.slice(0, 49)]) // Keep last 50 activities
  }, []) // Empty dependency array

  // System health monitoring - removed circular dependency  
  const checkSystemHealth = useCallback(async () => {
    try {
      // First check basic health - if this fails, don't spam with other calls
      const healthResponse = await Promise.allSettled([apiClient.health()])
      
      if (healthResponse[0].status === 'fulfilled') {
        // Backend is up, check other services
        const [voicesResponse, ragHealthResponse] = await Promise.allSettled([
          apiClient.getVoices(),
          apiClient.getRagHealth()
        ])

        setSystemHealth({
          backend: true,
          speech: voicesResponse.status === 'fulfilled',
          rag: ragHealthResponse.status === 'fulfilled',
          documents: ragHealthResponse.status === 'fulfilled' ? 
            (ragHealthResponse.value as any).total_documents || 0 : 0,
          voices: voicesResponse.status === 'fulfilled' ?
            (voicesResponse.value as any[]).length || 0 : 0
        })
      } else {
        // Backend is down, don't make additional calls
        setSystemHealth({
          backend: false,
          speech: false,
          rag: false,
          documents: 0,
          voices: 0
        })
      }
    } catch (error) {
      // Only log error occasionally to avoid spam
      if (Math.random() < 0.1) { // Log only 10% of errors
        console.warn('Health check failed - backend may be starting up:', error)
      }
      setSystemHealth({
        backend: false,
        speech: false,
        rag: false,
        documents: 0,
        voices: 0
      })
    }
  }, []) // Empty dependency array

  // Initialize system - removed circular dependencies
  useEffect(() => {
    const initialize = async () => {
      setIsLoading(true)
      await checkSystemHealth()
      addActivity('system', 'AIEDU system initialized', 'All services checked', true)
      setIsLoading(false)
    }

    initialize()
    
    // Periodic health checks - reduced frequency to avoid spam when backend is down
    const interval = setInterval(checkSystemHealth, 120000) // Every 2 minutes (was 30 seconds)
    return () => clearInterval(interval)
  }, []) // Only run on mount

  // Handle auto-send toggle
  const handleAutoSendToggle = useCallback((enabled: boolean) => {
    setSettings(prev => ({ ...prev, autoSendOnVoice: enabled }))
    addActivity('system', `Auto-send voice ${enabled ? 'enabled' : 'disabled'}`, undefined, true)
  }, [addActivity])

  // Update setting helper
  const updateSetting = useCallback(<K extends keyof UnifiedSettings>(
    key: K,
    value: UnifiedSettings[K]
  ) => {
    setSettings(prev => ({ ...prev, [key]: value }))
    addActivity('system', `Updated ${key} to ${value}`, undefined, true)
  }, [addActivity])

  // Clear activity log
  const clearActivityLog = () => {
    setActivityLog([])
    addActivity('system', 'Activity log cleared', undefined, true)
  }

  // Export system data
  const exportSystemData = () => {
    const exportData = {
      timestamp: new Date().toISOString(),
      settings,
      systemHealth,
      activityLog: activityLog.slice(0, 20), // Last 20 activities
      speechStats: {
        hasRecording: speech.hasRecording,
        hasAudio: speech.hasAudio,
        selectedVoice: speech.selectedVoice?.name,
        voicesCount: speech.voices.length
      }
    }
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `aiedu-system-export-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
    
    addActivity('system', 'System data exported', `${activityLog.length} activities included`, true)
  }

  // Handle errors
  const handleError = (error: string) => {
    addActivity('system', 'Error occurred', error, false)
  }

  // Quick actions
  const quickActions = [
    {
      id: 'voice-test',
      name: 'Test Voice',
      icon: '🎤',
      action: () => {
        speech.synthesizeText('Hello! AIEDU speech system is working perfectly.')
        addActivity('speech', 'Voice test initiated', 'Testing speech synthesis', true)
      }
    },
    {
      id: 'health-check',
      name: 'Health Check',
      icon: '🔍',
      action: () => {
        checkSystemHealth()
        addActivity('system', 'Manual health check', 'All services verified', true)
      }
    },
    {
      id: 'clear-all',
      name: 'Clear All',
      icon: '🧹',
      action: () => {
        speech.clearAll?.()
        clearActivityLog()
        addActivity('system', 'System cleared', 'All temporary data cleared', true)
      }
    }
  ]

  // Render system status indicator
  const renderSystemStatus = () => {
    const allHealthy = systemHealth.backend && systemHealth.speech && systemHealth.rag
    const issues = Object.entries(systemHealth).filter(([key, value]) => 
      key !== 'documents' && key !== 'voices' && !value
    ).length

    return (
      <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
        allHealthy 
          ? 'bg-green-100 text-green-800' 
          : issues > 1 
          ? 'bg-red-100 text-red-800'
          : 'bg-yellow-100 text-yellow-800'
      }`}>
        <div className={`w-2 h-2 rounded-full ${
          allHealthy ? 'bg-green-500 animate-pulse' : 
          issues > 1 ? 'bg-red-500' : 'bg-yellow-500'
        }`} />
        <span className="font-medium">
          {allHealthy ? 'All Systems Operational' : 
           issues > 1 ? `${issues} Systems Down` : 'System Issues Detected'}
        </span>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Starting AIEDU System</h2>
          <p className="text-gray-600">Initializing AI services...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {/* Logo and title */}
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white text-xl font-bold">
                🎓
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">AIEDU</h1>
                <p className="text-sm text-gray-600">Intelligent Learning Assistant</p>
              </div>
            </div>

            {/* Mode selector */}
            <div className="flex items-center space-x-4">
              <div className="flex bg-gray-100 rounded-lg p-1">
                {[
                  { key: 'unified' as const, name: 'Unified', icon: '🎯' },
                  { key: 'chat' as const, name: 'Chat', icon: '💬' },
                  { key: 'speech' as const, name: 'Speech', icon: '🎤' },
                  { key: 'knowledge' as const, name: 'Knowledge', icon: '📚' }
                ].map(mode => (
                  <button
                    key={mode.key}
                    onClick={() => updateSetting('mode', mode.key)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      settings.mode === mode.key
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-gray-600 hover:text-gray-800'
                    }`}
                  >
                    <span className="mr-1">{mode.icon}</span>
                    {mode.name}
                  </button>
                ))}
              </div>

              {/* System status */}
              {renderSystemStatus()}

              {/* Controls */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-all"
                  title="Settings"
                >
                  ⚙️
                </button>
                <button
                  onClick={() => setShowSystemPanel(!showSystemPanel)}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-all"
                  title="System Monitor"
                >
                  📊
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-7xl mx-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Settings Panel */}
          {showSettings && (
            <div className="lg:col-span-1">
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 sticky top-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">⚙️ Settings</h3>
                
                <div className="space-y-4">
                  {/* Interface Settings */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Interface</h4>
                    <div className="space-y-2">
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={settings.enableSpeech}
                          onChange={(e) => updateSetting('enableSpeech', e.target.checked)}
                          className="mr-2"
                        />
                        <span className="text-sm">Enable speech synthesis</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={settings.enableVoiceInput}
                          onChange={(e) => updateSetting('enableVoiceInput', e.target.checked)}
                          className="mr-2"
                        />
                        <span className="text-sm">Enable voice input</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={settings.autoSpeak}
                          onChange={(e) => updateSetting('autoSpeak', e.target.checked)}
                          className="mr-2"
                        />
                        <span className="text-sm">Auto-speak responses</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={settings.autoSendOnVoice}
                          onChange={(e) => updateSetting('autoSendOnVoice', e.target.checked)}
                          className="mr-2"
                        />
                        <span className="text-sm">Auto-send on voice input</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={settings.showDocuments}
                          onChange={(e) => updateSetting('showDocuments', e.target.checked)}
                          className="mr-2"
                        />
                        <span className="text-sm">Show retrieved documents</span>
                      </label>
                    </div>
                  </div>

                  {/* Language Settings */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Language</h4>
                    <select
                      value={settings.language}
                      onChange={(e) => updateSetting('language', e.target.value as any)}
                      className="w-full text-sm border border-gray-300 rounded px-3 py-2"
                    >
                      <option value="en-US">🇺🇸 English (US)</option>
                      <option value="es-ES">🇪🇸 Español (España)</option>
                    </select>
                  </div>

                  {/* Quick Actions */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Quick Actions</h4>
                    <div className="space-y-2">
                      {quickActions.map((action) => (
                        <button
                          key={action.id}
                          onClick={action.action}
                          className="w-full text-left px-3 py-2 text-sm bg-gray-50 hover:bg-gray-100 rounded-lg transition-all flex items-center space-x-2"
                        >
                          <span>{action.icon}</span>
                          <span>{action.name}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* System Actions */}
                  <div className="pt-4 border-t border-gray-200">
                    <button
                      onClick={exportSystemData}
                      className="w-full px-3 py-2 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg text-sm transition-all"
                    >
                      📥 Export System Data
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Main Interface */}
          <div className={`${showSettings ? 'lg:col-span-3' : 'lg:col-span-4'}`}>
            <div className="space-y-6">
              {/* Unified Mode */}
              {settings.mode === 'unified' && (
                <>
                  {/* Speech Controls - Top Section */}
                  <div className="bg-white rounded-xl shadow-sm border border-gray-200">
                    <div className="p-4 border-b border-gray-100">
                      <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                        <span className="mr-2">🎤</span>
                        Speech Interface
                      </h3>
                    </div>
                    <div className="p-4">
                      <SpeechControls
                        layout="horizontal"
                        language={settings.language}
                        onError={handleError}
                      />
                    </div>
                  </div>

                  {/* RAG Chat - Main Section */}
                  <div className="bg-white rounded-xl shadow-sm border border-gray-200 h-[600px]">
                    <RagChatInterface
                      onError={handleError}
                      enableSpeech={settings.enableSpeech}
                      enableVoiceInput={settings.enableVoiceInput}
                      autoSpeak={settings.autoSpeak}
                      showDocuments={settings.showDocuments}
                      autoSendOnVoice={settings.autoSendOnVoice}
                      initialLanguage={settings.language}
                      onAutoSendToggle={handleAutoSendToggle}
                      className="h-full"
                    />
                  </div>
                </>
              )}

              {/* Chat Only Mode */}
              {settings.mode === 'chat' && (
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 h-[700px]">
                  <RagChatInterface
                    onError={handleError}
                    enableSpeech={settings.enableSpeech}
                    enableVoiceInput={settings.enableVoiceInput}
                    autoSpeak={settings.autoSpeak}
                    showDocuments={settings.showDocuments}
                    autoSendOnVoice={settings.autoSendOnVoice}
                    initialLanguage={settings.language}
                    onAutoSendToggle={handleAutoSendToggle}
                    className="h-full"
                  />
                </div>
              )}

              {/* Speech Only Mode */}
                          {settings.mode === 'speech' && (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <SpeechControls
                  layout="vertical"
                  language={settings.language}
                  onError={handleError}
                />
              </div>
            )}

            {settings.mode === 'knowledge' && (
              <KnowledgeManager
                onDocumentUpdate={() => {
                  // Refresh system health when documents are updated
                  checkSystemHealth()
                }}
                onError={handleError}
              />
            )}
            </div>
          </div>
        </div>

        {/* System Monitor Panel */}
        {showSystemPanel && (
          <div className="mt-6 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">📊 System Monitor</h3>
              <button
                onClick={clearActivityLog}
                disabled={activityLog.length === 0}
                className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded transition-all disabled:opacity-50"
              >
                Clear Log ({activityLog.length})
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              {/* System Health */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <h4 className="font-medium text-gray-700 mb-2">System Health</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>Backend:</span>
                    <span className={systemHealth.backend ? 'text-green-600' : 'text-red-600'}>
                      {systemHealth.backend ? '✅ Online' : '❌ Offline'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Speech:</span>
                    <span className={systemHealth.speech ? 'text-green-600' : 'text-red-600'}>
                      {systemHealth.speech ? '✅ Ready' : '❌ Error'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>RAG:</span>
                    <span className={systemHealth.rag ? 'text-green-600' : 'text-red-600'}>
                      {systemHealth.rag ? '✅ Active' : '❌ Down'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Resource Stats */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <h4 className="font-medium text-gray-700 mb-2">Resources</h4>
                <div className="space-y-1 text-sm text-gray-600">
                  <div className="flex justify-between">
                    <span>Documents:</span>
                    <span className="font-mono">{systemHealth.documents}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Voices:</span>
                    <span className="font-mono">{systemHealth.voices}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Activities:</span>
                    <span className="font-mono">{activityLog.length}</span>
                  </div>
                </div>
              </div>

              {/* Current Session */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <h4 className="font-medium text-gray-700 mb-2">Session</h4>
                <div className="space-y-1 text-sm text-gray-600">
                  <div className="flex justify-between">
                    <span>Mode:</span>
                    <span className="capitalize">{settings.mode}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Language:</span>
                    <span>{settings.language}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Voice Status:</span>
                    <span>{speech.isRecording ? '🔴 Recording' : speech.hasAudio ? '🎵 Audio Ready' : '⚪ Idle'}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Activity Log */}
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Activity Log</h4>
              <div className="bg-gray-50 rounded-lg p-4 max-h-64 overflow-y-auto">
                {activityLog.length === 0 ? (
                  <p className="text-sm text-gray-500">No activities logged</p>
                ) : (
                  <div className="space-y-2">
                    {activityLog.slice(0, 10).map((activity) => (
                      <div key={activity.id} className="text-sm flex items-start space-x-2">
                        <span className="text-gray-500 font-mono text-xs mt-0.5 flex-shrink-0">
                          [{activity.timestamp.toLocaleTimeString()}]
                        </span>
                        <span className={`font-medium flex-shrink-0 ${
                          activity.type === 'speech' ? 'text-blue-600' :
                          activity.type === 'rag' ? 'text-green-600' :
                          'text-purple-600'
                        }`}>
                          {activity.type.toUpperCase()}:
                        </span>
                        <span className="text-gray-700">{activity.action}</span>
                        {!activity.success && <span className="text-red-500 text-xs">❌</span>}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 