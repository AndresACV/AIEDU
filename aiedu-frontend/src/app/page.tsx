import React from 'react'
import ProviderPanel from '@/components/providers/ProviderPanel'

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="container mx-auto max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-blue-600 mb-2">
            Sistema RAG con Texto a Voz
          </h1>
          <p className="text-gray-600">
            Next.js Frontend - Phase 3 Migration in Progress
          </p>
        </div>

        {/* Main Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Main Content Area */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6">
              <h2 className="text-xl font-semibold mb-4">Sistema RAG con Interacción por Voz</h2>
              
              {/* Placeholder for chat interface */}
              <div className="bg-gray-50 rounded-lg p-8 text-center mb-6">
                <p className="text-gray-500 mb-2">🚧 Chat Interface Coming Soon</p>
                <p className="text-sm text-gray-400">
                  The conversation interface will be migrated from the HTML template in the next steps
                </p>
              </div>

              {/* Placeholder for controls */}
              <div className="bg-gray-50 rounded-lg p-6 text-center">
                <p className="text-gray-500 mb-2">🎤 Voice Controls Coming Soon</p>
                <p className="text-sm text-gray-400">
                  Speech recording and RAG query functionality will be implemented next
                </p>
              </div>
            </div>
          </div>

          {/* Provider Panel Sidebar */}
          <div className="lg:col-span-1">
            <ProviderPanel />
            
            {/* Development Info */}
            <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-blue-800 mb-2">🚀 Phase 3 Progress</h3>
              <ul className="text-xs text-blue-700 space-y-1">
                <li>✅ Next.js Project Setup</li>
                <li>✅ TypeScript Interfaces</li>
                <li>✅ API Client</li>
                <li>✅ Provider Management</li>
                <li>🚧 Chat Interface (Next)</li>
                <li>🚧 Speech Integration (Next)</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500 text-sm">
          <p>LIAC. 2025 - Next.js Migration</p>
        </div>
      </div>
    </div>
  )
}
