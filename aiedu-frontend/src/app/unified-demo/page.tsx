'use client'

import React from 'react'
import { UnifiedInterface } from '@/components/unified'

export default function UnifiedDemoPage() {
  return (
    <div className="min-h-screen">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-sm font-medium">
            🎉 Phase 6C Complete: Unified AIEDU Interface with Speech + RAG + Knowledge Management
          </p>
        </div>
      </div>
      
      {/* Main Interface */}
      <UnifiedInterface />
    </div>
  )
} 