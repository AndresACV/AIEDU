'use client'

import React from 'react'
import Link from 'next/link'
import EnhancedProviderSelector from '@/components/providers/EnhancedProviderSelector'
import ProviderToggle from '@/components/providers/ProviderToggle'
import ProviderStatus from '@/components/providers/ProviderStatus'
import { ArrowLeft, Sparkles } from 'lucide-react'

export default function ProviderDemoPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-emerald-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/dashboard"
                className="flex items-center gap-2 text-gray-600 hover:text-blue-600 transition-colors"
              >
                <ArrowLeft size={20} />
                <span>Back to Dashboard</span>
              </Link>
            </div>
            <div className="flex items-center gap-2 bg-gradient-to-r from-blue-500 to-emerald-500 text-white px-3 py-1 rounded-full text-sm font-medium">
              <Sparkles size={16} />
              Phase 7A Demo
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-medium mb-4">
            🚀 Phase 7A: Enhanced User Experience
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            AI Provider Switching Interface
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
            Experience seamless switching between local (privacy-focused) and cloud (performance-optimized) AI providers. 
            See real-time status updates and quality differences in action.
          </p>
          
          {/* Feature Highlights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            <div className="bg-white rounded-lg p-4 shadow-sm border">
              <div className="text-2xl mb-2">🏠</div>
              <h3 className="font-semibold text-gray-900 mb-1">Local Processing</h3>
              <p className="text-sm text-gray-600">Complete privacy, offline capability, no data transmission</p>
            </div>
            <div className="bg-white rounded-lg p-4 shadow-sm border">
              <div className="text-2xl mb-2">☁️</div>
              <h3 className="font-semibold text-gray-900 mb-1">Cloud Processing</h3>
              <p className="text-sm text-gray-600">Neural voice quality, higher accuracy, latest AI models</p>
            </div>
            <div className="bg-white rounded-lg p-4 shadow-sm border">
              <div className="text-2xl mb-2">⚡</div>
              <h3 className="font-semibold text-gray-900 mb-1">Real-time Switching</h3>
              <p className="text-sm text-gray-600">Instant provider changes with live status monitoring</p>
            </div>
          </div>
        </div>

        {/* Main Demo Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Enhanced Provider Selector */}
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Complete Provider Control</h2>
              <p className="text-gray-600 mb-4">
                Full-featured provider selector with status monitoring, quality metrics, and educational information.
              </p>
              <EnhancedProviderSelector 
                defaultExpanded={true}
                showRefreshButton={true}
              />
            </div>
          </div>

          {/* Individual Components */}
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Component Showcase</h2>
              <p className="text-gray-600 mb-4">
                Individual components that make up the provider switching interface.
              </p>
              
              {/* Provider Toggle */}
              <div className="bg-white rounded-lg border p-4 mb-4">
                <h3 className="font-semibold text-gray-800 mb-3">Provider Toggle</h3>
                <ProviderToggle showTooltips={false} />
              </div>

              {/* Provider Status */}
              <div className="bg-white rounded-lg border p-4">
                <h3 className="font-semibold text-gray-800 mb-3">Service Status Dashboard</h3>
                <ProviderStatus compact={false} />
              </div>
            </div>
          </div>
        </div>

        {/* Technical Details */}
        <div className="bg-white rounded-xl shadow-sm border p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Technical Implementation</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Frontend Features</h3>
              <ul className="space-y-2 text-gray-600">
                <li className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>Real-time provider status polling (5-second intervals)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>Smooth animations and visual feedback</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>Quality metrics and performance indicators</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>Educational tooltips and provider comparison</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>Mobile-responsive design with touch support</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Backend Integration</h3>
              <ul className="space-y-2 text-gray-600">
                <li className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>Dynamic provider switching via API endpoints</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>Real provider availability detection (no mock data)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>Google Cloud credential management</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>Service-specific provider routing</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>Performance monitoring and error handling</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Quality Comparison */}
        <div className="bg-gradient-to-r from-blue-500 to-emerald-500 rounded-xl text-white p-8 mb-8">
          <h2 className="text-2xl font-bold mb-6">Quality Comparison Results</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white/10 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <span>🏠</span> Local Processing
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span>Voice Quality:</span>
                  <span className="font-mono">Synthetic (espeak)</span>
                </div>
                <div className="flex justify-between">
                  <span>Generation Speed:</span>
                  <span className="font-mono">~0.04s (Ultra Fast)</span>
                </div>
                <div className="flex justify-between">
                  <span>File Size:</span>
                  <span className="font-mono">~237KB</span>
                </div>
                <div className="flex justify-between">
                  <span>Privacy:</span>
                  <span className="font-mono">100% Private</span>
                </div>
              </div>
            </div>
            
            <div className="bg-white/10 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <span>☁️</span> Cloud Processing
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span>Voice Quality:</span>
                  <span className="font-mono">Neural (Google Cloud)</span>
                </div>
                <div className="flex justify-between">
                  <span>Generation Speed:</span>
                  <span className="font-mono">~0.99s (Natural)</span>
                </div>
                <div className="flex justify-between">
                  <span>File Size:</span>
                  <span className="font-mono">~401KB</span>
                </div>
                <div className="flex justify-between">
                  <span>Quality:</span>
                  <span className="font-mono">Human-like</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="text-center">
          <div className="space-x-4">
            <Link 
              href="/dashboard"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
            >
              <ArrowLeft size={16} />
              Back to Dashboard
            </Link>
            <Link 
              href="/speech-demo"
              className="inline-flex items-center gap-2 px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
            >
              Test Speech Interface
              <span>🎤</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
} 