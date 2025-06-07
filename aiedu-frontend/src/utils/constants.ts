// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000',
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
} as const

// Provider Status Refresh Interval
export const PROVIDER_STATUS_REFRESH_INTERVAL = 5000 // 5 seconds

// Backend Connection Instructions
export const BACKEND_CONNECTION_INFO = {
  title: 'FastAPI Backend Connection',
  description: 'The frontend connects to the FastAPI backend. If you see connection errors:',
  steps: [
    'Make sure the FastAPI backend is running on http://127.0.0.1:8000',
    'Start the backend with: cd backend && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000',
    'Check that the API is responding at http://127.0.0.1:8000/health',
    'Refresh this page to reconnect'
  ],
  troubleshooting: [
    'If the backend is not running, start it with: cd backend && python -m app.main',
    'If you see CORS errors, make sure both servers are running on the correct ports',
    'For development, FastAPI includes auto-reload by default with --reload flag'
  ]
} as const

// Development Commands
export const DEV_COMMANDS = {
  backend: 'cd local_deployment/web_app && source ../../venv/bin/activate && python app.py',
  backendWithReload: 'cd local_deployment/web_app && source ../../venv/bin/activate && export AIEDU_RELOAD=true && python app.py',
  frontend: 'cd aiedu-frontend && npm run dev',
} as const

// Service Status Colors for UI
export const STATUS_COLORS = {
  working: 'text-green-500',
  connecting: 'text-yellow-500',
  error: 'text-red-500',
  unknown: 'text-gray-400',
} as const

// Provider Information
export const PROVIDER_INFO = {
  local: {
    name: 'Local Providers',
    description: 'Privacy-focused, offline processing',
    services: {
      stt: 'Vosk (offline speech recognition)',
      tts: 'espeak (offline text-to-speech)',
      llm: 'Ollama (local language model)'
    }
  },
  cloud: {
    name: 'Cloud Providers', 
    description: 'Performance-optimized, online processing',
    services: {
      stt: 'Google Cloud Speech-to-Text',
      tts: 'Google Cloud Text-to-Speech (Neural voices)',
      llm: 'Google Gemini (advanced language model)'
    }
  }
} as const 