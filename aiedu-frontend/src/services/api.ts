import axios, { AxiosInstance, AxiosResponse } from 'axios'
import {
  ProviderType,
  ProviderResponse,
  ForceProviderRequest,
  ForceProviderResponse
} from '@/types/provider'
import {
  Voice,
  VoicesResponse,
  SynthesizeRequest,
  SynthesizeResponse,
  TranscribeResponse
} from '@/types/speech'
import {
  RagQueryRequest,
  RagQueryResponse,
  DocumentRequest,
  DocumentResponse,
  DocumentsListResponse,
  KnowledgeBaseStats
} from '@/types/rag'

class APIClient {
  private client: AxiosInstance
  private baseURL: string

  constructor() {
    this.baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
    console.log(`🔗 API Client initialized with baseURL: ${this.baseURL}`)
    
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000, // 30 seconds timeout
      headers: {
        'Content-Type': 'application/json',
      },
      // Handle self-signed certificates in development
      httpsAgent: typeof window === 'undefined' ? undefined : undefined,
      // Allow credentials for CORS
      withCredentials: false,
    })

    // Request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        console.log(`🌐 API Request: ${config.method?.toUpperCase()} ${config.url}`)
        return config
      },
      (error) => {
        console.error('❌ API Request Error:', error)
        return Promise.reject(error)
      }
    )

    // Response interceptor for logging
    this.client.interceptors.response.use(
      (response) => {
        console.log(`✅ API Response: ${response.status} ${response.config.url}`)
        return response
      },
      (error) => {
        // More specific error handling for different error types
        if (error.code === 'ERR_NETWORK' || error.message === 'Network Error') {
          console.warn(`🔌 Network Error: Backend server may not be running or accessible at ${this.baseURL}`)
        } else if (error.code === 'ECONNREFUSED') {
          console.warn(`🚫 Connection Refused: Backend server at ${this.baseURL} is not responding`)
        } else if (error.response?.status === 0) {
          console.warn(`⚠️ CORS/SSL Error: Check if backend server is running with HTTPS`)
        } else {
          console.error(`❌ API Error: ${error.response?.status} ${error.config?.url}`, error.response?.data || error.message)
        }
        return Promise.reject(error)
      }
    )
  }

  // Generic HTTP methods
  private async get<T>(endpoint: string): Promise<T> {
    const response: AxiosResponse<T> = await this.client.get(endpoint)
    return response.data
  }

  private async post<T>(endpoint: string, data?: any): Promise<T> {
    const response: AxiosResponse<T> = await this.client.post(endpoint, data)
    return response.data
  }

  private async postForm<T>(endpoint: string, formData: FormData): Promise<T> {
    const response: AxiosResponse<T> = await this.client.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  private async put<T>(endpoint: string, data?: any): Promise<T> {
    const response: AxiosResponse<T> = await this.client.put(endpoint, data)
    return response.data
  }

  private async delete<T>(endpoint: string): Promise<T> {
    const response: AxiosResponse<T> = await this.client.delete(endpoint)
    return response.data
  }

  // Provider Management API
  async forceProvider(provider: ProviderType): Promise<ForceProviderResponse> {
    return this.post<ForceProviderResponse>('/api/v1/providers/force', { provider })
  }

  async getCurrentProviders(): Promise<ProviderResponse> {
    return this.get<ProviderResponse>('/api/v1/providers/current')
  }

  // Speech API - FastAPI Endpoints
  async getVoices(language?: string): Promise<VoicesResponse> {
    try {
      const voices = await this.get<Voice[]>('/api/v1/speech/voices')
      return {
        success: true,
        voices: voices
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to load voices'
      }
    }
  }

  async synthesize(request: SynthesizeRequest): Promise<SynthesizeResponse> {
    return this.post<SynthesizeResponse>('/api/v1/speech/synthesize', request)
  }

  async transcribe(audioBlob: Blob, language?: string): Promise<TranscribeResponse> {
    const formData = new FormData()
    formData.append('file', audioBlob, 'recording.webm')
    if (language) {
      formData.append('language', language)
    }
    return this.postForm<TranscribeResponse>('/api/v1/speech/transcribe', formData)
  }

  async getSpeechStats(): Promise<{ success: boolean; stats?: any }> {
    return this.get<{ success: boolean; stats?: any }>('/api/v1/speech/stats')
  }

  async cleanupSpeechFiles(): Promise<{ success: boolean; message?: string }> {
    return this.post<{ success: boolean; message?: string }>('/api/v1/speech/cleanup')
  }

  // RAG API - FastAPI Endpoints  
  async ragQuery(request: RagQueryRequest): Promise<RagQueryResponse> {
    // Use extended timeout for RAG queries (Ollama model loading can take 60+ seconds)
    const response: AxiosResponse<RagQueryResponse> = await this.client.post('/api/v1/rag/query', request, {
      timeout: 120000, // 2 minutes timeout for RAG queries (model loading + inference)
    })
    return response.data
  }

  async addDocument(request: DocumentRequest): Promise<DocumentResponse> {
    return this.post<DocumentResponse>('/api/v1/rag/documents', {
      title: request.title,
      content: request.content,
      metadata: request.metadata || {}
    })
  }

  async listDocuments(): Promise<DocumentsListResponse> {
    return this.get<DocumentsListResponse>('/api/v1/rag/documents')
  }

  async ragDocuments(): Promise<any[]> {
    return this.get<any[]>('/api/v1/rag/documents')
  }

  async ragUpload(formData: FormData): Promise<DocumentResponse> {
    return this.postForm<DocumentResponse>('/api/v1/rag/upload', formData)
  }

  async ragDeleteDocument(docId: string): Promise<DocumentResponse> {
    return this.delete<DocumentResponse>(`/api/v1/rag/documents/${docId}`)
  }

  async deleteDocument(docId: string): Promise<DocumentResponse> {
    return this.delete<DocumentResponse>(`/api/v1/rag/documents/${docId}`)
  }

  async uploadFile(file: File): Promise<DocumentResponse> {
    const formData = new FormData()
    formData.append('file', file)
    return this.postForm<DocumentResponse>('/api/v1/rag/upload', formData)
  }

  async getRagStats(): Promise<{ success: boolean; stats?: any }> {
    return this.get<{ success: boolean; stats?: any }>('/api/v1/rag/stats')
  }

  async getRagHealth(): Promise<{ success: boolean; status?: string }> {
    return this.get<{ success: boolean; status?: string }>('/api/v1/rag/health')
  }

  async clearConversation(): Promise<{ success: boolean }> {
    return this.post<{ success: boolean }>('/clear_conversation')
  }

  // System API
  async health(): Promise<{ status: string; service: string }> {
    return this.get<{ status: string; service: string }>('/health')
  }

  async getSystemStatus(): Promise<{ success: boolean; status?: string }> {
    return this.get<{ success: boolean; status?: string }>('/system/status')
  }
}

// Export singleton instance
export const apiClient = new APIClient()
export default apiClient 