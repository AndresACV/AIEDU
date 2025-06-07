import axios, { AxiosInstance, AxiosResponse } from 'axios'
import {
  ProviderType,
  ProviderResponse,
  ForceProviderRequest,
  ForceProviderResponse
} from '@/types/provider'
import {
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

  // Speech API
  async getVoices(language?: string): Promise<VoicesResponse> {
    // Note: FastAPI endpoint doesn't use language parameter yet, will be added in Phase 2
    return this.get<VoicesResponse>('/api/v1/speech/voices')
  }

  async synthesize(request: SynthesizeRequest): Promise<SynthesizeResponse> {
    return this.post<SynthesizeResponse>('/synthesize', request)
  }

  async transcribe(audioBlob: Blob, language?: string): Promise<TranscribeResponse> {
    const formData = new FormData()
    formData.append('audio', audioBlob, 'recording.wav')
    if (language) {
      formData.append('language', language)
    }
    return this.postForm<TranscribeResponse>('/upload-audio', formData)
  }

  // RAG API
  async ragQuery(request: RagQueryRequest): Promise<RagQueryResponse> {
    return this.post<RagQueryResponse>('/rag_query', request)
  }

  async ragQueryWithSpeech(request: RagQueryRequest): Promise<RagQueryResponse> {
    return this.post<RagQueryResponse>('/rag_to_speech', request)
  }

  async addDocument(request: DocumentRequest): Promise<DocumentResponse> {
    return this.post<DocumentResponse>('/add_document', {
      text: request.content,
      metadata: { ...request.metadata, title: request.title }
    })
  }

  async listDocuments(): Promise<DocumentsListResponse> {
    return this.get<DocumentsListResponse>('/list_documents')
  }

  async updateDocument(docId: string, request: DocumentRequest): Promise<DocumentResponse> {
    return this.post<DocumentResponse>('/update_document', {
      doc_id: docId,
      text: request.content,
      metadata: { ...request.metadata, title: request.title }
    })
  }

  async deleteDocuments(docIds: string[]): Promise<DocumentResponse> {
    return this.post<DocumentResponse>('/delete_documents', { doc_ids: docIds })
  }

  async getKnowledgeBaseStats(): Promise<KnowledgeBaseStats> {
    return this.get<KnowledgeBaseStats>('/kb_stats')
  }

  async clearConversation(): Promise<{ success: boolean }> {
    return this.post<{ success: boolean }>('/clear_conversation')
  }

  // System API
  async getSystemStatus(): Promise<{ success: boolean; status?: string }> {
    return this.get<{ success: boolean; status?: string }>('/system/status')
  }
}

// Export singleton instance
export const apiClient = new APIClient()
export default apiClient 