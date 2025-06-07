export interface RagQueryRequest {
  query: string
  use_memory?: boolean
  voice_id?: string
}

export interface RagQueryResponse {
  success: boolean
  answer?: string
  retrieved_documents?: string[]
  audio_url?: string
  error?: string
}

export interface Document {
  id: string
  title: string
  content: string
  metadata: Record<string, any>
}

export interface DocumentRequest {
  title: string
  content: string
  metadata?: Record<string, any>
}

export interface DocumentResponse {
  success: boolean
  document_id?: string
  error?: string
}

export interface DocumentsListResponse {
  success: boolean
  documents?: {
    ids: string[]
    documents: string[]
    metadatas: Record<string, any>[]
  }
  error?: string
}

export interface KnowledgeBaseStats {
  success: boolean
  count?: number
  collection_name?: string
  status?: string
  error?: string
}

export interface Message {
  id: string
  type: 'user' | 'assistant' | 'error'
  content: string
  sources?: string[]
  timestamp: Date
} 