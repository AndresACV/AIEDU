'use client'

import React, { useState, useEffect, useCallback, useRef } from 'react'
import { apiClient } from '@/services/api'

interface Document {
  id: string
  title: string
  content_preview: string
  upload_date: string
  file_type: string
  size: number
}

interface KnowledgeManagerProps {
  onDocumentUpdate?: () => void
  onError?: (error: string) => void
  className?: string
}

export default function KnowledgeManager({
  onDocumentUpdate,
  onError,
  className = ''
}: KnowledgeManagerProps) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [dragActive, setDragActive] = useState(false)

  // Stable references for callbacks to prevent circular dependencies
  const onErrorRef = useRef(onError)
  const onDocumentUpdateRef = useRef(onDocumentUpdate)
  useEffect(() => { 
    onErrorRef.current = onError 
    onDocumentUpdateRef.current = onDocumentUpdate
  }, [onError, onDocumentUpdate])

  // Load documents - removed circular dependency
  const loadDocuments = useCallback(async () => {
    try {
      setIsLoading(true)
      const response = await apiClient.ragDocuments()
      if (Array.isArray(response)) {
        setDocuments(response)
      }
    } catch (error) {
      onErrorRef.current?.(`Failed to load documents: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsLoading(false)
    }
  }, []) // Empty dependency array

  useEffect(() => {
    loadDocuments()
  }, []) // Only run on mount

  // Handle file upload
  const handleFileUpload = async (files: FileList) => {
    if (!files || files.length === 0) return

    setIsUploading(true)
    setUploadProgress(0)

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const formData = new FormData()
        formData.append('file', file)

        await apiClient.ragUpload(formData)
        setUploadProgress(((i + 1) / files.length) * 100)
      }

      await loadDocuments()
      onDocumentUpdateRef.current?.()
      setSelectedFiles(null)
    } catch (error) {
      onErrorRef.current?.(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsUploading(false)
      setUploadProgress(0)
    }
  }

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(e.target.files)
    }
  }

  // Handle drag and drop
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files)
    }
  }

  // Delete document
  const handleDeleteDocument = async (documentId: string) => {
    if (!confirm('Are you sure you want to delete this document?')) return

    try {
      await apiClient.ragDeleteDocument(documentId)
      await loadDocuments()
      onDocumentUpdateRef.current?.()
    } catch (error) {
      onErrorRef.current?.(`Delete failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // Format date
  const formatDate = (dateString: string): string => {
    try {
      return new Date(dateString).toLocaleDateString()
    } catch {
      return 'Unknown'
    }
  }

  return (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <span className="mr-2">📚</span>
              Knowledge Base
            </h3>
            <p className="text-sm text-gray-600 mt-1">
              Manage your documents for AI-powered conversations
            </p>
          </div>
          <div className="text-sm text-gray-500">
            {documents.length} document{documents.length !== 1 ? 's' : ''}
          </div>
        </div>
      </div>

      {/* Upload Area */}
      <div className="p-6 border-b border-gray-200">
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center transition-all ${
            dragActive 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-300 hover:border-gray-400'
          } ${isUploading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => {
            if (!isUploading) {
              document.getElementById('file-upload')?.click()
            }
          }}
        >
          <input
            id="file-upload"
            type="file"
            multiple
            accept=".pdf,.txt,.doc,.docx,.md"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isUploading}
          />
          
          {isUploading ? (
            <div>
              <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-gray-600 mb-2">Uploading documents...</p>
              <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <p className="text-sm text-gray-500">{Math.round(uploadProgress)}% complete</p>
            </div>
          ) : (
            <div>
              <div className="text-4xl mb-4">📄</div>
              <p className="text-gray-600 mb-2">
                <strong>Click to upload</strong> or drag and drop files here
              </p>
              <p className="text-sm text-gray-500">
                Supports PDF, TXT, DOC, DOCX, MD files
              </p>
              {selectedFiles && (
                <div className="mt-4">
                  <p className="text-sm text-blue-600 mb-2">
                    {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''} selected:
                  </p>
                  <div className="text-xs text-gray-500">
                    {Array.from(selectedFiles).map((file, index) => (
                      <div key={index}>{file.name}</div>
                    ))}
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleFileUpload(selectedFiles)
                    }}
                    className="mt-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm transition-all"
                  >
                    Upload Files
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Documents List */}
      <div className="p-6">
        {isLoading ? (
          <div className="text-center py-8">
            <div className="w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-gray-500">Loading documents...</p>
          </div>
        ) : documents.length === 0 ? (
          <div className="text-center py-8">
            <div className="text-gray-400 text-4xl mb-4">📭</div>
            <p className="text-gray-500 mb-2">No documents uploaded yet</p>
            <p className="text-sm text-gray-400">
              Upload your first document to start building your knowledge base
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {documents.map((doc) => (
              <div key={doc.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-all">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-lg">
                        {doc.file_type === 'pdf' ? '📄' : 
                         doc.file_type === 'txt' ? '📝' : 
                         doc.file_type === 'doc' || doc.file_type === 'docx' ? '📄' : 
                         '📄'}
                      </span>
                      <h4 className="font-medium text-gray-900">{doc.title}</h4>
                    </div>
                    
                    <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                      {doc.content_preview}
                    </p>
                    
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span>📅 {formatDate(doc.upload_date)}</span>
                      <span>📊 {formatFileSize(doc.size)}</span>
                      <span>🏷️ {doc.file_type.toUpperCase()}</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2 ml-4">
                    <button
                      onClick={() => handleDeleteDocument(doc.id)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-all"
                      title="Delete document"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      {documents.length > 0 && (
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 rounded-b-xl">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <span>
              Total: {documents.length} document{documents.length !== 1 ? 's' : ''}
            </span>
            <button
              onClick={loadDocuments}
              className="text-blue-600 hover:text-blue-800 font-medium"
            >
              🔄 Refresh
            </button>
          </div>
        </div>
      )}
    </div>
  )
} 