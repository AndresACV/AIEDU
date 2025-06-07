'use client'

import React, { useState } from 'react'

interface RetrievedDocument {
  title: string
  content_preview: string
  doc_id: string
  similarity_score: number
  full_content?: string
}

interface DocumentViewerProps {
  documents: RetrievedDocument[]
  query?: string
  onDocumentSelect?: (docId: string) => void
  className?: string
  maxPreviewLength?: number
}

export default function DocumentViewer({
  documents,
  query = '',
  onDocumentSelect,
  className = '',
  maxPreviewLength = 300
}: DocumentViewerProps) {
  const [expandedDocs, setExpandedDocs] = useState<Set<string>>(new Set())
  const [sortBy, setSortBy] = useState<'relevance' | 'title'>('relevance')

  // Toggle document expansion
  const toggleExpansion = (docId: string) => {
    const newExpanded = new Set(expandedDocs)
    if (newExpanded.has(docId)) {
      newExpanded.delete(docId)
    } else {
      newExpanded.add(docId)
    }
    setExpandedDocs(newExpanded)
  }

  // Calculate relevance percentage (similarity scores are typically negative)
  const getRelevancePercentage = (score: number): number => {
    // Convert similarity score to percentage (assuming scores are between -2 and 0)
    return Math.max(0, Math.min(100, Math.round((1 + score / 2) * 100)))
  }

  // Sort documents
  const sortedDocuments = [...documents].sort((a, b) => {
    if (sortBy === 'relevance') {
      return b.similarity_score - a.similarity_score // Higher scores first
    } else {
      return a.title.localeCompare(b.title)
    }
  })

  // Highlight query terms in text
  const highlightText = (text: string, query: string): React.ReactNode => {
    if (!query.trim()) return text
    
    const terms = query.toLowerCase().split(/\s+/).filter(term => term.length > 2)
    let highlightedText = text
    
    terms.forEach(term => {
      const regex = new RegExp(`(${term})`, 'gi')
      highlightedText = highlightedText.replace(regex, '<mark>$1</mark>')
    })
    
    return <span dangerouslySetInnerHTML={{ __html: highlightedText }} />
  }

  if (!documents || documents.length === 0) {
    return (
      <div className={`bg-gray-50 rounded-lg p-6 text-center ${className}`}>
        <div className="text-gray-400 mb-2">📚</div>
        <p className="text-gray-600">No documents retrieved</p>
      </div>
    )
  }

  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">
              Retrieved Documents
            </h3>
            <p className="text-sm text-gray-500">
              {documents.length} document{documents.length !== 1 ? 's' : ''} found
            </p>
          </div>
          
          {/* Sort options */}
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'relevance' | 'title')}
              className="text-xs border border-gray-300 rounded px-2 py-1"
            >
              <option value="relevance">Relevance</option>
              <option value="title">Title</option>
            </select>
          </div>
        </div>
      </div>

      {/* Documents list */}
      <div className="max-h-96 overflow-y-auto">
        {sortedDocuments.map((doc, index) => {
          const isExpanded = expandedDocs.has(doc.doc_id)
          const relevancePercentage = getRelevancePercentage(doc.similarity_score)
          
          return (
            <div
              key={doc.doc_id}
              className="border-b border-gray-100 last:border-b-0"
            >
              {/* Document header */}
              <div className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900 mb-1">
                      {doc.title || `Document ${index + 1}`}
                    </h4>
                    
                    {/* Relevance score */}
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="flex items-center space-x-1">
                        <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className={`h-full transition-all ${
                              relevancePercentage > 70 
                                ? 'bg-green-500'
                                : relevancePercentage > 40
                                ? 'bg-yellow-500'
                                : 'bg-red-500'
                            }`}
                            style={{ width: `${relevancePercentage}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-500">
                          {relevancePercentage}% match
                        </span>
                      </div>
                      
                      <span className="text-xs text-gray-400 font-mono">
                        {doc.doc_id.slice(0, 8)}...
                      </span>
                    </div>
                  </div>

                  {/* Action buttons */}
                  <div className="flex items-center space-x-2 ml-4">
                    {onDocumentSelect && (
                      <button
                        onClick={() => onDocumentSelect(doc.doc_id)}
                        className="text-xs text-blue-600 hover:text-blue-800 px-2 py-1 hover:bg-blue-50 rounded"
                      >
                        View Full
                      </button>
                    )}
                    
                    <button
                      onClick={() => toggleExpansion(doc.doc_id)}
                      className="text-xs text-gray-600 hover:text-gray-800 px-2 py-1 hover:bg-gray-50 rounded"
                    >
                      {isExpanded ? 'Collapse' : 'Expand'}
                    </button>
                  </div>
                </div>

                {/* Content preview */}
                <div className="text-sm text-gray-700">
                  {isExpanded ? (
                    <div className="whitespace-pre-wrap">
                      {highlightText(doc.full_content || doc.content_preview, query)}
                    </div>
                  ) : (
                    <div>
                      {highlightText(
                        doc.content_preview.length > maxPreviewLength
                          ? doc.content_preview.slice(0, maxPreviewLength) + '...'
                          : doc.content_preview,
                        query
                      )}
                    </div>
                  )}
                </div>

                {/* Expand/collapse indicator */}
                {doc.content_preview.length > maxPreviewLength && (
                  <div className="mt-2">
                    <button
                      onClick={() => toggleExpansion(doc.doc_id)}
                      className="text-xs text-blue-600 hover:text-blue-800 flex items-center space-x-1"
                    >
                      <span>{isExpanded ? 'Show less' : 'Show more'}</span>
                      <svg
                        className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Footer with stats */}
      <div className="p-3 bg-gray-50 text-xs text-gray-500 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <span>
            Showing {documents.length} most relevant document{documents.length !== 1 ? 's' : ''}
          </span>
          
          {query && (
            <span>
              Query: "{query.length > 30 ? query.slice(0, 30) + '...' : query}"
            </span>
          )}
        </div>
      </div>
    </div>
  )
} 