from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# Document Models
class DocumentInfo(BaseModel):
    """Document information model."""
    id: str
    title: str
    content_preview: str
    added_at: Optional[str] = None
    content_length: int
    
class DocumentRequest(BaseModel):
    """Request model for adding a document."""
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    """Response model for document operations."""
    success: bool
    doc_id: Optional[str] = None
    title: Optional[str] = None
    content_length: Optional[int] = None
    duration: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None

class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    success: bool
    documents: List[DocumentInfo]
    total_count: int
    error: Optional[str] = None

# Query Models
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str
    n_results: Optional[int] = 3
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

class RetrievedDocument(BaseModel):
    """Information about a retrieved document."""
    content_preview: str
    title: Optional[str] = None
    doc_id: Optional[str] = None
    similarity_score: Optional[float] = None

class QueryTiming(BaseModel):
    """Timing information for query processing."""
    total_time: float
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None

class QueryMetadata(BaseModel):
    """Metadata about query processing."""
    n_documents_retrieved: int
    tokens_generated: int
    temperature: float

class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    success: bool
    response: Optional[str] = None
    query: str
    retrieved_documents: Optional[List[RetrievedDocument]] = None
    timing: QueryTiming
    metadata: Optional[QueryMetadata] = None
    error: Optional[str] = None

# Statistics Models
class RAGStats(BaseModel):
    """RAG service performance statistics."""
    documents_added: int
    documents_deleted: int
    queries_processed: int
    successful_queries: int
    failed_queries: int
    avg_query_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    total_documents: int
    success_rate: float
    service_status: str

class RAGComponents(BaseModel):
    """RAG service component information."""
    embedding_model: str
    llm_model: str
    vector_store: str

class RAGStatsResponse(BaseModel):
    """Response model for RAG statistics."""
    stats: RAGStats
    components: RAGComponents

# File Upload Models
class FileUploadResponse(BaseModel):
    """Response model for file uploads."""
    success: bool
    doc_id: Optional[str] = None
    filename: Optional[str] = None
    content_length: Optional[int] = None
    duration: Optional[float] = None
    error: Optional[str] = None
