from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Optional
import logging

from ...core.dependencies import get_rag_service
from ...services.rag_service import RAGService
from ...models.rag import (
    QueryRequest, QueryResponse,
    DocumentRequest, DocumentResponse, DocumentListResponse,
    RAGStatsResponse, FileUploadResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Process a query through the RAG pipeline.
    
    Args:
        request: Query request with text and optional parameters
        
    Returns:
        Response with AI-generated answer and retrieved documents
    """
    try:
        logger.info(f"RAG query request: '{request.query[:100]}...'")
        
        # Process query through RAG pipeline
        result = await rag_service.process_query(
            query=request.query,
            n_results=request.n_results,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        if result['success']:
            return QueryResponse(
                success=True,
                response=result['response'],
                query=result['query'],
                retrieved_documents=result.get('retrieved_documents', []),
                timing=result['timing'],
                metadata=result.get('metadata')
            )
        else:
            return QueryResponse(
                success=False,
                query=result['query'],
                timing=result['timing'],
                error=result['error']
            )
            
    except Exception as e:
        logger.error(f"Error in RAG query endpoint: {e}")
        return QueryResponse(
            success=False,
            query=request.query,
            timing={'total_time': 0.0},
            error=str(e)
        )

@router.post("/documents", response_model=DocumentResponse)
async def add_document(
    request: DocumentRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Add a document to the knowledge base.
    
    Args:
        request: Document data with title, content, and optional metadata
        
    Returns:
        Response with document ID and processing information
    """
    try:
        logger.info(f"Adding document: '{request.title}'")
        
        result = await rag_service.add_document(
            title=request.title,
            content=request.content,
            metadata=request.metadata
        )
        
        return DocumentResponse(**result)
        
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        return DocumentResponse(
            success=False,
            error=str(e)
        )

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    List all documents in the knowledge base.
    
    Returns:
        List of documents with metadata and previews
    """
    try:
        result = await rag_service.list_documents()
        return DocumentListResponse(**result)
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return DocumentListResponse(
            success=False,
            documents=[],
            total_count=0,
            error=str(e)
        )

@router.delete("/documents/{doc_id}", response_model=DocumentResponse)
async def delete_document(
    doc_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Delete a document from the knowledge base.
    
    Args:
        doc_id: ID of the document to delete
        
    Returns:
        Response with deletion status
    """
    try:
        logger.info(f"Deleting document: {doc_id}")
        
        result = await rag_service.delete_document(doc_id)
        return DocumentResponse(**result)
        
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        return DocumentResponse(
            success=False,
            error=str(e)
        )

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Upload and process a text file for the knowledge base.
    
    Args:
        file: Text file to upload and process
        
    Returns:
        Response with processing results
    """
    try:
        logger.info(f"File upload: {file.filename}")
        
        # Read file content
        content = await file.read()
        
        # Process file through RAG service
        result = await rag_service.upload_file(content, file.filename)
        
        return FileUploadResponse(
            success=result['success'],
            doc_id=result.get('doc_id'),
            filename=file.filename,
            content_length=result.get('content_length'),
            duration=result.get('duration'),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {e}")
        return FileUploadResponse(
            success=False,
            filename=file.filename,
            error=str(e)
        )

@router.get("/stats", response_model=RAGStatsResponse)
async def get_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get RAG service performance statistics.
    
    Returns:
        Performance statistics and component information
    """
    try:
        stats_data = rag_service.get_service_stats()
        
        return RAGStatsResponse(
            stats=stats_data,
            components=stats_data['components']
        )
        
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_health(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Check RAG service health and availability.
    
    Returns:
        Health status and component availability
    """
    try:
        is_available = rag_service.is_available()
        stats = rag_service.get_service_stats()
        
        return {
            "status": "healthy" if is_available else "degraded",
            "service": "rag",
            "available": is_available,
            "total_documents": stats.get('total_documents', 0),
            "components": {
                "embeddings": rag_service.embedding_generator is not None,
                "vector_store": rag_service.vector_store is not None,
                "llm": rag_service.llm is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking RAG health: {e}")
        return {
            "status": "error",
            "service": "rag",
            "available": False,
            "error": str(e)
        }
