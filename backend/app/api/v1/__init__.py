from fastapi import APIRouter
from .providers import router as providers_router
from .speech import router as speech_router
from .rag import router as rag_router

api_router = APIRouter(prefix="/api/v1")

# Include all endpoint routers
api_router.include_router(providers_router)
api_router.include_router(speech_router)
api_router.include_router(rag_router)
