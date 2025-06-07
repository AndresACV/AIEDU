from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .core.config import settings
from .core.dependencies import initialize_services, cleanup_services
from .api.v1 import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting AIEDU FastAPI Backend...")
    initialize_services()
    print(f"✅ FastAPI server ready on http://{settings.host}:{settings.port}")
    print(f"📚 API documentation available at http://{settings.host}:{settings.port}/docs")
    yield
    # Shutdown
    cleanup_services()
    print("👋 AIEDU FastAPI Backend shutdown complete")

# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API status."""
    return {
        "status": "AIEDU FastAPI Backend is running",
        "version": settings.api_version,
        "docs": f"http://{settings.host}:{settings.port}/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with service status."""
    import requests
    
    # Check Ollama status
    ollama_status = "unknown"
    ollama_models = []
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            ollama_models = [model["name"] for model in models]
            ollama_status = "connected" if "mistral:7b" in ollama_models else "model_missing"
        else:
            ollama_status = "error"
    except requests.exceptions.ConnectionError:
        ollama_status = "not_running"
    except Exception:
        ollama_status = "error"
    
    return {
        "status": "healthy", 
        "service": "aiedu-backend",
        "ollama": {
            "status": ollama_status,
            "models": ollama_models,
            "required_model": "mistral:7b",
            "ready": ollama_status == "connected"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
