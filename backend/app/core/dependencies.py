from typing import Generator
from ..services.provider_service import ProviderService
from ..services.speech_service import SpeechService
from ..services.rag_service import RAGService

# Global service instances (will be initialized at startup)
_provider_service: ProviderService = None
_speech_service: SpeechService = None
_rag_service: RAGService = None

def get_provider_service() -> ProviderService:
    """Dependency to get the provider service instance."""
    global _provider_service
    if _provider_service is None:
        _provider_service = ProviderService()
    return _provider_service

def get_speech_service() -> SpeechService:
    """Dependency to get the speech service instance."""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService()
    return _speech_service

def get_rag_service() -> RAGService:
    """Dependency to get the RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

# Service initialization function (called at startup)
def initialize_services():
    """Initialize all services at application startup."""
    global _provider_service, _speech_service, _rag_service
    print("🔧 Initializing services...")
    
    # Check Ollama availability first
    _check_ollama_status()
    
    try:
        _provider_service = ProviderService()
        print("✅ Provider service initialized")
        _speech_service = SpeechService()
        print("✅ Speech service initialized")
        _rag_service = RAGService()
        print("✅ RAG service initialized")
        print("✅ All services initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing services: {e}")
        # Still continue - services will be lazy-loaded when needed
        pass

def _check_ollama_status():
    """Check if Ollama is running and has the required model."""
    import requests
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if "mistral:7b" in model_names:
                print("✅ Ollama connected - mistral:7b model ready")
                # Warm up the model to prevent cold start delays
                _warm_up_model()
            else:
                print("⚠️  Ollama running but mistral:7b model not found")
                print("   Run: ollama pull mistral:7b")
                print(f"   Available models: {', '.join(model_names) if model_names else 'none'}")
        else:
            print("⚠️  Ollama API returned error")
            _print_ollama_help()
    except requests.exceptions.ConnectionError:
        print("⚠️  Ollama not running - RAG will use mock responses")
        _print_ollama_help()
    except requests.exceptions.Timeout:
        print("⚠️  Ollama connection timeout")
        _print_ollama_help()
    except Exception as e:
        print(f"⚠️  Ollama check failed: {e}")
        _print_ollama_help()

def _warm_up_model():
    """Warm up Ollama model to prevent cold start delays."""
    import requests
    import threading
    
    def warm_up():
        try:
            print("🔥 Warming up Ollama model (preventing cold starts)...")
            
            # Send a simple prompt to load the model into GPU memory
            warm_up_payload = {
                "model": "mistral:7b",
                "prompt": "Hello",
                "stream": False,
                "options": {
                    "num_predict": 5,  # Just generate a few tokens
                    "temperature": 0.1
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=warm_up_payload,
                timeout=60  # Give model time to load
            )
            
            if response.status_code == 200:
                print("✅ Model warmed up - future requests will be faster")
            else:
                print("⚠️  Model warm-up failed")
                
        except Exception as e:
            print(f"⚠️  Model warm-up error: {e}")
    
    # Run warm-up in background thread to not block startup
    warm_up_thread = threading.Thread(target=warm_up, daemon=True)
    warm_up_thread.start()

def _print_ollama_help():
    """Print help message for Ollama setup."""
    print("   📋 To fix this:")
    print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
    print("   2. Start Ollama: ollama serve")  
    print("   3. Install model: ollama pull mistral:7b")
    print("   4. Restart backend: python -m app.main")

# Service cleanup function (called at shutdown)
def cleanup_services():
    """Clean up services at application shutdown."""
    # Add cleanup logic here if needed
    print("✅ Services cleaned up")
