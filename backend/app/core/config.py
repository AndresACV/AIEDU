import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from the root .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env"))

class Settings(BaseSettings):
    # API Configuration
    api_title: str = "AIEDU API"
    api_version: str = "1.0.0"
    api_description: str = "Educational RAG System with Speech Integration"
    
    # Server Configuration
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    
    # CORS Configuration
    allowed_origins: List[str] = [
        f"http://localhost:{port}" for port in range(3000, 3020)
    ] + [
        f"http://127.0.0.1:{port}" for port in range(3000, 3020)
    ]
    
    # Provider Configuration
    default_provider: str = "local"
    
    # Google Cloud Configuration
    gemini_api_key: Optional[str] = None
    google_application_credentials: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from environment

# Global settings instance
settings = Settings()

# Ensure Google Cloud credentials are available in os.environ for Google Cloud SDK
if settings.google_application_credentials and not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = settings.google_application_credentials
    print(f"✅ Set GOOGLE_APPLICATION_CREDENTIALS in os.environ (length: {len(settings.google_application_credentials)})")

if settings.gemini_api_key and not os.environ.get('GEMINI_API_KEY'):
    os.environ['GEMINI_API_KEY'] = settings.gemini_api_key
    print(f"✅ Set GEMINI_API_KEY in os.environ")
