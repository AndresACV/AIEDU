import os
from typing import List
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
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
