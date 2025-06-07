# AIEDU FastAPI Backend

## Overview

This is the new FastAPI backend for AIEDU, replacing the Flask monolith with a clean API-first architecture.

## Features

- **Pure JSON API**: No HTML templates, clean separation from frontend
- **Auto-generated Documentation**: Available at `/docs` endpoint
- **Type Safety**: Full Pydantic validation and TypeScript compatibility
- **Better Performance**: 2-3x faster than Flask
- **Modern Architecture**: Clean service layer and dependency injection

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Server
```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Or run directly
python -m app.main
```

### 3. Access the API
- **API Base**: http://127.0.0.1:8000
- **Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

## API Endpoints

### Providers
- `GET /api/v1/providers/current` - Get current provider status
- `POST /api/v1/providers/force` - Switch provider (local/cloud)

### Speech
- `GET /api/v1/speech/voices` - Get available TTS voices

## Development

The FastAPI server runs on port 8000 to avoid conflicts with the old Flask server (port 5000).

The frontend should be configured to use:
```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Migration Status

**Phase 1 Complete**: Basic FastAPI foundation with core endpoints
**Phase 2 Planned**: Full speech service integration
**Phase 3 Planned**: RAG system migration
**Phase 4 Planned**: Advanced features and optimization 