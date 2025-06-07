# 🧠 AIEDU: Complete RAG System with Speech Interaction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Next.js](https://img.shields.io/badge/Next.js-15.3.3-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Status](https://img.shields.io/badge/Status-Phase%206C%20Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<p align="center">
  <img src="assets/logo.jpg" alt="AIEDU Logo" width="400">
</p>

**AIEDU** is a complete production-ready Retrieval-Augmented Generation (RAG) system with hybrid speech interaction capabilities. This educational AI platform combines **Next.js 15 + TypeScript frontend**, **FastAPI backend**, **local/cloud provider switching**, and comprehensive speech processing through multiple intuitive interfaces.

## 🎉 Current Status: Phase 6C Complete - All Systems Operational!

✅ **Backend API**: 24 FastAPI endpoints, 100% operational  
✅ **Frontend**: 5 production interfaces, 0 build errors  
✅ **RAG System**: 13 documents loaded, sub-second responses  
✅ **Speech Services**: 100% success rate, EN/ES support  
✅ **Unified Interface**: Complete multi-modal experience  

## ✨ Key Features

- **🚀 Modern Architecture**: Next.js 15.3.3 + TypeScript frontend, FastAPI backend (24 endpoints)
- **🧠 Complete RAG System**: ChromaDB vector store + sentence-transformers + Ollama/Gemini LLM
- **🔄 Hybrid Providers**: Switch between Local (privacy) and Cloud (performance) AI services
- **🗣️ Full Speech Interaction**: STT/TTS with Vosk+espeak (local) or Google Cloud (cloud)
- **🌐 Multilingual Support**: English (US) and Spanish (Latin America) with optimized voices
- **💻 Unified Interface**: Multi-modal interaction (voice + text + documents)
- **🎛️ Real-Time Monitoring**: System health tracking with 30-second updates
- **📚 Knowledge Management**: Modern React interface for document management
- **🔒 Type Safety**: Full TypeScript integration with comprehensive API types
- **🎨 Modern UI**: Tailwind CSS with responsive design and accessibility

## 🏗️ Architecture

```text
AIEDU/ (Phase 6C Complete)
├── backend/                     # FastAPI Backend (24 Endpoints)
│   ├── app/
│   │   ├── main.py             # FastAPI application
│   │   ├── api/v1/             # API v1 routes
│   │   │   ├── rag.py          # RAG endpoints (7 routes)
│   │   │   ├── speech.py       # Speech endpoints (8 routes)
│   │   │   ├── providers.py    # Provider endpoints (3 routes)
│   │   │   └── knowledge.py    # Knowledge endpoints (6 routes)
│   │   ├── core/               # Core configuration
│   │   ├── models/             # Pydantic models
│   │   └── services/           # Business logic
│   └── requirements.txt        # Backend dependencies
│
├── aiedu-frontend/             # Next.js Frontend (5 Interfaces)
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx        # Unified Interface (Main)
│   │   │   ├── dashboard/      # System Dashboard
│   │   │   ├── rag-demo/       # RAG Chat Interface
│   │   │   ├── speech-demo/    # Speech Interface
│   │   │   └── unified-demo/   # Phase 6C Demo
│   │   ├── components/
│   │   │   ├── unified/        # Unified Interface components
│   │   │   ├── knowledge/      # Document management
│   │   │   ├── chat/           # Chat interface
│   │   │   ├── speech/         # Speech components
│   │   │   └── providers/      # Provider management
│   │   ├── hooks/              # Custom React hooks
│   │   ├── services/           # API integration
│   │   └── types/              # TypeScript definitions
│   └── package.json            # Frontend dependencies
│
├── vector_db/                  # ChromaDB persistent storage (13 documents)
├── cursor_docs/                # Complete documentation
│   ├── activeContext.md        # Phase 6C Complete status
│   ├── progress.md             # All features achieved
│   ├── systemPatterns.md       # 22 architecture patterns
│   ├── productContext.md       # Complete product description
│   └── techContext.md          # Full tech stack details
├── requirements.txt            # Main dependencies
└── README.md                   # This documentation
```

## 🚀 Quick Start

### Prerequisites

1. **System Requirements**:
   - Python 3.10+ (tested with 3.10)
   - Node.js 18+ with npm
   - Windows WSL2/Ubuntu or native Linux
   - 8GB+ RAM recommended

2. **Install System Dependencies** (Linux/WSL2):
   ```bash
   # Install espeak for text-to-speech
   sudo apt update
   sudo apt install -y espeak espeak-data libespeak1 alsa-utils
   
   # Test espeak installation
   espeak "Hello world"
   ```

3. **Install Ollama (Required for Local LLM)**:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service (keeps running in background)
   ollama serve
   
   # Download the required model (first time only - ~4GB download)
   ollama pull mistral:7b
   
   # Verify installation
   ollama list  # Should show mistral:7b
   ```

4. **Setup Python Environment**:
   ```bash
   # Clone and setup
   git clone <repository-url>
   cd AIEDU
   python -m venv venv
   source venv/bin/activate  # Linux/WSL2
   pip install -r requirements.txt
   ```

5. **Setup Frontend**:
   ```bash
   cd aiedu-frontend
   npm install
   cd ..
   ```

### 🚀 Launch Application (Three-Service System)

You need **THREE terminals** for the complete system:

#### **Terminal 1: Ollama Service - Port 11434** 
```bash
# Start Ollama server (must be running for RAG to work)
ollama serve

# Leave this terminal running - Ollama will listen on http://localhost:11434
# The backend will connect to this automatically
```

#### **Terminal 2: Backend (FastAPI) - Port 8000**
```bash
source venv/bin/activate
cd backend
python -m app.main
```

**Backend startup (15-20 seconds):**
```
🔑 Loading environment variables...
🎤 Speech models: EN(✅) ES(✅)
🧠 Loading embedding model...
✅ Embedding model loaded and ready
🤖 Checking Ollama connection...
✅ Ollama connected - mistral:7b model ready
✅ All services initialized
🌐 Server ready - http://127.0.0.1:8000

⚠️  If you see "Ollama not available" warnings:
   Start Ollama in Terminal 1: ollama serve
```

#### **Terminal 3: Frontend (Next.js) - Port 3000**
```bash
cd aiedu-frontend
npm run dev
```

**Frontend startup (3-5 seconds):**
```
▲ Next.js 15.3.3
- Local:        http://localhost:3000
✓ Ready in 3.2s
```

### 🌐 Access the Complete System

#### **Production Interfaces** (All Operational):
- **🎯 Unified Interface**: http://localhost:3000 (Main - All features)
- **📊 System Dashboard**: http://localhost:3000/dashboard (Admin panel)
- **💬 RAG Chat**: http://localhost:3000/rag-demo (Document Q&A)
- **🎤 Speech Interface**: http://localhost:3000/speech-demo (Voice interaction)
- **🚀 Phase 6C Demo**: http://localhost:3000/unified-demo (Completion showcase)

#### **Backend API**:
- **API Base**: http://127.0.0.1:8000
- **Health Check**: http://127.0.0.1:8000/health
- **API Docs**: http://127.0.0.1:8000/docs (Auto-generated)

## 🎯 Complete Feature Set

### 🎛️ Unified Interface (Main Experience)
- **Multi-Modal Interaction**: Voice + Text + Documents in one interface
- **Real-Time System Monitoring**: Backend, Speech, RAG service status
- **Activity Logging**: 50-item history with timestamps
- **Quick Actions**: Voice test, health check, system clear
- **Settings Panel**: Speech synthesis, voice input, language selection
- **Interface Modes**: Unified, Chat-only, Speech-only, Knowledge Management

### 🧠 RAG System (Retrieval-Augmented Generation)
- **Vector Store**: ChromaDB with 13 pre-loaded documents
- **Embeddings**: sentence-transformers (GPU optimized)
- **Query Processing**: Sub-second response times (0.86s average)
- **Document Retrieval**: Semantic search with similarity scoring
- **Context Display**: Retrieved documents shown with queries
- **Multi-language**: English and Spanish document support

### 🎤 Speech Services (100% Success Rate)
- **Text-to-Speech**: espeak (local) + Google Cloud (cloud)
- **Speech-to-Text**: Vosk models (local) + Google Cloud (cloud)
- **Voice Selection**: English (US) and Spanish (Latin America)
- **Audio Generation**: WAV files with metadata
- **Real-time Processing**: Average 0.09s generation time
- **Quality Options**: Fast local vs. high-quality cloud

### 📚 Knowledge Management
- **Document Upload**: Drag-and-drop with progress tracking
- **File Support**: PDF, TXT, DOC, DOCX, MD formats
- **Document CRUD**: View, delete, organize operations
- **Metadata Display**: Size, type, upload date
- **Real-time Updates**: Document count and refresh

### 🔄 Provider Management
- **Hybrid System**: Local (privacy) vs Cloud (performance)
- **Real-time Switching**: Toggle between providers
- **Health Monitoring**: Service status indicators
- **Automatic Fallback**: Graceful degradation on failures
- **Status Tracking**: Live updates every 30 seconds

## 🔧 Technical Stack

### Frontend (Next.js 15.3.3)
- **Framework**: Next.js with App Router
- **Language**: TypeScript with strict typing
- **Styling**: Tailwind CSS 3.x
- **Icons**: Lucide React
- **HTTP Client**: Axios with interceptors
- **State Management**: React hooks + custom providers
- **Build Tool**: Turbopack for fast development

### Backend (FastAPI)
- **Framework**: FastAPI with auto-docs
- **Language**: Python 3.10 with Pydantic
- **Vector Store**: ChromaDB with persistence
- **Embeddings**: sentence-transformers
- **LLM**: Ollama (local) + Gemini (cloud)
- **Speech**: Vosk + espeak (local) + Google Cloud (cloud)
- **CORS**: Configured for frontend integration

### AI/ML Components
- **Local LLM**: Ollama + Mistral-7B (privacy-focused) - **Requires separate Ollama service**
- **Cloud LLM**: Google Gemini (performance-optimized)
- **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers)
- **STT Models**: Vosk English/Spanish (offline recognition)
- **TTS Engines**: espeak (local) + Google Neural (cloud)

### Service Architecture
```
Port 11434: Ollama LLM Service    (ollama serve)
Port 8000:  FastAPI Backend       (python -m app.main)
Port 3000:  Next.js Frontend      (npm run dev)
```

## 📊 System Performance

### Build Metrics (Latest)
```
✓ Compiled successfully in 14.0s
✓ Generating static pages (9/9)
✓ Bundle: 101kB shared JavaScript
✓ Pages: 5 interfaces, 0 errors
```

### Runtime Performance
- **Backend Startup**: 15-20 seconds (model loading)
- **Frontend Startup**: 3-5 seconds (Next.js compilation)
- **RAG Query Time**: 0.86s average (including retrieval)
- **Speech Generation**: 0.09s average
- **Document Retrieval**: Sub-second semantic search
- **System Health Checks**: 30-second intervals

### API Endpoints (24 Total)
- **RAG**: 7 endpoints (query, health, documents, etc.)
- **Speech**: 8 endpoints (TTS, STT, voices, stats, etc.)
- **Providers**: 3 endpoints (status, switch, config)
- **Knowledge**: 6 endpoints (upload, manage, search, etc.)

## 🔧 Troubleshooting

### Ollama Issues

**Problem**: Backend shows "Ollama not available" or RAG queries return mock responses
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start Ollama
ollama serve

# Check available models
ollama list

# If mistral:7b is missing, download it
ollama pull mistral:7b
```

**Problem**: Ollama model download is slow or fails
```bash
# Check disk space (model is ~4GB)
df -h

# Try pulling with specific version
ollama pull mistral:7b-instruct-v0.1

# Alternative: use smaller model (edit backend/app/services/rag_service.py)
ollama pull llama2:7b

# Keep model loaded in memory (prevents cold starts)
ollama pull mistral:7b
ollama run mistral:7b "Hello" # Loads model into memory
```

**Problem**: Backend can't connect to Ollama
```bash
# Check if port 11434 is in use
sudo lsof -i :11434

# Check Ollama logs
journalctl -u ollama --follow

# Restart Ollama
pkill ollama
ollama serve
```

### Speech Issues

**Problem**: "espeak not found" error
```bash
sudo apt update && sudo apt install -y espeak espeak-data libespeak1
```

**Problem**: Audio playback fails
- Check browser permissions for microphone/audio
- Verify backend is serving audio files correctly
- Test with: `curl http://127.0.0.1:8000/api/v1/speech/voices`

### General Issues

**Problem**: Port conflicts
```bash
# Kill processes using required ports
sudo lsof -ti:8000 | xargs kill  # Backend
sudo lsof -ti:3000 | xargs kill  # Frontend  
sudo lsof -ti:11434 | xargs kill # Ollama
```

## 🛠️ Development Guide

### Environment Setup
```bash
# Backend development
export AIEDU_DEBUG=true         # Enable debug mode
export AIEDU_RELOAD=true        # Auto-reload on changes

# Frontend development
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

### Development Workflow
1. **Edit Frontend**: Instant hot reload with state preservation
2. **Edit Backend**: Auto-restart with AIEDU_RELOAD=true
3. **Type Safety**: Full TypeScript integration across stack
4. **API Testing**: Live reload and reconnection

### Build Commands
```bash
# Frontend build
cd aiedu-frontend
npm run build        # Production build
npm run dev         # Development server

# Backend testing
cd backend
python -m pytest   # Run tests
uvicorn app.main:app --reload  # Development server
```

## 🎮 Usage Examples

### Unified Interface Usage
1. **Open**: http://localhost:3000
2. **Upload Documents**: Drag-and-drop files
3. **Ask Questions**: Type or use voice input
4. **Get Responses**: Text and audio output
5. **Monitor System**: Real-time health indicators

### API Usage
```bash
# Check system health
curl http://127.0.0.1:8000/health

# Query RAG system
curl -X POST http://127.0.0.1:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?"}'

# Generate speech
curl -X POST http://127.0.0.1:8000/api/v1/speech/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_id": "english_voice"}'
```

## ✅ Phase 6C Achievement Summary

### **All Requirements Achieved**:
- ✅ **Unified Interface**: Multi-modal experience complete
- ✅ **System Integration**: All components working together
- ✅ **Real-time Monitoring**: Health tracking operational
- ✅ **Production Ready**: Zero build errors, optimized bundles
- ✅ **Documentation**: Complete system documentation
- ✅ **Testing**: End-to-end verification successful

### **System Status**:
- **Interfaces**: 5 production-ready web interfaces
- **API Endpoints**: 24 FastAPI endpoints (100% operational)
- **Documents**: 13 loaded in vector database
- **Speech Services**: 100% success rate
- **Build Status**: 0 errors, optimized for production

### **Technology Achievement**:
- **Frontend**: Next.js 15 + TypeScript (modern React)
- **Backend**: FastAPI + Python (high-performance API)
- **AI/ML**: Complete RAG pipeline with speech integration
- **Architecture**: Microservices with hybrid provider system

## 🔍 System Monitoring

The system provides comprehensive real-time monitoring:

- **🟢 Healthy**: All services operational
- **🟡 Warning**: Service issues detected  
- **🔴 Error**: Service unavailable
- **📊 Metrics**: Response times, success rates, document counts

## 📄 License

MIT License - See LICENSE file for details.

## 👥 Contributors

**Andres Calvo** - [AndresACV](https://github.com/AndresACV)

---

**🎉 Phase 6C Complete: Production-Ready RAG System with Speech Interaction** 🚀

*Next.js 15 + FastAPI + TypeScript + AI/ML = Complete Educational Platform*