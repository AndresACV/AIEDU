# 🧠 AIEDU: RAG System with Speech Interaction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-Production-green)
![Ollama](https://img.shields.io/badge/Ollama-Mistral--7B-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<p align="center">
  <img src="assets/logo.jpg" alt="AIEDU Logo" width="400">
</p>

A Retrieval-Augmented Generation (RAG) system with comprehensive speech interaction capabilities. This educational AI platform combines high-quality embeddings, 100% offline speech processing, and local LLM inference through an intuitive web interface.

## ✨ Key Features

- **🧠 Complete RAG System**: ChromaDB vector store + sentence-transformers + Ollama LLM
- **🗣️ Full Speech Interaction**: Text-to-speech and speech-to-text with real-time processing
- **🌐 Multilingual Support**: English (US) and Spanish (Latin America) with optimized voices
- **💻 WSL2 Compatible**: Native espeak integration for Linux environments
- **🚀 Production Optimized**: Fast 45-second startup, instant question responses
- **💻 100% Local Processing**: No external APIs, complete privacy
- **⚡ GPU Accelerated**: RTX 3070 optimized with automatic Ollama acceleration
- **📚 Knowledge Management**: Add, update, delete documents with metadata
- **🧩 Conversation Memory**: Context-aware responses across interactions
- **🔒 HTTPS Ready**: SSL certificates with production deployment

## 📊 Performance Metrics

- **Startup Time**: ~45 seconds (optimized from 5+ minutes)
- **First Question**: Instant response (optimized from 2+ minutes)
- **LLM Inference**: 69 tokens/second (RTX 3070)
- **Memory Usage**: ~5.9 GiB VRAM efficiently managed
- **Speech Recognition**: >95% accuracy (Vosk models)
- **System Reliability**: Production-stable, crash-free

## 🏗️ Architecture

```text
AIEDU/
├── web_app/                    # Main Flask application
│   ├── app.py                  # Production-optimized Flask app
│   ├── embeddings.py           # sentence-transformers integration
│   ├── vector_store.py         # ChromaDB vector database  
│   ├── llm.py                  # Ollama API integration
│   ├── rag_pipeline.py         # Complete RAG implementation
│   ├── models/                 # Pre-installed speech models (required)
│   │   ├── vosk-model-small-en-us-0.15/  # English Vosk model (~40MB)
│   │   └── vosk-model-small-es-0.42/     # Spanish Vosk model (~40MB)
│   ├── static/                 # Web assets and generated audio
│   ├── ssl/                    # HTTPS certificates
│   └── templates/
│       └── index.html          # Responsive web interface
│
├── vector_db/                  # ChromaDB persistent storage
├── cursor_docs/                # Complete project documentation
│   ├── activeContext.md        # Current optimized status
│   ├── progress.md             # Development achievements
│   ├── productContext.md       # Project vision
│   ├── systemPatterns.md       # Technical architecture
│   └── techContext.md          # Implementation details
│
├── requirements.txt            # Streamlined dependencies
└── README.md                   # This documentation
```

## 🚀 Quick Start

### Prerequisites

1. **System Requirements**:
   - Python 3.10+ (tested with 3.10)
   - NVIDIA GPU (RTX 3070 optimized, others supported)
   - Windows WSL2/Ubuntu or native Linux
   - 8GB+ RAM recommended

2. **Install System Dependencies**:
   
   **For Linux/WSL2** (required for TTS):
   ```bash
   # Install espeak for text-to-speech
   sudo apt update
   sudo apt install -y espeak espeak-data libespeak1 alsa-utils
   
   # Test espeak installation
   espeak "Hello world"
   ```
   
   **For Windows** (TTS works automatically):
   ```bash
   # No additional TTS dependencies needed
   # Windows SAPI will be used automatically
   ```

3. **Install Ollama** (required for LLM):
   ```bash
   # Linux/WSL2
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows
   # Download from https://ollama.ai/download
   ```

4. **Setup Python Environment**:
   ```bash
   # Clone and setup
   git clone <repository-url>
   cd AIEDU
   python -m venv venv
   source venv/bin/activate  # Linux/WSL2
   # venv\Scripts\activate   # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

5. **Install Speech Recognition Models**:
   ```bash
   # The application requires pre-installed Vosk models in web_app/models/
   # These models are already included in the repository:
   
   # Required models:
   # - web_app/models/vosk-model-small-en-us-0.15/  (English)
   # - web_app/models/vosk-model-small-es-0.42/     (Spanish)
   
   # Verify models are present:
   ls -la web_app/models/
   # Should show both model directories
   
   # If models are missing, download them manually:
   # English model (~40MB):
   # wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   # unzip vosk-model-small-en-us-0.15.zip -d web_app/models/
   
   # Spanish model (~40MB):
   # wget https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
   # unzip vosk-model-small-es-0.42.zip -d web_app/models/
   ```

### Launch Application

1. **Start Ollama** (in separate terminal):
   ```bash
   ollama serve
   ollama pull mistral:7b  # Downloads ~4.1GB model
   ```

2. **Start AIEDU** (production optimized):
   ```bash
   python -m web_app.app
   
   # Wait for startup sequence:
   # 🎤 Speech models: EN(✅) ES(✅)           # ~5s
   # 🧠 Loading embedding model (~30s)...     # ~30s  
   # ✅ Embedding model loaded and ready
   # 🤖 Checking Ollama connection...         # ~2s
   # ✅ Ollama is running
   # 🔗 Initializing RAG pipeline...
   # ✅ RAG pipeline ready
   # 🌐 Server ready in 45s - https://127.0.0.1:5000
   ```

3. **Access Interface**: Open https://127.0.0.1:5000

## 🎯 Main Features

### 📚 Knowledge Base Management
- **Add Documents**: Upload text content with metadata
- **Query RAG System**: Ask questions about your documents
- **Conversation Memory**: Context-aware multi-turn conversations
- **Document Management**: View, update, delete stored knowledge

### 🎤 Speech Interaction
- **Speech-to-Text**: Record voice → automatic transcription → RAG query
- **Text-to-Speech**: Text input → RAG response → spoken answer
- **Language Support**: English (US) and Spanish (Latin America)
- **Voice Quality**: Optimized espeak parameters for natural speech
- **WSL2 Compatible**: Direct espeak integration for Linux environments
- **Audio Processing**: WebM/WAV support with real-time feedback

### 🔧 Technical Stack

#### **RAG Components**
- **Vector Store**: ChromaDB with persistent storage
- **Embeddings**: sentence-transformers (GPU accelerated)
- **LLM**: Ollama + Mistral-7B (local inference)
- **Retrieval**: Similarity search with metadata filtering

#### **Speech Processing** 
- **STT**: Vosk models (100% offline, Kaldi-based)
- **TTS**: Multi-platform text-to-speech
  - **Linux/WSL2**: espeak (Spanish Latin America, English variants)
  - **Windows**: SAPI5 (native Windows voices)
- **Voice Selection**: Simplified to English (US) and Spanish (Latin America)
- **Audio**: FFmpeg conversion, Web Audio API, WAV output

#### **Web Framework**
- **Backend**: Flask (production-optimized)
- **Frontend**: Responsive HTML/CSS/JS
- **Security**: HTTPS with self-signed certificates
- **Deployment**: Single-process, memory-safe

## 🎮 Usage Examples

### Text-based RAG Query
```bash
# Add a document
curl -X POST https://127.0.0.1:5000/add_document \
  -H "Content-Type: application/json" \
  -d '{"text": "Python is a programming language", "metadata": {"topic": "programming"}}'

# Query the knowledge base  
curl -X POST https://127.0.0.1:5000/rag_query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

### Speech-to-RAG Workflow
1. Click 🎤 **Record** button
2. Speak your question clearly
3. System transcribes → queries RAG → responds
4. Click 🔊 to hear the answer spoken aloud

### Knowledge Base Stats
```bash
curl https://127.0.0.1:5000/kb_stats
# Returns: document count, collection status, system health
```

## 🛠️ Configuration

### Environment Variables
```bash
# Optional: Override default production settings
export AIEDU_PRODUCTION=true    # Default: true (optimized)
export AIEDU_DEBUG=false        # Default: false (production)
```

### Performance Tuning
- **GPU Memory**: Automatically optimized for RTX 3070
- **Threading**: Single-threaded for stability
- **Model Loading**: Eager loading during startup
- **API Timeouts**: 30-second Ollama timeout with fallback

## 🔍 System Status

### ✅ Production Ready Features
- **Core RAG Pipeline**: 100% functional with optimized performance
- **Speech Interaction**: Full bidirectional voice processing
  - **TTS Fixed**: WSL2 compatibility achieved with direct espeak integration
  - **Voice Selection**: Simplified to 2 high-quality voices (EN-US, ES-LA)
  - **Audio Quality**: Optimized espeak parameters for natural speech
- **Knowledge Management**: Complete CRUD operations
- **Web Interface**: Production-stable with HTTPS
- **Error Handling**: Graceful failures with clear messaging
- **Memory Management**: Crash-free operation with safety measures

## 🔧 Troubleshooting

### Common Issues

**Ollama Connection Failed**:
```bash
# Ensure Ollama is running
ollama serve

# Verify model is available
ollama list
ollama pull mistral:7b  # If missing
```

**GPU Memory Issues**:
- System automatically manages GPU memory
- Monitor with: `nvidia-smi`
- Restart app if memory fragmentation occurs

**Speech Models Missing**:
- Models auto-download on first use
- Verify internet connection for initial setup
- Check `web_app/models/` directory

**TTS Not Working (Linux/WSL2)**:
```bash
# Install espeak if missing
sudo apt update
sudo apt install -y espeak espeak-data libespeak1

# Test espeak
espeak "Hello world"

# Check available voices
espeak --voices
```

## 📦 Dependencies

**Core Requirements**:
- Flask (web framework)
- sentence-transformers (embeddings)
- chromadb (vector store)
- requests (Ollama API)
- vosk (speech recognition)
- pyttsx3 (text-to-speech)

**GPU Support**:
- torch (PyTorch with CUDA)
- NVIDIA drivers (RTX 3070 optimized)

See `requirements.txt` for complete dependency list.

## 📄 License

MIT License - See LICENSE file for details.

## 👥 Contributors

**Andres Calvo** - [AndresACV](https://github.com/AndresACV)

---