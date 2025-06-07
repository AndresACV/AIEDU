# AIEDU Local Deployment

## Overview

This is the **local deployment model** of the AIEDU RAG system, designed for privacy-focused, on-premises operation with full offline capabilities.

## Features

- **Complete Privacy**: All data processing happens locally
- **Offline Capable**: Works without internet connection after initial setup
- **No API Costs**: Zero recurring costs, only hardware
- **Full Control**: Complete customization and data control

## Technology Stack

- **LLM**: Ollama (local Llama models)
- **STT**: Local Whisper models
- **TTS**: espeak (Linux/WSL2) / pyttsx3 (Windows)
- **Vector Store**: ChromaDB (local)
- **Embeddings**: sentence-transformers (local)
- **Frontend**: Flask + Jinja2 templates

## Prerequisites

### System Requirements
- **OS**: Windows 10/11 with WSL2 or Ubuntu 20.04+
- **RAM**: Minimum 16GB (32GB recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- **Storage**: 50GB+ free space for models
- **Python**: 3.9 or 3.10

### Required Software
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Install espeak (Linux/WSL2)
sudo apt install espeak espeak-data libespeak1 alsa-utils

# Install FFmpeg
sudo apt install ffmpeg
```

## Installation

### 1. Clone and Setup
```bash
cd AIEDU/local_deployment
python -m venv venv
source venv/bin/activate  # Linux/WSL2
# or
venv\Scripts\activate     # Windows

pip install -r requirements_local.txt
```

### 2. Download AI Models
```bash
# Download Ollama model (4GB)
ollama pull mistral:7b

# Speech models are already included in web_app/models/
```

### 3. Configure Environment
```bash
cp .env.example .env.local
# Edit .env.local with your preferences
```

## Usage

### 1. Start Ollama Service
```bash
ollama serve
```

### 2. Run Application
```bash
cd web_app
python app.py
```

### 3. Access Interface
Open browser to: `https://localhost:5000`

## Current Performance
- **Startup Time**: ~50 seconds (includes model pre-loading)
- **Response Time**: 3-5 seconds per query
- **First Query**: Instant (models pre-loaded)
- **Memory Usage**: ~5.9GB VRAM (RTX 3070)
- **Token Generation**: 30-60 tokens/second

## Troubleshooting

### Common Issues
1. **Ollama not starting**: Check if port 11434 is available
2. **GPU not detected**: Verify NVIDIA drivers and CUDA installation
3. **TTS not working**: Install espeak system package
4. **Out of memory**: Reduce batch size or use smaller model

### Support
See main project documentation in `cursor_docs/` for detailed troubleshooting.

## Architecture Migration
This local deployment will remain fully functional during the cloud deployment development. Both architectures will coexist to provide deployment flexibility. 