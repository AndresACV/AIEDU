# RAG System with Speech Interaction

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system with comprehensive speech interaction capabilities that emphasizes quality embeddings for knowledge retrieval and educational AI responses. It features both text-to-speech and speech-to-text functionality using 100% offline, open-source models.

> "La clave no es el LLM. Lo clave es el embedding." — The key is not the LLM, the key is the embedding.

## Current Status

Both the text-to-speech and speech-to-text components have been implemented with a web-based interface. The system provides high-quality voice synthesis using Windows SAPI5 (pyttsx3) and robust offline speech recognition using Vosk (based on the Kaldi speech recognition toolkit).

## Quick Start

### Prerequisites

1. Python 3.13 installed (tested with Python 3.13)
2. Virtual environment setup:

```bash
# Activate the virtual environment
.\venv\Scripts\activate  # Windows
```

3\. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Web Application

The web application provides a complete interface for both text-to-speech and speech-to-text functionality:

```bash
python web_app/app.py
```

Open your browser to http://127.0.0.1:5000 to access the interface.

### Features

#### Text-to-Speech
- Multiple voice options in English and Spanish
- Real-time voice synthesis
- High-quality audio output
- 100% local processing

#### Speech-to-Text
- Offline speech recognition using Vosk models
- Support for both English and Spanish
- Audio volume visualization
- Client-side audio format conversion
- Automatic model download and installation

## Project Structure

```text
├── requirements.txt         # Project dependencies
├── models/                  # Speech recognition models
│   ├── vosk-model-en-us/    # English Vosk model
│   └── vosk-model-es/       # Spanish Vosk model
├── web_app/                 # Web application
│   ├── app.py               # Flask application entry point
│   ├── static/              # Static assets
│   │   └── audio/           # Generated audio files
│   ├── models/              # Local speech models
│   └── templates/           # HTML templates
│       ├── index.html       # Original interface template
│       └── index_fixed.html # Current web interface
└── windsurf_docs/           # Project documentation
    ├── activeContext.md     # Current development focus
    ├── productContext.md    # Project purpose and background
    ├── progress.md          # Project progress tracking
    ├── systemPatterns.md    # System architecture
    └── techContext.md       # Technical context and constraints
```

### Key Files

- `web_app/app.py`: Main Flask application with endpoints for speech synthesis and recognition
- `web_app/templates/index_fixed.html`: Web interface with both TTS and STT functionality
- `models/`: Contains Vosk speech recognition models (downloaded automatically if not present)

## Next Steps

- Enhance speech recognition quality and accuracy
- Implement the embedding strategy component for the RAG system
- Add dialog memory for contextual conversations
- Develop the full RAG system with knowledge retrieval
- Optimize performance for production environments
- Add support for additional languages and voice models

## License

See the LICENSE file for details.