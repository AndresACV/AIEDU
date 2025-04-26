# 🧠 AIEDU: RAG System with Speech Interaction
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-Latest-green)
![Vosk](https://img.shields.io/badge/Vosk-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<p align="center">
  <img src="assets/logo.jpg" alt="AIEDU Logo" width="400">
</p>

An advanced Retrieval-Augmented Generation (RAG) system with comprehensive speech interaction capabilities. This application combines high-quality embeddings with 100% offline speech recognition and synthesis to provide educational AI responses through an intuitive web interface.

## ✨ Features

- **🗣️ Complete Speech Interaction**: Fully functional text-to-speech and speech-to-text capabilities
- **🌐 Multilingual Support**: Works with both English and Spanish languages with easy language selection
- **💻 100% Offline Processing**: Uses local models for privacy and reliability
- **🔊 Real-time Audio Visualization**: Interactive volume display for user feedback
- **🔄 Client-side Audio Processing**: Robust Web Audio API implementation for WAV conversion
- **🧩 Modular Architecture**: Designed for easy extension and enhancement
- **🌟 Web-based Interface**: User-friendly Flask application for all functionalities

## 🏗️ Project Structure

```text
AIEDU/
├── web_app/                # Web application
│   ├── app.py              # Flask application entry point
│   ├── static/             # Static assets
│   │   └── audio/          # Generated audio files
│   ├── models/             # Local speech models
│   └── templates/          # HTML templates
│       └── index.html      # Web interface
│
├── models/                 # Speech recognition models
│   ├── vosk-model-en-us/   # English Vosk model
│   └── vosk-model-es/      # Spanish Vosk model
│
├── assets/                 # Images and other static assets
│   └── logo.jpg            # Project logo
│
├── windsurf_docs/          # Project documentation (excluded from git)
│   ├── activeContext.md    # Current development focus
│   ├── productContext.md   # Project purpose and background
│   ├── progress.md         # Project progress tracking
│   ├── systemPatterns.md   # System architecture
│   └── techContext.md      # Technical context and constraints
│
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## 🚀 Installation

### Prerequisites

1. Python 3.13 installed (tested with Python 3.13)
2. Create and activate a virtual environment:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🎮 Usage

Start the Flask web application:

```bash
python web_app/app.py
```

Open your browser to http://127.0.0.1:5000 to access the interface.

Use the web interface to:
- Type text and convert it to speech in multiple voices
- Record your voice and convert speech to text
- View real-time audio volume visualization
- Switch between English and Spanish languages

## 🎤 Speech Components

### 🔊 Text-to-Speech
- Multiple voice options in English and Spanish
- Real-time voice synthesis using Windows SAPI5 (pyttsx3)
- High-quality audio output
- 100% local processing

### 🎧 Speech-to-Text
- Offline speech recognition using Vosk models (based on Kaldi)
- Support for both English and Spanish with dedicated language selector
- Audio volume visualization with Web Audio API
- Robust client-side audio format conversion (WebM to WAV)
- Error handling and comprehensive user feedback
- Automatic model download and installation

## 🔍 Current Status

The system now features a fully functional voice interaction pipeline with the following recent improvements:

- ✅ Robust client-side audio recording and WAV conversion
- ✅ Language selection dropdown for speech recognition (English/Spanish)
- ✅ Improved error handling and user feedback for voice recording
- ✅ Enhanced .gitignore to exclude sensitive data and temporary files
- ✅ Clean repository structure with documentation isolated from version control

Both the text-to-speech and speech-to-text components are now stable and production-ready with a user-friendly web interface.

## 🛣️ Roadmap

- 🔍 Enhance speech recognition quality and accuracy
- 🧩 Implement the embedding strategy component for the RAG system
- 💾 Add conversation memory for contextual interactions
- 🧠 Develop the full RAG system with knowledge retrieval
- ⚡ Optimize performance for production environments
- 🌐 Add support for additional languages and voice models

## 📦 Requirements

- Python 3.13+
- Flask
- Vosk (for speech recognition)
- pyttsx3 (for text-to-speech)
- Other dependencies listed in requirements.txt

## 📄 License

See the LICENSE file for details.

## 👥 Contributors

Andres Calvo - [AndresACV](https://github.com/AndresACV)