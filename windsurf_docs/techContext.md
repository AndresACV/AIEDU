# Technical Context

## Technologies and Libraries in Use
- **Python 3.13**: Primary programming language
- **Flask**: Web framework for the application
- **pyttsx3**: Text-to-speech library using Windows SAPI5
- **Vosk**: Offline speech recognition toolkit (based on Kaldi)
- **SpeechRecognition**: Python library for speech recognition (used as fallback)
- **JavaScript**: Client-side audio processing and interface
- **Web Audio API**: Browser-based audio recording and processing
- **Bootstrap**: Frontend UI framework for responsive design
- **jQuery**: JavaScript library for DOM manipulation
- **Fontawesome**: Icon library for UI elements

## Development Environment and Setup
- Windows-based local development environment
- Python virtual environment for dependency management
- 100% local computation without reliance on external APIs
- Flask development server in debug mode
- Git for version control

## Technical Constraints and Dependencies
- Must use open-source models only (no proprietary APIs)
- Focus on 100% offline speech processing
- Support for multiple languages (currently English and Spanish)
- Browser compatibility for audio recording and processing
- Efficient model loading and management
- Automatic download and setup of speech recognition models
- Real-time audio feedback and visualization
