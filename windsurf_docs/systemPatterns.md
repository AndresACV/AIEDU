# System Architecture and Design Patterns

## System Architecture
The RAG system with comprehensive speech interaction capabilities follows a modular architecture:

1. **Speech Input Module**: Captures and processes user voice input via browser
2. **Speech-to-Text Module**: Converts spoken language to text using Vosk models
3. **Text Input Module**: Handles direct text input as an alternative to speech
4. **Embedding Module**: Responsible for generating high-quality, structured embeddings
5. **Vector Database Module**: Stores and retrieves document embeddings
6. **Context Processing Module**: Assembles retrieved context for the LLM
7. **LLM Integration Module**: Manages interactions with the language model
8. **Text-to-Speech Module**: Converts text responses to speech using pyttsx3
9. **Output Delivery Module**: Manages the delivery of voice and text responses to users

## Key Technical Decisions
- Implement 100% offline speech recognition with Vosk
- Use client-side audio processing for better performance and privacy
- Provide real-time audio visualization for user feedback
- Support multilingual interaction (English and Spanish)
- Handle automatic model download and management
- Prioritize embedding quality over LLM complexity
- Use open-source models exclusively for speech processing
- Implement structured embeddings for enhanced semantic understanding
- Focus on educational output format and content
- Local computation without reliance on external APIs

## Core Framework and Module Interactions
- Flask-based web application serving HTML/JS frontend
- Client-server architecture with RESTful endpoints
- Browser-based audio capture and preprocessing
- Server-side speech recognition and synthesis
- Modular design with clear interfaces between components
- Event-driven communication between modules
- Asynchronous processing where appropriate
- Comprehensive error handling and fallback mechanisms
- Real-time feedback loops for audio processing
