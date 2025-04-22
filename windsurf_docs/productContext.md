# RAG System with Speech Interaction Capabilities

## Project Purpose and Background
This project aims to create a Retrieval-Augmented Generation (RAG) system with comprehensive speech interaction capabilities that emphasizes quality embeddings for knowledge retrieval and educational AI responses. The system can process both text and speech input, leverage optimized embeddings to retrieve relevant information, and generate educational responses that can be delivered as both text and voice using open-source models.

## Problems the Project Solves
1. Limited educational AI systems with bidirectional speech capabilities
2. Dependency on cloud services for speech recognition and synthesis
3. Privacy concerns with sending voice data to external APIs
4. Over-reliance on LLMs without optimized embedding strategies
5. Need for structured, semantically rich embeddings in educational contexts
6. Lack of multilingual speech interaction systems for education

## High-level Workflow and Intended Functionality
1. **User Input**: System accepts both speech and text queries from users
2. **Speech Recognition**: If input is speech, it is processed by Vosk for offline speech-to-text conversion
3. **Embedding and Retrieval**: Query is embedded and used to retrieve relevant information from a vector database
4. **LLM Processing**: Retrieved context is used by an LLM to generate educational responses
5. **Text-to-Speech Conversion**: Generated text is converted to natural-sounding voice using open-source models
6. **Response Delivery**: Both text and voice output are delivered to the user

The project emphasizes that "La clave no es el LLM. Lo clave es el embedding" (The key is not the LLM, the key is the embedding), focusing on optimizing embedding strategies for better knowledge retrieval rather than just improving the LLM component. The system prioritizes privacy and offline processing capabilities for all speech interaction.
