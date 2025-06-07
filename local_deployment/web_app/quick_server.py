#!/usr/bin/env python3
"""
Quick server script that starts Flask immediately without heavy model loading
"""

import sys
import os
from flask import Flask, jsonify

# Add project path
sys.path.append(os.path.dirname(__file__))

# Create a minimal Flask app first
app = Flask(__name__)

# Add CORS
try:
    from flask_cors import CORS
    allowed_origins = []
    for port in range(3000, 3020):
        allowed_origins.extend([f'http://localhost:{port}', f'http://127.0.0.1:{port}'])
    CORS(app, origins=allowed_origins, methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
    print("✅ CORS configured")
except ImportError:
    print("⚠️ flask_cors not available")

# Add minimal endpoints for testing
@app.route('/current-providers')
def current_providers():
    return jsonify({
        "stt_provider": "local",
        "tts_provider": "local", 
        "llm_provider": "local",
        "status": "Backend is running",
        "providers": {
            "local": {"stt": "available", "tts": "available", "llm": "available"},
            "cloud": {"stt": "available", "tts": "available", "llm": "available"}
        }
    })

@app.route('/voices')
def voices():
    return jsonify([
        {"id": "spanish_voice", "name": "Spanish (Latin America)", "language_type": "Spanish"},
        {"id": "english_voice", "name": "English (US)", "language_type": "English"}
    ])

@app.route('/')
def index():
    return jsonify({"status": "AIEDU Backend is running", "version": "quick-start"})

if __name__ == "__main__":
    print("🚀 Starting AIEDU Quick Server...")
    print("✅ Ready on http://127.0.0.1:5000")
    app.run(
        debug=False,
        host='127.0.0.1',
        port=5000,
        use_reloader=False,
        threaded=True
    ) 