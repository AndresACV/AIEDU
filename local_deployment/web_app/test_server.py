#!/usr/bin/env python3
"""
Simple test script to start the Flask server with minimal configuration
"""

import sys
import os

# Ensure we can import the app
sys.path.append(os.path.dirname(__file__))

try:
    from app import app
    print("✅ Flask app imported successfully")
    
    # Start server with basic configuration
    print("🚀 Starting server on http://127.0.0.1:5000...")
    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000,
        use_reloader=False,
        threaded=True
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Server error: {e}")
    import traceback
    traceback.print_exc() 