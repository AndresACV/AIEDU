#!/usr/bin/env python3
"""
Test script to start the Flask server with HTTPS like the original app
"""

import sys
import os

# Ensure we can import the app
sys.path.append(os.path.dirname(__file__))

try:
    from app import app
    print("✅ Flask app imported successfully")
    
    # Check if SSL certificates exist
    cert_file = 'ssl/cert.pem'
    key_file = 'ssl/key.pem'
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("✅ SSL certificates found")
        # Start server with HTTPS
        print("🚀 Starting server on https://127.0.0.1:5000...")
        app.run(
            debug=False,  # No debug to avoid reload issues
            host='127.0.0.1',
            port=5000,
            ssl_context=(cert_file, key_file),
            use_reloader=False,  # No reloader to avoid crashes
            threaded=True
        )
    else:
        print("⚠️ SSL certificates not found, using HTTP")
        print("🚀 Starting server on http://127.0.0.1:5000...")
        app.run(
            debug=False,
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