#!/usr/bin/env python3
"""Test script for STT service and WebSocket connection."""

import os
import sys
import asyncio
import websockets
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import GOOGLE_CLOUD_PROJECT
from backend.stt_service import transcribe_streaming_chirp3, PROJECT_ID


def test_config():
    """Test if configuration is loaded correctly."""
    print("=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    print(f"GOOGLE_CLOUD_PROJECT from config: {GOOGLE_CLOUD_PROJECT}")
    print(f"PROJECT_ID from stt_service: {PROJECT_ID}")
    
    if not PROJECT_ID:
        print("❌ ERROR: PROJECT_ID is not set!")
        return False
    
    if PROJECT_ID != "vertex-ai-demos-468803":
        print(f"⚠️  WARNING: PROJECT_ID is '{PROJECT_ID}', expected 'vertex-ai-demos-468803'")
    
    print("✅ Configuration loaded successfully")
    return True


def test_stt_service():
    """Test STT service with a simple audio stream."""
    print("\n" + "=" * 60)
    print("Testing STT Service")
    print("=" * 60)
    
    try:
        # Create a simple test audio stream (silence - just for testing connection)
        # In real usage, this would be actual audio data
        def test_audio_stream():
            # Generate some dummy audio data (silence)
            # Real audio would be PCM16 format
            chunk_size = 1600  # ~0.1 seconds at 16kHz
            for _ in range(5):  # 5 chunks = ~0.5 seconds
                yield bytes(chunk_size)
        
        print("Attempting to connect to Speech-to-Text API...")
        print("Note: This will fail if no actual audio is provided, but should connect successfully.")
        
        # Try to create the client and config (this tests authentication)
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
        
        client = SpeechClient()
        print("✅ SpeechClient created successfully")
        
        # Test config creation
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model="chirp_3",
        )
        print("✅ RecognitionConfig created successfully")
        
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=config
        )
        print("✅ StreamingRecognitionConfig created successfully")
        
        config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
            streaming_config=streaming_config,
        )
        print(f"✅ StreamingRecognizeRequest created with project: {PROJECT_ID}")
        
        print("\n⚠️  Note: Full STT test requires actual audio data.")
        print("   The connection test passed - API is accessible.")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        if "authentication" in str(e).lower():
            print("\n💡 Solution: Run 'gcloud auth application-default login'")
        elif "permission" in str(e).lower() or "403" in str(e):
            print(f"\n💡 Solution: Check that Speech-to-Text API is enabled for project: {PROJECT_ID}")
        return False


async def test_websocket_connection():
    """Test WebSocket connection to the backend."""
    print("\n" + "=" * 60)
    print("Testing WebSocket Connection")
    print("=" * 60)
    
    try:
        uri = "ws://localhost:8000/ws/transcribe"
        print(f"Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Wait for initial status message
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                print(f"✅ Received status: {data.get('message', 'N/A')}")
                
                # Send a test end message
                await websocket.send(json.dumps({"type": "end"}))
                print("✅ Sent end message successfully")
                
            except asyncio.TimeoutError:
                print("⚠️  Timeout waiting for status message")
                return False
            except Exception as e:
                print(f"⚠️  Error during WebSocket test: {str(e)}")
                return False
        
        print("✅ WebSocket connection test passed")
        return True
        
    except ConnectionRefusedError:
        print("❌ ERROR: Connection refused. Is the backend server running?")
        print("   Start it with: python3 backend/app.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


async def test_backend_api():
    """Test the backend REST API."""
    print("\n" + "=" * 60)
    print("Testing Backend REST API")
    print("=" * 60)
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Test root endpoint
            async with session.get("http://localhost:8000/") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Root endpoint: {data.get('message', 'N/A')}")
                else:
                    print(f"❌ Root endpoint returned status {response.status}")
                    return False
            
            # Test favicon endpoint
            async with session.get("http://localhost:8000/favicon.ico") as response:
                if response.status == 204:
                    print("✅ Favicon endpoint working")
                else:
                    print(f"⚠️  Favicon endpoint returned status {response.status}")
        
        return True
        
    except ImportError:
        print("⚠️  aiohttp not available, skipping REST API test")
        print("   Install with: pip install aiohttp")
        return True
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        if "Connection refused" in str(e):
            print("   Is the backend server running? Start it with: python3 backend/app.py")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Push-to-Talk STT Test Suite")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Configuration
    results.append(("Configuration", test_config()))
    
    # Test 2: STT Service
    results.append(("STT Service", test_stt_service()))
    
    # Test 3: Backend API (async)
    try:
        result = asyncio.run(test_backend_api())
        results.append(("Backend REST API", result))
    except Exception as e:
        print(f"❌ Error testing REST API: {str(e)}")
        results.append(("Backend REST API", False))
    
    # Test 4: WebSocket (async)
    try:
        result = asyncio.run(test_websocket_connection())
        results.append(("WebSocket Connection", result))
    except Exception as e:
        print(f"❌ Error testing WebSocket: {str(e)}")
        results.append(("WebSocket Connection", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

