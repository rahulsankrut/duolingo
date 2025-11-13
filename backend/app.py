"""
FastAPI Application for Push-to-Talk Language Tutoring

This is the main application entry point. It sets up the FastAPI server with
WebSocket support for real-time bidirectional communication with the frontend.

The application provides:
- WebSocket endpoint for real-time audio streaming and transcription
- CORS support for frontend communication
- Health check endpoints
"""

import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Add parent directory to Python path for imports
# This allows us to import from the 'backend' package
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import API_HOST, API_PORT
from backend.websocket_handler import handle_websocket_connection

# ============================================================================
# Application Setup
# ============================================================================

# Create FastAPI application instance
app = FastAPI(title="Push-to-Talk Language Tutoring API")

# Configure CORS (Cross-Origin Resource Sharing)
# This allows the frontend (running on different port) to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# ============================================================================
# HTTP Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - health check.
    
    Returns:
        Dictionary with API name and status
    """
    return {"message": "Push-to-Talk Language Tutoring API", "status": "running"}


@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests.
    
    Browsers automatically request favicon.ico. This endpoint prevents 404 errors.
    
    Returns:
        Empty response with 204 status (No Content)
    """
    return Response(status_code=204)


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for real-time language tutoring.
    
    This endpoint handles the complete real-time flow:
    - Receives audio chunks from client
    - Transcribes speech to text
    - Gets tutor response from Gemini
    - Converts response to speech
    - Sends audio back to client
    
    Protocol:
    - Client sends: Audio chunks (binary) and control messages (JSON)
    - Server sends: Transcripts, Gemini responses, TTS audio, status, errors
    
    Args:
        websocket: WebSocket connection from FastAPI
    """
    await handle_websocket_connection(websocket)


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run the application using uvicorn
    # This is used when running directly: python backend/app.py
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
