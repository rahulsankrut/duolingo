"""
WebSocket Utility Functions

This module provides helper functions for WebSocket communication between
the frontend and backend. It handles sending different types of messages
(status, transcripts, Gemini responses, TTS audio, errors) and receiving
audio chunks from the client.

All functions are designed to work with FastAPI's WebSocket implementation.
"""

import asyncio
import json
import queue as sync_queue
from typing import Callable, Optional
from fastapi import WebSocket


# ============================================================================
# Message Type Constants
# ============================================================================

# These constants define the types of messages we can send to the frontend
# Using constants prevents typos and makes the code more maintainable
MSG_TYPE_STATUS = "status"              # Status updates (e.g., "Connected")
MSG_TYPE_TRANSCRIPT = "transcript"      # Speech transcription results
MSG_TYPE_GEMINI_RESPONSE = "gemini_response"  # Gemini tutor's text response
MSG_TYPE_TTS_AUDIO = "tts_audio"       # Header for TTS audio data
MSG_TYPE_ERROR = "error"                # Error messages


# ============================================================================
# Message Sending Functions
# ============================================================================

async def send_status(websocket: WebSocket, message: str) -> None:
    """Send a status message to the client.
    
    Status messages inform the user about connection state or processing status.
    Example: "Connected. Ready for audio." or "Processing..."
    
    Args:
        websocket: The WebSocket connection to send to
        message: The status message to send
    """
    if websocket.client_state.name == "CONNECTED":
        await websocket.send_json({
            "type": MSG_TYPE_STATUS,
            "message": message
        })


async def send_transcript(
    websocket: WebSocket,
    text: str,
    is_final: bool
) -> None:
    """Send a transcription result to the client.
    
    Transcripts can be interim (partial, still being processed) or final (complete).
    The frontend displays interim results in italics and final results normally.
    
    Args:
        websocket: The WebSocket connection to send to
        text: The transcribed text
        is_final: True if this is the final transcription, False if interim
    """
    if websocket.client_state.name == "CONNECTED" and text:
        await websocket.send_json({
            "type": MSG_TYPE_TRANSCRIPT,
            "text": text,
            "is_final": is_final
        })


async def send_gemini_response(websocket: WebSocket, response: str) -> None:
    """Send Gemini's text response to the client.
    
    This is the tutor's response that will be displayed in the Gemini Response box
    on the frontend. It's also the text that will be converted to speech.
    
    Args:
        websocket: The WebSocket connection to send to
        response: The tutor's response text
    """
    if websocket.client_state.name == "CONNECTED" and response:
        await websocket.send_json({
            "type": MSG_TYPE_GEMINI_RESPONSE,
            "text": response
        })


async def send_tts_audio(websocket: WebSocket, audio_content: bytes) -> None:
    """Send TTS audio content to the client for playback.
    
    This function sends audio in two parts:
    1. First sends a JSON header with metadata (size, sample rate, format)
    2. Then sends the actual audio data as binary
    
    The frontend uses the header to know that binary data is coming and how to play it.
    
    Args:
        websocket: The WebSocket connection to send to
        audio_content: Audio data as bytes (LINEAR16 PCM format, 24kHz)
    """
    # Validate audio content
    if not audio_content:
        print("Warning: Empty audio content, skipping TTS send")
        return
        
    # Check connection is still active
    if websocket.client_state.name != "CONNECTED":
        print(f"Warning: WebSocket not connected (state: {websocket.client_state.name}), skipping TTS send")
        return
        
    try:
        # Step 1: Send JSON header to tell frontend audio is coming
        # Frontend uses this to prepare for binary data
        await websocket.send_json({
            "type": MSG_TYPE_TTS_AUDIO,
            "size": len(audio_content),
            "sample_rate": 24000,
            "format": "LINEAR16"
        })
        
        # Step 2: Send the actual audio data as binary
        await websocket.send_bytes(audio_content)
        print(f"Successfully sent TTS audio ({len(audio_content)} bytes)")
    except Exception as e:
        # Log error but don't crash
        error_msg = str(e) if e else "Unknown error"
        print(f"Error sending TTS audio: {error_msg}")
        import traceback
        traceback.print_exc()


async def send_error(websocket: WebSocket, message: str) -> None:
    """Send an error message to the client.
    
    Error messages inform the user when something goes wrong, like authentication
    failures or API errors.
    
    Args:
        websocket: The WebSocket connection to send to
        message: The error message to send
    """
    if websocket.client_state.name == "CONNECTED":
        try:
            await websocket.send_json({
                "type": MSG_TYPE_ERROR,
                "message": message
            })
        except Exception:
            # Connection may be closed, ignore silently
            pass


# ============================================================================
# Error Formatting
# ============================================================================

def format_authentication_error(error_msg: str) -> str:
    """Format authentication errors to be more user-friendly.
    
    Google Cloud authentication errors can be technical. This function makes
    them more understandable and provides clear instructions.
    
    Args:
        error_msg: The raw error message from the API
    
    Returns:
        A user-friendly error message with instructions
    """
    if "authentication" in error_msg.lower() or "reauthenticate" in error_msg.lower():
        return (
            "Google Cloud authentication required. "
            "Please run: gcloud auth application-default login"
        )
    return error_msg


# ============================================================================
# Audio Receiving Functions
# ============================================================================

async def receive_audio_chunks(
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    streaming_active: dict,
    language_state: dict,
    tts_model_state: dict
) -> None:
    """Receive audio chunks from the client and put them in a queue.
    
    This function runs in a loop, continuously receiving data from the WebSocket.
    It handles two types of data:
    - Binary data: Audio chunks from the microphone
    - Text data: Control messages (like "end" or "set_language")
    
    Args:
        websocket: The WebSocket connection to receive from
        audio_queue: Queue to put audio chunks into (for STT processing)
        streaming_active: Dictionary with 'value' key to control streaming state
                         When set to False, stops receiving
        language_state: Dictionary with 'value' key to store selected language
                        Updated when client sends language change message
        tts_model_state: Dictionary with 'value' key to store selected TTS model
                        Updated when client sends TTS model change message
    """
    while streaming_active["value"]:
        try:
            # Check if connection is still active
            if websocket.client_state.name != "CONNECTED":
                streaming_active["value"] = False
                break
            
            # Wait for data from client (with 1 second timeout)
            # Timeout allows us to check streaming_active periodically
            data = await asyncio.wait_for(websocket.receive(), timeout=1.0)
            
            # Handle binary data (audio chunks)
            if "bytes" in data:
                # Put audio chunk in queue for STT processing
                await audio_queue.put(data["bytes"])
            
            # Handle text data (control messages)
            elif "text" in data:
                message = json.loads(data["text"])
                
                # Client signals end of recording
                if message.get("type") == "end":
                    streaming_active["value"] = False
                    await audio_queue.put(None)  # Sentinel value to signal end
                    break
                
                # Client changes language selection
                elif message.get("type") == "set_language":
                    # Update the language state
                    language_state["value"] = message.get("language", "Spanish")
                    print(f"Language updated to: {language_state['value']}")
                elif message.get("type") == "set_tts_model":
                    # Update TTS model state
                    tts_model_state["value"] = message.get("model", "flash")
                    print(f"TTS model updated to: {tts_model_state['value']}")
                    
        except asyncio.TimeoutError:
            # Timeout is normal - just continue waiting
            continue
        except (RuntimeError, Exception) as e:
            # Handle disconnection or other errors
            error_msg = str(e)
            if "disconnect" in error_msg.lower() or "close" in error_msg.lower():
                streaming_active["value"] = False
                await audio_queue.put(None)
                break
            print(f"Error receiving audio: {error_msg}")
            streaming_active["value"] = False
            await audio_queue.put(None)
            break


# ============================================================================
# Queue Management Functions
# ============================================================================

async def transfer_audio_to_sync_queue(
    audio_queue: asyncio.Queue,
    sync_audio_queue: sync_queue.Queue,
    streaming_active: dict
) -> None:
    """Transfer audio chunks from async queue to sync queue.
    
    The STT API is blocking (synchronous), but we receive audio in an async queue.
    This function bridges the gap by transferring chunks from the async queue
    to a synchronous queue that the STT thread can read from.
    
    Args:
        audio_queue: Async queue receiving audio chunks from WebSocket
        sync_audio_queue: Synchronous queue for STT processing thread
        streaming_active: Dictionary with 'value' key to control streaming state
    """
    while streaming_active["value"]:
        try:
            # Get chunk from async queue (with short timeout)
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
            # Put it in sync queue for STT thread
            sync_audio_queue.put(chunk)
            
            # None is a sentinel value that means "end of stream"
            if chunk is None:
                break
        except asyncio.TimeoutError:
            # If streaming stopped, signal end to sync queue
            if not streaming_active["value"]:
                sync_audio_queue.put(None)
                break
            continue


def create_audio_stream_generator(
    sync_audio_queue: sync_queue.Queue,
    streaming_active: dict
) -> Callable[[], bytes]:
    """Create a generator function that yields audio chunks from a sync queue.
    
    The STT API expects an iterator/generator of audio chunks. This function
    creates a generator that reads from a synchronous queue and yields chunks.
    
    Args:
        sync_audio_queue: Synchronous queue containing audio chunks
        streaming_active: Dictionary with 'value' key to control streaming state
    
    Returns:
        A function that, when called, returns a generator yielding audio chunks
    """
    def audio_stream():
        """Generator that yields audio chunks from the sync queue."""
        while True:
            try:
                # Get chunk from queue (with timeout to allow checking streaming_active)
                chunk = sync_audio_queue.get(timeout=0.1)
                # None means end of stream
                if chunk is None:
                    break
                # Yield the chunk to STT API
                yield chunk
            except sync_queue.Empty:
                # If streaming stopped, exit generator
                if not streaming_active["value"]:
                    break
                continue
    
    return audio_stream


# ============================================================================
# Response Processing Functions
# ============================================================================

def extract_transcript_from_response(response) -> Optional[tuple[str, bool]]:
    """Extract transcript text and finality flag from STT API response.
    
    The STT API returns complex response objects. This function extracts
    the actual transcript text and whether it's final or interim.
    
    Args:
        response: StreamingRecognizeResponse object from STT API
    
    Returns:
        Tuple of (transcript_text, is_final) if transcript found, None otherwise
        is_final: True if transcription is complete, False if still processing
    """
    # Check if response has results
    if not hasattr(response, 'results') or not response.results:
        return None
    
    # Process each result in the response
    for result in response.results:
        # Get alternatives (different possible transcriptions)
        alternatives = getattr(result, 'alternatives', [])
        if alternatives and len(alternatives) > 0:
            # Get transcript from first (best) alternative
            transcript = getattr(alternatives[0], 'transcript', '')
            if transcript:
                # Check if this is a final result or interim
                is_final = getattr(result, 'is_final', False)
                return (transcript, is_final)
    
    # No transcript found in response
    return None
