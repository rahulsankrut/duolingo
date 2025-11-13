"""
WebSocket Handler for Real-Time Language Tutoring

This module orchestrates the complete flow:
1. Receives audio from client
2. Transcribes speech to text (STT)
3. Sends transcript to Gemini for tutor response
4. Converts Gemini response to speech (TTS)
5. Sends audio back to client

All processing happens in real-time using WebSocket for bidirectional communication.
"""

import asyncio
import concurrent.futures
import queue as sync_queue
import time
from fastapi import WebSocket, WebSocketDisconnect

from backend.stt_service import transcribe_streaming_chirp3
from backend.gemini_service import process_transcript_async, process_transcript_streaming_async
from backend.tts_service import synthesize_speech_async, synthesize_speech_streaming_async
from backend.websocket_utils import (
    send_status,
    send_transcript,
    send_gemini_response,
    send_gemini_chunk,
    send_tts_audio,
    send_tts_chunk,
    send_latency,
    send_error,
    format_authentication_error,
    receive_audio_chunks,
    transfer_audio_to_sync_queue,
    create_audio_stream_generator,
    extract_transcript_from_response,
)


# ============================================================================
# Main WebSocket Connection Handler
# ============================================================================

async def handle_websocket_connection(websocket: WebSocket) -> None:
    """Handle a new WebSocket connection for real-time language tutoring.
    
    This is the main entry point for WebSocket connections. It sets up the
    complete pipeline:
    - Audio reception from client
    - Speech-to-text transcription
    - Gemini processing for tutor responses
    - Text-to-speech synthesis
    - Audio playback to client
    
    Protocol:
    - Client sends: Audio chunks (binary) and control messages (JSON)
    - Server sends: Transcripts, Gemini responses, TTS audio, status, errors
    
    Args:
        websocket: The WebSocket connection from FastAPI
    """
    # Accept the WebSocket connection
    await websocket.accept()
    
    try:
        # Send initial status to let client know we're ready
        await send_status(websocket, "Connected. Ready for audio.")
        
        # ====================================================================
        # State Management
        # ====================================================================
        # Use dictionaries with 'value' key so they can be modified by reference
        # This allows multiple async tasks to share state
        
        audio_queue = asyncio.Queue()  # Queue for audio chunks from client
        response_queue = asyncio.Queue()  # Queue for STT transcription results
        streaming_active = {"value": True}  # Controls when to stop processing
        language_state = {"value": "Spanish"}  # Current selected language
        tts_model_state = {"value": "flash"}  # Current selected TTS model
        
        # ====================================================================
        # Start Processing Tasks
        # ====================================================================
        # Run two tasks in parallel:
        # 1. Receive audio chunks from client
        # 2. Process audio through STT -> Gemini -> TTS pipeline
        
        receive_task = asyncio.create_task(
            receive_audio_chunks(websocket, audio_queue, streaming_active, language_state, tts_model_state)
        )
        
        stt_task = asyncio.create_task(
            process_stt_streaming(websocket, audio_queue, response_queue, streaming_active, language_state, tts_model_state)
        )
        
        # Wait for both tasks to complete
        try:
            await asyncio.gather(receive_task, stt_task)
        except Exception as e:
            print(f"Error in WebSocket tasks: {e}")
            
    except WebSocketDisconnect:
        print("WebSocket disconnected by client")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await send_error(websocket, str(e))
    finally:
        # Cleanup: stop streaming and close connection
        streaming_active["value"] = False
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.close()
        except Exception:
            pass  # Connection may already be closed


# ============================================================================
# STT Processing Pipeline
# ============================================================================

async def process_stt_streaming(
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    response_queue: asyncio.Queue,
    streaming_active: dict,
    language_state: dict,
    tts_model_state: dict
) -> None:
    """Process audio through STT and send results to client in real-time.
    
    This function orchestrates the STT processing:
    1. Transfers audio from async queue to sync queue (for blocking STT API)
    2. Runs STT in a separate thread (so it doesn't block async operations)
    3. Processes STT responses and sends them to client
    4. For final transcripts, triggers Gemini and TTS processing
    
    Args:
        websocket: WebSocket connection for sending results
        audio_queue: Async queue receiving audio chunks from client
        response_queue: Queue for STT transcription results
        streaming_active: Dictionary controlling streaming state
        language_state: Dictionary storing selected language
    """
    try:
        # ====================================================================
        # Bridge Async and Sync Worlds
        # ====================================================================
        # STT API is blocking (synchronous), but we receive audio asynchronously
        # We need a sync queue that the STT thread can read from
        
        sync_audio_queue = sync_queue.Queue()
        
        # Start task to transfer audio from async queue to sync queue
        transfer_task = asyncio.create_task(
            transfer_audio_to_sync_queue(audio_queue, sync_audio_queue, streaming_active)
        )
        
        # Wait briefly for audio to accumulate before starting STT
        # This ensures we have enough audio for the API to process
        await asyncio.sleep(0.1)
        
        # Track STT start time for latency measurement
        stt_start_time = time.time()
        
        # Create generator function that yields audio chunks from sync queue
        # STT API expects an iterator/generator
        audio_stream = create_audio_stream_generator(sync_audio_queue, streaming_active)
        
        # Get event loop for thread-safe operations
        # We'll need this to put results back into async queue from STT thread
        loop = asyncio.get_event_loop()
        
        # ====================================================================
        # Run STT in Thread Pool
        # ====================================================================
        # STT API is blocking, so we run it in a thread pool
        # This allows other async operations to continue
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit STT processing to thread pool
            stt_future = executor.submit(
                run_stt_in_thread,
                audio_stream,
                response_queue,
                loop
            )
            
            # Process STT responses as they arrive
            # This also handles Gemini and TTS processing for final transcripts
            await process_stt_responses(websocket, response_queue, stt_future, language_state, tts_model_state, stt_start_time)
        
        # Wait for transfer task to complete
        await transfer_task
        
    except Exception as e:
        # Format error message to be user-friendly
        error_msg = format_authentication_error(str(e))
        print(f"Error in STT processing: {error_msg}")
        await send_error(websocket, error_msg)


def run_stt_in_thread(
    audio_stream_func,
    response_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop
) -> None:
    """Run STT processing in a separate thread.
    
    This function runs in a thread (not async) because the STT API is blocking.
    It processes audio chunks and puts results into an async queue using
    thread-safe operations.
    
    Args:
        audio_stream_func: Function that returns a generator yielding audio chunks
        response_queue: Async queue to put STT responses into
        loop: Event loop for thread-safe coroutine execution
    """
    try:
        # Call the function to get the actual generator
        audio_stream = audio_stream_func()
        
        # Process each STT response as it arrives
        for response in transcribe_streaming_chirp3(audio_stream):
            # Put response in async queue (thread-safe operation)
            # This allows the async code to process the response
            asyncio.run_coroutine_threadsafe(
                response_queue.put(response),
                loop
            )
        
        # Signal completion by putting None in queue
        asyncio.run_coroutine_threadsafe(
            response_queue.put(None),
            loop
        )
    except Exception as e:
        # Signal error by putting error tuple in queue
        asyncio.run_coroutine_threadsafe(
            response_queue.put(("error", str(e))),
            loop
        )


# ============================================================================
# Response Processing and Pipeline Orchestration
# ============================================================================

async def process_stt_responses(
    websocket: WebSocket,
    response_queue: asyncio.Queue,
    stt_future: concurrent.futures.Future,
    language_state: dict,
    tts_model_state: dict,
    stt_start_time: float,
    timeout: float = 30.0
) -> None:
    """Process STT responses and orchestrate the complete pipeline.
    
    This function:
    1. Receives STT transcription results
    2. Sends transcripts to client (interim and final)
    3. For final transcripts: triggers Gemini -> TTS pipeline
    4. Sends Gemini responses and TTS audio to client
    
    Args:
        websocket: WebSocket connection for sending results
        response_queue: Queue containing STT responses
        stt_future: Future for STT processing thread (to check if still running)
        language_state: Dictionary storing selected language
        tts_model_state: Dictionary storing selected TTS model
        stt_start_time: Timestamp when STT processing started (for latency calculation)
        timeout: How long to wait for responses before checking if STT is done
    """
    while True:
        try:
            # Wait for next STT response (with timeout)
            item = await asyncio.wait_for(response_queue.get(), timeout=timeout)
            
            # None means STT processing is complete
            if item is None:
                break
            
            # Error tuple means STT encountered an error
            if isinstance(item, tuple) and item[0] == "error":
                raise Exception(item[1])
            
            # ================================================================
            # Extract and Send Transcript
            # ================================================================
            transcript_data = extract_transcript_from_response(item)
            if transcript_data:
                transcript, is_final = transcript_data
                
                # Calculate and send STT latency for final transcripts
                if is_final and transcript:
                    stt_latency_ms = (time.time() - stt_start_time) * 1000
                    await send_latency(
                        websocket,
                        "stt",
                        stt_latency_ms,
                        {"transcript_length": len(transcript)}
                    )
                    print(f"[Latency] STT: {stt_latency_ms:.2f}ms")
                
                # Send transcript to client (both interim and final)
                await send_transcript(websocket, transcript, is_final)
                
                # ============================================================
                # Process Final Transcripts with Gemini and TTS
                # ============================================================
                # Only process final transcripts (not interim/partial ones)
                if is_final and transcript:
                    # Check connection before processing
                    if websocket.client_state.name != "CONNECTED":
                        print("WebSocket disconnected, skipping Gemini/TTS processing")
                        break
                    
                    # Get selected language from state
                    selected_lang = language_state["value"]
                    
                    # Step 1: Stream tutor response from Gemini (with timing)
                    gemini_start_time = time.time()
                    first_chunk_time = None
                    full_response = ""
                    
                    # Stream Gemini response chunk by chunk
                    async for chunk in process_transcript_streaming_async(
                        transcript,
                        tutor_language=selected_lang
                    ):
                        # Track time to first chunk
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            first_chunk_latency_ms = (first_chunk_time - gemini_start_time) * 1000
                            await send_latency(
                                websocket,
                                "gemini",
                                first_chunk_latency_ms,
                                {"model": "gemini-2.5-flash", "metric": "time_to_first_token"}
                            )
                            print(f"[Latency] Gemini (first token): {first_chunk_latency_ms:.2f}ms")
                        
                        # Accumulate full response for TTS
                        full_response += chunk
                        
                        # Check connection before sending chunk
                        if websocket.client_state.name != "CONNECTED":
                            print("WebSocket disconnected during Gemini streaming")
                            break
                        
                        # Send chunk to frontend
                        await send_gemini_chunk(websocket, chunk, is_complete=False)
                    
                    # Send final chunk marker
                    if full_response and websocket.client_state.name == "CONNECTED":
                        await send_gemini_chunk(websocket, "", is_complete=True)
                        
                        # Calculate total latency
                        gemini_total_latency_ms = (time.time() - gemini_start_time) * 1000
                        await send_latency(
                            websocket,
                            "gemini",
                            gemini_total_latency_ms,
                            {"model": "gemini-2.5-flash", "response_length": len(full_response), "metric": "total_time"}
                        )
                        
                        # Log conversation for debugging
                        print(f"\n[User]: {transcript}")
                        print(f"[Gemini]: {full_response}")
                        print(f"[Latency] Gemini (total): {gemini_total_latency_ms:.2f}ms\n")
                        
                        # Step 2: Stream Gemini response to speech (with timing)
                        # Get selected TTS model from state
                        selected_model_key = tts_model_state["value"]
                        # Map model key to actual model name
                        from backend.tts_service import GEMINI_TTS_MODELS
                        selected_model = GEMINI_TTS_MODELS.get(selected_model_key, GEMINI_TTS_MODELS["flash"])
                        print(f"Streaming speech synthesis from Gemini response using model: {selected_model}...")
                        
                        tts_start_time = time.time()
                        first_audio_time = None
                        total_audio_size = 0
                        chunk_count = 0
                        
                        # Stream TTS audio chunks as they arrive
                        async for audio_chunk in synthesize_speech_streaming_async(
                            full_response,
                            model_name=selected_model
                        ):
                            # Track time to first audio chunk
                            if first_audio_time is None:
                                first_audio_time = time.time()
                                first_chunk_latency_ms = (first_audio_time - tts_start_time) * 1000
                                await send_latency(
                                    websocket,
                                    "tts",
                                    first_chunk_latency_ms,
                                    {"model": selected_model, "metric": "time_to_first_audio"}
                                )
                                print(f"[Latency] TTS (first chunk): {first_chunk_latency_ms:.2f}ms")
                            
                            total_audio_size += len(audio_chunk)
                            chunk_count += 1
                            
                            # Check connection before sending chunk
                            if websocket.client_state.name != "CONNECTED":
                                print("WebSocket disconnected during TTS streaming")
                                break
                            
                            # Send chunk to frontend (first chunk sends header)
                            await send_tts_chunk(
                                websocket,
                                audio_chunk,
                                is_first=(chunk_count == 1),
                                is_complete=False
                            )
                        
                        # Send completion marker
                        if websocket.client_state.name == "CONNECTED" and total_audio_size > 0:
                            await send_tts_chunk(websocket, b"", is_first=False, is_complete=True)
                            
                            # Calculate total latency
                            tts_total_latency_ms = (time.time() - tts_start_time) * 1000
                            audio_size_kb = total_audio_size / 1024
                            await send_latency(
                                websocket,
                                "tts",
                                tts_total_latency_ms,
                                {"model": selected_model, "audio_size_kb": round(audio_size_kb, 2), "chunks": chunk_count, "metric": "total_time"}
                            )
                            print(f"[Latency] TTS (total): {tts_total_latency_ms:.2f}ms (audio: {audio_size_kb:.2f}KB, {chunk_count} chunks)")
                        elif total_audio_size == 0:
                            print("Warning: TTS streaming returned no audio content")
                
        except asyncio.TimeoutError:
            # Timeout is normal - check if STT is still running
            if stt_future.done():
                # STT finished, exit loop
                break
            # STT still running, continue waiting
            continue
        except Exception as e:
            # Log error and check if connection is still active
            print(f"Error processing STT response: {e}")
            if websocket.client_state.name != "CONNECTED":
                break
