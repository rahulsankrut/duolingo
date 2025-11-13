"""
Speech-to-Text Service using Google Cloud Speech-to-Text API

This module handles real-time speech transcription using Google's Chirp3 model.
It processes streaming audio and converts it to text using Application Default
Credentials (ADC) for authentication.

Key Features:
- Streaming recognition for real-time transcription
- Chirp3 model for high accuracy
- Supports 16kHz mono PCM audio format
- Uses US endpoint (required for Chirp3 model)
"""

import os
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
from typing import Iterator, Generator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

if not PROJECT_ID:
    raise ValueError(
        "GOOGLE_CLOUD_PROJECT environment variable is not set. "
        "Please set it in your .env file."
    )


# ============================================================================
# Configuration Creation
# ============================================================================

def create_streaming_config(
    language_code: str = "en-US",
    model: str = "chirp_3",
    sample_rate_hertz: int = 16000,
    audio_channel_count: int = 1
) -> cloud_speech.StreamingRecognitionConfig:
    """Create configuration for streaming speech recognition.
    
    This configures how the Speech-to-Text API should process the audio:
    - Audio format: LINEAR16 (PCM) at 16kHz, mono
    - Language: English (US) by default
    - Model: Chirp3 for best accuracy
    
    Args:
        language_code: Language to recognize (default: "en-US")
        model: Model to use (default: "chirp_3")
        sample_rate_hertz: Audio sample rate in Hz (default: 16000)
        audio_channel_count: Number of audio channels, 1 = mono (default: 1)
    
    Returns:
        StreamingRecognitionConfig object ready to use with the API
    """
    # Create recognition configuration
    # LINEAR16 = PCM encoding (raw audio data)
    recognition_config = cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hertz,
            audio_channel_count=audio_channel_count,
        ),
        language_codes=[language_code],
        model=model,
    )
    
    # Wrap in streaming config for real-time processing
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=recognition_config
    )
    
    return streaming_config


# ============================================================================
# Streaming Transcription
# ============================================================================

def transcribe_streaming_chirp3(
    audio_stream: Iterator[bytes],
    language_code: str = "en-US"
) -> Generator[cloud_speech.StreamingRecognizeResponse, None, None]:
    """Transcribe streaming audio in real-time using Chirp3 model.
    
    This function processes audio chunks as they arrive and yields transcription
    results immediately. This enables real-time transcription where users see
    results as they speak.
    
    The function:
    1. Creates a Speech-to-Text client connected to the US endpoint
    2. Configures streaming recognition with Chirp3 model
    3. Sends audio chunks to the API
    4. Yields transcription results as they arrive
    
    Reference: https://docs.cloud.google.com/speech-to-text/docs/streaming-recognize
    
    Args:
        audio_stream: Iterator that yields audio chunks as bytes
                    (should be PCM Int16 format, 16kHz, mono)
        language_code: Language to recognize (default: "en-US")
    
    Yields:
        StreamingRecognizeResponse objects containing transcription results
        Each response may contain interim (partial) or final transcriptions
    """
    # Create Speech-to-Text client
    # Must use US endpoint for Chirp3 model (it's not available in other regions)
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint="us-speech.googleapis.com",
        )
    )
    
    # Create streaming configuration
    streaming_config = create_streaming_config(language_code=language_code)
    
    # Validate project ID is set
    if not PROJECT_ID:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT is not set. "
            "Please set it in your .env file."
        )
    
    # Create the initial configuration request
    # This tells the API how to process the audio
    # Location "us" is required for Chirp3 model
    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/us/recognizers/_",
        streaming_config=streaming_config,
    )
    
    # Create generator for audio chunk requests
    # Each audio chunk becomes a StreamingRecognizeRequest
    audio_requests = (
        cloud_speech.StreamingRecognizeRequest(audio=audio_chunk)
        for audio_chunk in audio_stream
    )
    
    # Request generator function
    # The API requires: first send config, then send audio chunks
    def request_generator():
        yield config_request  # Send config first
        yield from audio_requests  # Then send all audio chunks
    
    # Start streaming recognition
    # This returns a generator that yields responses as they arrive
    responses = client.streaming_recognize(requests=request_generator())
    
    # Yield each response as it arrives
    # Responses come in real-time as the user speaks
    for response in responses:
        yield response
