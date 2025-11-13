"""
Text-to-Speech Service using Google Cloud Text-to-Speech API with Gemini-TTS

This module converts text responses from Gemini into natural-sounding speech
using Google's Gemini-TTS voices. It automatically detects the language of the
text and selects the appropriate voice.

Key Features:
- Automatic language detection
- Gemini-TTS voices for high-quality, expressive speech
- Support for 80+ locales and 30+ voice options
- Automatic voice selection based on detected language
- Uses Application Default Credentials (ADC) for authentication
- Optional prompts for controlling tone, style, and emotional expression

Reference: https://cloud.google.com/text-to-speech/docs/gemini-tts
"""

import os
import asyncio
import concurrent.futures
from typing import Optional, Tuple, AsyncIterator
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import texttospeech
from langdetect import detect, LangDetectException

# Load environment variables from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from backend.config import GEMINI_TTS_MODEL as CONFIG_TTS_MODEL

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
# Gemini-TTS Model Configuration
# ============================================================================

# Available Gemini-TTS models
GEMINI_TTS_MODELS = {
    "flash": "gemini-2.5-flash-tts",  # Fast, efficient model (recommended for most use cases)
    "lite": "gemini-2.5-flash-lite-preview-tts",  # Lightweight preview model
    "pro": "gemini-2.5-pro-tts",  # High-quality, more capable model
}

# Get the model key from config (defaults to "flash")
TTS_MODEL_KEY = CONFIG_TTS_MODEL.lower() if CONFIG_TTS_MODEL else "flash"

# Get the actual model name from the key
GEMINI_TTS_MODEL = GEMINI_TTS_MODELS.get(
    TTS_MODEL_KEY,
    GEMINI_TTS_MODELS["flash"]  # Default to flash if invalid key
)

# Log which model is being used
print(f"Using Gemini-TTS model: {GEMINI_TTS_MODEL} (key: {TTS_MODEL_KEY})")

# Default sample rate for Gemini-TTS (24kHz)
DEFAULT_SAMPLE_RATE = 24000


# ============================================================================
# Voice Configuration
# ============================================================================

# Mapping of language codes to Gemini-TTS voice names and language codes
# Format: language_code -> (voice_name, language_code_for_tts)
# 
# Gemini-TTS has 30 distinct voices with names like:
# - Zephyr (Bright, energetic)
# - Puck (Upbeat, cheerful)
# - Charon (Informative, clear)
# - And 27 more voices
# 
# See all available voices: https://cloud.google.com/text-to-speech/docs/gemini-tts#voice_options
# 
# IMPORTANT: The voice names below use "Zephyr" as a default. You can customize these
# based on your preferences. Different voices have different characteristics:
# - Gender, age, style, tone
# - Some voices work better for certain languages or contexts
# 
# You can list available voices programmatically or check the documentation to find
# the best voice for each language.
GEMINI_TTS_VOICES = {
    "en": ("Zephyr", "en-US"),  # English - US (Bright, energetic voice)
    "de": ("Zephyr", "de-DE"),  # German
    "fr": ("Zephyr", "fr-FR"),  # French
    "es": ("Zephyr", "es-ES"),  # Spanish - Spain
    "it": ("Zephyr", "it-IT"),  # Italian
    "pt": ("Zephyr", "pt-BR"),  # Portuguese - Brazil
    "ja": ("Zephyr", "ja-JP"),  # Japanese
    "ko": ("Zephyr", "ko-KR"),  # Korean
    "zh": ("Zephyr", "cmn-CN"),  # Chinese - Mandarin
    "ru": ("Zephyr", "ru-RU"),  # Russian
    "ar": ("Zephyr", "ar-XA"),  # Arabic
    "hi": ("Zephyr", "hi-IN"),  # Hindi
    "te": ("Zephyr", "te-IN"),  # Telugu
    "nl": ("Zephyr", "nl-NL"),  # Dutch
    "pl": ("Zephyr", "pl-PL"),  # Polish
    "tr": ("Zephyr", "tr-TR"),  # Turkish
}

# Default prompt for language tutor (can be customized)
# Prompts allow you to control tone, style, pace, and emotional expression
DEFAULT_PROMPT = "Say the following with a friendly and patient tone, as if teaching a language student"


# ============================================================================
# Language Detection
# ============================================================================

def detect_language(text: str) -> str:
    """Detect the language of the given text.
    
    Uses the langdetect library to automatically identify which language
    the text is written in. This allows us to select the correct TTS voice.
    
    Args:
        text: The text to analyze
    
    Returns:
        Two-letter language code (e.g., 'en', 'de', 'fr', 'te')
        Returns 'en' (English) if detection fails
    """
    try:
        # Detect language using langdetect library
        # Returns ISO 639-1 language code (e.g., 'en', 'de', 'fr')
        detected = detect(text)
        return detected
    except LangDetectException:
        # If detection fails (e.g., text too short or mixed languages),
        # default to English
        return "en"


def get_voice_for_language(language_code: str) -> Tuple[str, str]:
    """Get the appropriate Gemini-TTS voice for a given language.
    
    Takes a language code (like 'de' or 'de-DE') and returns the voice name
    and language code needed for the TTS API.
    
    Examples:
        'de' -> ('gemini-tts-voice-001', 'de-DE')
        'en' -> ('gemini-tts-voice-001', 'en-US')
        'te' -> ('gemini-tts-voice-001', 'te-IN')
    
    Note: You can customize voice selection by changing the voice name
    in GEMINI_TTS_VOICES. See available voices:
    https://cloud.google.com/text-to-speech/docs/gemini-tts#voice_options
    
    Args:
        language_code: Language code (can be 'de' or 'de-DE' format)
    
    Returns:
        Tuple of (voice_name, language_code_for_tts)
        Defaults to English if language not supported
    """
    # Extract base language code (e.g., 'de' from 'de-DE')
    base_lang = language_code.split('-')[0].lower()
    
    # Look up voice in our mapping, default to English if not found
    return GEMINI_TTS_VOICES.get(base_lang, GEMINI_TTS_VOICES["en"])


# ============================================================================
# Speech Synthesis
# ============================================================================

def synthesize_speech_gemini_tts(
    text: str,
    voice_name: Optional[str] = None,
    language_code: Optional[str] = None,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.LINEAR16,
    sample_rate_hertz: int = DEFAULT_SAMPLE_RATE
) -> Optional[bytes]:
    """Convert text to speech using Gemini-TTS voices.
    
    This function calls the Google Cloud Text-to-Speech API to generate
    natural-sounding speech from text using Gemini-TTS. Gemini-TTS provides
    high-quality, expressive speech with granular control over style, tone,
    pace, and emotional expression through text-based prompts.
    
    Reference: https://cloud.google.com/text-to-speech/docs/gemini-tts
    
    Args:
        text: The text to convert to speech
        voice_name: Name of the Gemini-TTS voice to use (optional, will auto-detect)
        language_code: Language code for the voice (optional, will auto-detect)
        prompt: Optional prompt to control tone, style, pace, or emotion
                (e.g., "Say the following with a friendly and patient tone")
        model_name: Gemini-TTS model to use (optional, uses default from config)
                    Options: "gemini-2.5-flash-tts", "gemini-2.5-flash-lite-preview-tts",
                    or "gemini-2.5-pro-tts"
        audio_encoding: Audio format (default: LINEAR16 = PCM)
        sample_rate_hertz: Sample rate in Hz (default: 24000)
    
    Returns:
        Audio content as bytes (PCM format), or None if there was an error
    """
    try:
        # Create TTS client (automatically uses Application Default Credentials)
        client = texttospeech.TextToSpeechClient()
        
        # Auto-detect language and voice if not provided
        if not language_code or not voice_name:
            detected_lang = detect_language(text)
            voice_name, language_code = get_voice_for_language(detected_lang)
            print(f"Detected language: {detected_lang}, using voice: {voice_name}, language: {language_code}")
        
        # Set the text to be synthesized
        # Gemini-TTS supports prompts for controlling tone/style
        synthesis_input = texttospeech.SynthesisInput(
            text=text,
            prompt=prompt or DEFAULT_PROMPT  # Use default prompt if not provided
        )
        
        # Use provided model or default from configuration
        tts_model = model_name or GEMINI_TTS_MODEL
        
        # Configure which voice to use with Gemini-TTS model
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
            model_name=tts_model  # Specify Gemini-TTS model
        )
        
        # Configure audio output format
        # LINEAR16 = PCM format (raw audio data)
        # 24kHz sample rate is standard for Gemini-TTS
        audio_config = texttospeech.AudioConfig(
            audio_encoding=audio_encoding,
            sample_rate_hertz=sample_rate_hertz,
        )
        
        # Call the API to generate speech
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        
        # Return the audio data as bytes
        return response.audio_content
        
    except Exception as e:
        # Log error but don't crash
        error_msg = str(e) if e else "Unknown error"
        print(f"Error synthesizing speech with Gemini-TTS: {error_msg}")
        import traceback
        traceback.print_exc()
        return None


async def synthesize_speech_async(
    text: str,
    voice_name: Optional[str] = None,
    language_code: Optional[str] = None,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None
) -> Optional[bytes]:
    """Convert text to speech asynchronously with automatic language detection.
    
    This function automatically detects the language of the text and selects
    the appropriate Gemini-TTS voice. It runs the blocking TTS API call in a
    thread pool so it doesn't block other async operations.
    
    Args:
        text: The text to convert to speech
        voice_name: Voice name (optional - will auto-detect if not provided)
        language_code: Language code (optional - will auto-detect if not provided)
        prompt: Optional prompt to control tone/style (e.g., "Say with a friendly tone")
        model_name: Gemini-TTS model to use (optional, uses default from config)
                    Options: "gemini-2.5-flash-tts", "gemini-2.5-flash-lite-preview-tts",
                    or "gemini-2.5-pro-tts"
    
    Returns:
        Audio content as bytes (PCM format), or None if there was an error
    """
    # Get the event loop for async execution
    loop = asyncio.get_event_loop()
    
    # Run the blocking TTS call in a thread pool
    # This prevents blocking other async operations
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the blocking function to the thread pool
        future = executor.submit(
            synthesize_speech_gemini_tts,
            text,
            voice_name,
            language_code,
            prompt,
            model_name
        )
        # Wait for result without blocking the event loop
        audio_content = await loop.run_in_executor(None, future.result)
        return audio_content


async def synthesize_speech_streaming_async(
    text: str,
    voice_name: Optional[str] = None,
    language_code: Optional[str] = None,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None
) -> AsyncIterator[bytes]:
    """Convert text to speech using streaming synthesis (yields audio chunks as they arrive).
    
    This function streams TTS audio chunks as they're generated, allowing the frontend
    to start playing audio before synthesis is complete.
    
    Reference: https://cloud.google.com/text-to-speech/docs/create-audio-text-streaming
    
    Args:
        text: The text to convert to speech
        voice_name: Voice name (optional - will auto-detect if not provided)
        language_code: Language code (optional - will auto-detect if not provided)
        prompt: Optional prompt to control tone/style
        model_name: Gemini-TTS model to use (optional, uses default from config)
    
    Yields:
        Audio chunks as bytes (PCM format) as they arrive from the TTS API
    """
    try:
        # Create TTS client
        client = texttospeech.TextToSpeechClient()
        
        # Auto-detect language and voice if not provided
        if not language_code or not voice_name:
            detected_lang = detect_language(text)
            voice_name, language_code = get_voice_for_language(detected_lang)
            print(f"Detected language: {detected_lang}, using voice: {voice_name}, language: {language_code}")
        
        # Use provided model or default from configuration
        tts_model = model_name or GEMINI_TTS_MODEL
        
        # Create streaming config request
        config_request = texttospeech.StreamingSynthesizeRequest(
            streaming_config=texttospeech.StreamingSynthesizeConfig(
                voice=texttospeech.VoiceSelectionParams(
                    language_code=language_code,
                    name=voice_name,
                    model_name=tts_model
                ),
                streaming_audio_config=texttospeech.StreamingAudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.PCM,  # Use PCM for streaming (not LINEAR16 which includes WAV header)
                    sample_rate_hertz=DEFAULT_SAMPLE_RATE
                )
            )
        )
        
        # Create text input request
        text_request = texttospeech.StreamingSynthesizeRequest(
            input=texttospeech.StreamingSynthesisInput(
                text=text,
                prompt=prompt or DEFAULT_PROMPT
            )
        )
        
        # Request generator function
        def request_generator():
            yield config_request  # Send config first
            yield text_request    # Then send text
        
        # Stream synthesis in thread pool since it's blocking
        loop = asyncio.get_event_loop()
        
        def stream_generator():
            """Generator that yields audio chunks from streaming response."""
            streaming_responses = client.streaming_synthesize(request_generator())
            for response in streaming_responses:
                if response.audio_content:
                    yield response.audio_content
        
        # Run streaming in thread pool and yield chunks asynchronously
        stream = stream_generator()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                try:
                    # Get next chunk from stream (blocking operation)
                    chunk = await loop.run_in_executor(
                        executor,
                        lambda: next(stream, None)
                    )
                    if chunk is None:
                        break
                    yield chunk
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error in TTS streaming chunk: {e}")
                    break
                    
    except Exception as e:
        print(f"Error synthesizing speech with Gemini-TTS streaming: {e}")
        import traceback
        traceback.print_exc()
        # Return empty iterator on error
        return
