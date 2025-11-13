"""
Text-to-Speech Service using Google Cloud Text-to-Speech API

This module converts text responses from Gemini into natural-sounding speech
using Google's Chirp3 HD voices. It automatically detects the language of the
text and selects the appropriate voice.

Key Features:
- Automatic language detection
- Chirp3 HD voices for high-quality speech
- Support for 15+ languages
- Automatic voice selection based on detected language
- Uses Application Default Credentials (ADC) for authentication
"""

import os
import asyncio
import concurrent.futures
from typing import Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import texttospeech
from langdetect import detect, LangDetectException

# Load environment variables from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

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
# Voice Configuration
# ============================================================================

# Mapping of language codes to Chirp3 HD voice names
# Format: language_code -> (voice_name, language_code_for_tts)
# These voice names were verified by querying the Google Cloud TTS API
CHIRP3_HD_VOICES = {
    "en": ("en-US-Chirp3-HD-Achernar", "en-US"),  # English - Female
    "de": ("de-DE-Chirp3-HD-Achernar", "de-DE"),  # German - Female
    "fr": ("fr-FR-Chirp3-HD-Achernar", "fr-FR"),  # French - Female
    "es": ("es-ES-Chirp3-HD-Achernar", "es-ES"),  # Spanish - Female
    "it": ("it-IT-Chirp3-HD-Achernar", "it-IT"),  # Italian - Female
    "pt": ("pt-BR-Chirp3-HD-Achernar", "pt-BR"),  # Portuguese - Female
    "ja": ("ja-JP-Chirp3-HD-Achernar", "ja-JP"),  # Japanese - Female
    "ko": ("ko-KR-Chirp3-HD-Achernar", "ko-KR"),  # Korean - Female
    "zh": ("cmn-CN-Chirp3-HD-Achernar", "cmn-CN"),  # Chinese - Female
    "ru": ("ru-RU-Chirp3-HD-Aoede", "ru-RU"),  # Russian - Female
    "ar": ("ar-XA-Chirp3-HD-Achernar", "ar-XA"),  # Arabic - Female
    "hi": ("hi-IN-Chirp3-HD-Achernar", "hi-IN"),  # Hindi - Female
    "te": ("te-IN-Chirp3-HD-Achernar", "te-IN"),  # Telugu - Female
    "nl": ("nl-NL-Chirp3-HD-Achernar", "nl-NL"),  # Dutch - Female
    "pl": ("pl-PL-Chirp3-HD-Achernar", "pl-PL"),  # Polish - Female
    "tr": ("tr-TR-Chirp3-HD-Achernar", "tr-TR"),  # Turkish - Female
}


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
    """Get the appropriate Chirp3 HD voice for a given language.
    
    Takes a language code (like 'de' or 'de-DE') and returns the voice name
    and language code needed for the TTS API.
    
    Examples:
        'de' -> ('de-DE-Chirp3-HD-Achernar', 'de-DE')
        'en' -> ('en-US-Chirp3-HD-Achernar', 'en-US')
        'te' -> ('te-IN-Chirp3-HD-Achernar', 'te-IN')
    
    Args:
        language_code: Language code (can be 'de' or 'de-DE' format)
    
    Returns:
        Tuple of (voice_name, language_code_for_tts)
        Defaults to English if language not supported
    """
    # Extract base language code (e.g., 'de' from 'de-DE')
    base_lang = language_code.split('-')[0].lower()
    
    # Look up voice in our mapping, default to English if not found
    return CHIRP3_HD_VOICES.get(base_lang, CHIRP3_HD_VOICES["en"])


# ============================================================================
# Speech Synthesis
# ============================================================================

def synthesize_speech_chirp3_hd(
    text: str,
    voice_name: str = "en-US-Chirp3-HD-Achernar",
    language_code: str = "en-US",
    audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.LINEAR16,
    sample_rate_hertz: int = 24000
) -> Optional[bytes]:
    """Convert text to speech using Chirp3 HD voices.
    
    This function calls the Google Cloud Text-to-Speech API to generate
    natural-sounding speech from text. It uses Chirp3 HD voices which
    provide high-quality, expressive speech.
    
    Reference: https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd
    
    Args:
        text: The text to convert to speech
        voice_name: Name of the voice to use (e.g., "en-US-Chirp3-HD-Achernar")
        language_code: Language code for the voice (e.g., "en-US", "de-DE")
        audio_encoding: Audio format (default: LINEAR16 = PCM)
        sample_rate_hertz: Sample rate in Hz (default: 24000 for Chirp3 HD)
    
    Returns:
        Audio content as bytes (PCM format), or None if there was an error
    """
    try:
        # Create TTS client (automatically uses Application Default Credentials)
        client = texttospeech.TextToSpeechClient()
        
        # Set the text to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Configure which voice to use
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )
        
        # Configure audio output format
        # LINEAR16 = PCM format (raw audio data)
        # 24kHz sample rate is standard for Chirp3 HD voices
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
        print(f"Error synthesizing speech: {error_msg}")
        import traceback
        traceback.print_exc()
        return None


async def synthesize_speech_async(
    text: str,
    voice_name: Optional[str] = None,
    language_code: Optional[str] = None
) -> Optional[bytes]:
    """Convert text to speech asynchronously with automatic language detection.
    
    This function automatically detects the language of the text and selects
    the appropriate voice. It runs the blocking TTS API call in a thread pool
    so it doesn't block other async operations.
    
    Args:
        text: The text to convert to speech
        voice_name: Voice name (optional - will auto-detect if not provided)
        language_code: Language code (optional - will auto-detect if not provided)
    
    Returns:
        Audio content as bytes (PCM format), or None if there was an error
    """
    # If voice/language not specified, detect automatically
    if not language_code or not voice_name:
        # Detect which language the text is in
        detected_lang = detect_language(text)
        # Get the appropriate voice for that language
        voice_name, language_code = get_voice_for_language(detected_lang)
        print(f"Detected language: {detected_lang}, using voice: {voice_name}")
    
    # Get the event loop for async execution
    loop = asyncio.get_event_loop()
    
    # Run the blocking TTS call in a thread pool
    # This prevents blocking other async operations
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the blocking function to the thread pool
        future = executor.submit(
            synthesize_speech_chirp3_hd,
            text,
            voice_name,
            language_code
        )
        # Wait for result without blocking the event loop
        audio_content = await loop.run_in_executor(None, future.result)
        return audio_content
