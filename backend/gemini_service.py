"""
Gemini LLM Service for Language Tutoring

This module handles communication with Google's Gemini AI model via Vertex AI.
It processes user transcripts and generates language tutor responses using
Application Default Credentials (ADC) for authentication.

Key Features:
- Language tutor persona (Lily) with customizable language support
- Automatic text cleaning for natural speech synthesis
- Support for both Latin and non-Latin script languages
- Async processing for non-blocking operations
"""

import os
import asyncio
import concurrent.futures
import re
from pathlib import Path
from typing import Optional, AsyncIterator, Iterator
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Load environment variables from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from backend.config import TUTOR_LANGUAGE

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Track if Vertex AI has been initialized (only needs to happen once)
_vertex_ai_initialized = False


# ============================================================================
# Text Processing Utilities
# ============================================================================

def clean_text_for_speech(text: str) -> str:
    """Clean text to make it natural for speech synthesis.
    
    Gemini sometimes returns text with markdown formatting (like **bold**, *italic*,
    or [links](url)). When these are read aloud by TTS, they sound unnatural.
    This function removes all formatting while preserving the actual content.
    
    Examples:
        "**Hello!** This is *great*" -> "Hello! This is great"
        "[Click here](url)" -> "Click here"
        "### Header" -> "Header"
    
    Args:
        text: Raw text from Gemini that may contain markdown formatting
    
    Returns:
        Cleaned text suitable for natural speech synthesis
    """
    if not text:
        return text
    
    # Step 1: Remove markdown formatting but keep the content inside
    # Bold text: **text** becomes just "text"
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Italic text: *text* or _text_ becomes just "text"
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    # Code blocks: `text` becomes just "text"
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Strikethrough: ~~text~~ becomes just "text"
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    
    # Step 2: Remove markdown headers (like # Header or ### Header)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Step 3: Remove markdown links but keep the link text
    # [Click here](https://example.com) becomes "Click here"
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Step 4: Remove list markers but keep the list content
    # "- Item" or "1. Item" becomes just "Item"
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Bullet points
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)   # Numbered lists
    
    # Step 5: Remove any remaining standalone markdown characters
    # These would be read aloud by TTS and sound weird
    text = re.sub(r'\*\*', '', text)  # Remove remaining **
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)  # Remove single * (but not part of **)
    text = re.sub(r'_', '', text)  # Remove underscores
    text = re.sub(r'`', '', text)  # Remove backticks
    text = re.sub(r'~', '', text)  # Remove tildes
    text = re.sub(r'\[', '', text)  # Remove brackets
    text = re.sub(r'\]', '', text)
    
    # Step 6: Remove parentheses that look like markdown artifacts
    # Keep natural parentheses, but remove empty ones or ones with URLs
    text = re.sub(r'\([^)]*http[^)]*\)', '', text)  # Remove URLs in parentheses
    text = re.sub(r'\(\)', '', text)  # Remove empty parentheses
    
    # Step 7: Clean up any remaining markdown artifacts
    text = re.sub(r'#{1,6}\s*', '', text)  # Remove any remaining # headers
    
    # Step 8: Normalize whitespace for natural speech flow
    # Multiple spaces become single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Multiple newlines become period and space (natural sentence break)
    text = re.sub(r'\n\s*\n+', '. ', text)
    # Single newlines become spaces
    text = re.sub(r'\n', ' ', text)
    
    # Step 9: Clean up spacing around punctuation
    # Remove space before punctuation (e.g., "word ." -> "word.")
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Remove duplicate punctuation (e.g., "word.." -> "word.")
    text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', text)
    
    # Step 10: Final cleanup
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single space
    text = text.strip()  # Remove leading/trailing whitespace
    
    return text


# ============================================================================
# System Instruction Generation
# ============================================================================

def get_tutor_system_instruction(language: str = "Spanish") -> str:
    """Generate system instruction that makes Gemini act as a language tutor.
    
    The system instruction tells Gemini how to behave - in this case, as a friendly
    language tutor named Lily. Different instructions are used for languages with
    non-Latin scripts (like Telugu, Hindi) vs Latin scripts (like Spanish, French).
    
    For non-Latin scripts:
    - Uses mixed English and target language (since student only knows English)
    - Provides examples of good teaching responses
    - Emphasizes not repeating content
    
    For Latin scripts:
    - Immerses student in target language
    - Responds primarily in the target language
    
    Args:
        language: The language to tutor (e.g., "Spanish", "Telugu", "German")
    
    Returns:
        A detailed system instruction string that configures Gemini's behavior
    """
    # Languages that use non-Latin scripts need special teaching approach
    # because students can't read the script yet, so we mix English explanations
    non_latin_script_languages = [
        "Telugu", "Hindi", "Bengali", "Tamil", "Kannada", "Malayalam",
        "Gujarati", "Marathi", "Punjabi", "Urdu", "Arabic", "Chinese",
        "Japanese", "Korean", "Thai"
    ]
    
    if language in non_latin_script_languages:
        return f"""You are a friendly and patient {language} language tutor named Lily. Your role is to help students learn {language} through natural conversation. The student only understands English, so you must teach using a mix of English and {language}.

CRITICAL TEACHING APPROACH:
1. Use a MIX of English and {language} in your responses - explain in English, demonstrate in {language}
2. When introducing {language} words/phrases, use this format: "English explanation [{language} word/phrase]"
3. NEVER repeat the same content twice - if you say something in {language}, don't repeat it in English (or vice versa)
4. Write {language} text using its native script (Telugu script for Telugu, Devanagari for Hindi, etc.)
5. Keep responses concise and conversational (2-3 sentences maximum)
6. If the student makes mistakes, correct them gently: "Almost! The correct way is [{language} word]"
7. Encourage the student: "Great job! You're learning quickly!"
8. Use simple vocabulary appropriate for beginners
9. Make learning fun and engaging with a warm, supportive tone

EXAMPLES OF GOOD RESPONSES:
- User: "How do you say hello?"
  You: "Hello! నమస్కారం is how you greet someone in Telugu."

- User: "What does thank you mean?"
  You: "Thank you is ధన్యవాదాలు. You can use it when someone helps you!"

- User: "How are you?"
  You: "I'm doing well! మీరు ఎలా ఉన్నారు? means 'How are you?' in Telugu."

Remember: Mix English (for understanding) with {language} (for learning), but never repeat the same thing twice!"""
    else:
        # For languages using Latin script (Spanish, French, German, etc.)
        # Students can read the script, so we can immerse them in the language
        return f"""You are a friendly and patient {language} language tutor named Lily. Your role is to help students learn {language} through natural conversation.

Guidelines:
1. Respond primarily in {language} to immerse the student in the language
2. Keep responses concise and conversational (2-3 sentences maximum)
3. If the student makes mistakes, gently correct them and provide the correct form
4. If the student asks a question in English, answer in {language} and explain in simple terms if needed
5. Encourage the student and celebrate their progress
6. Use simple vocabulary appropriate for language learners
7. If the student is struggling, provide helpful hints or break down complex concepts
8. Make learning fun and engaging with a warm, supportive tone

Remember: You're having a conversation, not giving a lecture. Keep it natural and friendly!"""


# ============================================================================
# Vertex AI Initialization
# ============================================================================

def initialize_vertex_ai() -> None:
    """Initialize Vertex AI client using Application Default Credentials.
    
    This only needs to be called once per application run. Vertex AI uses
    Application Default Credentials (ADC), which means it automatically uses
    credentials from gcloud auth application-default login.
    
    Raises:
        ValueError: If GOOGLE_CLOUD_PROJECT is not set in environment
    """
    global _vertex_ai_initialized
    
    # Only initialize once
    if _vertex_ai_initialized:
        return
    
    # Validate that project ID is configured
    if not PROJECT_ID:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT environment variable is not set.\n"
            "Please set it in your .env file."
        )
    
    # Initialize Vertex AI with project and location
    # Location determines which regional endpoint to use
    print(f"Initializing Vertex AI with project: {PROJECT_ID}, location: {LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    _vertex_ai_initialized = True


# ============================================================================
# Gemini Processing Functions
# ============================================================================

def process_transcript_with_gemini(
    transcript: str,
    model_name: str = "gemini-2.5-flash",
    tutor_language: Optional[str] = None
) -> Optional[str]:
    """Process user transcript with Gemini and return tutor's response.
    
    This function:
    1. Configures Gemini as a language tutor for the specified language
    2. Sends the user's transcript to Gemini
    3. Gets the tutor's response
    4. Cleans the response to remove markdown formatting
    5. Returns clean text ready for display and TTS
    
    Args:
        transcript: The user's spoken text (transcribed from speech)
        model_name: Which Gemini model to use (default: gemini-2.5-flash for speed)
        tutor_language: Language for tutoring (default: from config, usually "Spanish")
    
    Returns:
        Cleaned tutor response text, or None if there was an error
    """
    try:
        # Make sure Vertex AI is initialized
        initialize_vertex_ai()
        
        # Determine which language to use for tutoring
        # Use provided language or fall back to config default
        language = tutor_language or TUTOR_LANGUAGE
        
        # Get the system instruction that tells Gemini how to behave
        # This is what makes Gemini act like a language tutor
        system_instruction = get_tutor_system_instruction(language)
        
        # Create the Gemini model with the tutor persona
        # The system_instruction parameter is what configures the behavior
        model = GenerativeModel(
            model_name,
            system_instruction=system_instruction
        )
        
        # The prompt is simply the user's transcript
        # Gemini will respond based on the system instruction
        prompt = transcript
        
        # Configure response generation
        # Temperature 0.7 provides a balance between creativity and consistency
        generation_config = GenerationConfig(
            temperature=0.7,
        )
        
        # Send request to Gemini and get response
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract the text from Gemini's response
        # The response structure can vary, so we check multiple places
        raw_text = None
        if response and hasattr(response, 'text') and response.text:
            # Most common case: text is directly available
            raw_text = response.text.strip()
        elif response and hasattr(response, 'candidates') and response.candidates:
            # Fallback: extract from candidates structure
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                raw_text = candidate.content.parts[0].text
                if raw_text:
                    raw_text = raw_text.strip()
        
        # Clean the text to remove markdown and make it natural for speech
        # This ensures TTS will sound natural, not reading formatting characters
        if raw_text:
            cleaned_text = clean_text_for_speech(raw_text)
            return cleaned_text
        
        # If we couldn't extract any text, log a warning
        print(f"Warning: No text in Gemini response. Response: {response}")
        return None
        
    except Exception as e:
        # Log error but don't crash - return None so caller can handle gracefully
        print(f"Error processing transcript with Gemini: {e}")
        return None


async def process_transcript_async(
    transcript: str,
    model_name: str = "gemini-2.5-flash",
    tutor_language: Optional[str] = None
) -> Optional[str]:
    """Process transcript with Gemini asynchronously (non-blocking).
    
    The Gemini API is blocking (synchronous), but we're running in an async
    FastAPI application. This function runs the blocking Gemini call in a
    thread pool so it doesn't block other requests.
    
    Args:
        transcript: The user's spoken text (transcribed from speech)
        model_name: Which Gemini model to use (default: gemini-2.5-flash)
        tutor_language: Language for tutoring (default: from config)
    
    Returns:
        Cleaned tutor response text, or None if there was an error
    """
    # Get the event loop for this async context
    loop = asyncio.get_event_loop()
    
    # Run the blocking Gemini call in a thread pool
    # This allows other async operations to continue while waiting for Gemini
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the blocking function to the thread pool
        future = executor.submit(
            process_transcript_with_gemini,
            transcript,
            model_name,
            tutor_language
        )
        # Wait for the result without blocking the event loop
        response = await loop.run_in_executor(None, future.result)
        return response


async def process_transcript_streaming_async(
    transcript: str,
    model_name: str = "gemini-2.5-flash",
    tutor_language: Optional[str] = None
) -> AsyncIterator[str]:
    """Process transcript with Gemini using streaming (yields tokens as they arrive).
    
    This function streams Gemini's response token by token, allowing the frontend
    to display text incrementally for a better user experience.
    
    Args:
        transcript: The user's spoken text (transcribed from speech)
        model_name: Which Gemini model to use (default: gemini-2.5-flash)
        tutor_language: Language for tutoring (default: from config)
    
    Yields:
        Text chunks (tokens) as they arrive from Gemini
    """
    try:
        # Make sure Vertex AI is initialized
        initialize_vertex_ai()
        
        # Determine which language to use for tutoring
        language = tutor_language or TUTOR_LANGUAGE
        
        # Get the system instruction that tells Gemini how to behave
        system_instruction = get_tutor_system_instruction(language)
        
        # Create the Gemini model with the tutor persona
        model = GenerativeModel(
            model_name,
            system_instruction=system_instruction
        )
        
        # The prompt is simply the user's transcript
        prompt = transcript
        
        # Configure response generation
        generation_config = GenerationConfig(
            temperature=0.7,
        )
        
        # Stream response from Gemini
        # This yields chunks as they're generated
        response_stream = model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True  # Enable streaming
        )
        
        # Process stream in thread pool since it's blocking
        loop = asyncio.get_event_loop()
        
        accumulated_text = ""
        
        def stream_generator():
            """Generator that yields text chunks from streaming response."""
            nonlocal accumulated_text
            for chunk in response_stream:
                # Extract text from chunk
                chunk_text = None
                if hasattr(chunk, 'text') and chunk.text:
                    chunk_text = chunk.text
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        part = candidate.content.parts[0]
                        if hasattr(part, 'text') and part.text:
                            chunk_text = part.text
                
                if chunk_text:
                    # Get only the new text (incremental)
                    if len(chunk_text) > len(accumulated_text):
                        new_text = chunk_text[len(accumulated_text):]
                        accumulated_text = chunk_text
                        yield new_text
        
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
                    # Clean the chunk before yielding
                    cleaned_chunk = clean_text_for_speech(chunk)
                    if cleaned_chunk:
                        yield cleaned_chunk
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error in streaming chunk: {e}")
                    break
                    
    except Exception as e:
        print(f"Error processing transcript with Gemini streaming: {e}")
        # Return empty iterator on error
        return
