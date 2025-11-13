# Duolingo Language Tutoring Demo

A real-time language tutoring application that uses Google Cloud Speech-to-Text, Gemini AI, and Text-to-Speech to create an interactive language learning experience.

## Features

- **Push-to-Talk**: Hold a button to speak, release to transcribe
- **Real-time Transcription**: Uses Google Cloud Speech-to-Text with Chirp3 model
- **AI Language Tutor**: Powered by Gemini 2.5 Flash for natural conversations
- **Text-to-Speech**: Chirp3 HD voices for natural-sounding responses
- **Multi-language Support**: Spanish, German, French, Telugu, and more
- **Automatic Language Detection**: TTS automatically detects and uses the correct voice

## Architecture

```
User Speech → STT (Chirp3) → Gemini 2.5 Flash → TTS (Chirp3 HD) → Audio Playback
```

## Prerequisites

1. **Google Cloud Project** with the following APIs enabled:
   - Speech-to-Text API
   - Vertex AI API (for Gemini)
   - Text-to-Speech API

2. **Authentication**: Application Default Credentials (ADC)
   ```bash
   gcloud auth application-default login
   ```

3. **Python 3.8+** and **Node.js** (for local development)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rahulsankrut/duolingo.git
cd duolingo
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
TUTOR_LANGUAGE=Spanish
API_HOST=0.0.0.0
API_PORT=8000
```

### 5. Start Backend Server

```bash
cd backend
python app.py
```

The backend will run on `http://localhost:8000`

### 6. Start Frontend Server

**Important**: The frontend must be accessed via `localhost` or `https` for microphone access to work.

```bash
cd Frontend
python3 -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

## Troubleshooting

### "getUserMedia is not supported" Error

This error occurs when:
1. **Accessing via IP address** (e.g., `http://192.168.1.100:8080`)
2. **Not using HTTPS** (except for localhost)

**Solutions:**

**Option 1 (Recommended)**: Access via localhost on each machine
- On each laptop, run the frontend server locally
- Access via `http://localhost:8080`

**Option 2**: Set up HTTPS for local development
- Use a tool like [ngrok](https://ngrok.com/) to create an HTTPS tunnel
- Or set up a local HTTPS server with self-signed certificate

**Option 3**: Use a reverse proxy with HTTPS
- Set up nginx or similar with SSL certificate

### Microphone Permission Denied

1. Click the lock icon in your browser's address bar
2. Allow microphone access
3. Refresh the page

### Backend Connection Issues

- Ensure the backend is running on port 8000
- Check that `GOOGLE_CLOUD_PROJECT` is set correctly
- Verify ADC is configured: `gcloud auth application-default login`

## Project Structure

```
duolingo/
├── backend/
│   ├── app.py                 # FastAPI application
│   ├── config.py              # Configuration management
│   ├── stt_service.py         # Speech-to-Text service
│   ├── gemini_service.py       # Gemini LLM service
│   ├── tts_service.py         # Text-to-Speech service
│   ├── websocket_handler.py   # WebSocket orchestration
│   └── websocket_utils.py      # WebSocket utilities
├── Frontend/
│   ├── index.html             # Main HTML page
│   ├── styles.css             # Styling
│   └── app.js                 # Frontend logic
├── Test/
│   └── test_stt.py            # Test scripts
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables (not in git)
```

## Technologies Used

- **Backend**: FastAPI, Python
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Google Cloud Services**:
  - Speech-to-Text API (Chirp3)
  - Vertex AI (Gemini 2.5 Flash)
  - Text-to-Speech API (Chirp3 HD)
- **Real-time Communication**: WebSockets

## License

This project is for demonstration purposes.

