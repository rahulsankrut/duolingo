// API Configuration
const WS_BASE_URL = 'ws://localhost:8000';

// State
let websocket = null;
let mediaStream = null;
let isRecording = false;
let audioContext = null;
let processor = null;
let pendingTTSAudio = null; // Store TTS audio metadata
let ttsStreamingContext = null; // AudioContext for streaming TTS playback
let ttsAudioQueue = []; // Queue for streaming TTS audio chunks
let isTTSStreaming = false; // Track if TTS is currently streaming

// DOM Elements
const talkButton = document.getElementById('talkButton');
const status = document.getElementById('status');
const transcript = document.getElementById('transcript');
const geminiResponse = document.getElementById('geminiResponse');
const languageSelect = document.getElementById('languageSelect');
const ttsModelSelect = document.getElementById('ttsModelSelect');
const sttLatency = document.getElementById('sttLatency');
const geminiLatency = document.getElementById('geminiLatency');
const geminiBreakdown = document.getElementById('geminiBreakdown');
const ttsLatency = document.getElementById('ttsLatency');
const ttsBreakdown = document.getElementById('ttsBreakdown');

// Store latency metrics for components that have multiple metrics
const latencyMetrics = {
    gemini: {
        firstToken: null,
        total: null
    },
    tts: {
        firstAudio: null,
        total: null
    }
};

// Selected language (default: Spanish)
let selectedLanguage = 'Spanish';

// Selected TTS model (default: flash)
let selectedTTSModel = 'flash';

// Event Listeners
talkButton.addEventListener('mousedown', startRecording);
talkButton.addEventListener('mouseup', stopRecording);
talkButton.addEventListener('mouseleave', stopRecording);
talkButton.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startRecording();
});
talkButton.addEventListener('touchend', (e) => {
    e.preventDefault();
    stopRecording();
});

// Language selection handler
languageSelect.addEventListener('change', (e) => {
    selectedLanguage = e.target.value;
    console.log('Language changed to:', selectedLanguage);
    
    // Send language selection to backend if WebSocket is connected
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            type: 'set_language',
            language: selectedLanguage
        }));
    }
});

// TTS model selection handler
ttsModelSelect.addEventListener('change', (e) => {
    selectedTTSModel = e.target.value;
    console.log('TTS model changed to:', selectedTTSModel);
    
    // Send TTS model selection to backend if WebSocket is connected
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            type: 'set_tts_model',
            model: selectedTTSModel
        }));
    }
});

// Functions
async function connectWebSocket() {
    return new Promise((resolve, reject) => {
        const wsUrl = `${WS_BASE_URL}/ws/transcribe`;
        console.log('Connecting to WebSocket:', wsUrl);
        websocket = new WebSocket(wsUrl);
        
        // Set binary type to handle audio data
        websocket.binaryType = 'arraybuffer';
        
        const timeout = setTimeout(() => {
            if (websocket.readyState !== WebSocket.OPEN) {
                websocket.close();
                reject(new Error('Connection timeout'));
            }
        }, 10000);
        
        websocket.onopen = () => {
            console.log('WebSocket connected');
            clearTimeout(timeout);
            status.textContent = 'Connected';
            status.className = 'status connected';
            
                    // Send initial language selection
                    websocket.send(JSON.stringify({
                        type: 'set_language',
                        language: selectedLanguage
                    }));
                    
                    // Send initial TTS model selection
                    websocket.send(JSON.stringify({
                        type: 'set_tts_model',
                        model: selectedTTSModel
                    }));
            
            resolve();
        };
        
        websocket.onmessage = async (event) => {
            // Check if message is binary (TTS audio) or text (JSON)
            if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
                // Handle binary audio data (TTS audio)
                if (isTTSStreaming) {
                    // Streaming TTS - play chunk immediately
                    await handleTTSStreamingChunk(event.data);
                } else if (pendingTTSAudio) {
                    // Non-streaming TTS - complete audio
                    await handleTTSAudio(event.data, pendingTTSAudio);
                    pendingTTSAudio = null; // Clear after use
                }
            } else {
                // Handle JSON messages
                try {
                    const message = JSON.parse(event.data);
                    // If it's a TTS audio header, store metadata for next binary message
                    if (message.type === 'tts_audio') {
                        pendingTTSAudio = message;
                    } else {
                        handleWebSocketMessage(message);
                    }
                } catch (e) {
                    console.error('Error parsing message:', e);
                }
            }
        };
        
        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            clearTimeout(timeout);
            status.textContent = 'Connection error';
            status.className = 'status error';
            reject(error);
        };
        
        websocket.onclose = () => {
            console.log('WebSocket closed');
            status.textContent = 'Disconnected';
            status.className = 'status';
        };
    });
}

function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'status':
            console.log('Status:', message.message);
            break;
        
        case 'transcript':
            displayTranscript(message.text, message.is_final);
            break;
        
        case 'gemini_response':
            // Complete response (non-streaming mode, for backward compatibility)
            displayGeminiResponse(message.text);
            break;
        
        case 'gemini_chunk':
            // Streaming chunk - append to existing response
            displayGeminiChunk(message.text, message.is_complete);
            break;
        
        case 'tts_chunk':
            // Streaming TTS chunk - handle header or audio chunk
            if (message.is_first) {
                // Initialize streaming TTS playback
                handleTTSStreamingStart(message);
            } else if (message.is_complete) {
                // Finalize streaming TTS
                handleTTSStreamingComplete();
            }
            // Note: Audio chunks come as binary, handled in onmessage
            break;
        
        case 'latency':
            displayLatency(message.component, message.latency_ms, message.details);
            break;
        
        case 'error':
            console.error('Error:', message.message);
            status.textContent = 'Error: ' + message.message;
            status.className = 'status error';
            break;
    }
}

function displayLatency(component, latencyMs, details) {
    const metric = details?.metric;
    
    switch (component) {
        case 'stt':
            // STT only has one metric (total time)
            sttLatency.textContent = `${latencyMs}ms`;
            sttLatency.className = 'latency-value';
            applyLatencyColor(sttLatency, latencyMs);
            if (details) {
                sttLatency.title = Object.entries(details)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(', ');
            }
            break;
            
        case 'gemini':
            // Store both metrics
            if (metric === 'time_to_first_token') {
                latencyMetrics.gemini.firstToken = latencyMs;
                console.log(`[Gemini] Stored first token: ${latencyMs}ms`);
            } else if (metric === 'total_time') {
                latencyMetrics.gemini.total = latencyMs;
                console.log(`[Gemini] Stored total: ${latencyMs}ms`);
            } else {
                console.warn(`[Gemini] Unknown metric: ${metric}`);
            }
            
            // Display both metrics
            updateGeminiLatencyDisplay(details);
            break;
            
        case 'tts':
            // Store both metrics
            if (metric === 'time_to_first_audio') {
                latencyMetrics.tts.firstAudio = latencyMs;
                console.log(`[TTS] Stored first audio: ${latencyMs}ms`);
            } else if (metric === 'total_time') {
                latencyMetrics.tts.total = latencyMs;
                console.log(`[TTS] Stored total: ${latencyMs}ms`);
            } else {
                console.warn(`[TTS] Unknown metric: ${metric}`);
            }
            
            // Display both metrics
            updateTTSLatencyDisplay(details);
            break;
    }
    
    console.log(`[Latency] ${component}: ${latencyMs}ms`, details || '');
}

function updateGeminiLatencyDisplay(details) {
    const { firstToken, total } = latencyMetrics.gemini;
    
    console.log('[Gemini Latency]', { firstToken, total, details });
    
    if (firstToken !== null && total !== null) {
        // Show both metrics: "first/total"
        geminiLatency.textContent = `${firstToken}ms / ${total}ms`;
        if (geminiBreakdown) {
            geminiBreakdown.textContent = `(first: ${firstToken}ms, total: ${total}ms)`;
            geminiBreakdown.className = 'latency-breakdown';
            geminiBreakdown.style.display = 'inline';
        }
        
        // Color code based on first token latency (perceived latency)
        applyLatencyColor(geminiLatency, firstToken);
    } else if (firstToken !== null) {
        // Only first token available
        geminiLatency.textContent = `${firstToken}ms (first)`;
        if (geminiBreakdown) {
            geminiBreakdown.textContent = '';
            geminiBreakdown.style.display = 'none';
        }
        applyLatencyColor(geminiLatency, firstToken);
    } else if (total !== null) {
        // Only total available
        geminiLatency.textContent = `${total}ms (total)`;
        if (geminiBreakdown) {
            geminiBreakdown.textContent = '';
            geminiBreakdown.style.display = 'none';
        }
        applyLatencyColor(geminiLatency, total);
    }
    
    // Add tooltip with details if available
    if (details) {
        const detailsText = Object.entries(details)
            .filter(([key]) => key !== 'metric') // Exclude metric from tooltip
            .map(([key, value]) => `${key}: ${value}`)
            .join(', ');
        if (detailsText) {
            geminiLatency.title = detailsText;
        }
    }
}

function updateTTSLatencyDisplay(details) {
    const { firstAudio, total } = latencyMetrics.tts;
    
    console.log('[TTS Latency]', { firstAudio, total, details });
    
    if (firstAudio !== null && total !== null) {
        // Show both metrics: "first/total"
        ttsLatency.textContent = `${firstAudio}ms / ${total}ms`;
        if (ttsBreakdown) {
            ttsBreakdown.textContent = `(first: ${firstAudio}ms, total: ${total}ms)`;
            ttsBreakdown.className = 'latency-breakdown';
            ttsBreakdown.style.display = 'inline';
        }
        
        // Color code based on first audio latency (perceived latency)
        applyLatencyColor(ttsLatency, firstAudio);
    } else if (firstAudio !== null) {
        // Only first audio available
        ttsLatency.textContent = `${firstAudio}ms (first)`;
        if (ttsBreakdown) {
            ttsBreakdown.textContent = '';
            ttsBreakdown.style.display = 'none';
        }
        applyLatencyColor(ttsLatency, firstAudio);
    } else if (total !== null) {
        // Only total available
        ttsLatency.textContent = `${total}ms (total)`;
        if (ttsBreakdown) {
            ttsBreakdown.textContent = '';
            ttsBreakdown.style.display = 'none';
        }
        applyLatencyColor(ttsLatency, total);
    }
    
    // Add tooltip with details if available
    if (details) {
        const detailsText = Object.entries(details)
            .filter(([key]) => key !== 'metric') // Exclude metric from tooltip
            .map(([key, value]) => `${key}: ${value}`)
            .join(', ');
        if (detailsText) {
            ttsLatency.title = detailsText;
        }
    }
}

function applyLatencyColor(element, latencyMs) {
    // Remove existing color classes
    element.classList.remove('latency-good', 'latency-ok', 'latency-slow');
    
    // Add color based on latency
    if (latencyMs < 500) {
        element.classList.add('latency-good');
    } else if (latencyMs < 1500) {
        element.classList.add('latency-ok');
    } else {
        element.classList.add('latency-slow');
    }
}

function displayTranscript(text, isFinal) {
    // Remove placeholder if exists
    const placeholder = transcript.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    // Remove previous interim results
    if (!isFinal) {
        const interim = transcript.querySelector('.interim');
        if (interim) {
            interim.remove();
        }
    }
    
    // Add new transcript
    const p = document.createElement('p');
    p.className = isFinal ? 'final' : 'interim';
    p.textContent = text;
    transcript.appendChild(p);
    
    // Scroll to bottom
    transcript.scrollTop = transcript.scrollHeight;
}

function displayGeminiResponse(text) {
    // Remove placeholder if exists
    const placeholder = geminiResponse.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    // Add new response
    const p = document.createElement('p');
    p.className = 'gemini-response-text';
    p.textContent = text;
    geminiResponse.appendChild(p);
    
    // Scroll to bottom
    geminiResponse.scrollTop = geminiResponse.scrollHeight;
}

function displayGeminiChunk(chunk, isComplete) {
    // Remove placeholder if exists
    const placeholder = geminiResponse.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    // Create or get response element for streaming
    let responseElement = geminiResponse.querySelector('.gemini-response-text');
    if (!responseElement) {
        responseElement = document.createElement('p');
        responseElement.className = 'gemini-response-text';
        geminiResponse.appendChild(responseElement);
        responseElement.textContent = ''; // Initialize empty
    }
    
    // Append chunk to existing text (streaming)
    if (chunk) {
        responseElement.textContent += chunk;
        
        // Auto-scroll to bottom to show latest text
        geminiResponse.scrollTop = geminiResponse.scrollHeight;
    }
    
    // If complete, we can do any final formatting
    if (isComplete) {
        console.log('Gemini streaming complete');
    }
}

// TTS Streaming Functions
function handleTTSStreamingStart(metadata) {
    // Initialize streaming TTS playback.
    // Args:
    //   metadata: Metadata object with sample_rate, format, etc.
    console.log('Starting TTS streaming:', metadata);
    isTTSStreaming = true;
    ttsAudioQueue = [];
    
    // Create AudioContext for streaming playback
    const sampleRate = metadata.sample_rate || 24000;
    ttsStreamingContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: sampleRate
    });
    
    // Resume context if suspended
    if (ttsStreamingContext.state === 'suspended') {
        ttsStreamingContext.resume();
    }
}

async function handleTTSStreamingChunk(audioData) {
    // Handle a streaming TTS audio chunk and play it immediately.
    // Args:
    //   audioData: Audio chunk as ArrayBuffer or Blob
    try {
        if (!ttsStreamingContext || ttsStreamingContext.state === 'closed') {
            console.warn('TTS streaming context not available');
            return;
        }
        
        // Convert ArrayBuffer or Blob to ArrayBuffer
        let arrayBuffer;
        if (audioData instanceof Blob) {
            arrayBuffer = await audioData.arrayBuffer();
        } else if (audioData instanceof ArrayBuffer) {
            arrayBuffer = audioData;
        } else {
            console.error('Unexpected audio data type:', audioData.constructor.name);
            return;
        }
        
        if (!arrayBuffer || arrayBuffer.byteLength === 0) {
            return;
        }
        
        const sampleRate = ttsStreamingContext.sampleRate;
        
        // Convert Int16 PCM to Float32 for Web Audio API
        const int16Array = new Int16Array(arrayBuffer);
        const float32Array = new Float32Array(int16Array.length);
        
        for (let i = 0; i < int16Array.length; i++) {
            // Normalize Int16 (-32768 to 32767) to Float32 (-1.0 to 1.0)
            float32Array[i] = int16Array[i] / 32768.0;
        }
        
        // Create audio buffer
        const audioBuffer = ttsStreamingContext.createBuffer(1, float32Array.length, sampleRate);
        audioBuffer.getChannelData(0).set(float32Array);
        
        // Create and play audio source
        const source = ttsStreamingContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ttsStreamingContext.destination);
        
        // Queue the source to play
        const currentTime = ttsStreamingContext.currentTime;
        const nextStartTime = ttsAudioQueue.length > 0 
            ? ttsAudioQueue[ttsAudioQueue.length - 1].endTime 
            : currentTime;
        
        source.start(nextStartTime);
        
        // Track this chunk
        ttsAudioQueue.push({
            source: source,
            startTime: nextStartTime,
            endTime: nextStartTime + audioBuffer.duration
        });
        
        // Don't clean up sources while streaming - we need to track them for completion
        // Only clean up if we're not streaming anymore
        if (!isTTSStreaming) {
            const now = ttsStreamingContext.currentTime;
            ttsAudioQueue = ttsAudioQueue.filter(item => item.endTime > now);
        }
        
    } catch (error) {
        console.error('Error playing TTS streaming chunk:', error);
    }
}

function handleTTSStreamingComplete() {
    // Finalize streaming TTS playback.
    console.log('TTS streaming complete');
    isTTSStreaming = false;
    
    // Wait for all queued audio to finish playing before closing context
    if (ttsAudioQueue.length > 0 && ttsStreamingContext) {
        // Find the latest end time from all queued chunks
        const latestEndTime = Math.max(...ttsAudioQueue.map(item => item.endTime));
        const currentTime = ttsStreamingContext.currentTime;
        const remainingDuration = Math.max(0, latestEndTime - currentTime);
        
        console.log(`Waiting ${(remainingDuration * 1000).toFixed(0)}ms for audio to finish`);
        
        // Wait for all audio to finish, plus a small buffer
        setTimeout(() => {
            if (ttsStreamingContext && ttsStreamingContext.state !== 'closed') {
                ttsStreamingContext.close();
                ttsStreamingContext = null;
            }
            ttsAudioQueue = [];
            console.log('TTS audio playback finished');
        }, (remainingDuration * 1000) + 500); // Add 500ms buffer
    } else {
        // No audio queued, close immediately
        if (ttsStreamingContext && ttsStreamingContext.state !== 'closed') {
            ttsStreamingContext.close();
            ttsStreamingContext = null;
        }
        ttsAudioQueue = [];
    }
}

async function handleTTSAudio(audioData, metadata) {
    try {
        console.log('Received TTS audio data:', {
            type: audioData.constructor.name,
            size: audioData.byteLength || audioData.size,
            metadata: metadata
        });
        
        // Convert ArrayBuffer or Blob to ArrayBuffer if needed
        let arrayBuffer;
        if (audioData instanceof Blob) {
            arrayBuffer = await audioData.arrayBuffer();
        } else if (audioData instanceof ArrayBuffer) {
            arrayBuffer = audioData;
        } else {
            console.error('Unexpected audio data type:', audioData.constructor.name);
            return;
        }
        
        if (!arrayBuffer || arrayBuffer.byteLength === 0) {
            console.error('Empty audio buffer received');
            return;
        }
        
        const sampleRate = metadata?.sample_rate || 24000;
        console.log(`Processing audio: ${arrayBuffer.byteLength} bytes at ${sampleRate}Hz`);
        
        // Resume AudioContext if suspended (required by some browsers)
        let playbackContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: sampleRate
        });
        
        if (playbackContext.state === 'suspended') {
            await playbackContext.resume();
        }
        
        // Convert Int16 PCM to Float32 for Web Audio API
        const int16Array = new Int16Array(arrayBuffer);
        const float32Array = new Float32Array(int16Array.length);
        
        for (let i = 0; i < int16Array.length; i++) {
            // Normalize Int16 (-32768 to 32767) to Float32 (-1.0 to 1.0)
            float32Array[i] = int16Array[i] / 32768.0;
        }
        
        // Create audio buffer
        const audioBuffer = playbackContext.createBuffer(1, float32Array.length, sampleRate);
        audioBuffer.getChannelData(0).set(float32Array);
        
        // Create and play audio source
        const source = playbackContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(playbackContext.destination);
        
        // Handle playback completion
        source.onended = () => {
            console.log('TTS audio playback completed');
            playbackContext.close();
        };
        
        source.start();
        console.log(`Playing TTS audio: ${(audioBuffer.duration).toFixed(2)}s`);
        
    } catch (error) {
        console.error('Error playing TTS audio:', error);
        console.error('Error details:', {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
    }
}

async function startRecording() {
    if (isRecording) return;
    
    // Reset latency metrics for new conversation
    latencyMetrics.gemini = { firstToken: null, total: null };
    latencyMetrics.tts = { firstAudio: null, total: null };
    geminiLatency.textContent = '-';
    geminiBreakdown.textContent = '';
    ttsLatency.textContent = '-';
    ttsBreakdown.textContent = '';
    sttLatency.textContent = '-';
    
    try {
        // Check if we're in a secure context (HTTPS or localhost)
        const isSecureContext = window.isSecureContext || 
                                window.location.protocol === 'https:' || 
                                window.location.hostname === 'localhost' || 
                                window.location.hostname === '127.0.0.1';
        
        if (!isSecureContext) {
            throw new Error(
                'Microphone access requires a secure connection (HTTPS).\n' +
                'Please access this page via:\n' +
                '- https://your-domain.com (HTTPS)\n' +
                '- http://localhost:8080 (localhost is secure)\n' +
                'Current URL: ' + window.location.href
            );
        }
        
        // Check if getUserMedia is available
        let getUserMedia = null;
        
        // Try modern API first
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
        }
        // Fallback to legacy API (deprecated but still used by some browsers)
        else if (navigator.getUserMedia) {
            getUserMedia = (constraints) => {
                return new Promise((resolve, reject) => {
                    navigator.getUserMedia(constraints, resolve, reject);
                });
            };
        }
        // Check if WebRTC is supported at all
        else if (!navigator.mediaDevices) {
            throw new Error(
                'WebRTC is not supported in this browser.\n' +
                'Please use a modern browser like Chrome, Firefox, Safari, or Edge.\n' +
                'Make sure you\'re using the latest version.'
            );
        }
        else {
            throw new Error(
                'getUserMedia is not available.\n' +
                'This might be due to:\n' +
                '1. Browser not supporting WebRTC\n' +
                '2. Page not loaded over HTTPS or localhost\n' +
                '3. Browser permissions blocked\n' +
                'Current URL: ' + window.location.href
            );
        }
        
        // Check protocol
        const protocol = window.location.protocol;
        if (protocol === 'file:') {
            throw new Error(
                'Cannot access microphone from file:// URL.\n' +
                'Please run a local server:\n' +
                'cd Frontend && python3 -m http.server 8080\n' +
                'Then access: http://localhost:8080'
            );
        }
        
        // Connect WebSocket if not connected
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            await connectWebSocket();
        }
        
        // Get user media using the appropriate API
        if (!mediaStream) {
            mediaStream = await getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
        }
        
        // Create AudioContext for processing
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });
        
        const source = audioContext.createMediaStreamSource(mediaStream);
        
        // Create ScriptProcessorNode for audio processing
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        processor.onaudioprocess = (event) => {
            if (!isRecording || !websocket || websocket.readyState !== WebSocket.OPEN) {
                return;
            }
            
            const inputData = event.inputBuffer.getChannelData(0);
            
            // Convert Float32Array to Int16Array (PCM format)
            const int16Array = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const s = Math.max(-1, Math.min(1, inputData[i]));
                int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            
            // Send audio chunk via WebSocket
            try {
                websocket.send(int16Array.buffer);
            } catch (e) {
                console.error('Error sending audio:', e);
            }
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        isRecording = true;
        talkButton.classList.add('recording');
        talkButton.querySelector('.button-text').textContent = 'Recording...';
        status.textContent = 'Recording...';
        status.className = 'status recording';
        
        // Clear interim results when starting new recording
        const interim = transcript.querySelector('.interim');
        if (interim) {
            interim.remove();
        }
        
    } catch (error) {
        console.error('Error starting recording:', error);
        let errorMessage = error.message || 'Unknown error';
        
        // Handle specific error types
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            errorMessage = 'Microphone access denied.\n\nPlease:\n1. Click the lock icon in your browser address bar\n2. Allow microphone access\n3. Refresh the page';
        } else if (error.name === 'NotFoundError') {
            errorMessage = 'No microphone found. Please connect a microphone and try again.';
        } else if (error.message && error.message.includes('file://')) {
            errorMessage = 'Cannot access microphone from file:// URL.\n\nPlease:\n1. Start a local server: cd Frontend && python3 -m http.server 8080\n2. Open http://localhost:8080 in your browser';
        } else if (error.message && (error.message.includes('secure connection') || error.message.includes('HTTPS'))) {
            // This is the secure context error - provide detailed instructions
            const hostname = window.location.hostname;
            const isIPAddress = /^\d+\.\d+\.\d+\.\d+$/.test(hostname);
            
            if (isIPAddress) {
                errorMessage = 'Microphone access requires HTTPS when accessing via IP address.\n\n' +
                              'SOLUTIONS:\n\n' +
                              'Option 1 (Recommended): Access via localhost on each machine\n' +
                              'On each laptop, run:\n' +
                              '  cd Frontend && python3 -m http.server 8080\n' +
                              'Then open: http://localhost:8080\n\n' +
                              'Option 2: Set up HTTPS\n' +
                              'Use a tool like ngrok or set up a local HTTPS server.\n\n' +
                              'Current URL: ' + window.location.href;
            } else {
                errorMessage = error.message;
            }
        }
        
        status.textContent = errorMessage;
        status.className = 'status error';
        alert(errorMessage);
    }
}

function stopRecording() {
    if (!isRecording) return;
    
    isRecording = false;
    talkButton.classList.remove('recording');
    talkButton.querySelector('.button-text').textContent = 'Hold to Talk';
    status.textContent = 'Processing...';
    status.className = 'status';
    
    // Stop audio processing
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    
    // Send end signal to server
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        try {
            websocket.send(JSON.stringify({ type: 'end' }));
        } catch (e) {
            console.error('Error sending end signal:', e);
        }
    }
}

// Initialize - connect on page load
window.addEventListener('load', () => {
    connectWebSocket().catch(error => {
        console.error('Failed to connect:', error);
        status.textContent = 'Failed to connect to server. Make sure backend is running on port 8000.';
        status.className = 'status error';
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (isRecording) {
        stopRecording();
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
        audioContext.close();
    }
    if (websocket) {
        websocket.close();
    }
});

