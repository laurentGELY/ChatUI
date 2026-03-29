# MyThott

A local, privacy-first web UI for chatbots powered by [Ollama](https://ollama.ai).
All inference runs on your machine — no cloud, no data leaving your network.

---

## Features

| Feature | Details |
|---|---|
| **Streaming responses** | Tokens appear in real time via Server-Sent Events (SSE) |
| **Stop streaming** | Interrupt a response mid-generation |
| **Speech-to-text** | Record your voice with the mic button; transcribed by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper `small`, CPU) |
| **Text-to-speech** | Bot replies read aloud via [Piper](https://github.com/rhasspy/piper) (optional) |
| **Language-adaptive voice** | Detects French vs English in the reply and switches voice automatically |
| **Model selector** | Pick any Ollama model loaded on your machine |
| **Prompt selector** | Switch system prompts from the UI; add `.txt` files to `prompts/` |
| **Custom prompt** | Edit the system prompt inline in the expert panel |
| **Temperature slider** | Adjust creativity (0 = deterministic, 1 = creative; default 0.7) |
| **Regenerate** | Re-run the last user message with a different random seed |
| **Token counter** | Live prompt-token / context-window gauge with colour warning |
| **Conversation export** | Download the full conversation as a `.txt` file |
| **Session reset** | Clear history without losing settings |
| **Browser notification** | Notified when the response completes while the tab is in background |

---

## Requirements

### Runtime

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| [Ollama](https://ollama.ai) | any recent | must be running (`ollama serve`) |
| ffmpeg | system package | required for STT audio decoding |

### Python packages

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
jinja2==3.1.4
httpx==0.27.0
python-multipart==0.0.9
faster-whisper==1.2.1
```

### Optional — TTS

- `bin/piper` — Piper binary (Linux x86-64)
- `bin/piper_amd64/` — Piper shared libraries + espeak-ng data
- `voices/*.onnx` + `voices/*.onnx.json` — Piper voice models

Piper and voice models are **not included** in the repository due to size.
Download them from the [Piper releases page](https://github.com/rhasspy/piper/releases).

---

## Installation

```bash
# 1. Clone
git clone <repo-url> ChatUI
cd ChatUI

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Python dependencies
pip install -r requirements.txt

# 4. System dependency for STT
sudo apt install ffmpeg

# 5. Create the prompts directory and a default system prompt
mkdir -p prompts
# → edit prompts/system_prompt.txt  (see Prompts section below)

# 6. (Optional) Install Piper TTS
#    Place the piper binary at bin/piper
#    Place piper_amd64/ libs at bin/piper_amd64/
#    Place .onnx + .onnx.json voice files in voices/
```

---

## Running

```bash
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

Ollama must be running separately:

```bash
ollama serve
```

---

## Configuration

Environment variables (set in shell or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Default model shown in selector |

---

## Prompts

Place `.txt` files in the `prompts/` directory.
Each file is a system prompt selectable from the UI.

The default file is `prompts/system_prompt.txt`.
You can add as many as you like — they appear automatically in the dropdown.

The expert panel also lets you edit the prompt directly in the browser for the current session without touching the files.

---

## Speech-to-Text

The first STT request downloads the Whisper `small` model (~240 MB) from Hugging Face.
Subsequent requests reuse the cached model (loaded once per server process).

Audio is captured in the browser via `MediaRecorder`, capped at **30 seconds**.
The transcribed text is inserted into the message input for review before sending.

The server automatically detects the spoken language and returns it alongside the transcript.

---

## Text-to-Speech

TTS is optional. If `bin/piper` is not found, the speaker button is hidden.

The UI auto-detects whether the bot's reply is French or English and selects the matching voice. You can also choose the voice manually from the dropdown.

Voice files must be placed in `voices/` as `<name>.onnx` + `<name>.onnx.json` pairs.
The default voice is `fr_FR-gilles-low.onnx`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Main chat UI |
| `POST` | `/stream` | Send message, stream response (SSE) |
| `POST` | `/send` | Send message, blocking (legacy) |
| `POST` | `/stt` | Transcribe audio file |
| `GET` | `/tts` | Synthesize speech (`?text=...&voice=...`) |
| `GET` | `/models` | List available Ollama models |
| `GET` | `/prompts` | List available prompt files |
| `POST` | `/session/prompt` | Set prompt for current session |
| `POST` | `/reset` | Clear conversation history |
| `GET` | `/export` | Download conversation as `.txt` |
| `GET` | `/voices` | List available Piper voice files |
| `GET` | `/health` | Health check (Ollama, Piper, sessions) |

---

## Testing

```bash
# Install dev dependencies
python3 -m pip install -r requirements-dev.txt

# Run all tests
python3 -m pytest tests/ -v
```

54 tests covering: helper functions, all HTTP routes, SSE streaming, STT endpoint.
No real Ollama or Piper needed — all external calls are mocked.

---

## Project Structure

```
ChatUI/
├── main.py                  # FastAPI application
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Test dependencies (pytest)
├── pytest.ini               # Pytest configuration
├── prompts/                 # System prompt files (.txt)
│   └── system_prompt.txt
├── templates/
│   └── index.html           # Single-page UI (Jinja2)
├── static/                  # CSS, JS, icons
├── tests/                   # Pytest test suite
│   ├── conftest.py
│   ├── test_helpers.py
│   ├── test_routes.py
│   └── test_stt.py
├── bin/
│   ├── piper                # Piper TTS binary (not in repo)
│   └── piper_amd64/         # Piper shared libraries (not in repo)
└── voices/                  # Piper voice models (.onnx, not in repo)
```

---

## Licence

MIT
