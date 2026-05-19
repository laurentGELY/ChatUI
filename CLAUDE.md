# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the dev server
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Run all tests
python3 -m pytest tests/ -v

# Run a single test file
python3 -m pytest tests/test_helpers.py -v

# Run a single test by name
python3 -m pytest tests/test_routes.py::TestStream::test_stream_basic -v

# Install runtime dependencies
pip install -r requirements.txt

# Install test dependencies
pip install -r requirements-dev.txt

# Run the full pipeline (tests → requirements update → git commit + push)
./pipeline.sh "your commit message"
```

## Architecture

The entire backend is a single file: `main.py`. There is no package structure — everything (routes, helpers, state, config) lives there.

**Session model**: Sessions are keyed by a UUID stored in an `httponly` cookie. All session state lives in module-level dicts (`CONVERSATIONS`, `PROMPT_STATES`, `CUSTOM_PROMPTS`, `SESSION_STATS`, `MODEL_CTX_CACHE`). There is no database or persistent storage — all state is lost on server restart.

**Streaming**: `/stream` is the primary chat endpoint. It opens a streaming request to Ollama's `/api/chat` and forwards tokens via Server-Sent Events (SSE). It detects client disconnection via `await request.is_disconnected()` and rolls back the pending user message if no complete reply was received.

**Ollama integration**: A single shared `httpx.AsyncClient` is created at startup via the `lifespan` context manager and closed on shutdown. Context-window size per model is fetched once from `/api/show` and cached in `MODEL_CTX_CACHE`.

**TTS (Piper)**: Optional. The `bin/piper` binary and `bin/piper_amd64/` library directory are not in the repo. Piper runs as a subprocess in a thread pool executor. Path traversal is guarded by resolving voice paths and confirming they stay within `VOICES_DIR`.

**STT (faster-whisper)**: Optional. The Whisper `small` model is loaded lazily on first `/stt` request using double-checked locking (`_STT_LOAD_LOCK`) to prevent concurrent loads. Transcription runs in a thread pool executor to avoid blocking the event loop.

**Prompt system**: System prompts are `.txt` files in `prompts/`. The active prompt per session is tracked by filename in `PROMPT_STATES`, or `"__custom__"` when the user edits inline. `get_session_prompt()` validates that resolved paths stay within `PROMPTS_DIR` (path-traversal guard).

**Frontend**: Single Jinja2 template at `templates/index.html`. The UI is server-rendered with session state hydrated on page load. JS on the page handles SSE consumption, STT recording via `MediaRecorder`, and TTS playback.

## Test setup

Tests use `starlette.testclient.TestClient` with `monkeypatch` to replace `PROMPTS_DIR`, `VOICES_DIR`, and `http_client` (the Ollama HTTP client) with test doubles — no real Ollama or Piper needed. The shared fixtures live in `tests/conftest.py`. Helper functions `make_ollama_response()` and `make_show_response()` build mock httpx responses.

## Key environment variables

| Variable | Default |
|---|---|
| `OLLAMA_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.1:8b-instruct-q4_K_M` |
