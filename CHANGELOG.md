# Changelog

All notable changes to this project are documented here.

---

## [2026-03-28] — docs

### Changed
- Added Google-style docstrings to all test classes in `tests/test_routes.py`,
  `tests/test_helpers.py`, and `tests/test_stt.py`.
- Updated `client` fixture docstring in `tests/conftest.py` to proper Google
  style (Args / Yields sections).
- Added docstring to `make_audio_upload` helper in `tests/test_stt.py`.
- Added `pipeline.sh` to the project structure listing in README.

---

## [2.5.0] — 2026-03-28

### Added
- Pytest test suite (54 tests) covering helper functions, all HTTP routes,
  SSE streaming, and the STT endpoint — no real Ollama or Piper required.
- `requirements-dev.txt` listing test dependencies (`pytest`, `pytest-asyncio`).
- Testing section in README with run instructions.
- Google-style docstrings on all Python functions in `main.py` and `tests/conftest.py`.

### Changed
- Updated module docstring in `main.py` to document all v2.5 features
  (streaming SSE, stop/regenerate, TTS, STT, token counter, export, prompt selector).
- Fixed version label mismatch in module docstring.
- README: added 25 MB server-side audio upload limit to the STT section.

---

## [2.5.0-initial] — 2025 (initial release)

### Added
- FastAPI web UI for Ollama-powered chatbots.
- Streaming responses token by token via Server-Sent Events (SSE).
- Stop streaming via client-side `AbortController` and server-side disconnect detection.
- Response regeneration (re-runs last user message with a new random seed).
- Text-to-speech via Piper binary with automatic French/English voice selection.
- Speech-to-text via faster-whisper (`small` model, lazy-loaded on first use).
- Model selector, prompt file selector, custom inline prompt editor (expert panel).
- Temperature slider (0.0–1.0, default 0.7).
- Token counter with context-window usage gauge (colour-coded warning levels).
- Conversation export as plain-text `.txt` file.
- Browser notification when response completes in a background tab.
- Session reset (clears history, preserves settings).
- `/health` endpoint reporting Ollama status, Piper availability, and session count.
- Path-traversal guard on voice and prompt file lookups.
- `OLLAMA_URL` and `OLLAMA_MODEL` environment variable configuration.
