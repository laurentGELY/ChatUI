"""
Shared fixtures for all test modules.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from starlette.testclient import TestClient

import main as app_module


SAMPLE_PROMPT = "You are a test assistant. Answer concisely."


@pytest.fixture
def prompts_dir(tmp_path):
    """Temporary prompts directory with a default system_prompt.txt."""
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "system_prompt.txt").write_text(SAMPLE_PROMPT)
    (d / "other_prompt.txt").write_text("You are another assistant.")
    return d


@pytest.fixture
def voices_dir(tmp_path):
    """Temporary voices directory with a fake .onnx file."""
    d = tmp_path / "voices"
    d.mkdir()
    (d / "fr_FR-gilles-low.onnx").write_bytes(b"fake-model")
    (d / "fr_FR-gilles-low.onnx.json").write_text("{}")
    (d / "en_US-ryan-low.onnx").write_bytes(b"fake-model")
    (d / "en_US-ryan-low.onnx.json").write_text("{}")
    return d


@pytest.fixture
def mock_http():
    """Return an AsyncMock that replaces ``main.http_client`` in tests.

    Injected by the ``client`` fixture so no real Ollama server is needed.
    Individual tests configure return values on this mock as required.
    """
    return AsyncMock()


@pytest.fixture
def client(prompts_dir, voices_dir, mock_http, monkeypatch):
    """
    Full TestClient with:
    - isolated prompts and voices directories
    - http_client replaced by an AsyncMock (no real Ollama needed)
    - fresh conversation state
    """
    monkeypatch.setattr(app_module, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(app_module, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(app_module, "DEFAULT_VOICE", "fr_FR-gilles-low.onnx")

    # Clear all in-memory state before each test
    app_module.CONVERSATIONS.clear()
    app_module.SESSION_STATS.clear()
    app_module.PROMPT_STATES.clear()
    app_module.MODEL_CTX_CACHE.clear()

    with TestClient(app_module.app, raise_server_exceptions=True) as c:
        # Replace the real httpx client created by lifespan with our mock
        monkeypatch.setattr(app_module, "http_client", mock_http)
        yield c


def make_ollama_response(content="Hello!", prompt_tokens=10):
    """Build a mock httpx response for a non-streaming Ollama call."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "message": {"content": content},
        "prompt_eval_count": prompt_tokens,
    }
    return resp


def make_show_response(num_ctx=None, native_ctx=None):
    """Build a mock httpx response for /api/show (model context detection)."""
    params = f"num_ctx {num_ctx}\ntemperature 0.7" if num_ctx else ""
    model_info = {}
    if native_ctx:
        model_info["llama.context_length"] = native_ctx
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"parameters": params, "model_info": model_info}
    return resp
