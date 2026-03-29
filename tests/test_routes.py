"""
Integration tests for HTTP routes.
Uses a TestClient with mocked Ollama (no real server needed).
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import main as app_module
from tests.conftest import make_ollama_response, make_show_response


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:

    def test_ollama_up(self, client, mock_http):
        resp = MagicMock()
        resp.status_code = 200
        mock_http.get = AsyncMock(return_value=resp)

        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["ollama_status"] == "OK"
        assert "version" in data

    def test_ollama_down(self, client, mock_http):
        mock_http.get = AsyncMock(side_effect=Exception("refused"))

        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["ollama_status"] == "DOWN"


# ── /models ───────────────────────────────────────────────────────────────────

class TestModels:

    def test_returns_model_list(self, client, mock_http):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]}
        mock_http.get = AsyncMock(return_value=resp)

        r = client.get("/models")
        assert r.status_code == 200
        assert "llama3:8b" in r.json()["models"]

    def test_ollama_unavailable_returns_empty(self, client, mock_http):
        mock_http.get = AsyncMock(side_effect=Exception("connection refused"))

        r = client.get("/models")
        assert r.status_code == 200
        assert r.json()["models"] == []


# ── /prompts ──────────────────────────────────────────────────────────────────

class TestPrompts:

    def test_lists_txt_files(self, client, prompts_dir):
        r = client.get("/prompts")
        assert r.status_code == 200
        data = r.json()
        assert "system_prompt.txt" in data["prompts"]
        assert "other_prompt.txt" in data["prompts"]

    def test_default_is_system_prompt(self, client):
        r = client.get("/prompts")
        assert r.json()["default"] == "system_prompt.txt"


# ── /voices ───────────────────────────────────────────────────────────────────

class TestVoices:

    def test_lists_onnx_files(self, client, voices_dir):
        r = client.get("/voices")
        assert r.status_code == 200
        voices = r.json()["voices"]
        assert "fr_FR-gilles-low.onnx" in voices
        assert "en_US-ryan-low.onnx" in voices

    def test_empty_dir_returns_empty_list(self, client, tmp_path, monkeypatch):
        empty = tmp_path / "empty_voices"
        empty.mkdir()
        monkeypatch.setattr(app_module, "VOICES_DIR", empty)
        r = client.get("/voices")
        assert r.json()["voices"] == []


# ── /session/prompt ───────────────────────────────────────────────────────────

class TestSessionPrompt:

    def test_set_prompt_by_file(self, client):
        r = client.post("/session/prompt",
                        json={"file": "system_prompt.txt"},
                        cookies={"session_id": "abc123"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_set_custom_prompt(self, client):
        r = client.post("/session/prompt",
                        json={"text": "You are a custom bot."},
                        cookies={"session_id": "abc123"})
        assert r.status_code == 200
        assert r.json()["file"] == "__custom__"

    def test_missing_file_returns_404(self, client):
        r = client.post("/session/prompt",
                        json={"file": "nonexistent.txt"})
        assert r.status_code == 404

    def test_invalid_extension_rejected(self, client):
        r = client.post("/session/prompt",
                        json={"file": "malicious.sh"})
        assert r.status_code == 400

    def test_empty_body_returns_400(self, client):
        r = client.post("/session/prompt", json={})
        assert r.status_code == 400


# ── /reset ────────────────────────────────────────────────────────────────────

class TestReset:

    def test_clears_conversation(self, client):
        sid = "reset-test-session"
        app_module.CONVERSATIONS[sid] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        client.post("/reset", cookies={"session_id": sid},
                    allow_redirects=False)
        assert sid not in app_module.CONVERSATIONS

    def test_redirects_to_home(self, client):
        r = client.post("/reset", allow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/"


# ── /export ───────────────────────────────────────────────────────────────────

class TestExport:

    def test_no_history_returns_404(self, client):
        r = client.get("/export", cookies={"session_id": "empty-session"})
        assert r.status_code == 404

    def test_returns_text_file(self, client):
        sid = "export-test-session"
        app_module.CONVERSATIONS[sid] = [
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        r = client.get("/export", cookies={"session_id": sid})
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        body = r.text
        assert "What is 2+2?" in body
        assert "4" in body

    def test_filename_in_content_disposition(self, client):
        sid = "export-cd-session"
        app_module.CONVERSATIONS[sid] = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        r = client.get("/export", cookies={"session_id": sid})
        assert "attachment" in r.headers.get("content-disposition", "")
        assert ".txt" in r.headers["content-disposition"]


# ── /send (blocking) ──────────────────────────────────────────────────────────

class TestSend:

    def test_happy_path_redirects(self, client, mock_http):
        # /api/show for context size
        mock_http.post = AsyncMock(side_effect=[
            make_ollama_response("The answer is 42.", prompt_tokens=20),
            make_show_response(num_ctx=4096),
        ])
        # second call is get_model_ctx → /api/show uses post
        mock_http.post = AsyncMock(return_value=make_ollama_response("The answer is 42."))
        # Also mock get_model_ctx
        async def fake_ctx(model):
            return 4096, False
        with patch.object(app_module, "get_model_ctx", fake_ctx):
            r = client.post("/send",
                            data={"user_input": "What is 6×7?", "model": "llama3:8b"},
                            allow_redirects=False,
                            cookies={"session_id": "send-session"})
        assert r.status_code == 303

    def test_history_updated(self, client, mock_http):
        sid = "send-history-session"
        mock_http.post = AsyncMock(return_value=make_ollama_response("Bonjour!"))
        async def fake_ctx(model):
            return 2048, True
        with patch.object(app_module, "get_model_ctx", fake_ctx):
            client.post("/send",
                        data={"user_input": "Hello", "model": "llama3:8b"},
                        cookies={"session_id": sid})
        history = app_module.CONVERSATIONS.get(sid, [])
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_input_too_long_is_truncated(self, client, mock_http):
        sid = "truncate-session"
        mock_http.post = AsyncMock(return_value=make_ollama_response("ok"))
        async def fake_ctx(model):
            return 2048, True
        long_input = "x" * (app_module.MAX_USER_INPUT_LEN + 500)
        with patch.object(app_module, "get_model_ctx", fake_ctx):
            client.post("/send",
                        data={"user_input": long_input, "model": "llama3:8b"},
                        cookies={"session_id": sid})
        history = app_module.CONVERSATIONS.get(sid, [])
        assert len(history[0]["content"]) == app_module.MAX_USER_INPUT_LEN


# ── /stream (SSE) ─────────────────────────────────────────────────────────────

class TestStream:

    def _make_stream_mock(self, mock_http, tokens, prompt_tokens=15):
        """Configure mock_http.stream to yield the given tokens then done."""
        lines = [json.dumps({"message": {"content": t}, "done": False}) for t in tokens]
        lines.append(json.dumps({
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": prompt_tokens,
        }))

        async def aiter_lines():
            for line in lines:
                yield line

        stream_resp = MagicMock()
        stream_resp.raise_for_status = MagicMock()
        stream_resp.aiter_lines = aiter_lines
        stream_resp.__aenter__ = AsyncMock(return_value=stream_resp)
        stream_resp.__aexit__ = AsyncMock(return_value=False)
        # stream() is used as `async with`, not awaited — must be a plain Mock
        mock_http.stream = MagicMock(return_value=stream_resp)

    def test_returns_sse_content_type(self, client, mock_http):
        self._make_stream_mock(mock_http, ["Hello", " world"])
        async def fake_ctx(model): return 2048, True
        with patch.object(app_module, "get_model_ctx", fake_ctx):
            r = client.post("/stream",
                            data={"user_input": "Hi", "model": "llama3:8b"})
        assert "text/event-stream" in r.headers["content-type"]

    def test_tokens_in_response(self, client, mock_http):
        self._make_stream_mock(mock_http, ["Bon", "jour"])
        async def fake_ctx(model): return 2048, True
        with patch.object(app_module, "get_model_ctx", fake_ctx):
            r = client.post("/stream",
                            data={"user_input": "Hi", "model": "llama3:8b"})
        assert "Bon" in r.text
        assert "jour" in r.text

    def test_done_event_in_response(self, client, mock_http):
        self._make_stream_mock(mock_http, ["Hi"])
        async def fake_ctx(model): return 2048, True
        with patch.object(app_module, "get_model_ctx", fake_ctx):
            r = client.post("/stream",
                            data={"user_input": "Hi", "model": "llama3:8b"})
        assert '"done": true' in r.text or '"done":true' in r.text

    def test_history_saved_after_stream(self, client, mock_http):
        sid = "stream-history-session"
        self._make_stream_mock(mock_http, ["Reply"])
        async def fake_ctx(model): return 2048, True
        with patch.object(app_module, "get_model_ctx", fake_ctx):
            client.post("/stream",
                        data={"user_input": "Hello", "model": "llama3:8b"},
                        cookies={"session_id": sid})
        history = app_module.CONVERSATIONS.get(sid, [])
        assert len(history) == 2
        assert history[0]["content"] == "Hello"

    def test_ollama_error_yields_error_event(self, client, mock_http):
        mock_http.stream.side_effect = Exception("connection refused")
        r = client.post("/stream",
                        data={"user_input": "Hi", "model": "llama3:8b"})
        assert "error" in r.text

    def test_history_not_saved_on_error(self, client, mock_http):
        sid = "stream-error-session"
        mock_http.stream.side_effect = Exception("boom")
        client.post("/stream",
                    data={"user_input": "Hello", "model": "llama3:8b"},
                    cookies={"session_id": sid})
        history = app_module.CONVERSATIONS.get(sid, [])
        assert history == []


# ── get_model_ctx ─────────────────────────────────────────────────────────────

class TestGetModelCtx:

    @pytest.mark.asyncio
    async def test_reads_explicit_num_ctx(self, mock_http):
        app_module.MODEL_CTX_CACHE.clear()
        mock_http.post = AsyncMock(return_value=make_show_response(num_ctx=8192))
        app_module.http_client = mock_http

        ctx, approx = await app_module.get_model_ctx("test-model")
        assert ctx == 8192
        assert approx is False

    @pytest.mark.asyncio
    async def test_falls_back_to_native_ctx(self, mock_http):
        app_module.MODEL_CTX_CACHE.clear()
        mock_http.post = AsyncMock(return_value=make_show_response(native_ctx=32768))
        app_module.http_client = mock_http

        ctx, approx = await app_module.get_model_ctx("test-model-2")
        assert ctx == 32768
        assert approx is True

    @pytest.mark.asyncio
    async def test_falls_back_to_2048_on_error(self, mock_http):
        app_module.MODEL_CTX_CACHE.clear()
        mock_http.post = AsyncMock(side_effect=Exception("network error"))
        app_module.http_client = mock_http

        ctx, approx = await app_module.get_model_ctx("unknown-model")
        assert ctx == 2048
        assert approx is True

    @pytest.mark.asyncio
    async def test_result_is_cached(self, mock_http):
        app_module.MODEL_CTX_CACHE.clear()
        mock_http.post = AsyncMock(return_value=make_show_response(num_ctx=4096))
        app_module.http_client = mock_http

        await app_module.get_model_ctx("cached-model")
        await app_module.get_model_ctx("cached-model")
        assert mock_http.post.call_count == 1  # second call used cache
