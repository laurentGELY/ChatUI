"""
Tests for the /stt (speech-to-text) endpoint.
faster-whisper is mocked so no model download is needed.
"""
import io
import pytest
from unittest.mock import MagicMock, patch

import main as app_module


def make_audio_upload(content=b"fake-audio-data", filename="recording.webm"):
    return ("audio", (filename, io.BytesIO(content), "audio/webm"))


class TestSTT:

    def test_transcribes_audio(self, client):
        segment = MagicMock()
        segment.text = "Bonjour tout le monde"
        mock_info = MagicMock()
        mock_info.language = "fr"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([segment], mock_info)

        with patch.object(app_module, "get_stt_model", return_value=mock_model):
            r = client.post("/stt", files={"audio": ("rec.webm", b"data", "audio/webm")})

        assert r.status_code == 200
        data = r.json()
        assert data["text"] == "Bonjour tout le monde"
        assert data["language"] == "fr"

    def test_multiple_segments_joined(self, client):
        seg1, seg2 = MagicMock(), MagicMock()
        seg1.text = "Hello"
        seg2.text = "world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        with patch.object(app_module, "get_stt_model", return_value=mock_model):
            r = client.post("/stt", files={"audio": ("rec.webm", b"data", "audio/webm")})

        assert r.json()["text"] == "Hello world"

    def test_missing_faster_whisper_returns_503(self, client):
        with patch.object(app_module, "get_stt_model",
                          side_effect=RuntimeError("faster-whisper non installé")):
            r = client.post("/stt", files={"audio": ("rec.webm", b"data", "audio/webm")})
        assert r.status_code == 503

    def test_oversized_audio_returns_413(self, client):
        big = b"x" * (app_module.STT_MAX_AUDIO_MB * 1024 * 1024 + 1)
        with patch.object(app_module, "get_stt_model", return_value=MagicMock()):
            r = client.post("/stt", files={"audio": ("rec.webm", big, "audio/webm")})
        assert r.status_code == 413

    def test_empty_transcript_returns_empty_string(self, client):
        mock_info = MagicMock()
        mock_info.language = "fr"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)

        with patch.object(app_module, "get_stt_model", return_value=mock_model):
            r = client.post("/stt", files={"audio": ("rec.webm", b"data", "audio/webm")})

        assert r.status_code == 200
        assert r.json()["text"] == ""
