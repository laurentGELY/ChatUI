"""
Unit tests for pure helper functions (no I/O, no mocking needed).
"""
import pytest
import main as app_module
from main import strip_markdown, build_messages


# ── strip_markdown ────────────────────────────────────────────────────────────

class TestStripMarkdown:

    def test_removes_bold(self):
        assert strip_markdown("**bold**") == "bold"

    def test_removes_italic_stars(self):
        assert strip_markdown("*italic*") == "italic"

    def test_removes_italic_underscores(self):
        assert strip_markdown("_italic_") == "italic"

    def test_removes_bold_underscores(self):
        assert strip_markdown("__bold__") == "bold"

    def test_removes_inline_code(self):
        assert strip_markdown("`code`") == "code"

    def test_removes_heading(self):
        assert strip_markdown("## Title") == "Title"

    def test_removes_heading_all_levels(self):
        for level in range(1, 7):
            assert strip_markdown(f"{'#' * level} Heading") == "Heading"

    def test_removes_bullet_dash(self):
        result = strip_markdown("- item one")
        assert result == "item one"

    def test_removes_bullet_star(self):
        result = strip_markdown("* item one")
        assert result == "item one"

    def test_removes_horizontal_rule(self):
        result = strip_markdown("---")
        assert result == ""

    def test_collapses_multiple_blank_lines(self):
        result = strip_markdown("line1\n\n\n\nline2")
        assert result == "line1\n\nline2"

    def test_plain_text_unchanged(self):
        text = "This is plain text with no markdown."
        assert strip_markdown(text) == text

    def test_mixed_content(self):
        text = "**Bold** and _italic_ and `code`."
        assert strip_markdown(text) == "Bold and italic and code."

    def test_empty_string(self):
        assert strip_markdown("") == ""


# ── build_messages ────────────────────────────────────────────────────────────

class TestBuildMessages:

    def test_starts_with_system_message(self, prompts_dir, monkeypatch):
        monkeypatch.setattr(app_module, "PROMPTS_DIR", prompts_dir)
        session_id = "test-session-id"
        messages = build_messages([], "Hello", session_id)
        assert messages[0]["role"] == "system"
        assert "test assistant" in messages[0]["content"]

    def test_appends_user_input_last(self, prompts_dir, monkeypatch):
        monkeypatch.setattr(app_module, "PROMPTS_DIR", prompts_dir)
        messages = build_messages([], "My question", "sess")
        assert messages[-1] == {"role": "user", "content": "My question"}

    def test_includes_history(self, prompts_dir, monkeypatch):
        monkeypatch.setattr(app_module, "PROMPTS_DIR", prompts_dir)
        history = [
            {"role": "user",      "content": "First"},
            {"role": "assistant", "content": "Reply"},
        ]
        messages = build_messages(history, "Second", "sess")
        # system + 2 history + new user = 4
        assert len(messages) == 4
        assert messages[1] == history[0]
        assert messages[2] == history[1]
        assert messages[3]["content"] == "Second"

    def test_empty_history(self, prompts_dir, monkeypatch):
        monkeypatch.setattr(app_module, "PROMPTS_DIR", prompts_dir)
        messages = build_messages([], "Hi", "sess")
        assert len(messages) == 2  # system + user
