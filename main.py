#!/usr/bin/env python3
"""
Chatbot UI
==========
Version: 2.5.0
Licence: MIT

Interface web générique pour chatbots propulsés par Ollama.
Les prompts sont des fichiers .txt dans prompts/ (interchangeables via l'UI).

Fonctionnalités :
  - Réponses en streaming token par token (SSE)
  - Arrêt du streaming (bouton Stop + AbortController côté client)
  - Régénération de la dernière réponse
  - TTS via Piper (optionnel) avec détection de langue automatique
  - STT via faster-whisper (optionnel, chargé à la demande)
  - Sélecteur de modèle, de prompt, de voix et de température dans l'UI
  - Compteur de tokens avec détection du contexte Ollama
  - Export de la conversation en .txt
  - Notification navigateur quand la réponse est prête (onglet en arrière-plan)

LANCEMENT
---------
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000

DÉPENDANCES OPTIONNELLES
------------------------
TTS  : binaire Piper dans bin/piper + modèles .onnx dans voices/
STT  : pip install faster-whisper  &&  sudo apt install ffmpeg
       Le modèle Whisper 'small' (~240 Mo) est téléchargé au premier appel /stt.
"""

import asyncio
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import uuid
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, File, Request, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Configuration ──────────────────────────────────────────────────────────────

CODE_VERSION = "2.5.0"

OLLAMA_URL           = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

BASE_DIR      = Path(__file__).parent
PROMPTS_DIR   = BASE_DIR / "prompts"
TEMPLATES_DIR = BASE_DIR / "templates"

PIPER_BIN      = BASE_DIR / "bin" / "piper"
PIPER_LIBS_DIR = BASE_DIR / "bin" / "piper_amd64"
VOICES_DIR     = BASE_DIR / "voices"
DEFAULT_VOICE  = "fr_FR-gilles-low.onnx"

# Environnement Piper précalculé une fois au démarrage
PIPER_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": str(PIPER_LIBS_DIR),
    "ESPEAK_DATA_PATH": str(PIPER_LIBS_DIR / "espeak-ng-data"),
}

STT_MAX_AUDIO_MB = 25  # limite de taille pour les uploads audio STT

# Filet de sécurité absolu — pas de truncation silencieuse,
# le compteur de caractères dans l'UI prend le relais pour avertir.
MAX_USER_INPUT_LEN = 8192

DEFAULT_PROMPT_FILE = "system_prompt.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Client HTTP partagé ────────────────────────────────────────────────────────

http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application lifespan and shared HTTP client.

    Creates a shared ``httpx.AsyncClient`` on startup and closes it cleanly
    on shutdown, ensuring connection pools are released.
    """
    global http_client
    http_client = httpx.AsyncClient(timeout=120.0)
    yield
    await http_client.aclose()

# ── État ───────────────────────────────────────────────────────────────────────

CONVERSATIONS:   Dict[str, List[Dict[str, str]]] = {}
PROMPT_STATES:   Dict[str, str]                  = {}  # session → filename | "__custom__"
CUSTOM_PROMPTS:  Dict[str, str]                  = {}  # session → texte personnalisé
SESSION_STATS:   Dict[str, Dict]                 = {}  # session → {prompt_tokens, ctx_size}
MODEL_CTX_CACHE: Dict[str, tuple[int, bool]]     = {}  # model → (num_ctx, is_approx)
STT_MODEL:      Optional[object] = None           # faster-whisper (chargé à la demande)
_STT_LOAD_LOCK: threading.Lock  = threading.Lock()  # évite les chargements concurrents

app = FastAPI(title="MyThott", version=CODE_VERSION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ── Helpers ────────────────────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """Remove Markdown formatting markers from text.

    Strips headings, bold/italic, inline code, horizontal rules, and
    bullet points. Used to clean LLM output before TTS synthesis and
    plain-text display.

    Args:
        text: Input text potentially containing Markdown markers.

    Returns:
        Plain text with all Markdown markers removed and consecutive
        blank lines collapsed to at most one.
    """
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_{2}(.+?)_{2}',   r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(.+?)\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_(.+?)_',   r'\1', text, flags=re.DOTALL)
    text = re.sub(r'`(.+?)`',   r'\1', text)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_session_prompt(session_id: str) -> str:
    """Return the active system prompt text for a session.

    Resolves the prompt from ``PROMPT_STATES``:
    - ``"__custom__"`` → returns the inline text from ``CUSTOM_PROMPTS``.
    - A filename → reads and returns that file from ``PROMPTS_DIR``.
    - Not set → falls back to ``DEFAULT_PROMPT_FILE``.

    Args:
        session_id: The session identifier.

    Returns:
        The system prompt text. Returns a descriptive error string if the
        file is missing or the resolved path is outside ``PROMPTS_DIR``.
    """
    state = PROMPT_STATES.get(session_id)
    if state == "__custom__":
        return CUSTOM_PROMPTS.get(session_id, "[ERREUR : prompt personnalisé non défini]")
    filename = state or DEFAULT_PROMPT_FILE
    path = (PROMPTS_DIR / filename).resolve()
    if path.parent != PROMPTS_DIR.resolve() or path.suffix != ".txt":
        logger.error(f"Chemin prompt invalide : {path}")
        return "[ERREUR : chemin prompt invalide]"
    if not path.exists():
        logger.error(f"Prompt manquant : {path}")
        return "[ERREUR : créez prompts/system_prompt.txt]"
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def get_or_create_session_id(request: Request) -> str:
    """Return the session ID from the request cookie, or generate a new one.

    Args:
        request: The incoming HTTP request.

    Returns:
        A UUID session ID string (existing from cookie, or freshly generated).
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Nouvelle session : {session_id[:8]}")
    return session_id


async def get_model_ctx(model: str) -> tuple[int, bool]:
    """Retrieve the effective context window size for an Ollama model.

    Queries ``/api/show`` and resolves the context size using this priority:

    1. Explicit ``num_ctx`` in Modelfile parameters → exact value (``is_approx=False``).
    2. Native context length from ``model_info``    → approximate (may be inflated by VRAM).
    3. Ollama historical default of 2048            → safe fallback.

    Results are cached in ``MODEL_CTX_CACHE`` to avoid repeated API calls.

    Args:
        model: Ollama model identifier (e.g. ``"llama3.1:8b-instruct-q4_K_M"``).

    Returns:
        A ``(num_ctx, is_approx)`` tuple. ``is_approx`` is ``True`` when the
        value is an estimate rather than the explicitly configured context size.
    """
    if model in MODEL_CTX_CACHE:
        return MODEL_CTX_CACHE[model]

    ctx, approx = 2048, True
    try:
        resp = await http_client.post(
            f"{OLLAMA_URL}/api/show",
            json={"model": model},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # Priorité 1 : num_ctx explicite dans le Modelfile (valeur réelle chargée)
        m = re.search(r'\bnum_ctx\s+(\d+)', data.get("parameters", ""))
        if m:
            ctx, approx = int(m.group(1)), False
        else:
            # Priorité 2 : contexte natif du modèle (approximatif — peut être surestimé)
            info = data.get("model_info", {})
            native = (
                info.get("llama.context_length")
                or info.get("general.context_length")
                or info.get("context_length")
            )
            if native and isinstance(native, int) and native > 0:
                ctx, approx = native, True

        logger.info(f"Contexte {model} : {ctx} tokens {'(approx)' if approx else '(exact)'}")
    except Exception as e:
        logger.warning(f"Impossible de lire le contexte de {model} : {e}")

    MODEL_CTX_CACHE[model] = (ctx, approx)
    return ctx, approx


def build_messages(history: List[Dict[str, str]], user_input: str, session_id: str) -> List[Dict[str, str]]:
    """Build the message list for an Ollama API call.

    Prepends the session system prompt, appends the full conversation
    history, and adds the new user message at the end. The full history
    is always included; the token counter in the UI warns when the
    context window is nearly full.

    Args:
        history: Previous conversation turns as ``{"role": ..., "content": ...}`` dicts.
        user_input: The new user message text.
        session_id: Session identifier used to look up the active system prompt.

    Returns:
        A list of message dicts (system + history + new user message) ready
        for the Ollama ``/api/chat`` endpoint.
    """
    messages = [{"role": "system", "content": get_session_prompt(session_id)}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    return messages


async def call_ollama(messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> tuple[str, int]:
    """Call the Ollama /api/chat endpoint and return the full reply (non-streaming).

    Args:
        messages: Conversation messages including the system prompt.
        model: Ollama model identifier.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).

    Returns:
        A ``(reply_text, prompt_eval_count)`` tuple. On connection or API
        errors, returns a descriptive error string and ``0`` for token count.
    """
    try:
        resp = await http_client.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"temperature": temperature}},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"], data.get("prompt_eval_count", 0)
    except httpx.ConnectError:
        logger.error("Ollama non accessible")
        return "[ERREUR : Ollama non démarré — lancez 'ollama serve']", 0
    except Exception as e:
        logger.error(f"Erreur Ollama : {e}")
        return f"[ERREUR : {e}]", 0

# ── STT (faster-whisper) ───────────────────────────────────────────────────────

def get_stt_model():
    """Return the faster-whisper model, loading it on first call (lazy, thread-safe).

    Uses double-checked locking via ``_STT_LOAD_LOCK`` to ensure the model
    is loaded only once even under concurrent requests.

    Returns:
        A ``WhisperModel`` instance (``small``, CPU, int8).

    Raises:
        RuntimeError: If ``faster-whisper`` is not installed.
    """
    global STT_MODEL
    if STT_MODEL is not None:
        return STT_MODEL
    with _STT_LOAD_LOCK:  # double-checked locking : un seul thread charge le modèle
        if STT_MODEL is None:
            try:
                from faster_whisper import WhisperModel
                STT_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
                logger.info("Whisper 'small' chargé")
            except ImportError:
                raise RuntimeError("faster-whisper non installé — lancez : pip install faster-whisper")
    return STT_MODEL


@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Transcribe an uploaded audio file to text using faster-whisper.

    Accepts webm, wav, or ogg audio. Files larger than ``STT_MAX_AUDIO_MB``
    are rejected. Transcription runs in a thread pool to avoid blocking the
    event loop.

    Args:
        audio: Uploaded audio file.

    Returns:
        JSON ``{"text": ..., "language": ...}`` on success, or
        ``{"error": ...}`` with an appropriate HTTP status on failure.
    """
    tmp_path = None
    try:
        content = await audio.read()
        if len(content) > STT_MAX_AUDIO_MB * 1024 * 1024:
            return JSONResponse({"error": f"Fichier audio trop grand (max {STT_MAX_AUDIO_MB} Mo)"}, status_code=413)
        suffix  = Path(audio.filename or "rec.webm").suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        def transcribe():
            model          = get_stt_model()
            segments, info = model.transcribe(tmp_path, beam_size=5)
            text           = " ".join(s.text.strip() for s in segments).strip()
            return text, info.language

        text, lang = await asyncio.get_running_loop().run_in_executor(None, transcribe)
        logger.info(f"STT ({lang}) : «{text[:60]}»")
        return JSONResponse({"text": text, "language": lang})

    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        logger.error(f"STT erreur : {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── TTS (Piper) ────────────────────────────────────────────────────────────────

@app.get("/voices")
def list_voices():
    """List available Piper TTS voice model files.

    Returns:
        JSON with ``voices`` (sorted list of ``.onnx`` filenames) and ``default``.
    """
    models = sorted(p.name for p in VOICES_DIR.glob("*.onnx"))
    return {"voices": models, "default": DEFAULT_VOICE}


@app.get("/tts")
async def tts_piper(text: str, voice: str = DEFAULT_VOICE):
    """Synthesize speech from text using the Piper TTS binary.

    Runs Piper in a thread pool to avoid blocking the event loop. Voice path
    is validated to stay within ``VOICES_DIR`` (path-traversal guard).

    Args:
        text: Text to synthesize.
        voice: Filename of the ``.onnx`` voice model in the voices directory.

    Returns:
        A WAV audio stream on success, or an HTML error response if Piper
        is missing, the voice file is invalid, or synthesis fails.
    """
    model_path = (VOICES_DIR / voice).resolve()
    if model_path.parent != VOICES_DIR.resolve() or model_path.suffix != ".onnx":
        return HTMLResponse("Voix invalide", status_code=400)
    if not PIPER_BIN.exists():
        return HTMLResponse(f"Piper introuvable : {PIPER_BIN}", status_code=500)
    if not model_path.exists():
        return HTMLResponse(f"Modèle introuvable : {model_path}", status_code=500)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        proc = await asyncio.get_running_loop().run_in_executor(None, lambda: subprocess.run(
            [str(PIPER_BIN), "--model", str(model_path), "--output_file", tmp_path],
            input=text.strip().encode("utf-8"),
            capture_output=True, timeout=30, env=PIPER_ENV,
        ))
        if proc.returncode != 0:
            logger.error(f"Piper stderr : {proc.stderr.decode()}")
            return HTMLResponse("Piper a échoué", status_code=500)
        with open(tmp_path, "rb") as f:
            wav = f.read()
        return StreamingResponse(io.BytesIO(wav), media_type="audio/wav",
                                 headers={"Cache-Control": "no-cache"})
    except subprocess.TimeoutExpired:
        return HTMLResponse("TTS timeout", status_code=500)
    except Exception as e:
        logger.error(f"TTS erreur : {e}")
        return HTMLResponse(f"TTS erreur : {e}", status_code=500)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/models")
async def list_models():
    """List Ollama models available on the local server.

    Returns:
        JSON with ``models`` (sorted list of model name strings) and ``default``.
        Returns an empty ``models`` list if Ollama is unreachable.
    """
    try:
        resp = await http_client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        names = sorted(m["name"] for m in resp.json().get("models", []))
        return {"models": names, "default": OLLAMA_DEFAULT_MODEL}
    except Exception as e:
        logger.warning(f"Impossible de lister les modèles Ollama : {e}")
        return {"models": [], "default": OLLAMA_DEFAULT_MODEL}


@app.get("/prompts")
def list_prompts():
    """List system prompt files available in the prompts directory.

    Returns:
        JSON with ``prompts`` (sorted list of ``.txt`` filenames) and ``default``.
    """
    files = sorted(p.name for p in PROMPTS_DIR.glob("*.txt"))
    return {"prompts": files, "default": DEFAULT_PROMPT_FILE}


@app.post("/session/prompt")
async def set_session_prompt(request: Request):
    """Set the active system prompt for the current session.

    Accepts a JSON body with either:
    - ``{"file": "<filename.txt>"}`` — selects a file from ``prompts/``.
    - ``{"text": "<prompt text>"}`` — sets an inline custom prompt.

    Args:
        request: HTTP request with JSON body.

    Returns:
        JSON ``{"ok": true, "file": filename}`` on success, or
        ``{"ok": false, "error": ...}`` with an appropriate HTTP status.
    """
    session_id = get_or_create_session_id(request)
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "JSON invalide"}, status_code=400)

    if "text" in data and data["text"].strip():
        CUSTOM_PROMPTS[session_id] = data["text"].strip()
        PROMPT_STATES[session_id]  = "__custom__"
        logger.info(f"Prompt personnalisé — session {session_id[:8]}, {len(data['text'])} chars")
        return JSONResponse({"ok": True, "file": "__custom__"})

    if "file" in data:
        filename = Path(data["file"]).name
        if not filename.endswith(".txt"):
            return JSONResponse({"ok": False, "error": "fichier invalide"}, status_code=400)
        if not (PROMPTS_DIR / filename).exists():
            return JSONResponse({"ok": False, "error": "fichier introuvable"}, status_code=404)
        PROMPT_STATES[session_id] = filename
        CUSTOM_PROMPTS.pop(session_id, None)
        logger.info(f"Prompt → {filename} — session {session_id[:8]}")
        return JSONResponse({"ok": True, "file": filename})

    return JSONResponse({"ok": False, "error": "paramètre manquant"}, status_code=400)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main chat UI page.

    Sets a ``session_id`` cookie on first visit. Passes conversation history
    and token statistics (prompt tokens vs. context window) to the template.

    Args:
        request: The incoming HTTP request (session cookie read here).

    Returns:
        HTML response rendered from ``templates/index.html``.
    """
    session_id     = get_or_create_session_id(request)
    history        = CONVERSATIONS.get(session_id, [])
    current_prompt = PROMPT_STATES.get(session_id, DEFAULT_PROMPT_FILE)

    # Statistiques de tokens pour la statusbar
    raw   = SESSION_STATS.get(session_id)
    token_stats = None
    if raw and history:
        pt  = raw["prompt_tokens"]
        ctx = raw["ctx_size"]
        pct = round(pt / ctx * 100) if ctx else 0
        token_stats = {
            "prompt_tokens": pt,
            "ctx_size":      ctx,
            "ctx_approx":    raw["ctx_approx"],
            "pct":           pct,
            "level": "danger" if pct > 90 else "warn" if pct > 75 else "ok",
        }

    response = templates.TemplateResponse("index.html", {
        "request":        request,
        "history":        history,
        "session_id":     session_id[:8],
        "code_version":   CODE_VERSION,
        "default_model":  OLLAMA_DEFAULT_MODEL,
        "current_prompt": current_prompt,
        "token_stats":    token_stats,
        "max_input_len":  MAX_USER_INPUT_LEN,
    })
    if "session_id" not in request.cookies:
        response.set_cookie("session_id", session_id, httponly=True)
    return response


@app.post("/send", response_class=RedirectResponse)
async def send_message(
    request: Request,
    user_input: str   = Form(...),
    model: str        = Form(OLLAMA_DEFAULT_MODEL),
    temperature: float = Form(0.7),
):
    """Send a message and wait for the complete Ollama reply (blocking).

    Calls Ollama synchronously, appends the exchange to session history,
    updates token stats, and redirects to ``/``. Prefer ``/stream`` for
    interactive use to avoid request timeouts on long replies.

    Args:
        request: HTTP request (session cookie).
        user_input: The user's message text (truncated to ``MAX_USER_INPUT_LEN``).
        model: Ollama model identifier.
        temperature: Sampling temperature.

    Returns:
        A 303 redirect to ``/``.
    """
    if len(user_input) > MAX_USER_INPUT_LEN:
        user_input = user_input[:MAX_USER_INPUT_LEN]

    session_id = get_or_create_session_id(request)
    history    = CONVERSATIONS.setdefault(session_id, [])

    messages              = build_messages(history, user_input, session_id)
    reply, prompt_tokens  = await call_ollama(messages, model, temperature)
    reply                 = strip_markdown(reply)

    ctx_size, ctx_approx      = await get_model_ctx(model)
    SESSION_STATS[session_id] = {
        "prompt_tokens": prompt_tokens,
        "ctx_size":      ctx_size,
        "ctx_approx":    ctx_approx,
    }

    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": reply})

    logger.info(
        f"Tour ajouté — session {session_id[:8]}, modèle {model}, "
        f"{len(history)} messages, {prompt_tokens}/{ctx_size} tokens"
    )
    return RedirectResponse(url="/", status_code=303)


@app.get("/export")
async def export_conversation(request: Request):
    """Download the current session's conversation as a plain-text file.

    Args:
        request: HTTP request (session cookie).

    Returns:
        A plain-text file attachment named
        ``conversation_YYYYMMDD_HHMM.txt``, or a 404 response if the
        session has no conversation history.
    """
    session_id = get_or_create_session_id(request)
    history    = CONVERSATIONS.get(session_id, [])
    if not history:
        return HTMLResponse("Aucune conversation à exporter.", status_code=404)

    lines = [f"Conversation — {datetime.now().strftime('%Y-%m-%d %H:%M')}", "=" * 50, ""]
    for msg in history:
        role = "Vous" if msg["role"] == "user" else "Bot"
        lines += [f"[{role}]", msg["content"], ""]

    filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    return StreamingResponse(
        io.BytesIO("\n".join(lines).encode("utf-8")),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/stream")
async def stream_message_sse(
    request: Request,
    user_input: str    = Form(...),
    model: str         = Form(OLLAMA_DEFAULT_MODEL),
    temperature: float = Form(0.7),
    regenerate: bool   = Form(False),
):
    """Send a message and stream the Ollama reply token by token via SSE.

    Each SSE event is a JSON object with one of:
    - ``{"token": "..."}`` — partial reply text.
    - ``{"done": true, "prompt_tokens": N, "ctx_size": N, ...}`` — final stats.
    - ``{"error": "..."}`` — error message.

    Streaming stops immediately if the client disconnects (e.g. Stop button).
    If no complete response is received, the pending user message is rolled back.

    Args:
        request: HTTP request (session cookie, disconnect detection).
        user_input: The user's message text (truncated to ``MAX_USER_INPUT_LEN``).
        model: Ollama model identifier.
        temperature: Sampling temperature.
        regenerate: If ``True``, remove the last user/assistant exchange before
            re-submitting (response regeneration).

    Returns:
        A ``text/event-stream`` streaming response.
    """
    if len(user_input) > MAX_USER_INPUT_LEN:
        user_input = user_input[:MAX_USER_INPUT_LEN]

    session_id = get_or_create_session_id(request)
    history    = CONVERSATIONS.setdefault(session_id, [])

    # Régénération : retire le dernier échange (user + assistant) de l'historique
    if regenerate and len(history) >= 2:
        history.pop()  # assistant
        history.pop()  # user

    messages = build_messages(history, user_input, session_id)
    history.append({"role": "user", "content": user_input})

    async def generate():
        full_reply    = ""
        done_received = False
        try:
            async with http_client.stream(
                "POST", f"{OLLAMA_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": True,
                      "options": {"temperature": temperature}},
                timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=5.0),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    # Abandon si le client s'est déconnecté (bouton Stop ou fermeture)
                    if await request.is_disconnected():
                        logger.info(f"Client déconnecté — session {session_id[:8]}")
                        break
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("message", {}).get("content", "")
                    if token:
                        full_reply += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if data.get("done"):
                        done_received        = True
                        prompt_tokens        = data.get("prompt_eval_count", 0)
                        clean_reply          = strip_markdown(full_reply)
                        history.append({"role": "assistant", "content": clean_reply})
                        ctx_size, ctx_approx = await get_model_ctx(model)
                        SESSION_STATS[session_id] = {
                            "prompt_tokens": prompt_tokens,
                            "ctx_size":      ctx_size,
                            "ctx_approx":    ctx_approx,
                        }
                        pct   = round(prompt_tokens / ctx_size * 100) if ctx_size else 0
                        level = "danger" if pct > 90 else "warn" if pct > 75 else "ok"
                        logger.info(
                            f"Stream — session {session_id[:8]}, modèle {model}, "
                            f"{len(history)} msg, {prompt_tokens}/{ctx_size} tokens"
                        )
                        yield f"data: {json.dumps({'done': True, 'prompt_tokens': prompt_tokens, 'ctx_size': ctx_size, 'ctx_approx': ctx_approx, 'pct': pct, 'level': level})}\n\n"

        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': '[ERREUR : Ollama non démarré — lancez ollama serve]'})}\n\n"
        except Exception as e:
            logger.error(f"Stream erreur : {e}")
            yield f"data: {json.dumps({'error': f'[ERREUR : {e}]'})}\n\n"
        finally:
            # Cleanup unifié : retire le message user si aucune réponse complète reçue
            if not done_received and history and history[-1]["role"] == "user":
                history.pop()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reset", response_class=RedirectResponse)
async def reset_conversation(request: Request):
    """Clear the current session's conversation history and token stats.

    Model, prompt, voice, and temperature settings are preserved.

    Args:
        request: HTTP request (session cookie).

    Returns:
        A 303 redirect to ``/``.
    """
    session_id = get_or_create_session_id(request)
    CONVERSATIONS.pop(session_id, None)
    SESSION_STATS.pop(session_id, None)
    logger.info(f"Session reset : {session_id[:8]}")
    return RedirectResponse(url="/", status_code=303)


@app.get("/health")
async def health_check():
    """Return server health and configuration status.

    Returns:
        JSON with app version, Ollama connectivity status, Piper binary
        availability, available voice model names, and active session count.
    """
    ollama_ok = False
    try:
        resp = await http_client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        ollama_ok = resp.status_code == 200
    except Exception:
        pass
    return {
        "version":          CODE_VERSION,
        "ollama_default":   OLLAMA_DEFAULT_MODEL,
        "ollama_status":    "OK" if ollama_ok else "DOWN",
        "piper_ok":         PIPER_BIN.exists(),
        "voices_available": sorted(p.name for p in VOICES_DIR.glob("*.onnx")),
        "active_sessions":  len(CONVERSATIONS),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
