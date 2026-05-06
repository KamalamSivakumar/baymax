import asyncio
import base64
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import (
    ModeRequest,
    VoiceCommandRequest,
    RobotActionRequest,
    LLMChatRequest,
    EyeRequest,
    LedRequest,
)
from app.state import current_state
from app.mode_manager import (
    get_current_mode,
    set_current_mode,
    get_mode_config,
    get_mode_description,
)
from app.voice_commands import handle_voice_command
from app.robot_controller import (
    get_robot_behavior_for_mode,
    validate_robot_action,
)
from app.pi_client import (
    push_mode_to_pi,
    push_action_to_pi,
    get_pi_status,
    push_eye_expression,
    push_led_color,
    trigger_audio_speak,
    record_audio_from_pi,
)

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Baymax Jetson Backend",
    description="LLM-driven robot brain running on Jetson Orin Nano.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── WebSocket connection manager ──────────────────────────────────────────────

class _ConnectionManager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket):
        self._clients.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        for ws in list(self._clients):
            try:
                await ws.send_text(msg)
            except Exception:
                self._clients.remove(ws)


ws_manager = _ConnectionManager()


async def _broadcast_state():
    """Push current state to all connected WebSocket clients."""
    mode = get_current_mode()
    await ws_manager.broadcast({
        "type": "state_update",
        "state": {
            "mode":               mode,
            "mode_description":   get_mode_description(mode),
            "mode_config":        get_mode_config(mode),
            "last_voice_command": current_state["last_voice_command"],
            "last_robot_action":  current_state["last_robot_action"],
            "pi_connected":       current_state["pi_connected"],
            "eye_state":          current_state["eye_state"],
            "led_state":          current_state["led_state"],
        },
    })


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    from app.llm_engine import register_builtin_tools
    register_builtin_tools()
    logger.info("Baymax Jetson backend started — tools registered")


# ── Static / root ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"status": "running", "message": "Baymax Jetson backend is active."}


@app.get("/health")
def health_check():
    return {"status": "ok", "model": os.environ.get("OLLAMA_MODEL", "llama3.2:3b")}


# ── State / Mode ──────────────────────────────────────────────────────────────

@app.get("/state")
def get_state():
    mode = get_current_mode()
    return {
        "status": "success",
        "state": {
            "mode":               mode,
            "mode_description":   get_mode_description(mode),
            "mode_config":        get_mode_config(mode),
            "last_voice_command": current_state["last_voice_command"],
            "last_robot_action":  current_state["last_robot_action"],
            "pi_connected":       current_state["pi_connected"],
            "eye_state":          current_state["eye_state"],
            "led_state":          current_state["led_state"],
        },
    }


@app.get("/mode")
def get_mode():
    mode = get_current_mode()
    return {
        "status":      "success",
        "mode":        mode,
        "description": get_mode_description(mode),
        "config":      get_mode_config(mode),
    }


@app.post("/mode")
async def update_mode(request: ModeRequest):
    backend_result = set_current_mode(request.mode)
    pi_result      = push_mode_to_pi(request.mode)
    current_state["pi_connected"] = pi_result["success"]
    await _broadcast_state()
    return {
        "status":       "success",
        "message":      f"Mode changed to {request.mode}.",
        "backend":      backend_result,
        "raspberry_pi": pi_result,
    }


# ── Voice command (LLM-backed) ────────────────────────────────────────────────

@app.post("/voice-command")
async def voice_command(request: VoiceCommandRequest):
    result    = handle_voice_command(request.command)
    pi_result = None
    if result["recognized"] and result["mode_changed"]:
        pi_result = push_mode_to_pi(result["new_mode"])
        current_state["pi_connected"] = pi_result["success"]
    await _broadcast_state()
    return {
        "status":       "success",
        "message":      result["message"],
        "voice":        result,
        "raspberry_pi": pi_result,
    }


# ── LLM Chat ──────────────────────────────────────────────────────────────────

@app.post("/llm/chat")
async def llm_chat(request: LLMChatRequest):
    from app.llm_engine import chat
    result = await asyncio.get_event_loop().run_in_executor(None, chat, request.message)
    await _broadcast_state()
    return {
        "status":     "success",
        "response":   result["response"],
        "tool_calls": result["tool_calls"],
        "rounds":     result["rounds"],
    }


@app.post("/llm/reset")
async def llm_reset():
    from app.llm_engine import reset_conversation
    reset_conversation()
    return {"status": "success", "message": "Conversation history cleared."}


# ── Robot actions ─────────────────────────────────────────────────────────────

@app.get("/robot/behavior")
def get_robot_behavior():
    mode     = get_current_mode()
    behavior = get_robot_behavior_for_mode(mode)
    return {"status": "success", "mode": mode, "behavior": behavior}


@app.post("/robot/action")
async def robot_action(request: RobotActionRequest):
    validation = validate_robot_action(request.action)
    if not validation["allowed"]:
        return {"status": "blocked", "message": validation["message"], "data": validation}
    pi_result = push_action_to_pi(request.action)
    current_state["pi_connected"] = pi_result["success"]
    await _broadcast_state()
    return {
        "status":       "success" if pi_result["success"] else "partial_failure",
        "message":      validation["message"],
        "validation":   validation,
        "raspberry_pi": pi_result,
    }


# ── Eye control ───────────────────────────────────────────────────────────────

@app.post("/eyes")
async def set_eyes(request: EyeRequest):
    result = push_eye_expression(request.expression)
    if result["success"]:
        current_state["eye_state"] = request.expression
    current_state["pi_connected"] = result["success"]
    await _broadcast_state()
    return {"status": "success" if result["success"] else "error", **result}


# ── LED control ───────────────────────────────────────────────────────────────

@app.post("/led")
async def set_led_endpoint(request: LedRequest):
    result = push_led_color(request.color)
    if result["success"]:
        current_state["led_state"] = request.color
    current_state["pi_connected"] = result["success"]
    await _broadcast_state()
    return {"status": "success" if result["success"] else "error", **result}


# ── Vision ────────────────────────────────────────────────────────────────────

@app.get("/vision/analyze")
def vision_analyze():
    from app.vision_processor import analyze_twister
    result = analyze_twister()
    return {"status": "success", **result}


@app.get("/vision/describe")
def vision_describe():
    from app.vision_processor import describe_scene
    result = describe_scene()
    return {"status": "success", **result}


# ── Audio ─────────────────────────────────────────────────────────────────────

@app.post("/audio/speak")
async def audio_speak(request: LLMChatRequest):
    """Synthesize text on Jetson and send to Pi for playback."""
    result = trigger_audio_speak(request.message)
    current_state["pi_connected"] = result["success"]
    return {"status": "success" if result["success"] else "error", **result}


@app.post("/audio/transcribe")
async def audio_transcribe():
    """Record from Pi mic and return transcript."""
    from app.audio_manager import transcribe
    audio_result = record_audio_from_pi(duration=5)
    if not audio_result["success"]:
        return {"status": "error", "message": audio_result.get("error", "Record failed")}
    wav_bytes = base64.b64decode(audio_result["audio_b64"])
    result    = transcribe(wav_bytes)
    return {"status": "success" if result["success"] else "error", **result}


# ── Pi status ─────────────────────────────────────────────────────────────────

@app.get("/pi/status")
def pi_status():
    result = get_pi_status()
    current_state["pi_connected"] = result["success"]
    return {"status": "success" if result["success"] else "error", **result}


# ── WebSocket push ────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    await _broadcast_state()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "chat" and msg.get("message"):
                from app.llm_engine import chat
                result = await asyncio.get_event_loop().run_in_executor(
                    None, chat, msg["message"]
                )
                await websocket.send_text(json.dumps({
                    "type":       "chat_response",
                    "response":   result["response"],
                    "tool_calls": result["tool_calls"],
                }))
                await _broadcast_state()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)