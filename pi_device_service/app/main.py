"""
main.py — Raspberry Pi 3B body service.

Exposes all hardware subsystems (motors, LED, eyes, audio, camera) as a
REST API that the Jetson brain calls over the Ethernet link.
"""

import base64
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    Mode,
    DeviceModeRequest,
    DeviceActionRequest,
    EyeExpressionRequest,
    LedColorRequest,
    AudioRecordRequest,
    AudioSpeakRequest,
)
from app.state import device_state
from app.display_controller import update_display, set_expression
from app.audio_controller import play_prompt, play_text, record_audio, play_audio
from app.motor_controller import apply_motor_profile, execute_motor_action, cleanup_gpio
from app.button_controller import read_buttons
from app.led_controller import set_led

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Baymax Raspberry Pi Body Service",
    description="Hardware interface service running on Raspberry Pi 3B.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    from app.camera_stream import start as start_camera
    start_camera()
    logger.info("Baymax Pi body service started")


@app.on_event("shutdown")
async def shutdown():
    cleanup_gpio()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "message": "Baymax Raspberry Pi body service is active."}


@app.get("/device/status")
def device_status():
    buttons = read_buttons()
    return {"status": "success", "device_state": device_state, "buttons": buttons}


# ── Mode ──────────────────────────────────────────────────────────────────────

def _get_mode_profile(mode: Mode) -> dict:
    profiles = {
        Mode.kids: {
            "display_message": "Kids Mode",
            "speaker_prompt":  "Let's play!",
            "motor_profile":   "slow_drive",
        },
        Mode.young_adult: {
            "display_message": "Young Adult Mode",
            "speaker_prompt":  "Choose a challenge.",
            "motor_profile":   "medium_drive",
        },
        Mode.adult: {
            "display_message": "Adult Mode",
            "speaker_prompt":  "How can I help you focus today?",
            "motor_profile":   "slow_drive",
        },
    }
    return profiles[mode]


@app.post("/device/mode")
def set_device_mode(request: DeviceModeRequest):
    profile = _get_mode_profile(request.mode)

    device_state["mode"]            = request.mode
    device_state["display_message"] = profile["display_message"]
    device_state["speaker_prompt"]  = profile["speaker_prompt"]
    device_state["motor_profile"]   = profile["motor_profile"]

    display_result = update_display(profile["display_message"])
    audio_result   = play_prompt(profile["speaker_prompt"])
    motor_result   = apply_motor_profile(profile["motor_profile"])

    return {
        "status":  "applied",
        "mode":    request.mode,
        "profile": profile,
        "display": display_result,
        "audio":   audio_result,
        "motor":   motor_result,
    }


# ── Action ────────────────────────────────────────────────────────────────────

ACTION_PROMPTS = {
    "start_trajectory_mapping_mat":     "Starting trajectory mapping on mat.",
    "start_storytelling":               "Story time starting.",
    "start_video_game_snake":           "Snake game starting.",
    "start_video_game_memory":          "Memory game starting.",
    "start_puzzles":                    "Puzzle challenge starting.",
    "start_peer_tutoring":              "Peer tutoring session starting.",
    "start_trajectory_mapping_desktop": "Starting trajectory mapping on desktop.",
    "start_guided_meditation":          "Starting guided meditation.",
    "start_facial_recognition":         "Starting facial recognition.",
    "stop": "Stopping.",
}


@app.post("/device/action")
def device_action(request: DeviceActionRequest):
    current_mode    = device_state["mode"]
    motor_profile   = device_state["motor_profile"]
    device_state["last_action"] = request.action

    motor_result = execute_motor_action(
        action=request.action,
        mode=current_mode.value,
        motor_profile=motor_profile,
    )

    prompt = (
        device_state["speaker_prompt"]
        if request.action == "play_prompt"
        else ACTION_PROMPTS.get(request.action)
    )
    audio_result = (
        play_prompt(prompt)
        if prompt
        else {"audio_played": False, "message": "No audio prompt for this action."}
    )

    return {
        "status": "executed",
        "mode":   current_mode,
        "action": request.action,
        "motor":  motor_result,
        "audio":  audio_result,
    }


# ── Eyes ──────────────────────────────────────────────────────────────────────

@app.post("/device/eyes")
def set_device_eyes(request: EyeExpressionRequest):
    result = set_expression(request.expression)
    device_state["eye_expression"] = request.expression
    return {"status": "ok" if result.get("success") else "error", **result}


# ── LED ───────────────────────────────────────────────────────────────────────

@app.post("/device/led")
def set_device_led(request: LedColorRequest):
    result = set_led(request.color)
    device_state["led_color"] = request.color
    return {"status": "ok" if result.get("success") else "error", **result}


# ── Audio ─────────────────────────────────────────────────────────────────────

@app.post("/device/audio/speak")
def audio_speak(request: AudioSpeakRequest):
    result = play_text(request.text)
    return {"status": "ok" if result.get("audio_played") else "error", **result}


@app.post("/device/audio/record")
def audio_record(request: AudioRecordRequest):
    wav_bytes = record_audio(duration=request.duration)
    if not wav_bytes:
        return {"success": False, "audio_b64": "", "message": "Recording failed or PyAudio unavailable"}
    encoded = base64.b64encode(wav_bytes).decode("utf-8")
    return {"success": True, "audio_b64": encoded, "size_bytes": len(wav_bytes)}
