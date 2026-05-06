"""
main.py — Raspberry Pi 3B body service.

Exposes all hardware subsystems (motors, LED, eyes, audio, camera) as a
REST API that the Jetson brain calls over the Ethernet link.

Also exposes a live MJPEG camera stream at:
    GET /camera/stream

And a browser preview page at:
    GET /
"""

import base64
import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

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


# ── Camera Setup ──────────────────────────────────────────────────────────────

try:
    from picamera2 import Picamera2
    import cv2

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
    )

    CAMERA_AVAILABLE = True

except Exception as e:
    logger.exception(f"Camera unavailable: {e}")
    picam2 = None
    CAMERA_AVAILABLE = False


def generate_frames():
    while True:
        if not CAMERA_AVAILABLE or picam2 is None:
            time.sleep(1)
            continue

        frame = picam2.capture_array()

        # Convert RGB to BGR for OpenCV JPEG encoding
        #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        jpg_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpg_bytes
            + b"\r\n"
        )

        time.sleep(0.03)  # ~30 FPS max


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global picam2

    if CAMERA_AVAILABLE and picam2 is not None:
        try:
            picam2.start()
            logger.info("Pi camera started")
        except Exception as e:
            logger.exception(f"Failed to start camera: {e}")

    logger.info("Baymax Pi body service started")


@app.on_event("shutdown")
async def shutdown():
    global picam2

    try:
        if CAMERA_AVAILABLE and picam2 is not None:
            picam2.stop()
            logger.info("Pi camera stopped")
    except Exception as e:
        logger.exception(f"Failed to stop camera cleanly: {e}")

    cleanup_gpio()


# ── Health / Root ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return HTMLResponse("""
    <html>
        <head>
            <title>Baymax Pi Body Service</title>
        </head>
        <body>
            <h1>Baymax Raspberry Pi Body Service</h1>

            <p>Status: running</p>

            <h2>Camera Stream</h2>
            <img src="/camera/stream" width="640" height="480" />

            <h2>Useful Endpoints</h2>
            <ul>
                <li><a href="/docs">FastAPI Docs</a></li>
                <li><a href="/health">Health</a></li>
                <li><a href="/device/status">Device Status</a></li>
                <li><a href="/camera/stream">Camera Stream</a></li>
            </ul>
        </body>
    </html>
    """)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Baymax Raspberry Pi Body Service",
        "camera_available": CAMERA_AVAILABLE,
    }


@app.get("/device/status")
def device_status():
    buttons = read_buttons()
    return {
        "status": "success",
        "device_state": device_state,
        "buttons": buttons,
        "camera_available": CAMERA_AVAILABLE,
    }


# ── Camera ────────────────────────────────────────────────────────────────────

@app.get("/camera/stream")
def camera_stream():
    if not CAMERA_AVAILABLE or picam2 is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Camera not available"}
        )

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ── Mode ──────────────────────────────────────────────────────────────────────

def _get_mode_profile(mode: Mode) -> dict:
    profiles = {
        Mode.kids: {
            "display_message": "Kids Mode",
            "speaker_prompt": "Let's play!",
            "motor_profile": "slow_drive",
        },
        Mode.young_adult: {
            "display_message": "Young Adult Mode",
            "speaker_prompt": "Choose a challenge.",
            "motor_profile": "medium_drive",
        },
        Mode.adult: {
            "display_message": "Adult Mode",
            "speaker_prompt": "How can I help you focus today?",
            "motor_profile": "slow_drive",
        },
    }
    return profiles[mode]


@app.post("/device/mode")
def set_device_mode(request: DeviceModeRequest):
    profile = _get_mode_profile(request.mode)

    device_state["mode"] = request.mode
    device_state["display_message"] = profile["display_message"]
    device_state["speaker_prompt"] = profile["speaker_prompt"]
    device_state["motor_profile"] = profile["motor_profile"]

    display_result = update_display(profile["display_message"])
    audio_result = play_prompt(profile["speaker_prompt"])
    motor_result = apply_motor_profile(profile["motor_profile"])

    return {
        "status": "applied",
        "mode": request.mode,
        "profile": profile,
        "display": display_result,
        "audio": audio_result,
        "motor": motor_result,
    }


# ── Action ────────────────────────────────────────────────────────────────────

ACTION_PROMPTS = {
    "start_trajectory_mapping_mat": "Starting trajectory mapping on mat.",
    "start_storytelling": "Story time starting.",
    "start_video_game_snake": "Snake game starting.",
    "start_video_game_memory": "Memory game starting.",
    "start_puzzles": "Puzzle challenge starting.",
    "start_peer_tutoring": "Peer tutoring session starting.",
    "start_trajectory_mapping_desktop": "Starting trajectory mapping on desktop.",
    "start_guided_meditation": "Starting guided meditation.",
    "start_facial_recognition": "Starting facial recognition.",
    "stop": "Stopping.",
}


@app.post("/device/action")
def device_action(request: DeviceActionRequest):
    current_mode = device_state["mode"]
    motor_profile = device_state["motor_profile"]
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
        "mode": current_mode,
        "action": request.action,
        "motor": motor_result,
        "audio": audio_result,
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
        return {
            "success": False,
            "audio_b64": "",
            "message": "Recording failed or PyAudio unavailable",
        }

    encoded = base64.b64encode(wav_bytes).decode("utf-8")

    return {
        "success": True,
        "audio_b64": encoded,
        "size_bytes": len(wav_bytes),
    }
