from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ModeRequest, VoiceCommandRequest, RobotActionRequest
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
)


app = FastAPI(
    title="DeskBot Jetson Backend",
    description="Main backend running on Jetson Orin Nano.",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this later to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "DeskBot Jetson backend is active.",
        "default_mode": "kids",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Jetson backend is healthy.",
    }


@app.get("/state")
def get_state():
    mode = get_current_mode()

    return {
        "status": "success",
        "state": {
            "mode": mode,
            "mode_description": get_mode_description(mode),
            "mode_config": get_mode_config(mode),
            "last_voice_command": current_state["last_voice_command"],
            "last_robot_action": current_state["last_robot_action"],
            "pi_connected": current_state["pi_connected"],
        },
    }


@app.get("/mode")
def get_mode():
    mode = get_current_mode()

    return {
        "status": "success",
        "mode": mode,
        "description": get_mode_description(mode),
        "config": get_mode_config(mode),
    }


@app.post("/mode")
def update_mode(request: ModeRequest):
    backend_result = set_current_mode(request.mode)
    pi_result = push_mode_to_pi(request.mode)

    current_state["pi_connected"] = pi_result["success"]

    return {
        "status": "success",
        "message": f"Mode changed to {request.mode}.",
        "backend": backend_result,
        "raspberry_pi": pi_result,
    }


@app.post("/voice-command")
def voice_command(request: VoiceCommandRequest):
    result = handle_voice_command(request.command)

    pi_result = None

    if result["recognized"] and result["mode_changed"]:
        pi_result = push_mode_to_pi(result["new_mode"])
        current_state["pi_connected"] = pi_result["success"]

    return {
        "status": "success",
        "message": result["message"],
        "voice": result,
        "raspberry_pi": pi_result,
    }


@app.get("/robot/behavior")
def get_robot_behavior():
    mode = get_current_mode()
    behavior = get_robot_behavior_for_mode(mode)

    return {
        "status": "success",
        "mode": mode,
        "behavior": behavior,
    }


@app.post("/robot/action")
def robot_action(request: RobotActionRequest):
    validation = validate_robot_action(request.action)

    if not validation["allowed"]:
        return {
            "status": "blocked",
            "message": validation["message"],
            "data": validation,
        }

    pi_result = push_action_to_pi(request.action)
    current_state["pi_connected"] = pi_result["success"]

    return {
        "status": "success" if pi_result["success"] else "partial_failure",
        "message": validation["message"],
        "validation": validation,
        "raspberry_pi": pi_result,
    }


@app.get("/pi/status")
def pi_status():
    result = get_pi_status()
    current_state["pi_connected"] = result["success"]

    return {
        "status": "success" if result["success"] else "error",
        "raspberry_pi": result,
    }