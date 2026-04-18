from fastapi import FastAPI

from app.schemas import Mode, DeviceModeRequest, DeviceActionRequest
from app.state import device_state
from app.display_controller import update_display
from app.audio_controller import play_prompt
from app.motor_controller import apply_motor_profile, execute_motor_action
from app.button_controller import read_buttons


app = FastAPI(
    title="DeskBot Raspberry Pi Device Service",
    description="Hardware interface service running on Raspberry Pi.",
    version="0.1.0",
)


def get_mode_profile(mode: Mode) -> dict:
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


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "DeskBot Raspberry Pi device service is active.",
    }


@app.get("/device/status")
def device_status():
    buttons = read_buttons()

    return {
        "status": "success",
        "device_state": device_state,
        "buttons": buttons,
    }


@app.post("/device/mode")
def set_device_mode(request: DeviceModeRequest):
    profile = get_mode_profile(request.mode)

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


@app.post("/device/action")
def device_action(request: DeviceActionRequest):
    current_mode = device_state["mode"]
    current_motor_profile = device_state["motor_profile"]
    device_state["last_action"] = request.action

    motor_result = execute_motor_action(
        action=request.action,
        mode=current_mode.value,
        motor_profile=current_motor_profile,
    )

    if request.action == "play_prompt":
        audio_result = play_prompt(device_state["speaker_prompt"])

    elif request.action == "start_memory_game":
        audio_result = play_prompt("Memory game starting.")

    elif request.action == "start_story":
        audio_result = play_prompt("Story time starting.")

    elif request.action == "start_reaction_game":
        audio_result = play_prompt("Reaction game starting.")

    elif request.action == "start_focus_sprint":
        audio_result = play_prompt("Focus sprint starting.")

    elif request.action == "start_guided_meditation":
        audio_result = play_prompt("Starting guided meditation.")

    elif request.action == "start_task_planner":
        audio_result = play_prompt("Opening task planner.")

    elif request.action == "stop":
        audio_result = play_prompt("Stopping.")

    else:
        audio_result = {
            "audio_played": False,
            "message": "No audio prompt configured for this action.",
        }

    return {
        "status": "executed",
        "mode": current_mode,
        "action": request.action,
        "motor": motor_result,
        "audio": audio_result,
    }