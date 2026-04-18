from app.schemas import Mode
from app.mode_manager import set_current_mode
from app.state import current_state


VOICE_MODE_COMMANDS = {
    "kids mode": Mode.kids,
    "switch to kids mode": Mode.kids,
    "go to kids mode": Mode.kids,
    "child mode": Mode.kids,

    "young adult mode": Mode.young_adult,
    "switch to young adult mode": Mode.young_adult,
    "go to young adult mode": Mode.young_adult,
    "teen mode": Mode.young_adult,

    "adult mode": Mode.adult,
    "switch to adult mode": Mode.adult,
    "go to adult mode": Mode.adult,
    "productivity mode": Mode.adult,
}


def normalize_command(command: str) -> str:
    return command.strip().lower()


def handle_voice_command(command: str) -> dict:
    normalized = normalize_command(command)
    current_state["last_voice_command"] = normalized

    if normalized in VOICE_MODE_COMMANDS:
        new_mode = VOICE_MODE_COMMANDS[normalized]
        result = set_current_mode(new_mode)

        return {
            "recognized": True,
            "command": normalized,
            "mode_changed": True,
            "new_mode": result["mode"],
            "message": f"Mode changed to {result['mode']}.",
        }

    return {
        "recognized": False,
        "command": normalized,
        "mode_changed": False,
        "new_mode": current_state["mode"],
        "message": "Command not recognized. Mode was not changed.",
    }