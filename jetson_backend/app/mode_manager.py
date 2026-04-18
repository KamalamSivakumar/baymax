from app.schemas import Mode
from app.state import current_state


def get_current_mode() -> Mode:
    return current_state["mode"]


def set_current_mode(mode: Mode) -> dict:
    current_state["mode"] = mode

    return {
        "mode": mode,
        "description": get_mode_description(mode),
        "config": get_mode_config(mode),
    }


def get_mode_description(mode: Mode) -> str:
    descriptions = {
        Mode.kids: (
            "Safe default mode with simple games, visual prompts, "
            "speaker prompts, and restricted motor speed."
        ),
        Mode.young_adult: (
            "Interactive mode with games, focus activities, voice commands, "
            "and moderate motor speed."
        ),
        Mode.adult: (
            "Productivity and wellness mode with task planning, guided meditation, "
            "and calm motor behavior."
        ),
    }

    return descriptions[mode]


def get_mode_config(mode: Mode) -> dict:
    configs = {
        Mode.kids: {
            "allowed_features": [
                "simple_video_game",
                "storytelling",
                "memory_game",
                "speaker_prompts",
                "lcd_icons",
                "basic_drive",
            ],
            "voice_enabled": False,
            "motor_speed": "low",
            "tone": "playful",
        },
        Mode.young_adult: {
            "allowed_features": [
                "reaction_game",
                "focus_sprint",
                "quiz_game",
                "voice_commands",
                "basic_drive",
            ],
            "voice_enabled": True,
            "motor_speed": "medium",
            "tone": "casual",
        },
        Mode.adult: {
            "allowed_features": [
                "guided_meditation",
                "task_planning",
                "daily_checkin",
                "voice_commands",
                "basic_drive",
            ],
            "voice_enabled": True,
            "motor_speed": "low",
            "tone": "calm",
        },
    }

    return configs[mode]