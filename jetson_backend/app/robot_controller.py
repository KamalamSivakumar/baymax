from app.schemas import Mode
from app.state import current_state


def get_robot_behavior_for_mode(mode: Mode) -> dict:
    behaviors = {
        Mode.kids: {
            "lcd_message": "Kids Mode",
            "speaker_prompt": "Let's play!",
            "motor_profile": "slow_drive",
            "safety_level": "maximum",
        },
        Mode.young_adult: {
            "lcd_message": "Young Adult Mode",
            "speaker_prompt": "Choose a challenge.",
            "motor_profile": "medium_drive",
            "safety_level": "medium",
        },
        Mode.adult: {
            "lcd_message": "Adult Mode",
            "speaker_prompt": "How can I help you focus today?",
            "motor_profile": "slow_drive",
            "safety_level": "standard",
        },
    }

    return behaviors[mode]


def validate_robot_action(action: str) -> dict:
    mode = current_state["mode"]
    current_state["last_robot_action"] = action

    allowed_actions_by_mode = {
        Mode.kids: [
            "drive_forward",
            "drive_backward",
            "turn_left",
            "turn_right",
            "play_prompt",
            "start_memory_game",
            "start_story",
            "stop",
        ],
        Mode.young_adult: [
            "drive_forward",
            "drive_backward",
            "turn_left",
            "turn_right",
            "play_prompt",
            "start_reaction_game",
            "start_focus_sprint",
            "stop",
        ],
        Mode.adult: [
            "drive_forward",
            "drive_backward",
            "turn_left",
            "turn_right",
            "play_prompt",
            "start_guided_meditation",
            "start_task_planner",
            "stop",
        ],
    }

    if action not in allowed_actions_by_mode[mode]:
        return {
            "allowed": False,
            "mode": mode,
            "action": action,
            "message": f"Action '{action}' is not allowed in {mode} mode.",
        }

    return {
        "allowed": True,
        "mode": mode,
        "action": action,
        "message": f"Action '{action}' is allowed in {mode} mode.",
    }