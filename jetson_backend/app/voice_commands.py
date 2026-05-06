"""
voice_commands.py — Routes text commands through the LLM brain.

The LLM handles all intent recognition; mode-switch shortcut phrases are kept
as a fast path so the UI voice box still works without LLM latency.
"""

from app.schemas import Mode
from app.mode_manager import set_current_mode
from app.state import current_state


# Fast-path phrases that bypass the LLM (mode switches only)
VOICE_MODE_COMMANDS = {
    "kids mode":              Mode.kids,
    "switch to kids mode":    Mode.kids,
    "go to kids mode":        Mode.kids,
    "child mode":             Mode.kids,

    "young adult mode":              Mode.young_adult,
    "switch to young adult mode":    Mode.young_adult,
    "go to young adult mode":        Mode.young_adult,
    "teen mode":                     Mode.young_adult,

    "adult mode":              Mode.adult,
    "switch to adult mode":    Mode.adult,
    "go to adult mode":        Mode.adult,
    "productivity mode":       Mode.adult,
}


def normalize_command(command: str) -> str:
    return command.strip().lower()


def handle_voice_command(command: str) -> dict:
    """
    First checks fast-path mode-switch phrases, then delegates to the LLM
    for any command that isn't a direct mode switch.
    """
    normalized = normalize_command(command)
    current_state["last_voice_command"] = normalized

    # Fast path: direct mode switch
    if normalized in VOICE_MODE_COMMANDS:
        new_mode = VOICE_MODE_COMMANDS[normalized]
        result   = set_current_mode(new_mode)
        return {
            "recognized":   True,
            "command":      normalized,
            "mode_changed": True,
            "new_mode":     result["mode"],
            "message":      f"Mode changed to {result['mode']}.",
            "llm_used":     False,
        }

    # LLM path: send to Baymax brain
    try:
        from app.llm_engine import chat
        llm_result = chat(command)
        return {
            "recognized":   True,
            "command":      normalized,
            "mode_changed": False,
            "new_mode":     current_state["mode"],
            "message":      llm_result["response"],
            "tool_calls":   llm_result["tool_calls"],
            "llm_used":     True,
        }
    except Exception as exc:
        return {
            "recognized":   False,
            "command":      normalized,
            "mode_changed": False,
            "new_mode":     current_state["mode"],
            "message":      f"Command not processed: {exc}",
            "llm_used":     True,
        }