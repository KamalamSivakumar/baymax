"""
llm_engine.py — Ollama-backed LLM brain with tool-calling loop.

Uses the Ollama Python client to run a local model (default: llama3.2:3b).
On each turn:
  1. Append the user message to conversation history.
  2. Send full history + registered tools to Ollama.
  3. If the model calls a tool, dispatch it, append result, loop back to step 2.
  4. Return the final text response.

Configure via environment variables:
  OLLAMA_MODEL   — model tag (default: llama3.2:3b)
  OLLAMA_HOST    — Ollama server URL (default: http://localhost:11434)
"""

from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING

import ollama

from app.tools_registry import get_ollama_tools, dispatch
from app.state import current_state

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")

_client = ollama.Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = """You are Baymax, a friendly and helpful healthcare companion robot.
You control a physical robot body through a set of tools.

Available subsystems you can control:
- Drive motors (move forward, backward, turn left/right, stop)
- Tri-color LED (red, green, yellow, or off — use to signal mood/status)
- Animated eyes (normal, excited, disappointed, blink, sleeping)
- Camera vision (analyze Twister game state)
- Audio speech synthesis (say something out loud)

Guidelines:
- Be warm, reassuring, and concise. You are modelled after Disney's Baymax.
- When asked to do a physical action, call the relevant tool immediately.
- For Twister game mode, always analyze the camera before declaring success/failure.
- After completing a physical action, briefly acknowledge it in your text response.
- If a tool fails, report the issue simply and suggest a fix.
- Never invent tool results — only report what the tool actually returned.
"""


def _get_history() -> list[dict]:
    return current_state.setdefault("conversation_history", [])


def _ensure_system() -> None:
    history = _get_history()
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})


def chat(user_message: str, max_tool_rounds: int = 8) -> dict:
    """
    Process one user message through the LLM tool-calling loop.

    Returns:
        {
          "response": str,          # final assistant text
          "tool_calls": list[dict], # log of tools that were called
          "rounds": int,
        }
    """
    _ensure_system()
    history = _get_history()
    history.append({"role": "user", "content": user_message})
    current_state["last_voice_command"] = user_message

    tools = get_ollama_tools()
    tool_call_log: list[dict] = []

    for round_num in range(max_tool_rounds):
        try:
            response = _client.chat(
                model=OLLAMA_MODEL,
                messages=history,
                tools=tools if tools else None,
            )
        except Exception as exc:
            logger.exception("Ollama chat failed")
            error_text = f"I'm having trouble thinking right now: {exc}"
            history.append({"role": "assistant", "content": error_text})
            return {"response": error_text, "tool_calls": tool_call_log, "rounds": round_num}

        msg = response.message

        # No tool calls — final text response
        if not msg.tool_calls:
            text = msg.content or ""
            history.append({"role": "assistant", "content": text})
            return {"response": text, "tool_calls": tool_call_log, "rounds": round_num + 1}

        # Append assistant's tool-call message to history
        history.append(msg)

        # Dispatch all tool calls in this turn
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = tc.function.arguments or {}
            logger.info("Tool call: %s(%s)", fn_name, fn_args)

            result = dispatch(fn_name, fn_args)
            tool_call_log.append({"tool": fn_name, "args": fn_args, "result": result})
            logger.info("Tool result: %s", result)

            # Feed result back into history as a tool message
            history.append({
                "role": "tool",
                "content": result,
            })

    # Exceeded max rounds — get a final text response without tools
    logger.warning("Reached max_tool_rounds (%d), forcing text response", max_tool_rounds)
    try:
        final = _client.chat(model=OLLAMA_MODEL, messages=history)
        text = final.message.content or "I ran out of steps. Please try again."
    except Exception:
        text = "I ran out of steps trying to complete your request."
    history.append({"role": "assistant", "content": text})
    return {"response": text, "tool_calls": tool_call_log, "rounds": max_tool_rounds}


def reset_conversation() -> None:
    """Clear conversation history (keeps system prompt)."""
    current_state["conversation_history"] = []
    _ensure_system()


def register_builtin_tools() -> None:
    """
    Register all built-in Baymax tools.
    Called once at application startup from main.py.
    Deferred here to avoid circular imports.
    """
    import app.pi_client as pi
    import app.vision_processor as vision
    from app.tools_registry import register_tool

    # ── Motor ──────────────────────────────────────────────────────────────
    def move_robot(action: str) -> str:
        allowed = ["drive_forward", "drive_backward", "turn_left", "turn_right", "stop"]
        if action not in allowed:
            return f"Unknown action '{action}'. Allowed: {allowed}"
        result = pi.push_action_to_pi(action)
        current_state["pi_connected"] = result["success"]
        return f"Motor action '{action}': {'ok' if result['success'] else result.get('error', 'failed')}"

    register_tool(
        name="move_robot",
        description=(
            "Drive or stop the robot. Use this to move Baymax physically. "
            "action must be one of: drive_forward, drive_backward, turn_left, turn_right, stop."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["drive_forward", "drive_backward", "turn_left", "turn_right", "stop"],
                    "description": "The movement command to execute.",
                }
            },
            "required": ["action"],
        },
        handler=move_robot,
        tags=["motor", "actuator"],
    )

    # ── LED ───────────────────────────────────────────────────────────────
    def set_led(color: str) -> str:
        allowed = ["red", "green", "yellow", "all_on", "off"]
        if color not in allowed:
            return f"Unknown color '{color}'. Allowed: {allowed}"
        result = pi.push_led_color(color)
        current_state["led_state"] = color
        current_state["pi_connected"] = result["success"]
        return f"LED set to '{color}': {'ok' if result['success'] else result.get('error', 'failed')}"

    register_tool(
        name="set_led",
        description=(
            "Control the tri-color LED on the robot body. "
            "Use red for alerts/errors, green for success/ready, "
            "yellow for processing/thinking, off to turn it off."
        ),
        parameters={
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "enum": ["red", "green", "yellow", "all_on", "off"],
                    "description": "Color preset to set.",
                }
            },
            "required": ["color"],
        },
        handler=set_led,
        tags=["led", "actuator"],
    )

    # ── Eyes ──────────────────────────────────────────────────────────────
    def set_eye_expression(expression: str) -> str:
        allowed = ["normal", "excited", "disappointed", "blink", "sleeping"]
        if expression not in allowed:
            return f"Unknown expression '{expression}'. Allowed: {allowed}"
        result = pi.push_eye_expression(expression)
        current_state["eye_state"] = expression
        current_state["pi_connected"] = result["success"]
        return f"Eye expression '{expression}': {'ok' if result['success'] else result.get('error', 'failed')}"

    register_tool(
        name="set_eye_expression",
        description=(
            "Change Baymax's eye display expression. "
            "excited: happy crescent eyes with glow flutter. "
            "disappointed: hooded heavy eyes. "
            "blink: single blink. "
            "sleeping: eyes nearly closed. "
            "normal: return to default open eyes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "enum": ["normal", "excited", "disappointed", "blink", "sleeping"],
                    "description": "The eye expression to display.",
                }
            },
            "required": ["expression"],
        },
        handler=set_eye_expression,
        tags=["eyes", "actuator"],
    )

    # ── Speech ────────────────────────────────────────────────────────────
    def speak(text: str) -> str:
        result = pi.trigger_audio_speak(text)
        current_state["pi_connected"] = result["success"]
        return f"Spoke: '{text}' — {'ok' if result['success'] else result.get('error', 'failed')}"

    register_tool(
        name="speak",
        description=(
            "Make Baymax speak a sentence out loud through the robot's speaker. "
            "Use this to give verbal feedback to the user. Keep phrases short and friendly."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to synthesize and play through the speaker.",
                }
            },
            "required": ["text"],
        },
        handler=speak,
        tags=["audio", "actuator"],
    )

    # ── Vision ────────────────────────────────────────────────────────────
    def analyze_twister() -> str:
        result = vision.analyze_twister()
        return result.get("details", str(result))

    register_tool(
        name="analyze_twister",
        description=(
            "Capture a frame from the Pi Camera and analyze the Twister game state. "
            "Uses pose estimation (MediaPipe) and color detection (OpenCV) to determine "
            "whether the player's hands/feet are on the correct colored circles. "
            "Returns a summary of what was detected and whether the move is correct."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=analyze_twister,
        tags=["vision", "sensor"],
    )

    def capture_frame_description() -> str:
        result = vision.describe_scene()
        return result.get("description", str(result))

    register_tool(
        name="describe_scene",
        description=(
            "Capture a frame from the Pi Camera and describe what is visible. "
            "Useful for situational awareness, confirming the robot's surroundings, "
            "or verifying that a task was completed correctly."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=capture_frame_description,
        tags=["vision", "sensor"],
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    def change_mode(mode: str) -> str:
        from app.schemas import Mode
        from app.mode_manager import set_current_mode
        try:
            m = Mode(mode)
        except ValueError:
            return f"Unknown mode '{mode}'. Choose from: kids, young_adult, adult"
        set_current_mode(m)
        result = pi.push_mode_to_pi(m)
        current_state["pi_connected"] = result["success"]
        return f"Mode changed to '{mode}': {'ok' if result['success'] else result.get('error', 'failed')}"

    register_tool(
        name="change_mode",
        description=(
            "Switch Baymax's operating mode. "
            "kids: safe, slow, playful — trajectory mapping, storytelling, games. "
            "young_adult: medium speed, puzzles, tutoring. "
            "adult: wellness, guided meditation, facial recognition."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["kids", "young_adult", "adult"],
                    "description": "The operating mode to activate.",
                }
            },
            "required": ["mode"],
        },
        handler=change_mode,
        tags=["mode"],
    )

    logger.info("Built-in tools registered: %s", list_tools())


def list_tools() -> list[str]:
    from app.tools_registry import list_tools as _lt
    return _lt()
