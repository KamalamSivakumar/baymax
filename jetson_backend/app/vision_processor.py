"""
vision_processor.py — Gemini Vision pipeline for Baymax.

Connects to the Pi Camera MJPEG stream at PI_STREAM_URL, captures JPEG frames,
and sends them to Gemini Vision for analysis.

  analyze_hand_raise(expected)  — checks if the player is raising the expected hand
  describe_scene()              — general scene description

Configure via environment variables:
  PI_STREAM_URL   — Pi camera MJPEG stream (default: http://192.168.5.1:8080/stream)
  GEMINI_API_KEY  — Google AI Studio API key
  GEMINI_MODEL    — model for vision calls (default: gemini-2.0-flash)
"""

from __future__ import annotations
import logging
import os
import urllib.request

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
PI_STREAM_URL  = os.environ.get("PI_STREAM_URL", "http://192.168.5.1:8080/stream")
FRAME_TIMEOUT  = int(os.environ.get("PI_STREAM_TIMEOUT_MS", "3000"))

_client = genai.Client(api_key=GEMINI_API_KEY)


def _fetch_jpeg() -> bytes | None:
    """Grab one JPEG frame from the Pi camera MJPEG stream."""
    try:
        req = urllib.request.Request(PI_STREAM_URL, headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=FRAME_TIMEOUT / 1000) as resp:
            raw = resp.read(1 << 20)  # max 1 MB

        # MJPEG streams embed JPEG boundaries; find the first complete JPEG
        start = raw.find(b'\xff\xd8')
        end   = raw.find(b'\xff\xd9')
        jpeg = raw[start:end + 2] if (start != -1 and end != -1) else raw
        return jpeg if jpeg else None
    except Exception as exc:
        logger.warning("Frame fetch failed: %s", exc)
        return None


def _gemini_vision(jpeg: bytes, prompt: str) -> str:
    """Send a JPEG frame to Gemini Vision with the given prompt. Returns the text reply."""
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=jpeg)),
            types.Part(text=prompt),
        ],
    )
    return (response.text or "").strip()


def analyze_hand_raise(expected: str) -> dict:
    """
    Capture a frame and ask Gemini Vision whether the player is raising
    their right or left hand (or neither).

    Returns:
        {
            success: bool,       # True if expected hand detected
            detected: str,       # "right" | "left" | "neither" | "error"
            details: str,
        }
    """
    jpeg = _fetch_jpeg()
    if jpeg is None:
        return {
            "success": False,
            "detected": "error",
            "details": "Could not reach the Pi camera stream.",
        }

    try:
        answer = _gemini_vision(
            jpeg,
            "Look at the person in the image. Are they raising their right hand, "
            "their left hand, or neither? Reply with exactly one word: right, left, or neither.",
        )
        detected = answer.lower().strip().split()[0] if answer else "neither"
        if detected not in ("right", "left", "neither"):
            detected = "neither"
        success = detected == expected.lower()
        details = (
            f"Expected '{expected}' hand raised. Detected: '{detected}'. "
            f"{'Correct!' if success else 'Not quite — try again!'}"
        )
        logger.info("Hand raise check: expected=%s detected=%s success=%s", expected, detected, success)
        return {"success": success, "detected": detected, "details": details}

    except Exception as exc:
        logger.exception("Gemini Vision hand-raise analysis failed")
        return {"success": False, "detected": "error", "details": str(exc)}


def describe_scene() -> dict:
    """
    Capture a frame and ask Gemini Vision to describe what it sees.
    Used for general awareness (Adults mode emotion check, engagement check, etc.).
    """
    jpeg = _fetch_jpeg()
    if jpeg is None:
        return {
            "description": "Camera unavailable — cannot see the scene.",
            "available": False,
        }

    try:
        description = _gemini_vision(
            jpeg,
            "Describe what you see in this image in 1-2 concise sentences. "
            "Focus on the person's posture, expression, and general activity.",
        )
        logger.info("Scene described: %s", description)
        return {"description": description, "available": True}

    except Exception as exc:
        logger.exception("Gemini Vision scene description failed")
        return {"description": f"Vision error: {exc}", "available": False}
