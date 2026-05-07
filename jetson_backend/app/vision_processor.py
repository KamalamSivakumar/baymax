"""
vision_processor.py — OpenCV + MediaPipe vision pipeline for Baymax.

Connects to the Pi Camera MJPEG stream at PI_STREAM_URL, captures frames,
and runs two analyses:

  analyze_hand_raise(expected)  — MediaPipe Pose: detects if the expected wrist
                                   is raised above the corresponding shoulder.
  describe_scene()              — lightweight scene description (brightness,
                                   dominant colors, edge activity).

Configure via environment variables:
  PI_STREAM_URL        — Pi camera MJPEG stream (default: http://192.168.5.1:8080/stream)
  PI_STREAM_TIMEOUT_MS — Frame fetch timeout in ms (default: 3000)
"""

from __future__ import annotations
import logging
import os
import urllib.request

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PI_STREAM_URL = os.environ.get("PI_STREAM_URL", "http://192.168.5.1:8080/stream")
FRAME_TIMEOUT = int(os.environ.get("PI_STREAM_TIMEOUT_MS", "3000"))

_COLOR_RANGES = {
    "red":    (np.array([0,  120,  70]), np.array([10, 255, 255])),
    "green":  (np.array([36,  50,  70]), np.array([86, 255, 255])),
    "blue":   (np.array([94,  80,  70]), np.array([126,255, 255])),
    "yellow": (np.array([25,  50,  70]), np.array([35, 255, 255])),
}


def _fetch_frame() -> np.ndarray | None:
    """Grab one JPEG frame from the Pi camera MJPEG stream."""
    try:
        req = urllib.request.Request(PI_STREAM_URL, headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=FRAME_TIMEOUT / 1000) as resp:
            raw = resp.read(1 << 20)  # max 1 MB

        start = raw.find(b'\xff\xd8')
        end   = raw.find(b'\xff\xd9')
        jpeg_bytes = raw[start:end + 2] if (start != -1 and end != -1) else raw

        arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        logger.warning("Frame fetch failed: %s", exc)
        return None


def _detect_pose_keypoints(frame: np.ndarray) -> list[dict] | None:
    """
    Run MediaPipe Pose on the frame.
    Returns list of landmark dicts {name, x_px, y_px, visibility},
    or None if MediaPipe is unavailable.
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5,
        ) as pose:
            results = pose.process(rgb)

        if not results.pose_landmarks:
            return []

        landmarks_of_interest = {
            "left_wrist":     mp_pose.PoseLandmark.LEFT_WRIST,
            "right_wrist":    mp_pose.PoseLandmark.RIGHT_WRIST,
            "left_shoulder":  mp_pose.PoseLandmark.LEFT_SHOULDER,
            "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        }
        out = []
        for name, idx in landmarks_of_interest.items():
            lm = results.pose_landmarks.landmark[idx]
            out.append({
                "name":       name,
                "x_px":       int(lm.x * w),
                "y_px":       int(lm.y * h),
                "visibility": round(lm.visibility, 2),
            })
        return out

    except ImportError:
        logger.warning("mediapipe not installed; skipping pose detection")
        return None


def analyze_hand_raise(expected: str = "right") -> dict:
    """
    Detect whether the expected hand ('left' or 'right') is raised above
    the corresponding shoulder using MediaPipe Pose.

    Returns:
        {
            success: bool,   # True if the correct hand is raised
            detected: str,   # "right" | "left" | "both" | "none"
            details: str,
        }
    """
    frame = _fetch_frame()
    if frame is None:
        return {
            "success": False,
            "detected": "none",
            "details": "Could not reach the Pi camera stream. Check that the Pi is online.",
        }

    keypoints = _detect_pose_keypoints(frame)
    if keypoints is None:
        return {
            "success": False,
            "detected": "none",
            "details": "Pose estimation unavailable (mediapipe not installed).",
        }
    if not keypoints:
        return {
            "success": False,
            "detected": "none",
            "details": "No person detected in frame.",
        }

    kp = {lm["name"]: lm for lm in keypoints}
    raised = []
    for side in ("left", "right"):
        wrist    = kp.get(f"{side}_wrist")
        shoulder = kp.get(f"{side}_shoulder")
        if (wrist and shoulder
                and wrist["visibility"] >= 0.4
                and shoulder["visibility"] >= 0.4):
            # Image coords: y increases downward, so raised wrist has lower y value
            if wrist["y_px"] < shoulder["y_px"]:
                raised.append(side)

    expected = expected.lower().strip()
    if expected not in ("left", "right"):
        expected = "right"

    success  = expected in raised
    detected = raised[0] if len(raised) == 1 else ("both" if raised else "none")
    details  = (
        f"Raised: {', '.join(raised) if raised else 'none'}. "
        f"Expected: {expected}. "
        f"{'Correct!' if success else 'Not quite — try again!'}"
    )
    logger.info("Hand raise: expected=%s detected=%s success=%s", expected, detected, success)
    return {"success": success, "detected": detected, "details": details}


def describe_scene() -> dict:
    """
    Lightweight scene description: brightness, dominant colors, edge activity.
    No heavy ML — fast and reliable.
    """
    frame = _fetch_frame()
    if frame is None:
        return {
            "description": "Camera unavailable — cannot see the scene.",
            "available": False,
        }

    h, w   = frame.shape[:2]
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = int(hsv[:, :, 2].mean())

    color_notes = []
    for name, (lo, hi) in _COLOR_RANGES.items():
        mask = cv2.inRange(hsv, lo, hi)
        frac = mask.sum() / (h * w * 255)
        if frac > 0.02:
            color_notes.append(f"{int(frac * 100)}% {name}")

    edge_count = int(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 80, 200).sum() / 255)
    activity   = "busy" if edge_count > 5000 else "calm"

    desc_parts = [f"{w}x{h} frame", f"brightness={brightness}/255", f"scene={activity}"]
    desc_parts.append("colors: " + ", ".join(color_notes) if color_notes else "no strong colors detected")

    return {
        "description": "; ".join(desc_parts),
        "available":   True,
        "width":       w,
        "height":      h,
        "brightness":  brightness,
    }
