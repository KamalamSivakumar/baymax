"""
vision_processor.py — OpenCV + MediaPipe vision pipeline for Baymax.

Connects to the Pi Camera MJPEG stream at PI_STREAM_URL, captures frames,
and runs two analyses:

  analyze_twister()  — HSV color circle detection + MediaPipe Pose
                       → determines if hand/foot is on the correct Twister circle.
  describe_scene()   — lightweight scene description (object count, dominant colors).

Configure via environment variables:
  PI_STREAM_URL — Pi camera MJPEG stream (default: http://192.168.5.1:8080/stream)
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


# ── Twister color definitions (HSV ranges) ────────────────────────────────────
# Format: name -> (lower_hsv, upper_hsv)
TWISTER_COLORS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "red":    (np.array([0,  120,  70]), np.array([10, 255, 255])),
    "red2":   (np.array([170,120,  70]), np.array([180,255, 255])),  # red wraps hue
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

        # MJPEG streams embed JPEG boundaries; find the first complete JPEG
        start = raw.find(b'\xff\xd8')
        end   = raw.find(b'\xff\xd9')
        if start == -1 or end == -1:
            # Might be a plain JPEG endpoint
            jpeg_bytes = raw
        else:
            jpeg_bytes = raw[start:end + 2]

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        logger.warning("Frame fetch failed: %s", exc)
        return None


def _detect_color_circles(frame: np.ndarray) -> list[dict]:
    """
    Find colored circles (Twister mat dots) in the frame.
    Returns list of: {color, x, y, radius}
    """
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (11, 11), 0)
    detections = []

    for color_name, (lo, hi) in TWISTER_COLORS.items():
        mask = cv2.inRange(blurred, lo, hi)
        if color_name == "red2":
            # Merge red2 into red mask
            red_mask = detections[-1].get("_mask") if detections and detections[-1]["color"] == "red" else None
            if red_mask is not None:
                mask = cv2.bitwise_or(red_mask, mask)
                detections.pop()
                color_name = "red"

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # ignore noise
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            detections.append({
                "color": color_name,
                "x": int(x),
                "y": int(y),
                "radius": int(radius),
                "_mask": mask,
            })

    # Strip internal mask field
    for d in detections:
        d.pop("_mask", None)

    return detections


def _detect_pose_keypoints(frame: np.ndarray) -> list[dict] | None:
    """
    Run MediaPipe Pose on the frame.
    Returns list of landmark dicts: {name, x_px, y_px, visibility}
    or None if MediaPipe is unavailable.
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5,
        ) as pose:
            results = pose.process(rgb)

        if not results.pose_landmarks:
            return []

        # We only need extremities for Twister
        landmarks_of_interest = {
            "left_wrist":  mp_pose.PoseLandmark.LEFT_WRIST,
            "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
            "left_ankle":  mp_pose.PoseLandmark.LEFT_ANKLE,
            "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
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


def _limb_on_circle(limb: dict, circles: list[dict], tolerance_px: int = 40) -> dict | None:
    """
    Return the circle that a limb landmark is closest to (within tolerance),
    or None if no circle is close enough.
    """
    best = None
    best_dist = float("inf")
    for c in circles:
        dist = ((limb["x_px"] - c["x"]) ** 2 + (limb["y_px"] - c["y"]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = c
    if best and best_dist <= (best["radius"] + tolerance_px):
        return best
    return None


def analyze_twister() -> dict:
    """
    Full Twister analysis: detect mat circles + player pose.
    Returns:
        {
            success: bool,
            details: str,
            circles_found: int,
            limbs_detected: list,
            limb_placements: dict,   # limb -> color circle they're on
        }
    """
    frame = _fetch_frame()
    if frame is None:
        return {
            "success": False,
            "details": "Could not reach the Pi camera stream. Check that the Pi is online and the stream is running.",
            "circles_found": 0,
            "limbs_detected": [],
            "limb_placements": {},
        }

    circles = _detect_color_circles(frame)
    keypoints = _detect_pose_keypoints(frame)

    if not circles:
        return {
            "success": False,
            "details": "No colored circles detected. Make sure the Twister mat is visible and well-lit.",
            "circles_found": 0,
            "limbs_detected": [],
            "limb_placements": {},
        }

    if keypoints is None:
        # MediaPipe not available — report color detection only
        color_summary = ", ".join(f"{c['color']} at ({c['x']},{c['y']})" for c in circles)
        return {
            "success": True,
            "details": f"Detected {len(circles)} circle(s): {color_summary}. Pose estimation unavailable.",
            "circles_found": len(circles),
            "limbs_detected": [],
            "limb_placements": {},
        }

    placements: dict[str, str] = {}
    if keypoints:
        for limb in keypoints:
            if limb["visibility"] < 0.4:
                continue
            circle = _limb_on_circle(limb, circles)
            if circle:
                placements[limb["name"]] = circle["color"]

    circle_summary = ", ".join(f"{c['color']}@({c['x']},{c['y']})" for c in circles[:8])
    if placements:
        placement_text = "; ".join(f"{k}→{v}" for k, v in placements.items())
        details = f"Circles detected: {circle_summary}. Limb placements: {placement_text}."
    elif keypoints:
        details = f"Circles detected: {circle_summary}. Player detected but no limbs are on circles yet."
    else:
        details = f"Circles detected: {circle_summary}. No player detected in frame."

    return {
        "success": bool(placements),
        "details": details,
        "circles_found": len(circles),
        "limbs_detected": keypoints or [],
        "limb_placements": placements,
    }


def describe_scene() -> dict:
    """
    Lightweight scene description: object count, dominant colors, brightness.
    No heavy ML — fast enough for general awareness checks.
    """
    frame = _fetch_frame()
    if frame is None:
        return {
            "description": "Camera unavailable — cannot see the scene.",
            "available": False,
        }

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = int(hsv[:, :, 2].mean())

    # Dominant color bins
    color_notes = []
    for name, (lo, hi) in list(TWISTER_COLORS.items())[:4]:  # skip red2
        mask = cv2.inRange(hsv, lo, hi)
        frac = mask.sum() / (h * w * 255)
        if frac > 0.02:
            color_notes.append(f"{int(frac * 100)}% {name}")

    edge_count = int(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 80, 200).sum() / 255)
    activity = "busy" if edge_count > 5000 else "calm"

    desc_parts = [f"{w}×{h} frame", f"brightness={brightness}/255", f"scene={activity}"]
    if color_notes:
        desc_parts.append("colors: " + ", ".join(color_notes))
    else:
        desc_parts.append("no strong colors detected")

    return {
        "description": "; ".join(desc_parts),
        "available": True,
        "width": w,
        "height": h,
        "brightness": brightness,
    }
