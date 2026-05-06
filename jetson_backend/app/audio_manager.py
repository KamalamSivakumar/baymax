"""
audio_manager.py — Jetson audio manager for mic recording, speaker playback,
STT using Whisper, and TTS using pyttsx3.

Jetson-local pipeline:
  mic → record_audio() → transcribe()
  synthesize() / uploaded WAV bytes → play_audio()

Tested manually with:
  Record: arecord -D plughw:2,0 -f cd -t wav -d 5 test.wav
  Play:   aplay speech.wav
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ----------------------------
# Config
# ----------------------------

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")

# Your working Jetson mic command used: arecord -D plughw:2,0 ...
AUDIO_INPUT_DEVICE = os.environ.get("AUDIO_INPUT_DEVICE", "plughw:2,0")

# Your working playback command was simply: aplay speech.wav
# So default output device is fine unless you override it.
AUDIO_OUTPUT_DEVICE = os.environ.get("AUDIO_OUTPUT_DEVICE", "")

DEFAULT_RECORD_SECONDS = int(os.environ.get("AUDIO_RECORD_SECONDS", "5"))
DEFAULT_SAMPLE_RATE = int(os.environ.get("AUDIO_SAMPLE_RATE", "44100"))
DEFAULT_CHANNELS = int(os.environ.get("AUDIO_CHANNELS", "2"))

# Lazy-loaded globals
_whisper_model = None
_whisper_lock = threading.Lock()
_tts_engine = None
_tts_lock = threading.Lock()


# ----------------------------
# Internal helpers
# ----------------------------

def _get_whisper():
    global _whisper_model

    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                import whisper

                logger.info("Loading Whisper model: %s", WHISPER_MODEL_SIZE)
                _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
                logger.info("Whisper ready")

    return _whisper_model


def _get_tts():
    global _tts_engine

    if _tts_engine is None:
        with _tts_lock:
            if _tts_engine is None:
                import pyttsx3

                _tts_engine = pyttsx3.init()
                _tts_engine.setProperty("rate", 165)
                _tts_engine.setProperty("volume", 0.9)
                logger.info("pyttsx3 TTS ready")

    return _tts_engine


def _safe_unlink(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


# ----------------------------
# Hardware audio: mic recording
# ----------------------------

def record_audio(
    duration_seconds: int = DEFAULT_RECORD_SECONDS,
    device: str = AUDIO_INPUT_DEVICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> bytes | None:
    """
    Record audio from the Jetson-connected mic using arecord.

    Returns:
        WAV bytes, or None on failure.
    """

    if duration_seconds <= 0:
        duration_seconds = DEFAULT_RECORD_SECONDS

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "arecord",
            "-D", device,
            "-f", "S16_LE",
            "-r", str(sample_rate),
            "-c", str(channels),
            "-t", "wav",
            "-d", str(duration_seconds),
            tmp_path,
        ]

        logger.info("Recording audio: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=duration_seconds + 5,
        )

        if result.returncode != 0:
            logger.error("arecord failed: %s", result.stderr)
            return None

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()

        logger.info("Recorded %d bytes of WAV audio", len(wav_bytes))
        return wav_bytes

    except subprocess.TimeoutExpired:
        logger.exception("arecord timed out")
        return None

    except Exception as exc:
        logger.exception("Audio recording failed")
        return None

    finally:
        if tmp_path:
            _safe_unlink(tmp_path)


# ----------------------------
# Hardware audio: speaker playback
# ----------------------------

def play_audio(wav_bytes: bytes, device: str = AUDIO_OUTPUT_DEVICE) -> bool:
    """
    Play WAV audio bytes through the Jetson-connected speaker using aplay.

    Args:
        wav_bytes: Raw WAV file bytes.
        device: Optional ALSA output device. Empty means default aplay device.

    Returns:
        True if playback succeeded, False otherwise.
    """

    if not wav_bytes:
        logger.warning("No audio bytes provided to play_audio()")
        return False

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        cmd = ["aplay"]

        if device:
            cmd.extend(["-D", device])

        cmd.append(tmp_path)

        logger.info("Playing audio: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.error("aplay failed: %s", result.stderr)
            return False

        logger.info("Audio playback completed")
        return True

    except Exception:
        logger.exception("Audio playback failed")
        return False

    finally:
        if tmp_path:
            _safe_unlink(tmp_path)


def play_audio_file(path: str | Path, device: str = AUDIO_OUTPUT_DEVICE) -> bool:
    """
    Play an existing WAV file through the Jetson speaker.
    Useful for local testing.
    """

    path = Path(path)

    if not path.exists():
        logger.error("Audio file does not exist: %s", path)
        return False

    cmd = ["aplay"]

    if device:
        cmd.extend(["-D", device])

    cmd.append(str(path))

    try:
        logger.info("Playing audio file: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.error("aplay failed: %s", result.stderr)
            return False

        return True

    except Exception:
        logger.exception("Audio file playback failed")
        return False


# ----------------------------
# STT
# ----------------------------

def transcribe(wav_bytes: bytes) -> dict:
    """
    Transcribe WAV audio bytes to text using Whisper.

    Returns:
        {
            "text": str,
            "language": str,
            "success": bool
        }
    """

    if not wav_bytes:
        return {
            "text": "",
            "language": "unknown",
            "success": False,
            "error": "No WAV bytes provided",
        }

    tmp_path = None

    try:
        model = _get_whisper()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        result = model.transcribe(tmp_path, fp16=False)

        text = result.get("text", "").strip()
        lang = result.get("language", "unknown")

        logger.info("Transcribed (%s): %s", lang, text)

        return {
            "text": text,
            "language": lang,
            "success": True,
        }

    except Exception as exc:
        logger.exception("Transcription failed")

        return {
            "text": "",
            "language": "unknown",
            "success": False,
            "error": str(exc),
        }

    finally:
        if tmp_path:
            _safe_unlink(tmp_path)


def listen_and_transcribe(duration_seconds: int = DEFAULT_RECORD_SECONDS) -> dict:
    """
    Convenience function:
      record from Jetson mic → transcribe with Whisper
    """

    wav_bytes = record_audio(duration_seconds=duration_seconds)

    if wav_bytes is None:
        return {
            "text": "",
            "language": "unknown",
            "success": False,
            "error": "Recording failed",
        }

    result = transcribe(wav_bytes)
    result["duration_seconds"] = duration_seconds

    return result


# ----------------------------
# TTS
# ----------------------------

def synthesize(text: str) -> bytes | None:
    """
    Synthesize text to WAV audio bytes using pyttsx3.

    Returns:
        Raw WAV bytes or None on failure.
    """

    if not text or not text.strip():
        return None

    tmp_path = None

    try:
        engine = _get_tts()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()

        logger.info("Synthesized %d chars → %d bytes WAV", len(text), len(wav_bytes))

        return wav_bytes

    except Exception:
        logger.exception("TTS synthesis failed")
        return None

    finally:
        if tmp_path:
            _safe_unlink(tmp_path)


def speak_text(text: str) -> bool:
    """
    Convenience function:
      text → pyttsx3 WAV → Jetson speaker
    """

    wav_bytes = synthesize(text)

    if wav_bytes is None:
        return False

    return play_audio(wav_bytes)
