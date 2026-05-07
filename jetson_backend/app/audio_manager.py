"""
audio_manager.py — Jetson audio: hardware I/O from tested pipeline + Gemini STT/TTS.

Hardware pipeline (tested on Jetson with plughw:2,0):
  arecord → WAV bytes → Gemini STT → text
  text → Gemini TTS → WAV bytes → aplay

Configure via environment variables:
  AUDIO_INPUT_DEVICE   — ALSA device for recording (default: plughw:2,0)
  AUDIO_OUTPUT_DEVICE  — ALSA device for playback (default: system default)
  AUDIO_RECORD_SECONDS — default recording duration (default: 5)
  AUDIO_SAMPLE_RATE    — sample rate for arecord (default: 44100)
  AUDIO_CHANNELS       — channel count for arecord (default: 2)
  GEMINI_API_KEY       — Google AI Studio API key
  GEMINI_STT_MODEL     — model for transcription (default: gemini-2.0-flash)
  GEMINI_TTS_MODEL     — model for synthesis (default: gemini-2.5-flash-preview-tts)
  TTS_VOICE            — Gemini TTS voice name (default: Kore)
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ── Hardware config (from tested pipeline) ────────────────────────────────────

AUDIO_INPUT_DEVICE  = os.environ.get("AUDIO_INPUT_DEVICE", "plughw:2,0")
AUDIO_OUTPUT_DEVICE = os.environ.get("AUDIO_OUTPUT_DEVICE", "")

DEFAULT_RECORD_SECONDS = int(os.environ.get("AUDIO_RECORD_SECONDS", "5"))
DEFAULT_SAMPLE_RATE    = int(os.environ.get("AUDIO_SAMPLE_RATE", "44100"))
DEFAULT_CHANNELS       = int(os.environ.get("AUDIO_CHANNELS", "2"))

# ── Gemini config ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
STT_MODEL      = os.environ.get("GEMINI_STT_MODEL", "gemini-2.0-flash")
TTS_MODEL      = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE      = os.environ.get("TTS_VOICE", "Kore")

_client = genai.Client(api_key=GEMINI_API_KEY)


def _safe_unlink(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ── Hardware recording (tested: arecord -D plughw:2,0) ───────────────────────

def record_audio(
    duration_seconds: int = DEFAULT_RECORD_SECONDS,
    device: str = AUDIO_INPUT_DEVICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> bytes | None:
    """
    Record audio from the Jetson-connected mic using arecord.
    Returns WAV bytes, or None on failure.
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
    except Exception:
        logger.exception("Audio recording failed")
        return None
    finally:
        if tmp_path:
            _safe_unlink(tmp_path)


# ── Hardware playback (tested: aplay) ─────────────────────────────────────────

def play_audio(wav_bytes: bytes, device: str = AUDIO_OUTPUT_DEVICE) -> bool:
    """
    Play WAV audio bytes through the Jetson speaker using aplay.
    Returns True if playback succeeded, False otherwise.
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
    """Play an existing WAV file through the Jetson speaker."""
    path = Path(path)
    if not path.exists():
        logger.error("Audio file does not exist: %s", path)
        return False
    cmd = ["aplay"]
    if device:
        cmd.extend(["-D", device])
    cmd.append(str(path))
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error("aplay failed: %s", result.stderr)
            return False
        return True
    except Exception:
        logger.exception("Audio file playback failed")
        return False


# ── STT: Gemini ───────────────────────────────────────────────────────────────

def transcribe(wav_bytes: bytes) -> dict:
    """
    Transcribe WAV audio bytes to text using Gemini.
    Returns {"text": str, "language": str, "success": bool}
    """
    if not wav_bytes:
        return {"text": "", "language": "unknown", "success": False, "error": "No WAV bytes provided"}

    try:
        response = _client.models.generate_content(
            model=STT_MODEL,
            contents=[
                types.Part(inline_data=types.Blob(mime_type="audio/wav", data=wav_bytes)),
                types.Part(text="Transcribe this audio. Reply with only the spoken words, nothing else."),
            ],
        )
        text = (response.text or "").strip()
        logger.info("Transcribed: %s", text)
        return {"text": text, "language": "auto", "success": True}

    except Exception as exc:
        logger.exception("Transcription failed")
        return {"text": "", "language": "unknown", "success": False, "error": str(exc)}


def listen_and_transcribe(duration_seconds: int = DEFAULT_RECORD_SECONDS) -> dict:
    """Record from Jetson mic then transcribe with Gemini."""
    wav_bytes = record_audio(duration_seconds=duration_seconds)
    if wav_bytes is None:
        return {"text": "", "language": "unknown", "success": False, "error": "Recording failed"}
    result = transcribe(wav_bytes)
    result["duration_seconds"] = duration_seconds
    return result


# ── TTS: Gemini ───────────────────────────────────────────────────────────────

def synthesize(text: str) -> bytes | None:
    """
    Synthesize text to WAV audio bytes using Gemini TTS.
    Returns raw WAV bytes or None on failure.
    """
    if not text or not text.strip():
        return None

    try:
        response = _client.models.generate_content(
            model=TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                    )
                ),
            ),
        )
        pcm = response.candidates[0].content.parts[0].inline_data.data
        wav = _pcm_to_wav(pcm)
        logger.info("Synthesized %d chars → %d bytes WAV", len(text), len(wav))
        return wav

    except Exception:
        logger.exception("TTS synthesis failed")
        return None


def speak_text(text: str) -> bool:
    """text → Gemini TTS WAV → Jetson speaker."""
    wav_bytes = synthesize(text)
    if wav_bytes is None:
        return False
    return play_audio(wav_bytes)


# ── Startup hook ──────────────────────────────────────────────────────────────

def _warm_cache() -> None:
    """No-op — Gemini TTS needs no pre-warming in this configuration."""
    logger.info("Audio manager ready (arecord/aplay + Gemini STT/TTS)")


# ── Compatibility aliases ─────────────────────────────────────────────────────
record = record_audio
play   = play_audio
speak  = speak_text
