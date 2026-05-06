"""
audio_manager.py — STT (Whisper) and TTS (pyttsx3) on the Jetson.

Pipeline:
  record → (Pi sends WAV bytes to Jetson) → transcribe() → LLM → synthesize() → (Jetson sends audio to Pi) → play

Environment variables:
  WHISPER_MODEL  — openai-whisper model size: tiny, base, small (default: tiny)
                   Use 'base' on 8 GB Orin Nano for better accuracy.
"""

from __future__ import annotations
import io
import logging
import os
import tempfile
import threading

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")

# Lazy-loaded globals (expensive to init — load once at first use)
_whisper_model = None
_whisper_lock  = threading.Lock()
_tts_engine    = None
_tts_lock      = threading.Lock()


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
                _tts_engine.setProperty("rate", 165)   # words per minute
                _tts_engine.setProperty("volume", 0.9)
                logger.info("pyttsx3 TTS ready")
    return _tts_engine


def transcribe(wav_bytes: bytes) -> dict:
    """
    Transcribe WAV audio bytes to text using Whisper.

    Returns:
        {"text": str, "language": str, "success": bool}
    """
    if not wav_bytes:
        return {"text": "", "language": "unknown", "success": False}

    try:
        model = _get_whisper()

        # Write to a temp file — Whisper reads from disk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        result = model.transcribe(tmp_path, fp16=False)

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        text = result.get("text", "").strip()
        lang = result.get("language", "unknown")
        logger.info("Transcribed (%s): %s", lang, text)
        return {"text": text, "language": lang, "success": True}

    except Exception as exc:
        logger.exception("Transcription failed")
        return {"text": "", "language": "unknown", "success": False, "error": str(exc)}


def synthesize(text: str) -> bytes | None:
    """
    Synthesize text to WAV audio bytes using pyttsx3.
    Returns raw WAV bytes or None on failure.
    """
    if not text.strip():
        return None

    try:
        engine = _get_tts()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        logger.info("Synthesized %d chars → %d bytes WAV", len(text), len(wav_bytes))
        return wav_bytes

    except Exception as exc:
        logger.exception("TTS synthesis failed")
        return None
