"""
audio_manager.py — Gemini STT and TTS for the Jetson.

The USB mic/speaker hybrid is connected directly to the Jetson.
Full audio pipeline runs locally on the Jetson:
  record() → WAV bytes → transcribe() → LLM → synthesize() → play()

Latency optimisations:
  - Common phrases are pre-generated at startup and returned instantly from cache.
  - VAD (webrtcvad) stops recording early on silence.
  - All API calls use the lightweight flash model.

Environment variables:
  GEMINI_API_KEY   — Google AI Studio API key (shared with llm_engine)
  GEMINI_STT_MODEL — model for audio transcription (default: gemini-2.0-flash)
  GEMINI_TTS_MODEL — model for speech synthesis (default: gemini-2.5-flash-preview-tts)
  TTS_VOICE        — Gemini TTS voice name (default: Kore)
  USB_AUDIO_INDEX  — override PyAudio device index (auto-detected if unset)
"""

from __future__ import annotations
import io
import logging
import os
import subprocess
import tempfile
import wave

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
STT_MODEL       = os.environ.get("GEMINI_STT_MODEL", "gemini-2.0-flash")
TTS_MODEL       = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE       = os.environ.get("TTS_VOICE", "Kore")

_client = genai.Client(api_key=GEMINI_API_KEY)

# ── Phrase cache ──────────────────────────────────────────────────────────────
# Pre-generate WAV bytes for high-frequency phrases at startup so they play
# immediately without an API round-trip.
_CACHE_PHRASES = [
    "Great job!",
    "Try again!",
    "Raise your right hand!",
    "Raise your left hand!",
    "Well done!",
    "Let's play!",
    "I am Baymax, your personal healthcare companion.",
]
_phrase_cache: dict[str, bytes] = {}


def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _synthesize_api(text: str) -> bytes | None:
    """Call Gemini TTS API and return WAV bytes."""
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
        audio_part = response.candidates[0].content.parts[0]
        pcm = audio_part.inline_data.data
        wav = _pcm_to_wav(pcm)
        logger.info("TTS synthesized %d chars → %d bytes WAV", len(text), len(wav))
        return wav
    except Exception as exc:
        logger.exception("TTS synthesis failed: %s", exc)
        return None


def _warm_cache() -> None:
    """Pre-generate WAV for all cached phrases. Called at startup."""
    for phrase in _CACHE_PHRASES:
        wav = _synthesize_api(phrase)
        if wav:
            _phrase_cache[phrase.lower().strip()] = wav
    logger.info("TTS cache warmed: %d phrases", len(_phrase_cache))


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe(wav_bytes: bytes) -> dict:
    """
    Transcribe WAV audio bytes to text using Gemini.

    Returns:
        {"text": str, "language": str, "success": bool}
    """
    if not wav_bytes:
        return {"text": "", "language": "unknown", "success": False}

    try:
        # Write WAV to temp file — Gemini file upload works from bytes via inline data
        response = _client.models.generate_content(
            model=STT_MODEL,
            contents=[
                types.Part(
                    inline_data=types.Blob(mime_type="audio/wav", data=wav_bytes)
                ),
                types.Part(text="Transcribe this audio. Reply with only the spoken words, nothing else."),
            ],
        )
        text = (response.text or "").strip()
        logger.info("Transcribed: %s", text)
        return {"text": text, "language": "auto", "success": True}

    except Exception as exc:
        logger.exception("Transcription failed")
        return {"text": "", "language": "unknown", "success": False, "error": str(exc)}


def synthesize(text: str) -> bytes | None:
    """
    Synthesize text to WAV audio bytes.
    Returns cached bytes instantly for known phrases, calls Gemini API otherwise.
    """
    if not text.strip():
        return None

    # Check cache first (case-insensitive exact match)
    cached = _phrase_cache.get(text.lower().strip())
    if cached:
        logger.info("TTS cache hit: %s", text)
        return cached

    return _synthesize_api(text)


# ── Local USB audio (Jetson-side) ─────────────────────────────────────────────

SAMPLE_RATE  = 16000
CHANNELS     = 1
SAMPLE_WIDTH = 2       # 16-bit PCM
CHUNK_SIZE   = 1024

# VAD config
VAD_FRAME_MS   = 10
VAD_FRAME_SAMP = SAMPLE_RATE * VAD_FRAME_MS // 1000   # 160 samples
SILENCE_LIMIT  = 0.5   # seconds of silence before stopping
MAX_RECORD_SEC = 8

_PA_AVAILABLE  = False
_AUDIO_INDEX: int | None = None


def _init_audio() -> None:
    global _PA_AVAILABLE, _AUDIO_INDEX
    env_idx = os.environ.get("USB_AUDIO_INDEX")
    if env_idx is not None:
        _AUDIO_INDEX = int(env_idx)
        _PA_AVAILABLE = True
        return
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            if "usb" in name and info.get("maxInputChannels", 0) > 0:
                _AUDIO_INDEX = i
                logger.info("USB audio device found at index %d: %s", i, info["name"])
                break
        pa.terminate()
        _PA_AVAILABLE = True
    except Exception as exc:
        logger.warning("PyAudio init failed: %s — audio recording disabled", exc)


_init_audio()


def record() -> bytes:
    """
    Record from the Jetson USB mic until silence (VAD) or MAX_RECORD_SEC.
    Returns raw WAV bytes.
    """
    if not _PA_AVAILABLE:
        logger.warning("PyAudio unavailable — cannot record")
        return b""

    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)
        use_vad = True
    except ImportError:
        use_vad = False
        logger.warning("webrtcvad not installed — fixed-duration recording")

    import pyaudio
    pa = pyaudio.PyAudio()
    frame_size = VAD_FRAME_SAMP if use_vad else CHUNK_SIZE
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=_AUDIO_INDEX,
            frames_per_buffer=frame_size,
        )
        frames: list[bytes] = []
        silent_frames = 0
        silence_trigger = int(SILENCE_LIMIT * 1000 / VAD_FRAME_MS)
        max_frames = int(MAX_RECORD_SEC * 1000 / VAD_FRAME_MS) if use_vad else int(SAMPLE_RATE / CHUNK_SIZE * MAX_RECORD_SEC)

        for _ in range(max_frames):
            chunk = stream.read(frame_size, exception_on_overflow=False)
            frames.append(chunk)
            if use_vad:
                is_speech = vad.is_speech(chunk, SAMPLE_RATE)
                silent_frames = 0 if is_speech else silent_frames + 1
                if silent_frames >= silence_trigger and len(frames) > silence_trigger:
                    logger.info("VAD: silence — stopping early (%d frames)", len(frames))
                    break

        stream.stop_stream()
        stream.close()
    finally:
        pa.terminate()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    logger.info("Recorded %d frames → %d bytes WAV", len(frames), buf.tell())
    return buf.getvalue()


def play(wav_bytes: bytes) -> bool:
    """
    Play WAV bytes through the Jetson USB speaker using aplay.
    Returns True on success.
    """
    if not wav_bytes:
        return False
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        cmd = ["aplay"]
        if _AUDIO_INDEX is not None:
            cmd += ["-D", f"hw:{_AUDIO_INDEX},0"]
        cmd.append(tmp_path)
        subprocess.run(cmd, check=True, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.unlink(tmp_path)
        logger.info("Played %d bytes WAV", len(wav_bytes))
        return True
    except Exception as exc:
        logger.warning("play() failed: %s", exc)
        return False


def speak(text: str) -> bool:
    """
    Synthesize text (Gemini TTS or cache) and play it immediately on the Jetson.
    Returns True on success.
    """
    wav = synthesize(text)
    if wav is None:
        logger.warning("speak(): synthesis returned None for '%s'", text)
        return False
    return play(wav)

