"""
Audio module: transcription via Groq Whisper, multi-speaker diarization,
TTS voice replies, and live microphone interaction.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

from groq import Groq

WHISPER_MODEL = "whisper-large-v3"
SUPPORTED_AUDIO = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac"}
# Groq audio file size limit
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB


def _client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# Single file transcription
# ---------------------------------------------------------------------------

def transcribe(audio_path: str, language: str | None = None) -> dict:
    """Transcribe a single audio file. Returns text + segments with timestamps."""
    path = Path(audio_path)
    if path.suffix.lower() not in SUPPORTED_AUDIO:
        raise ValueError(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_AUDIO)}"
        )

    client = _client()
    with open(audio_path, "rb") as f:
        kwargs: dict = dict(
            file=(path.name, f),
            model=WHISPER_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )
        if language:
            kwargs["language"] = language
        result = client.audio.transcriptions.create(**kwargs)

    return {
        "text": result.text,
        "language": getattr(result, "language", "unknown"),
        "duration": getattr(result, "duration", None),
        "segments": getattr(result, "segments", []),
        "words": getattr(result, "words", []),
    }


# ---------------------------------------------------------------------------
# Multiple file transcription
# ---------------------------------------------------------------------------

def transcribe_multiple(audio_paths: list[str]) -> list[dict]:
    """Transcribe multiple audio files, returning one result per file."""
    results = []
    for path in audio_paths:
        result = transcribe(path)
        result["file"] = path
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Multi-speaker diarization (segment-based, no external model needed)
# ---------------------------------------------------------------------------

def diarize(audio_path: str, min_speakers: int = 2) -> dict:
    """
    Approximate speaker diarization using Whisper verbose segments.

    Strategy: segments with long pauses (>0.8s) between them likely represent
    speaker turns. We label speakers sequentially (Speaker 1, 2, ...).
    For >4 speakers this produces a reasonable approximation without
    requiring pyannote.audio or HuggingFace access.
    """
    result = transcribe(audio_path)
    segments = result.get("segments", [])

    if not segments:
        return {"text": result["text"], "speakers": [], "speaker_count": 0}

    PAUSE_THRESHOLD = 0.8  # seconds
    speaker_id = 1
    labeled: list[dict] = []
    prev_end = 0.0

    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()

        if start - prev_end > PAUSE_THRESHOLD and labeled:
            speaker_id += 1

        labeled.append({
            "speaker": f"Speaker {speaker_id}",
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text,
        })
        prev_end = end

    actual_speakers = speaker_id
    return {
        "text": result["text"],
        "speaker_count": actual_speakers,
        "speakers": labeled,
        "note": (
            f"Detected ~{actual_speakers} speakers via pause-based segmentation. "
            "For precise diarization, use pyannote.audio."
        ),
    }


# ---------------------------------------------------------------------------
# TTS — voice reply
# ---------------------------------------------------------------------------

def text_to_speech(text: str, output_path: str = "data/reply.mp3", lang: str = "en") -> str:
    """Convert text to speech using gTTS. Returns path to saved audio file."""
    from gtts import gTTS
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    return output_path


def transcribe_and_reply(audio_path: str, system_context: str = "") -> dict:
    """
    Full voice round-trip:
      1. Transcribe user audio
      2. Generate AI response (Groq LLM)
      3. Convert response to speech
    Returns transcript, AI text response, and path to reply audio.
    """
    # Step 1: transcribe
    transcript = transcribe(audio_path)
    user_text = transcript["text"]

    # Step 2: generate response
    client = _client()
    messages = []
    if system_context:
        messages.append({"role": "system", "content": system_context})
    messages.append({"role": "user", "content": user_text})

    llm_resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=512,
    )
    reply_text = llm_resp.choices[0].message.content

    # Step 3: TTS
    reply_audio = text_to_speech(reply_text, output_path="data/reply.mp3")

    return {
        "user_transcript": user_text,
        "ai_response": reply_text,
        "reply_audio": reply_audio,
    }


# ---------------------------------------------------------------------------
# Live microphone interaction
# ---------------------------------------------------------------------------

def live_voice_interaction(duration_seconds: int = 5, system_context: str = "") -> dict:
    """
    Record from the microphone, transcribe, respond, and play back the reply.
    Requires: sounddevice, soundfile, gTTS, afplay (macOS) or aplay (Linux).
    """
    import sounddevice as sd
    import soundfile as sf
    import numpy as np

    SAMPLE_RATE = 16000
    print(f"Recording for {duration_seconds}s... speak now!")
    audio_data = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording complete.")

    # Save to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio_data, SAMPLE_RATE)

    result = transcribe_and_reply(tmp_path, system_context=system_context)
    os.unlink(tmp_path)

    # Play reply audio
    reply_path = result["reply_audio"]
    if os.path.exists(reply_path):
        if os.uname().sysname == "Darwin":
            os.system(f"afplay {reply_path}")
        else:
            os.system(f"aplay {reply_path}")

    return result
