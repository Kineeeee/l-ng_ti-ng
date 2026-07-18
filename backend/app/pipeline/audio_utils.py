"""
audio_utils.py — Shared audio utilities dùng chung trong pipeline.

Các hàm này bị trùng lặp giữa tts.py và align.py — đã được extract
vào đây để có một source of truth duy nhất.
"""

import os
import wave
from pydub import AudioSegment


def get_wav_duration_fast(wav_path: str) -> float:
    """
    Get WAV duration using wave module (header-only, ~10x faster than pydub).
    Falls back to size-based estimate if header read fails.
    """
    try:
        with wave.open(wav_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / rate
    except Exception:
        pass
    # Fallback: estimate from file size (16-bit mono PCM, 44-byte header)
    try:
        size = os.path.getsize(wav_path)
        return max(0.0, (size - 44) / (2 * 16000))
    except Exception:
        return 0.5


def is_valid_wav(wav_path: str, min_size: int = 100) -> bool:
    """Check if a WAV file exists, is non-trivially sized, and has frames."""
    if not os.path.exists(wav_path):
        return False
    if os.path.getsize(wav_path) <= min_size:
        return False
    try:
        with wave.open(wav_path, 'rb') as wf:
            return wf.getnframes() > 0
    except Exception:
        return False


def detect_leading_silence(
    sound: AudioSegment,
    silence_threshold: float = -50.0,
    chunk_size: int = 10,
) -> int:
    """Return the number of leading silent milliseconds in an AudioSegment."""
    trim_ms = 0
    duration = len(sound)
    while trim_ms < duration and sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
    return trim_ms


def trim_silence(
    sound: AudioSegment,
    silence_threshold: float = -50.0,
    chunk_size: int = 10,
    keep_silence_ms: int = 30,
) -> AudioSegment:
    """
    Trim leading and trailing silence from an AudioSegment.
    Keeps keep_silence_ms of cushion on each end to avoid clipping.
    Returns a minimal silent segment if the entire clip is silence.
    """
    duration = len(sound)
    start_trim = detect_leading_silence(sound, silence_threshold, chunk_size)

    reversed_sound = sound.reverse()
    end_trim = detect_leading_silence(reversed_sound, silence_threshold, chunk_size)

    trimmed_start = max(0, start_trim - keep_silence_ms)
    trimmed_end = min(duration, (duration - end_trim) + keep_silence_ms)

    if trimmed_start >= trimmed_end:
        return AudioSegment.silent(duration=100, frame_rate=sound.frame_rate).set_channels(sound.channels)

    return sound[trimmed_start:trimmed_end]


def create_silence_wav(output_path: str, duration_sec: float = 0.5, sample_rate: int = 24000):
    """Create a short silence WAV file (used as fallback for failed TTS)."""
    silence = AudioSegment.silent(duration=int(duration_sec * 1000), frame_rate=sample_rate)
    silence = silence.set_frame_rate(sample_rate).set_channels(1)
    silence.export(output_path, format="wav")
