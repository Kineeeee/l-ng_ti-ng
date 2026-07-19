import os
import math
import time
import shutil
import subprocess
from backend.app.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE,
    WHISPER_CPU_THREADS,
    WHISPER_DEVICE,
    GROQ_API_KEY,
    ENABLE_OCR_SUBTITLE,
)

# Module-level model cache to avoid reloading on repeated calls
_model_cache = {}


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

class _WordLike:
    """
    Lightweight word object used by _get_effective_speech_bounds.
    Defined at module level to avoid re-creating a class inside loops.
    """
    __slots__ = ("word", "start", "end")

    def __init__(self, word: str, start: float, end: float):
        self.word  = word
        self.start = start
        self.end   = end


def _resolve_device() -> str:
    """
    Resolve the effective transcription backend.
      'auto'  → try mlx first (Apple Silicon), fall back to cpu
      'mlx'   → force Apple GPU via mlx-whisper
      'cpu'   → force faster-whisper CPU
      'groq'  → Groq cloud API
    """
    device = WHISPER_DEVICE.lower()

    if device == "auto":
        try:
            import mlx_whisper  # noqa: F401
            import platform
            if platform.machine() in ("arm64", "aarch64"):
                return "mlx"
        except ImportError:
            pass
        return "cpu"

    return device


def _get_effective_speech_bounds(words: list, default_start: float, default_end: float, max_gap_sec: float = 1.5) -> tuple:
    """
    Finds the effective start and end times of the main speech in a segment,
    ignoring isolated words separated by large gaps of silence.
    """
    if not words:
        return default_start, default_end

    word_list = []
    for w in words:
        try:
            word_list.append({"text": w.word.strip(), "start": w.start, "end": w.end})
        except AttributeError:
            pass

    if not word_list:
        return default_start, default_end

    num_words = len(word_list)
    large_gap_idx = -1
    max_gap = 0.0

    for i in range(num_words - 1):
        gap = word_list[i + 1]["start"] - word_list[i]["end"]
        if gap > max_gap:
            max_gap = gap
            large_gap_idx = i

    if max_gap > max_gap_sec and large_gap_idx != -1:
        text_before = "".join([w["text"] for w in word_list[: large_gap_idx + 1]])
        text_after  = "".join([w["text"] for w in word_list[large_gap_idx + 1 :]])

        if len(text_after) >= len(text_before):
            print(
                f"           [Sync Adjust] Segment '{word_list[0]['text'][:5]}...' "
                f"has large gap of {max_gap:.2f}s. "
                f"Shifting start: {default_start:.2f}s -> {word_list[large_gap_idx + 1]['start']:.2f}s"
            )
            return word_list[large_gap_idx + 1]["start"], word_list[-1]["end"]
        else:
            print(
                f"           [Sync Adjust] Segment '{word_list[0]['text'][:5]}...' "
                f"has large gap of {max_gap:.2f}s. "
                f"Shifting end: {default_end:.2f}s -> {word_list[large_gap_idx]['end']:.2f}s"
            )
            return word_list[0]["start"], word_list[large_gap_idx]["end"]

    return word_list[0]["start"], word_list[-1]["end"]


# ─────────────────────────────────────────────
# Backend A: mlx-whisper  (Apple GPU / MLX)
# ─────────────────────────────────────────────

# Map from WHISPER_MODEL_SIZE to mlx-community HuggingFace repos
_MLX_MODEL_MAP = {
    "tiny":           "mlx-community/whisper-tiny-mlx",
    "base":           "mlx-community/whisper-base-mlx",
    "small":          "mlx-community/whisper-small-mlx",
    "medium":         "mlx-community/whisper-medium-mlx",
    "large-v2":       "mlx-community/whisper-large-v2-mlx",
    "large-v3":       "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _detect_speech_intervals(
    audio_path: str,
    top_db: float = 35.0,
    speech_pad_ms: int = 200,
    merge_gap_sec: float = 1.5,
):
    """
    Detect speech intervals using librosa energy thresholding (VAD for MLX).
    Uses librosa which is already in requirements.txt — no new dependencies needed.

    Returns:
        list of (start_sec, end_sec)  — speech windows to transcribe individually
        []                            — no speech detected (return empty results)
        None                          — VAD unavailable or speech covers full file
                                        (caller should transcribe full audio)
    """
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(y) / sr

        # Non-silent intervals: returns Nx2 array of [start_frame, end_frame]
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=512, hop_length=128)

        if len(intervals) == 0:
            print("[Module 2] [MLX/GPU] VAD: audio appears fully silent.")
            return []

        pad_sec = speech_pad_ms / 1000.0

        # Convert frames to seconds, apply padding, clamp to file bounds
        speech_windows = []
        for start_frame, end_frame in intervals:
            start_sec = max(0.0, float(start_frame) / sr - pad_sec)
            end_sec   = min(total_duration, float(end_frame) / sr + pad_sec)
            speech_windows.append((start_sec, end_sec))

        # Merge nearby intervals
        merged = [speech_windows[0]]
        for start, end in speech_windows[1:]:
            last_start, last_end = merged[-1]
            if start - last_end <= merge_gap_sec:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        # If speech covers >=90% of the file, chunking adds overhead with no benefit
        total_speech = sum(e - s for s, e in merged)
        coverage = total_speech / total_duration if total_duration > 0 else 1.0
        if coverage >= 0.90:
            print(
                f"[Module 2] [MLX/GPU] VAD: speech covers {coverage:.0%} of audio "
                f"— skipping chunking, transcribing full file."
            )
            return None  # Signal: transcribe full audio

        print(
            f"[Module 2] [MLX/GPU] VAD: {len(merged)} speech region(s) detected "
            f"({total_speech:.1f}s / {total_duration:.1f}s, {coverage:.0%})"
        )
        return merged

    except Exception as e:
        print(f"[Module 2] VAD warning: {e}. Transcribing full audio.")
        return None


def _transcribe_mlx_chunk(audio_path: str, model_repo: str, language_hint: str) -> dict:
    """Run mlx_whisper.transcribe on a single audio file. Returns raw result dict."""
    import mlx_whisper
    return mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
        language=language_hint,
        word_timestamps=True,
        fp16=True,
        # Note: mlx-whisper uses greedy decoding only — beam_size is not supported.
        condition_on_previous_text=True,
    )


def _transcribe_with_mlx(audio_path: str, language_hint: str = None) -> list:
    """
    Transcribe using mlx-whisper — runs natively on Apple Silicon GPU via MLX.
    Typically 5-10x faster than faster-whisper CPU on M-series chips.

    Uses librosa-based VAD to skip silence/noise before transcription,
    preventing Whisper hallucinations on non-speech regions.
    """
    model_repo = _MLX_MODEL_MAP.get(
        WHISPER_MODEL_SIZE,
        f"mlx-community/whisper-{WHISPER_MODEL_SIZE}-mlx",
    )

    print(f"[Module 2] [MLX/GPU] Transcribing '{audio_path}'")
    print(f"           Model repo : {model_repo}")
    t0 = time.time()

    # ── VAD: detect speech regions ────────────────────────────────────────
    speech_intervals = _detect_speech_intervals(audio_path)

    if speech_intervals is not None and len(speech_intervals) == 0:
        # VAD confirmed: audio is fully silent
        print("[Module 2] [MLX/GPU] VAD: no speech detected. Returning empty.")
        return []

    # raw_segments_with_offset: list of (segment_dict, offset_sec)
    raw_segments_with_offset = []
    detected_language = "unknown"

    if speech_intervals is None:
        # Transcribe full file (VAD unavailable or speech covers full audio)
        result = _transcribe_mlx_chunk(audio_path, model_repo, language_hint)
        detected_language = result.get("language", "unknown")
        for seg in result.get("segments", []):
            raw_segments_with_offset.append((seg, 0.0))

    else:
        # Chunk audio by speech intervals and transcribe each separately
        tmp_dir = os.path.join(
            os.path.dirname(os.path.abspath(audio_path)), "_mlx_vad_tmp"
        )
        os.makedirs(tmp_dir, exist_ok=True)
        print(f"[Module 2] [MLX/GPU] Transcribing {len(speech_intervals)} chunk(s)...")
        try:
            for idx, (start_sec, end_sec) in enumerate(speech_intervals):
                chunk_path = os.path.join(tmp_dir, f"chunk_{idx:03d}.wav")
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(start_sec),
                        "-t",  str(end_sec - start_sec),
                        "-i",  audio_path,
                        "-ar", "16000", "-ac", "1",
                        chunk_path,
                    ],
                    capture_output=True,
                )
                if not os.path.exists(chunk_path):
                    continue

                chunk_result = _transcribe_mlx_chunk(chunk_path, model_repo, language_hint)
                if detected_language == "unknown":
                    detected_language = chunk_result.get("language", "unknown")

                for seg in chunk_result.get("segments", []):
                    raw_segments_with_offset.append((seg, start_sec))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Build output segments (same format as before) ─────────────────────
    elapsed = time.time() - t0
    print(f"[Module 2] Detected language: {detected_language}")

    results = []
    for seg, offset in raw_segments_with_offset:
        words = [
            _WordLike(
                word=w.get("word", ""),
                start=w.get("start", seg["start"]) + offset,
                end=w.get("end",   seg["end"])   + offset,
            )
            for w in seg.get("words", [])
        ]

        start_time, end_time = _get_effective_speech_bounds(
            words,
            seg["start"] + offset,
            seg["end"]   + offset,
            max_gap_sec=1.5,
        )

        results.append({
            "id":                len(results),
            "start":             start_time,
            "end":               end_time,
            "text":              seg["text"].strip(),
            "detected_language": detected_language,
        })

    audio_duration = results[-1]["end"] if results else 0
    speed_ratio    = audio_duration / elapsed if elapsed > 0 and audio_duration > 0 else 0
    print(f"[Module 2] [MLX/GPU] {len(results)} segments in {elapsed:.1f}s ({speed_ratio:.1f}x realtime)")
    return results


# ─────────────────────────────────────────────
# Backend B: faster-whisper  (CPU)
# ─────────────────────────────────────────────

def _get_faster_whisper_model(model_size: str, compute_type: str, cpu_threads: int):
    """Get or create a cached WhisperModel (faster-whisper) instance."""
    from faster_whisper import WhisperModel

    cache_key = f"{model_size}_{compute_type}_{cpu_threads}"
    if cache_key not in _model_cache:
        print(
            f"[Module 2] Loading faster-whisper model '{model_size}' "
            f"(compute_type: {compute_type}, cpu_threads: {cpu_threads})..."
        )
        print("           (This may take a while if downloading the model for the first time)")
        t0 = time.time()
        _model_cache[cache_key] = WhisperModel(
            model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads
        )
        print(f"           Model loaded in {time.time() - t0:.1f}s")
    else:
        print(f"[Module 2] Using cached model '{model_size}'")
    return _model_cache[cache_key]


def _transcribe_with_cpu(audio_path: str, language_hint: str = None) -> list:
    """
    Transcribe using faster-whisper on CPU (int8 quantized).
    Original fallback / non-Apple-Silicon path.
    """
    model = _get_faster_whisper_model(WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE, WHISPER_CPU_THREADS)

    print(f"[Module 2] [CPU] Transcribing '{audio_path}' (beam_size={WHISPER_BEAM_SIZE})...")
    t0 = time.time()

    segments, info = model.transcribe(
        audio_path,
        language=language_hint,
        beam_size=WHISPER_BEAM_SIZE,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
            speech_pad_ms=200,
        ),
        condition_on_previous_text=True,
    )

    detected_language = info.language
    print(f"[Module 2] Detected language: {detected_language} (p={info.language_probability:.2f})")

    results = []
    for segment in segments:
        start_time, end_time = _get_effective_speech_bounds(
            segment.words, segment.start, segment.end, max_gap_sec=1.5
        )
        results.append({
            "id":                segment.id,
            "start":             start_time,
            "end":               end_time,
            "text":              segment.text.strip(),
            "detected_language": detected_language,
        })

    elapsed        = time.time() - t0
    audio_duration = results[-1]["end"] if results else 0
    speed_ratio    = audio_duration / elapsed if elapsed > 0 and audio_duration > 0 else 0
    print(f"[Module 2] [CPU] {len(results)} segments in {elapsed:.1f}s ({speed_ratio:.1f}x realtime)")
    return results


# ─────────────────────────────────────────────
# Backend C: Groq Whisper API  (online)
# ─────────────────────────────────────────────

_GROQ_MAX_BYTES          = 25 * 1024 * 1024   # 25 MB hard limit per request
_GROQ_CHUNK_DURATION_SEC = 600                 # Split into 10-minute chunks


def _split_audio_for_groq(audio_path: str, job_tmp_dir: str) -> list:
    """
    Split audio into <=10-minute chunks using ffmpeg so each fits under 25 MB.
    Returns list of (chunk_path, start_offset_seconds).
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    total_duration = float(probe.stdout.strip())
    num_chunks = math.ceil(total_duration / _GROQ_CHUNK_DURATION_SEC)

    if num_chunks == 1:
        return [(audio_path, 0.0)]

    os.makedirs(job_tmp_dir, exist_ok=True)
    chunks = []
    for i in range(num_chunks):
        start      = i * _GROQ_CHUNK_DURATION_SEC
        chunk_path = os.path.join(job_tmp_dir, f"groq_chunk_{i:03d}.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-t", str(_GROQ_CHUNK_DURATION_SEC),
             "-i", audio_path, "-ar", "16000", "-ac", "1", "-b:a", "64k", chunk_path],
            capture_output=True,
        )
        if os.path.exists(chunk_path):
            chunks.append((chunk_path, start))
    return chunks


def _groq_get_attr(obj, key: str, default=None):
    """
    Safely get a field from either a dict or an object attribute.
    Defined at module level (not inside a loop) for correctness and efficiency.
    """
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _transcribe_with_groq(audio_path: str, language_hint: str = None) -> list:
    """
    Transcribe using Groq's hosted Whisper Large v3 Turbo API.
    Free tier: 28,800 seconds of audio/day.
    Each request is limited to 25 MB — large files are split automatically.
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "groq package is not installed.\n"
            "Run: pip install groq\n"
            "Or set WHISPER_DEVICE=auto to use the local MLX/CPU backend."
        )

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set in .env.\n"
            "Get a free key at https://console.groq.com and add:\n"
            "  GROQ_API_KEY=gsk_...\n"
            "to your .env file."
        )

    client = Groq(api_key=GROQ_API_KEY)

    file_size = os.path.getsize(audio_path)
    tmp_dir   = os.path.join(os.path.dirname(audio_path), "_groq_tmp")

    if file_size > _GROQ_MAX_BYTES:
        print(f"[Module 2] [Groq] File is {file_size / 1e6:.1f} MB > 25 MB limit — splitting into chunks...")
        chunks = _split_audio_for_groq(audio_path, tmp_dir)
    else:
        chunks = [(audio_path, 0.0)]

    print(f"[Module 2] [Groq] Sending {len(chunks)} chunk(s) to Groq Whisper API...")
    t0 = time.time()

    all_segments      = []
    detected_language = "unknown"

    for chunk_path, offset in chunks:
        with open(chunk_path, "rb") as f:
            groq_response = client.audio.transcriptions.create(
                file=(os.path.basename(chunk_path), f),
                model="whisper-large-v3-turbo",
                language=language_hint,
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"],
            )

        detected_language = getattr(groq_response, "language", detected_language)
        raw_segs = getattr(groq_response, "segments", []) or []

        for seg in raw_segs:
            seg_start = (_groq_get_attr(seg, "start") or 0.0)
            seg_end   = (_groq_get_attr(seg, "end")   or 0.0)
            seg_text  = (_groq_get_attr(seg, "text")  or "")

            raw_words = _groq_get_attr(seg, "words") or []
            words = [
                _WordLike(
                    word=(_groq_get_attr(w, "word")  or ""),
                    start=(_groq_get_attr(w, "start") or seg_start) + offset,
                    end=(_groq_get_attr(w, "end")   or seg_end)   + offset,
                )
                for w in raw_words
            ]

            adj_start = seg_start + offset
            adj_end   = seg_end   + offset

            refined_start, refined_end = _get_effective_speech_bounds(
                words, adj_start, adj_end, max_gap_sec=1.5
            )

            all_segments.append({
                "id":                len(all_segments),
                "start":             refined_start,
                "end":               refined_end,
                "text":              seg_text.strip(),
                "detected_language": detected_language,
            })

    # Cleanup temporary split files
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed        = time.time() - t0
    audio_duration = all_segments[-1]["end"] if all_segments else 0
    speed_ratio    = audio_duration / elapsed if elapsed > 0 and audio_duration > 0 else 0
    print(f"[Module 2] [Groq] {len(all_segments)} segments in {elapsed:.1f}s ({speed_ratio:.1f}x realtime)")
    print(f"[Module 2] Detected language: {detected_language}")
    return all_segments


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def transcribe_audio(audio_path: str, language_hint: str = None, device_override: str = None, video_path: str = None) -> list:
    """
    Transcribes audio using the configured backend, and optionally fuses OCR hardsub subtitles.

    device_override (optional) — override backend for this single call:
        'auto'  → auto-detect (mlx if Apple Silicon, else cpu)
        'mlx'   → Apple GPU via mlx-whisper
        'cpu'   → faster-whisper CPU
        'groq'  → Groq cloud API

    Config (via .env):
        WHISPER_DEVICE=auto|mlx|cpu|groq
        GROQ_API_KEY=gsk_...  (required for groq backend)
        ENABLE_OCR_SUBTITLE=True|False

    Returns a list of segment dicts: { id, start, end, text, detected_language }
    """
    device = (device_override or WHISPER_DEVICE).lower()
    if device == "auto":
        device = _resolve_device()

    backend_label = {
        "mlx":  "Apple GPU (MLX)",
        "cpu":  "CPU (faster-whisper)",
        "groq": "Groq Cloud API",
    }.get(device, device)

    print(f"[Module 2] Backend: {backend_label}")

    if device == "mlx":
        try:
            stt_segments = _transcribe_with_mlx(audio_path, language_hint)
        except Exception as e:
            print(f"[Module 2] WARNING: MLX transcription failed ({e}). Falling back to CPU...")
            stt_segments = _transcribe_with_cpu(audio_path, language_hint)

    elif device == "groq":
        stt_segments = _transcribe_with_groq(audio_path, language_hint)

    else:  # 'cpu' or unknown → safer default
        stt_segments = _transcribe_with_cpu(audio_path, language_hint)

    # Perform Video OCR Subtitle Fusion if video_path is available
    if ENABLE_OCR_SUBTITLE and video_path and os.path.exists(video_path):
        try:
            from backend.app.pipeline.ocr_subtitle import extract_subtitles_from_video, merge_stt_and_ocr_segments
            ocr_segs = extract_subtitles_from_video(video_path)
            if ocr_segs:
                fused_segments, stats = merge_stt_and_ocr_segments(stt_segments, ocr_segs)
                return fused_segments
        except Exception as e:
            print(f"[Module 2] WARNING: OCR subtitle extraction/fusion failed ({e}). Continuing with STT segments.")

    return stt_segments


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dev = sys.argv[2] if len(sys.argv) > 2 else None
        res = transcribe_audio(sys.argv[1], device_override=dev)
        for r in res:
            print(r)
