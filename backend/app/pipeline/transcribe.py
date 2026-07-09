import os
import time
from faster_whisper import WhisperModel
from backend.app.config import WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE, WHISPER_BEAM_SIZE, WHISPER_CPU_THREADS

# Module-level model cache to avoid reloading on repeated calls
_model_cache = {}


def _get_model(model_size: str, compute_type: str, cpu_threads: int) -> WhisperModel:
    """Get or create a cached WhisperModel instance."""
    cache_key = f"{model_size}_{compute_type}_{cpu_threads}"
    if cache_key not in _model_cache:
        print(f"[Module 2] Loading faster-whisper model '{model_size}' (compute_type: {compute_type}, cpu_threads: {cpu_threads})...")
        print("           (This may take a while if downloading the model for the first time)")
        t0 = time.time()
        _model_cache[cache_key] = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)
        print(f"           Model loaded in {time.time() - t0:.1f}s")
    else:
        print(f"[Module 2] Using cached model '{model_size}'")
    return _model_cache[cache_key]


def _get_effective_speech_bounds(words: list, default_start: float, default_end: float, max_gap_sec: float = 1.5) -> tuple:
    """
    Finds the effective start and end times of the main speech in a segment,
    ignoring isolated words separated by large gaps of silence.
    """
    if not words:
        return default_start, default_end
        
    # Extract words with text and timings
    word_list = []
    for w in words:
        try:
            word_list.append({
                "text": w.word.strip(),
                "start": w.start,
                "end": w.end
            })
        except AttributeError:
            pass
            
    if not word_list:
        return default_start, default_end
        
    # Find the largest gap of silence between consecutive words
    num_words = len(word_list)
    large_gap_idx = -1
    max_gap = 0.0
    
    for i in range(num_words - 1):
        gap = word_list[i+1]["start"] - word_list[i]["end"]
        if gap > max_gap:
            max_gap = gap
            large_gap_idx = i
            
    # If the largest gap is greater than our threshold (e.g. 1.5 seconds)
    if max_gap > max_gap_sec and large_gap_idx != -1:
        # Count characters/words before and after the gap
        text_before = "".join([w["text"] for w in word_list[:large_gap_idx + 1]])
        text_after = "".join([w["text"] for w in word_list[large_gap_idx + 1:]])
        
        # If the main content is after the gap, adjust start time
        if len(text_after) >= len(text_before):
            print(f"           [Sync Adjust] Segment '{word_list[0]['text'][:5]}...' has large gap of {max_gap:.2f}s. Shifting start: {default_start:.2f}s -> {word_list[large_gap_idx + 1]['start']:.2f}s")
            return word_list[large_gap_idx + 1]["start"], word_list[-1]["end"]
        else:
            # Main content is before the gap, adjust end time
            print(f"           [Sync Adjust] Segment '{word_list[0]['text'][:5]}...' has large gap of {max_gap:.2f}s. Shifting end: {default_end:.2f}s -> {word_list[large_gap_idx]['end']:.2f}s")
            return word_list[0]["start"], word_list[large_gap_idx]["end"]
            
    # Otherwise, return the exact start of the first word and end of the last word to trim VAD pad
    return word_list[0]["start"], word_list[-1]["end"]


def transcribe_audio(audio_path: str, language_hint: str = None) -> list:
    """
    Transcribes audio using faster-whisper.
    Returns a list of segment dicts: { id, start, end, text, detected_language }
    
    Optimizations:
    - Model caching: avoids reloading on repeated calls
    - VAD filter: skips silence → 20-40% faster + reduces hallucination
    - beam_size: configurable (e.g., 1 or 3 for speed, 5 for quality)
    - Word timestamps: enabled to precisely detect voice starts/ends and avoid lip-sync lags
    """
    model = _get_model(WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE, WHISPER_CPU_THREADS)
    
    print(f"[Module 2] Transcribing {audio_path} (beam_size={WHISPER_BEAM_SIZE})...")
    t0 = time.time()
    
    segments, info = model.transcribe(
        audio_path,
        language=language_hint,
        beam_size=WHISPER_BEAM_SIZE,
        word_timestamps=True,               # Enable word timestamps for exact boundaries
        vad_filter=True,                    # Skip silence regions → faster + less hallucination
        vad_parameters=dict(
            min_silence_duration_ms=300,     # Minimum silence to split on (ms)
            speech_pad_ms=200,               # Padding around detected speech
        ),
        condition_on_previous_text=True,     # Maintain context continuity (explicit)
    )
    
    detected_language = info.language
    print(f"[Module 2] Detected language: {detected_language} with probability {info.language_probability:.2f}")
    
    results = []
    for segment in segments:
        # Determine precise speech boundaries from word-level timestamps if available
        start_time, end_time = _get_effective_speech_bounds(
            segment.words, 
            segment.start, 
            segment.end,
            max_gap_sec=1.5
        )
        
        results.append({
            "id": segment.id,
            "start": start_time,
            "end": end_time,
            "text": segment.text.strip(),
            "detected_language": detected_language
        })
    
    elapsed = time.time() - t0
    audio_duration = results[-1]["end"] if results else 0
    speed_ratio = audio_duration / elapsed if elapsed > 0 and audio_duration > 0 else 0
    
    print(f"[Module 2] Extracted {len(results)} segments in {elapsed:.1f}s ({speed_ratio:.1f}x realtime)")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = transcribe_audio(sys.argv[1])
        for r in res:
            print(r)
