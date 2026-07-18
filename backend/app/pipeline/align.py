import os
import time
import subprocess
import wave
from pydub import AudioSegment
from backend.app.config import (
    TTS_DYNAMIC_SPEEDUP,
    MIX_ORIGINAL_AUDIO,
    DUCK_VOLUME_DB,
    ORIGINAL_VOLUME_DB,
    TTS_VOLUME_DB,
    AUDIO_SYNC_OFFSET_MS
)
from backend.app.pipeline.audio_utils import (
    get_wav_duration_fast as _get_wav_duration_fast,
    trim_silence as _trim_silence,
)

# Target sample rate for dubbed audio
# 24kHz matches edge-tts native output → avoids quality loss from downsample+upsample
DUBBED_SAMPLE_RATE = 24000


def _speed_up_with_ffmpeg(input_path: str, output_path: str, speed_ratio: float) -> bool:
    """
    Speed up audio using FFmpeg atempo filter.
    Higher quality than pydub.speedup() — no chunk overlap artifacts.
    """
    speed_ratio = max(1.01, min(speed_ratio, 2.0))
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter:a", f"atempo={speed_ratio:.4f}",
        "-ar", str(DUBBED_SAMPLE_RATE), "-ac", "1",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def merge_intervals(intervals: list, gap_threshold_ms: int = 800) -> list:
    """Merge overlapping or nearby (start_ms, end_ms) intervals."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end + gap_threshold_ms:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged


def align_and_merge_audio(
    segments: list, 
    video_duration_sec: float, 
    output_dir: str, 
    job_id: str,
    original_audio_path: str = None,
    mix_original_audio: bool = MIX_ORIGINAL_AUDIO,
    duck_volume_db: float = DUCK_VOLUME_DB,
    original_volume_db: float = ORIGINAL_VOLUME_DB,
    tts_volume_db: float = TTS_VOLUME_DB,
    sync_offset_ms: float = AUDIO_SYNC_OFFSET_MS
) -> str:
    """
    Aligns TTS audio segments to the original timestamps and merges them into a single track,
    optionally mixing in the original background audio with ducking transitions.
    
    Returns the path to the merged dubbed audio WAV file.
    """
    print(f"[Module 5] Aligning and merging {len(segments)} segments...")
    t0 = time.time()
    
    total_ms = int(video_duration_sec * 1000)
    
    # 1. Pre-process and trim all speech segments first, adjusting their speed and volume
    processed_segments = []
    speedup_count = 0
    
    for segment in segments:
        if not segment.get("tts_audio_path") or not os.path.exists(segment["tts_audio_path"]):
            continue
            
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        target_duration_ms = end_ms - start_ms
        
        wav_path = segment["tts_audio_path"]
        segment_audio = AudioSegment.from_file(wav_path)
        
        # Ensure consistent sample rate and channel count
        if segment_audio.frame_rate != DUBBED_SAMPLE_RATE:
            segment_audio = segment_audio.set_frame_rate(DUBBED_SAMPLE_RATE)
        segment_audio = segment_audio.set_channels(1)
        
        # Trim silence (both leading and trailing to get true spoken duration)
        segment_audio = _trim_silence(segment_audio)
        
        # Apply voice volume boost
        if tts_volume_db != 0.0:
            segment_audio = segment_audio + tts_volume_db
            
        tts_duration_ms = len(segment_audio)
        
        # Perform dynamic speed up if the clip is too long for its slot
        if TTS_DYNAMIC_SPEEDUP and tts_duration_ms > target_duration_ms * 1.15:
            speed_ratio = tts_duration_ms / target_duration_ms
            speed_ratio = min(speed_ratio, 1.3)
            
            if speed_ratio > 1.05:
                # Use high-quality FFmpeg speed up
                sped_up_path = wav_path.replace(".wav", "_fast.wav")
                if _speed_up_with_ffmpeg(wav_path, sped_up_path, speed_ratio):
                    segment_audio = AudioSegment.from_file(sped_up_path)
                    segment_audio = segment_audio.set_frame_rate(DUBBED_SAMPLE_RATE).set_channels(1)
                    segment_audio = _trim_silence(segment_audio)
                    if tts_volume_db != 0.0:
                        segment_audio = segment_audio + tts_volume_db
                    tts_duration_ms = len(segment_audio)
                    
                    try:
                        os.remove(sped_up_path)
                    except OSError:
                        pass
                    speedup_count += 1
                else:
                    # Fallback to Pydub speedup
                    segment_audio = segment_audio.speedup(playback_speed=speed_ratio, chunk_size=50, crossfade=25)
                    tts_duration_ms = len(segment_audio)
                    speedup_count += 1
                    
        processed_segments.append({
            "audio": segment_audio,
            "start_ms": start_ms,
            "duration_ms": tts_duration_ms
        })
        
    # 2. Build the list of original speech intervals based on the original segment start/end times
    speech_intervals = []
    for segment in segments:
        if not segment.get("tts_audio_path") or not os.path.exists(segment["tts_audio_path"]):
            continue
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        if start_ms >= end_ms:
            continue
        speech_intervals.append((start_ms, end_ms))
        
    # 3. Initialize background audio and apply ducking
    mixed_bg_audio = None
    
    if mix_original_audio and original_audio_path and os.path.exists(original_audio_path):
        try:
            print(f"           Loading original audio for mixing: {original_audio_path}")
            original_audio = AudioSegment.from_file(original_audio_path)
            original_audio = original_audio.set_frame_rate(DUBBED_SAMPLE_RATE)
            
            if original_volume_db != 0.0:
                original_audio = original_audio + original_volume_db
                
            # Merge intervals using the generic helper
            merged_intervals = merge_intervals(speech_intervals, gap_threshold_ms=800)
            print(f"           Applying audio ducking ({duck_volume_db}dB) across {len(merged_intervals)} regions")
            
            # Slice-based ducking to prevent relative gain accumulation
            ducked_audio = AudioSegment.silent(duration=0, frame_rate=DUBBED_SAMPLE_RATE)
            if original_audio.channels > 1:
                ducked_audio = ducked_audio.set_channels(original_audio.channels)
            else:
                ducked_audio = ducked_audio.set_channels(1)
                
            curr_ms = 0
            fade_ms = 200
            total_len = len(original_audio)
            
            for start_ms, end_ms in merged_intervals:
                # Apply sync offset to the ducking window so it matches the voice timing
                shifted_start = max(0, start_ms + int(sync_offset_ms))
                shifted_end = max(0, end_ms + int(sync_offset_ms))
                
                # Cap boundaries to total duration
                shifted_start = min(total_len, shifted_start)
                shifted_end = min(total_len, shifted_end)
                
                s_start = max(curr_ms, shifted_start - fade_ms)
                s_end = min(total_len, shifted_end + fade_ms)
                
                # A. Segment before ducking transition (0dB)
                if s_start > curr_ms:
                    ducked_audio += original_audio[curr_ms:s_start]
                    
                # B. Fade down transition (0dB -> duck_volume_db)
                if shifted_start > s_start:
                    fade_down_part = original_audio[s_start:shifted_start]
                    fade_down_part = fade_down_part.fade(to_gain=duck_volume_db, start=0, duration=shifted_start - s_start)
                    ducked_audio += fade_down_part
                    
                # C. Ducked speech region (constant duck_volume_db)
                if shifted_end > shifted_start:
                    speech_part = original_audio[shifted_start:shifted_end] + duck_volume_db
                    ducked_audio += speech_part
                    
                # D. Fade up transition (duck_volume_db -> 0dB)
                if s_end > shifted_end:
                    fade_up_part = original_audio[shifted_end:s_end] + duck_volume_db
                    fade_up_part = fade_up_part.fade(to_gain=-duck_volume_db, start=0, duration=s_end - shifted_end)
                    ducked_audio += fade_up_part
                    
                curr_ms = s_end
                
            # E. Append remaining audio at the end of the video (0dB)
            if curr_ms < total_len:
                ducked_audio += original_audio[curr_ms:]
                
            mixed_bg_audio = ducked_audio
        except Exception as e:
            print(f"           ⚠️ Failed to load or mix original audio: {e}. Falling back to silent background.")
            
    if mixed_bg_audio is None:
        mixed_bg_audio = AudioSegment.silent(duration=total_ms, frame_rate=DUBBED_SAMPLE_RATE)
        mixed_bg_audio = mixed_bg_audio.set_channels(1)
        
    # 4. Overlay the processed dubbed voices onto the background at shifted timestamps
    for seg in processed_segments:
        segment_audio = seg["audio"]
        shifted_start_ms = max(0, seg["start_ms"] + int(sync_offset_ms))
        mixed_bg_audio = mixed_bg_audio.overlay(segment_audio, position=shifted_start_ms)
        
    dubbed_audio_path = os.path.join(output_dir, f"dubbed_{job_id}.wav")
    mixed_bg_audio.export(dubbed_audio_path, format="wav")
    
    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(dubbed_audio_path) / (1024 * 1024)
    print(f"[Module 5] Merged audio saved to {dubbed_audio_path}")
    print(f"           {len(processed_segments)} segments, {speedup_count} sped up, {file_size_mb:.1f}MB, {elapsed:.1f}s")
    return dubbed_audio_path


if __name__ == "__main__":
    pass
