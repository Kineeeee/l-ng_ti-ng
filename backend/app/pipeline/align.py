import os
import time
import subprocess
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


def resolve_voice_collisions(processed_segments: list, min_gap_ms: int = 40) -> tuple:
    """
    STRICT NON-OVERLAPPING ALIGNMENT:
    Guarantees that consecutive TTS voice segments NEVER overlap with each other.
    
    1. If segment N is too long and collides with segment N+1, speeds up segment N using FFmpeg atempo.
    2. If segment N still touches segment N+1, slides segment N+1 forward so there is at least min_gap_ms gap.
    
    Returns:
        tuple: (resolved_segments: list, speedup_count: int, collision_resolved_count: int)
    """
    if not processed_segments:
        return [], 0, 0

    processed_segments.sort(key=lambda x: x["start_ms"])
    
    speedup_count = 0
    collision_resolved_count = 0
    prev_end_ms = 0

    for i in range(len(processed_segments)):
        seg = processed_segments[i]
        
        # A. Ensure segment does not overlap with PREVIOUS segment
        if seg["start_ms"] < prev_end_ms + min_gap_ms:
            seg["start_ms"] = prev_end_ms + min_gap_ms
            collision_resolved_count += 1
            
        dur_ms = len(seg["audio"])
        
        # B. Check available gap before NEXT segment
        if i < len(processed_segments) - 1:
            next_start_ms = processed_segments[i + 1]["start_ms"]
            available_ms = next_start_ms - seg["start_ms"] - min_gap_ms
            
            # Perform speedup if the current audio duration exceeds available gap
            if TTS_DYNAMIC_SPEEDUP and available_ms > 200 and dur_ms > available_ms:
                speed_ratio = dur_ms / available_ms
                speed_ratio = min(speed_ratio, 1.45)  # Cap speedup ratio at 1.45x
                
                if speed_ratio > 1.05:
                    wav_path = seg.get("wav_path", "")
                    sped_up = False
                    if wav_path and os.path.exists(wav_path):
                        sped_up_path = wav_path.replace(".wav", "_fast.wav")
                        if _speed_up_with_ffmpeg(wav_path, sped_up_path, speed_ratio):
                            new_audio = AudioSegment.from_file(sped_up_path)
                            new_audio = new_audio.set_frame_rate(DUBBED_SAMPLE_RATE).set_channels(1)
                            new_audio = _trim_silence(new_audio)
                            if seg.get("tts_volume_db", 0.0) != 0.0:
                                new_audio = new_audio + seg["tts_volume_db"]
                            seg["audio"] = new_audio
                            try: os.remove(sped_up_path)
                            except OSError: pass
                            sped_up = True
                            speedup_count += 1
                    
                    if not sped_up:
                        seg["audio"] = seg["audio"].speedup(playback_speed=speed_ratio, chunk_size=50, crossfade=25)
                        speedup_count += 1
                    
                    dur_ms = len(seg["audio"])

        # C. Double check: if audio still extends past next segment start, push next segment start forward
        if i < len(processed_segments) - 1:
            next_start = processed_segments[i + 1]["start_ms"]
            if seg["start_ms"] + dur_ms > next_start - min_gap_ms:
                processed_segments[i + 1]["start_ms"] = seg["start_ms"] + dur_ms + min_gap_ms
                collision_resolved_count += 1

        prev_end_ms = seg["start_ms"] + dur_ms

    return processed_segments, speedup_count, collision_resolved_count


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
    Aligns TTS audio segments to original timestamps and merges them into a single track.
    STRICT NON-OVERLAPPING GUARANTEE: Ensures consecutive TTS voice lines NEVER overlap.
    
    Returns the path to the merged dubbed audio WAV file.
    """
    print(f"[Module 5] Aligning and merging {len(segments)} segments...")
    t0 = time.time()
    
    total_ms = int(video_duration_sec * 1000)
    
    # 1. Pre-process and trim all speech segments first
    raw_segments = []
    
    for segment in segments:
        if not segment.get("tts_audio_path") or not os.path.exists(segment["tts_audio_path"]):
            continue
            
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        
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
            
        raw_segments.append({
            "id": segment.get("id"),
            "audio": segment_audio,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "wav_path": wav_path,
            "tts_volume_db": tts_volume_db
        })
        
    # 2. Strict non-overlapping collision resolution
    processed_segments, speedup_count, collisions_resolved = resolve_voice_collisions(raw_segments, min_gap_ms=40)
    
    # 3. Build speech intervals for audio ducking using non-overlapping positions
    speech_intervals = []
    for seg in processed_segments:
        s_start = seg["start_ms"]
        s_end = s_start + len(seg["audio"])
        if s_start < s_end:
            speech_intervals.append((s_start, s_end))
        
    # 4. Initialize background audio and apply ducking
    mixed_bg_audio = None
    
    if mix_original_audio and original_audio_path and os.path.exists(original_audio_path):
        try:
            print(f"           Loading original audio for mixing: {original_audio_path}")
            original_audio = AudioSegment.from_file(original_audio_path)
            original_audio = original_audio.set_frame_rate(DUBBED_SAMPLE_RATE)
            
            if original_volume_db != 0.0:
                original_audio = original_audio + original_volume_db
                
            merged_intervals = merge_intervals(speech_intervals, gap_threshold_ms=800)
            print(f"           Applying audio ducking ({duck_volume_db}dB) across {len(merged_intervals)} regions")
            
            ducked_audio = AudioSegment.silent(duration=0, frame_rate=DUBBED_SAMPLE_RATE)
            if original_audio.channels > 1:
                ducked_audio = ducked_audio.set_channels(original_audio.channels)
            else:
                ducked_audio = ducked_audio.set_channels(1)
                
            curr_ms = 0
            fade_ms = 200
            total_len = len(original_audio)
            
            for start_ms, end_ms in merged_intervals:
                shifted_start = max(0, start_ms + int(sync_offset_ms))
                shifted_end = max(0, end_ms + int(sync_offset_ms))
                
                shifted_start = min(total_len, shifted_start)
                shifted_end = min(total_len, shifted_end)
                
                s_start = max(curr_ms, shifted_start - fade_ms)
                s_end = min(total_len, shifted_end + fade_ms)
                
                if s_start > curr_ms:
                    ducked_audio += original_audio[curr_ms:s_start]
                    
                if shifted_start > s_start:
                    fade_down_part = original_audio[s_start:shifted_start]
                    fade_down_part = fade_down_part.fade(to_gain=duck_volume_db, start=0, duration=shifted_start - s_start)
                    ducked_audio += fade_down_part
                    
                if shifted_end > shifted_start:
                    speech_part = original_audio[shifted_start:shifted_end] + duck_volume_db
                    ducked_audio += speech_part
                    
                if s_end > shifted_end:
                    fade_up_part = original_audio[shifted_end:s_end] + duck_volume_db
                    fade_up_part = fade_up_part.fade(to_gain=-duck_volume_db, start=0, duration=s_end - shifted_end)
                    ducked_audio += fade_up_part
                    
                curr_ms = s_end
                
            if curr_ms < total_len:
                ducked_audio += original_audio[curr_ms:]
                
            mixed_bg_audio = ducked_audio
        except Exception as e:
            print(f"           ⚠️ Failed to load or mix original audio: {e}. Falling back to silent background.")
            
    if mixed_bg_audio is None:
        mixed_bg_audio = AudioSegment.silent(duration=total_ms, frame_rate=DUBBED_SAMPLE_RATE)
        mixed_bg_audio = mixed_bg_audio.set_channels(1)
        
    # 5. Overlay non-overlapping dubbed voice segments
    for seg in processed_segments:
        segment_audio = seg["audio"]
        shifted_start_ms = max(0, seg["start_ms"] + int(sync_offset_ms))
        mixed_bg_audio = mixed_bg_audio.overlay(segment_audio, position=shifted_start_ms)
        
    dubbed_audio_path = os.path.join(output_dir, f"dubbed_{job_id}.wav")
    mixed_bg_audio.export(dubbed_audio_path, format="wav")
    
    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(dubbed_audio_path) / (1024 * 1024)
    print(f"[Module 5] 🔒 Strict Non-Overlapping Alignment: Merged audio saved to {dubbed_audio_path}")
    print(f"           {len(processed_segments)} segments (0 overlaps guaranteed), {speedup_count} sped up, {collisions_resolved} gaps resolved, {file_size_mb:.1f}MB, {elapsed:.1f}s")
    return dubbed_audio_path


if __name__ == "__main__":
    pass
