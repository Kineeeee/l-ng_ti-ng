import math
import os
import subprocess
import json
import time

from backend.app.pipeline.audio_utils import get_wav_duration_fast

def get_video_duration(video_path: str) -> float:
    """Probes video duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        return float(data.get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0


def _speed_up_audio_file(input_path: str, output_path: str, speed_ratio: float) -> bool:
    """Speed up audio file using FFmpeg atempo filter (supports multi-pass speed > 2.0x)."""
    speed_ratio = max(1.01, min(speed_ratio, 10.0))
    atempo_filter = _build_atempo_filter(speed_ratio)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-i", input_path,
        "-filter:a", atempo_filter,
        "-ar", "24000", "-ac", "1",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _get_seg_start(seg: dict) -> float:
    """Returns segment start time in seconds (prioritizes orig_start)."""
    if "orig_start" in seg and seg["orig_start"] is not None:
        return float(seg["orig_start"])
    if "start_ms" in seg and seg["start_ms"] is not None:
        return seg["start_ms"] / 1000.0
    if "start" in seg and seg["start"] is not None:
        return float(seg["start"])
    if "new_start" in seg and seg["new_start"] is not None:
        return float(seg["new_start"])
    return 0.0


def _get_seg_end(seg: dict) -> float:
    """Returns segment end time in seconds (prioritizes orig_end)."""
    if "orig_end" in seg and seg["orig_end"] is not None:
        return float(seg["orig_end"])
    if "end_ms" in seg and seg["end_ms"] is not None:
        return seg["end_ms"] / 1000.0
    if "end" in seg and seg["end"] is not None:
        return float(seg["end"])
    if "new_end" in seg and seg["new_end"] is not None:
        return float(seg["new_end"])
    return 0.0


def _build_atempo_filter(speed: float) -> str:
    """Builds a chain of atempo filters to handle values outside FFmpeg's [0.5, 2.0] limit."""
    speed = max(0.1, min(speed, 10.0))
    filters = []
    curr = speed
    while curr > 2.0:
        filters.append("atempo=2.0")
        curr /= 2.0
    while curr < 0.5:
        filters.append("atempo=0.5")
        curr /= 0.5
    filters.append(f"atempo={curr:.5f}")
    return ",".join(filters)


def _render_clip_batch(video_path: str, clips: list, output_v_mp4: str, output_a_wav: str) -> bool:
    """
    Renders a batch of video & original audio clip tasks in a single FFmpeg pass using -filter_complex_script.
    Retimes BOTH video track AND background audio track to maintain 100% sync.
    Outputs video and audio into separate files to avoid MP4 muxing drift during concat.
    """
    filter_lines = []
    v_inputs = []
    a_inputs = []
    ffmpeg_inputs = []

    for idx, clip in enumerate(clips):
        orig_s = clip["orig_start"]
        orig_e = clip["orig_end"]
        orig_dur = max(0.02, orig_e - orig_s)
        target_dur = clip["target_dur"]
        v_speed = target_dur / orig_dur
        a_speed = 1.0 / v_speed

        # Accurate duration seeking
        ffmpeg_inputs.extend([
            "-ss", f"{orig_s:.3f}",
            "-t", f"{orig_dur:.3f}",
            "-i", video_path
        ])

        # Adjust video speed (constant 30fps frame-locked PTS + trim to exact target_dur) and audio speed
        filter_lines.append(
            f"[{idx}:v]fps=30,setpts=N/(30*TB)*{v_speed:.5f},trim=duration={target_dur:.4f}[v{idx}];"
        )
        filter_lines.append(
            f"[{idx}:a]asetpts=PTS,{_build_atempo_filter(a_speed)},aresample=async=1:first_pts=0[a{idx}];"
        )
        v_inputs.append(f"[v{idx}]")
        a_inputs.append(f"[a{idx}]")

    v_concat = "".join(v_inputs) + f"concat=n={len(clips)}:v=1:a=0[vout];"
    a_concat = "".join(a_inputs) + f"concat=n={len(clips)}:v=0:a=1[aout]"
    filter_lines.append(v_concat)
    filter_lines.append(a_concat)

    script_content = "\n".join(filter_lines)
    script_path = output_v_mp4 + "_filter.txt"

    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        # Render batch clips with -bf 0 (no B-frames) for 100% strictly monotonic PTS across concat batches
        cmd = ["ffmpeg", "-y", "-nostdin"] + ffmpeg_inputs + [
            "-filter_complex_script", script_path,
            "-map", "[vout]", "-r", "30", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-bf", "0", output_v_mp4,
            "-map", "[aout]", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", output_a_wav
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if res.returncode != 0 or not os.path.exists(output_v_mp4) or os.path.getsize(output_v_mp4) == 0:
            cmd_fallback = ["ffmpeg", "-y", "-nostdin"] + ffmpeg_inputs + [
                "-filter_complex_script", script_path,
                "-map", "[vout]", "-r", "30", "-c:v", "h264_videotoolbox", "-b:v", "4M", output_v_mp4,
                "-map", "[aout]", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", output_a_wav
            ]
            subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        return (os.path.exists(output_v_mp4) and os.path.getsize(output_v_mp4) > 0 and 
                os.path.exists(output_a_wav) and os.path.getsize(output_a_wav) > 0)
    finally:
        if os.path.exists(script_path):
            try: os.remove(script_path)
            except Exception: pass


def retime_video_and_segments(
    video_path: str,
    processed_segments: list,
    output_dir: str,
    job_id: str
) -> tuple:
    """
    Retimes original video segments AND original audio so that each speech segment matches its TTS audio duration.

    1. Cuts video & audio into non-overlapping gap clips and speech clips.
    2. Uses FFmpeg setpts and atempo filters to stretch/compress each segment to match TTS duration.
    3. Concatenates retimed video & audio clips into smooth output files without timeline drift.
    4. Updates each segment in processed_segments with exact new_start and new_end timestamps.

    Returns:
        tuple: (retimed_video_path: str, updated_segments: list, retimed_audio_path: str, retimed_total_duration: float)
    """
    from pydub import AudioSegment
    from backend.app.pipeline.audio_utils import trim_silence

    total_duration = get_video_duration(video_path)
    if total_duration <= 0 or not processed_segments:
        print("[Video Retime] ⚠️ Video duration empty or no segments. Returning original video.")
        return video_path, processed_segments, None, total_duration

    print(f"[Video Retime] 🎬 Retiming video & original background audio based on TTS audio durations ({len(processed_segments)} segments)...")
    t0 = time.time()

    tmp_dir = os.path.join(output_dir, f"_retime_tmp_{job_id}")
    os.makedirs(tmp_dir, exist_ok=True)

    # Filter & sort valid segments
    valid_segs = sorted(processed_segments, key=_get_seg_start)

    clip_tasks = []  # list of dicts: {type: 'gap'|'speech', orig_start, orig_end, target_dur, seg_ref}
    curr_orig_t = 0.0
    curr_retimed_t = 0.0

    for seg in valid_segs:
        raw_start = _get_seg_start(seg)
        raw_end = _get_seg_end(seg)

        # Enforce strict non-overlapping monotonicity on original video timeline
        s_start = max(curr_orig_t, raw_start)
        s_end = max(s_start + 0.05, raw_end)

        # Preserve original source video bounds
        if "orig_start" not in seg or seg["orig_start"] is None:
            seg["orig_start"] = s_start
        if "orig_end" not in seg or seg["orig_end"] is None:
            seg["orig_end"] = s_end

        # Audio duration from trimmed AudioSegment or WAV file
        seg_source = str(seg.get("source", ""))
        is_ocr_or_no_tts = seg_source in ("ocr_recovered", "ocr_only") or seg_source.startswith("ocr")

        if is_ocr_or_no_tts:
            tts_dur = max(0.2, s_end - s_start)
        elif seg.get("tts_audio_path") and os.path.exists(seg["tts_audio_path"]):
            try:
                audio_obj = AudioSegment.from_file(seg["tts_audio_path"])
                audio_obj = trim_silence(audio_obj)
                trimmed_dur = len(audio_obj) / 1000.0
                if trimmed_dur > 0.35:
                    tts_dur = trimmed_dur
                    seg["audio"] = audio_obj
                else:
                    tts_dur = max(0.2, s_end - s_start)
            except Exception:
                tts_dur = get_wav_duration_fast(seg["tts_audio_path"])
                if tts_dur <= 0.35:
                    tts_dur = max(0.2, s_end - s_start)
        elif "audio" in seg and seg["audio"] is not None:
            tts_dur = len(seg["audio"]) / 1000.0
        elif "tts_duration" in seg and seg["tts_duration"] is not None and seg["tts_duration"] > 0.35:
            tts_dur = float(seg["tts_duration"])
        elif "dur_ms" in seg and seg["dur_ms"] is not None and seg["dur_ms"] > 350:
            tts_dur = seg["dur_ms"] / 1000.0
        else:
            tts_dur = max(0.2, s_end - s_start)

        # Natural Ripple-Edit Retiming: Slow down video clip for speech if TTS is longer
        orig_dur = max(0.05, s_end - s_start)

        # Gap before this segment (1.000x exact original speed)
        if s_start > curr_orig_t + 0.02:
            orig_gap = s_start - curr_orig_t
            clip_tasks.append({
                "type": "gap",
                "orig_start": curr_orig_t,
                "orig_end": s_start,
                "target_dur": orig_gap,
                "seg_ref": None
            })
            curr_retimed_t += orig_gap
            curr_orig_t = s_start

        if is_ocr_or_no_tts:
            target_dur = orig_dur
        else:
            if tts_dur > orig_dur:
                # Fully stretch video clip to match TTS audio duration so voice is never rushed
                target_dur = tts_dur
                speed_ratio = tts_dur / target_dur
                if speed_ratio > 1.03:
                    wav_path = seg.get("tts_audio_path", "")
                    if wav_path and os.path.exists(wav_path):
                        sped_up_path = wav_path.replace(".wav", "_retime_fast.wav")
                        if _speed_up_audio_file(wav_path, sped_up_path, speed_ratio):
                            try:
                                new_audio = AudioSegment.from_file(sped_up_path)
                                new_audio = trim_silence(new_audio)
                                seg["audio"] = new_audio
                                tts_dur = len(new_audio) / 1000.0
                                os.replace(sped_up_path, wav_path)
                            except Exception:
                                if os.path.exists(sped_up_path):
                                    try: os.remove(sped_up_path)
                                    except OSError: pass
            else:
                target_dur = orig_dur

        # Speech segment clip
        clip_tasks.append({
            "type": "speech",
            "orig_start": s_start,
            "orig_end": s_end,
            "target_dur": target_dur,
            "seg_ref": seg
        })

        curr_orig_t = s_end
        curr_retimed_t += target_dur

    # Gap after last segment to end of video
    if total_duration > curr_orig_t + 0.02:
        clip_tasks.append({
            "type": "gap",
            "orig_start": curr_orig_t,
            "orig_end": total_duration,
            "target_dur": total_duration - curr_orig_t,
            "seg_ref": None
        })

    # Update segment new_start and new_end timestamps
    curr_retimed_t = 0.0
    cum_frames = 0
    for clip in clip_tasks:
        next_cum_frames = round((curr_retimed_t + clip["target_dur"]) * 30.0)
        clip["target_dur"] = max(1, next_cum_frames - cum_frames) / 30.0
        cum_frames += max(1, next_cum_frames - cum_frames)
        
        target_dur = clip["target_dur"]
        if clip["type"] == "speech" and clip["seg_ref"] is not None:
            n_start = round(curr_retimed_t, 3)
            n_end = round(curr_retimed_t + target_dur, 3)
            n_start_ms = int(round(curr_retimed_t * 1000.0))
            n_end_ms = int(round((curr_retimed_t + target_dur) * 1000.0))

            ref = clip["seg_ref"]
            ref["orig_start"] = clip["orig_start"]
            ref["orig_end"] = clip["orig_end"]
            ref["new_start"] = n_start
            ref["new_end"] = n_end
            ref["new_start_ms"] = n_start_ms
            ref["new_end_ms"] = n_end_ms
            ref["start"] = n_start
            ref["end"] = n_end
            ref["start_ms"] = n_start_ms
            ref["end_ms"] = n_end_ms

            # 1-to-1 Duration Lock: Guarantee TTS audio duration equals exact video clip millisecond duration
            if ref.get("audio") is not None and isinstance(ref["audio"], AudioSegment):
                audio_obj = ref["audio"]
                target_ms = n_end_ms - n_start_ms
                curr_ms = len(audio_obj)
                if curr_ms > 0 and target_ms > 0 and curr_ms != target_ms:
                    if curr_ms > target_ms:
                        fitted_audio = audio_obj[:target_ms]
                    else:
                        pad_ms = target_ms - curr_ms
                        fitted_audio = audio_obj + AudioSegment.silent(duration=pad_ms, frame_rate=audio_obj.frame_rate)
                    
                    ref["audio"] = fitted_audio
                    ref["dur_ms"] = target_ms
                    ref["tts_duration"] = round(target_ms / 1000.0, 3)
                    if ref.get("tts_audio_path") and os.path.exists(ref["tts_audio_path"]):
                        try:
                            fitted_audio.export(ref["tts_audio_path"], format="wav")
                        except Exception:
                            pass

        curr_retimed_t += target_dur

    # Render clip tasks in batches (50 clips per batch) to avoid memory or keyframe drift
    BATCH_SIZE = 50
    batch_files_v = []
    batch_files_a = []
    
    for b_idx in range(0, len(clip_tasks), BATCH_SIZE):
        batch_clips = clip_tasks[b_idx:b_idx + BATCH_SIZE]
        batch_v_mp4 = os.path.join(tmp_dir, f"batch_v_{b_idx // BATCH_SIZE:03d}.mp4")
        batch_a_wav = os.path.join(tmp_dir, f"batch_a_{b_idx // BATCH_SIZE:03d}.wav")
        if _render_clip_batch(video_path, batch_clips, batch_v_mp4, batch_a_wav):
            batch_files_v.append(batch_v_mp4)
            batch_files_a.append(batch_a_wav)

    if not batch_files_v or not batch_files_a:
        print("[Video Retime] ⚠️ Batch clip rendering failed, returning original video.")
        return video_path, processed_segments, None, total_duration

    # Concat all rendered batch files into single retimed mp4
    concat_list_v_path = os.path.join(tmp_dir, "concat_list_v.txt")
    with open(concat_list_v_path, "w", encoding="utf-8") as f:
        for p in batch_files_v:
            f.write(f"file '{os.path.abspath(p)}'\n")

    concat_list_a_path = os.path.join(tmp_dir, "concat_list_a.txt")
    with open(concat_list_a_path, "w", encoding="utf-8") as f:
        for p in batch_files_a:
            f.write(f"file '{os.path.abspath(p)}'\n")

    retimed_video_path = os.path.join(output_dir, f"retimed_{job_id}.mp4")
    concat_v_cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_v_path,
        "-c:v", "h264_videotoolbox", "-b:v", "4M", "-r", "30",
        retimed_video_path
    ]
    subprocess.run(concat_v_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    retimed_audio_path = os.path.join(output_dir, f"retimed_audio_{job_id}.wav")
    
    # Construct 100% sample-exact retimed background audio using PyDub PCM slicing to eliminate AAC truncation drift
    try:
        orig_audio_bg = AudioSegment.from_file(video_path)
        retimed_bg_parts = []
        for clip in clip_tasks:
            o_s_ms = int(clip["orig_start"] * 1000)
            o_e_ms = int(clip["orig_end"] * 1000)
            target_dur_ms = int(clip["target_dur"] * 1000)
            
            clip_bg = orig_audio_bg[o_s_ms:o_e_ms]
            orig_dur_ms = len(clip_bg)
            
            if orig_dur_ms > 0 and target_dur_ms > 0:
                speed_ratio = orig_dur_ms / float(target_dur_ms)
                if abs(speed_ratio - 1.0) > 0.005:
                    new_frame_rate = int(clip_bg.frame_rate * speed_ratio)
                    if new_frame_rate > 1000:
                        clip_retimed = clip_bg._spawn(clip_bg.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(clip_bg.frame_rate)
                    else:
                        clip_retimed = clip_bg
                else:
                    clip_retimed = clip_bg
                
                # Trim or pad to exact target_dur_ms
                if len(clip_retimed) > target_dur_ms:
                    clip_retimed = clip_retimed[:target_dur_ms]
                elif len(clip_retimed) < target_dur_ms:
                    clip_retimed += AudioSegment.silent(duration=target_dur_ms - len(clip_retimed), frame_rate=clip_bg.frame_rate)
                
                retimed_bg_parts.append(clip_retimed)

        if retimed_bg_parts:
            full_retimed_bg = retimed_bg_parts[0]
            for part in retimed_bg_parts[1:]:
                full_retimed_bg += part
            full_retimed_bg.export(retimed_audio_path, format="wav")
    except Exception as e:
        print(f"[Video Retime] ⚠️ PyDub bg audio export fallback: {e}")
        concat_a_cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_a_path,
            "-c", "copy",
            retimed_audio_path
        ]
        subprocess.run(concat_a_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    # Cleanup temp directory
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    for seg in processed_segments:
        if isinstance(seg, dict):
            seg.pop("audio", None)

    if os.path.exists(retimed_video_path) and os.path.getsize(retimed_video_path) > 0:
        elapsed = time.time() - t0
        print(f"[Video Retime] ✅ Retimed video & background audio created successfully ({curr_retimed_t:.1f}s total duration, took {elapsed:.1f}s) -> {retimed_video_path}")
        return retimed_video_path, processed_segments, retimed_audio_path, curr_retimed_t

    return video_path, processed_segments, None, total_duration

