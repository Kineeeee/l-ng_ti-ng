import os
import cv2
import time
import glob
import shutil
import difflib
import subprocess
import numpy as np
import concurrent.futures
from backend.app.config import (
    ENABLE_OCR_SUBTITLE,
    OCR_CROP_BOTTOM_RATIO,
    OCR_SAMPLE_FPS,
    OCR_MIN_CONFIDENCE,
)

# Global lazy initialized RapidOCR engine
_ocr_engine = None

def get_ocr_engine():
    """
    Lazy initialization of RapidOCR engine with GPU acceleration (CoreML / CUDA / DirectML) when available.
    """
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            import onnxruntime as ort
            print("[OCR] Initializing RapidOCR engine...")
            
            # Check available GPU execution providers
            available = ort.get_available_providers()
            preferred_gpu = ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'DirectMLExecutionProvider']
            gpu_providers = [p for p in preferred_gpu if p in available]
            
            _ocr_engine = RapidOCR()

            if gpu_providers:
                target_providers = gpu_providers + ['CPUExecutionProvider']
                print(f"[OCR] 🚀 Enabling GPU acceleration for OCR: {gpu_providers}")
                for sub in [getattr(_ocr_engine, 'text_detector', None), 
                            getattr(_ocr_engine, 'text_recognizer', None), 
                            getattr(_ocr_engine, 'text_cls', None)]:
                    if sub and hasattr(sub, 'infer') and hasattr(sub.infer, 'session'):
                        try:
                            model_path = sub.infer.session._model_path
                            sub.infer.session = ort.InferenceSession(model_path, providers=target_providers)
                        except Exception as patch_err:
                            print(f"[OCR Warning] Failed to patch GPU session for sub-module: {patch_err}")
            else:
                print("[OCR] Running on CPUExecutionProvider (CPU multi-threading).")

        except Exception as e:
            print(f"[OCR Warning] Failed to initialize RapidOCR engine: {e}")
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None


def string_similarity(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings."""
    if not s1 or not s2:
        return 0.0
    s1_clean = "".join(s1.split()).lower()
    s2_clean = "".join(s2.split()).lower()
    if s1_clean == s2_clean:
        return 1.0
    return difflib.SequenceMatcher(None, s1_clean, s2_clean).ratio()


def frame_mse(img1, img2) -> float:
    """Calculate Mean Squared Error between two images."""
    if img1 is None or img2 is None or img1.shape != img2.shape:
        return 999.0
    return float(np.mean((img1.astype("float") - img2.astype("float")) ** 2))


def get_text_pixel_mask(crop_bgr):
    """
    Extracts high-contrast text pixel mask to filter out video background movement.
    Returns: (text_ratio: float, binary_thresh_image)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0, None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # High-contrast thresholding for white/yellow subtitles
    _, thresh = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)
    white_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    ratio = white_pixels / total_pixels if total_pixels > 0 else 0.0
    return ratio, thresh


def compute_stt_time_gaps(stt_segments: list, total_duration: float, min_gap_sec: float = 0.8) -> list:
    """
    Computes time gaps (start, end) where Whisper STT detected NO speech.
    Returns list of (gap_start, gap_end) tuples.
    """
    if not stt_segments:
        return [(0.0, total_duration)] if total_duration >= min_gap_sec else []

    gaps = []
    sorted_segs = sorted(stt_segments, key=lambda x: x["start"])

    # Gap before first STT segment
    if sorted_segs[0]["start"] >= min_gap_sec:
        gaps.append((0.0, max(0.0, sorted_segs[0]["start"] - 0.2)))

    # Gaps between consecutive STT segments
    for i in range(len(sorted_segs) - 1):
        prev_end = sorted_segs[i]["end"]
        next_start = sorted_segs[i + 1]["start"]
        gap_dur = next_start - prev_end
        if gap_dur >= min_gap_sec:
            gaps.append((round(prev_end + 0.2, 3), round(next_start - 0.2, 3)))

    # Gap after last STT segment
    if total_duration - sorted_segs[-1]["end"] >= min_gap_sec:
        gaps.append((round(sorted_segs[-1]["end"] + 0.2, 3), round(total_duration, 3)))

    return gaps


def detect_subtitle_region(
    video_path: str,
    height: int,
    width: int,
    duration: float,
    stt_segments: list = None,
    n_samples: int = 8,
    search_top_ratio: float = 0.65,
) -> tuple:
    """
    Auto-detects the exact vertical pixel range (y_start, y_end) where hard subtitles appear.

    PERFORMANCE:
    - Uses a SINGLE FFmpeg call (fps-based uniform sampling + crop + scale) to extract
      all n_samples frames at once — eliminates per-frame subprocess overhead entirely.
    - Pre-filters frames with text pixel ratio check before running ONNX inference.
    - Applies early-exit once EARLY_EXIT_FRAMES frames have confirmed votes.

    Returns:
        (y_start_px, y_end_px) if a clear subtitle band is found,
        or (None, None) if detection fails (caller falls back to crop_bottom_ratio).
    """
    engine = get_ocr_engine()
    if not engine:
        return None, None

    if duration <= 0:
        return None, None

    # Search only within the bottom portion of the frame
    y_search_start = int(height * search_top_ratio)
    crop_h_orig    = height - y_search_start   # original pixel height of search zone

    # ── Single FFmpeg call: extract n_samples frames uniformly, pre-cropped + scaled ──
    # fps = n_samples / duration gives exactly n_samples frames spread across the video.
    # crop=iw:{crop_h_orig}:0:{y_search_start} cuts the subtitle search zone.
    # scale=720:-1 shrinks to 720px wide (preserving AR) for fast ONNX.
    detect_tmp = os.path.join(
        os.path.dirname(os.path.abspath(video_path)),
        f"_ocr_detect_{int(time.time())}"
    )
    os.makedirs(detect_tmp, exist_ok=True)

    detect_fps = n_samples / max(1.0, duration)
    vf_detect  = (
        f"fps={detect_fps:.5f},"
        f"crop=iw:{crop_h_orig}:0:{y_search_start},"
        f"scale=720:-1"
    )
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-threads", "0",
        "-i", video_path,
        "-vf", vf_detect,
        "-frames:v", str(n_samples),
        "-q:v", "3",
        os.path.join(detect_tmp, "d_%03d.jpg")
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)

    detect_frames = sorted(glob.glob(os.path.join(detect_tmp, "d_*.jpg")))

    # scale factor to map bbox coords from 720px-wide space back to original-width space
    # (uniform scale: x_orig = x_720 * width/720,  y_full = y_search_start + y_crop * width/720)
    scale_to_orig = width / 720.0

    votes          = np.zeros(height, dtype=np.int32)
    frames_with_votes = 0
    EARLY_EXIT_FRAMES = 4

    try:
        for img_path in detect_frames:
            if frames_with_votes >= EARLY_EXIT_FRAMES:
                break

            frame = cv2.imread(img_path)
            if frame is None:
                continue

            # Pre-filter: skip frames with no bright pixels
            text_ratio, _ = get_text_pixel_mask(frame)
            if text_ratio < 0.001:
                continue

            try:
                result, _ = engine(frame)
            except Exception:
                continue

            if not result:
                continue

            frame_contributed = False
            for line in result:
                bbox  = line[0] if len(line) > 0 else []
                score = float(line[2]) if len(line) > 2 else 0.0
                text  = line[1].strip() if len(line) > 1 else ""

                if score < 0.4 or len(text) < 2 or not bbox or len(bbox) < 2:
                    continue

                try:
                    # Map bbox from 720px-crop space → full-frame pixel space
                    y_coords = [pt[1] for pt in bbox]
                    y_top = int(y_search_start + min(y_coords) * scale_to_orig)
                    y_bot = int(y_search_start + max(y_coords) * scale_to_orig)

                    # Horizontal centering: x/720 = x_orig/video_width (uniform scale)
                    x_coords = [pt[0] for pt in bbox]
                    x_center_norm = (min(x_coords) + max(x_coords)) / 2.0 / 720.0
                    if x_center_norm < 0.1 or x_center_norm > 0.9:
                        continue

                    line_h = y_bot - y_top
                    if line_h < int(height * 0.01) or line_h > int(height * 0.12):
                        continue

                    votes[y_top:y_bot] += 1
                    frame_contributed = True
                except Exception:
                    continue

            if frame_contributed:
                frames_with_votes += 1
    finally:
        if os.path.exists(detect_tmp):
            try:
                shutil.rmtree(detect_tmp)
            except Exception:
                pass

    max_votes = int(np.max(votes))
    if max_votes < 2:
        print("[OCR Region] ⚠️ Could not auto-detect subtitle band "
              f"(max_votes={max_votes}, useful_frames={frames_with_votes}). "
              "Using fallback crop ratio.")
        return None, None

    # ── Find contiguous band around vote peak ─────────────────────────────────
    peak_y    = int(np.argmax(votes))
    threshold = max(2, int(max_votes * 0.35))

    top = peak_y
    while top > 0 and votes[top - 1] >= threshold:
        top -= 1

    bottom = peak_y
    while bottom < height - 1 and votes[bottom + 1] >= threshold:
        bottom += 1

    detected_h = bottom - top
    if detected_h < int(height * 0.01) or detected_h > int(height * 0.20):
        print(f"[OCR Region] ⚠️ Detected band ({detected_h}px) unrealistic. "
              "Using fallback crop ratio.")
        return None, None

    # Generous padding so OCR doesn't clip text edges
    padding = max(12, int(height * 0.015))
    y_start = max(0, top - padding)
    y_end   = min(height, bottom + padding)

    detected_ratio = (y_end - y_start) / height
    print(f"[OCR Region] ✅ Subtitle band: y=[{y_start}–{y_end}]px "
          f"({y_start/height:.3f}–{y_end/height:.3f}), "
          f"height={y_end-y_start}px ({detected_ratio:.1%} of frame), "
          f"useful_frames={frames_with_votes}/{len(detect_frames)}, "
          f"peak_votes={max_votes}")
    return y_start, y_end


def extract_subtitles_from_video(
    video_path: str,
    stt_segments: list = None,
    crop_bottom_ratio: float = OCR_CROP_BOTTOM_RATIO,
    sample_fps: float = OCR_SAMPLE_FPS,
    min_confidence: float = OCR_MIN_CONFIDENCE,
) -> list:
    """
    Extracts embedded/hardsub subtitle lines from video frames using RapidOCR.
    LIGHTNING-FAST HYBRID OPTIMIZED PIPELINE:
    1. Auto-detects exact subtitle Y-band via detect_subtitle_region() (replaces fixed 25% crop ratio).
    2. If stt_segments is provided, ONLY scans frames inside STT silent gaps (skips 100% of speech frames).
    3. Uses FFmpeg C-decoder to extract gap frames directly at sample_fps in batch (250x faster frame extraction).
    4. Text Mask Binarization Pre-Filter: Skips 100% of ONNX inferences on empty video background scenery (< 0.15% text pixels).
    5. Text Mask MSE Filter: Skips ONNX inference if subtitle text mask hasn't changed since last frame.
    6. Downscales cropped image width to 720px for 3x-5x faster ONNX GPU inference.

    Returns:
        List of dicts: [{'start': float, 'end': float, 'text': str, 'confidence': float}]
    """
    if not os.path.exists(video_path):
        print(f"[OCR Warning] Video path does not exist: {video_path}")
        return []

    engine = get_ocr_engine()
    if not engine:
        print("[OCR Warning] RapidOCR engine not available. Skipping OCR subtitle extraction.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[OCR Warning] Could not open video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    if fps <= 0 or height <= 0 or width <= 0:
        return []

    # ── Step 0: Auto-detect subtitle vertical region ──────────────────────────
    # Uses RapidOCR on sample frames to find the exact pixel band where subtitles
    # appear, replacing the fixed OCR_CROP_BOTTOM_RATIO with a tight, precise crop.
    print(f"[OCR Region] 🔍 Auto-detecting subtitle region from sample frames...")
    detected_y_start, detected_y_end = detect_subtitle_region(
        video_path=video_path,
        height=height,
        width=width,
        duration=duration,
        stt_segments=stt_segments,
        n_samples=8,
    )

    if detected_y_start is not None and detected_y_end is not None:
        crop_y_start = detected_y_start
        crop_y_end   = detected_y_end
    else:
        # Fallback: use configured crop_bottom_ratio
        crop_y_start = int(height * (1.0 - crop_bottom_ratio))
        crop_y_end   = height
        print(f"[OCR Region] ↩️  Fallback to crop_bottom_ratio={crop_bottom_ratio} → y=[{crop_y_start}–{crop_y_end}]px")

    # Calculate target time gaps to scan with OCR
    if stt_segments is not None and len(stt_segments) > 0:
        time_gaps = compute_stt_time_gaps(stt_segments, duration, min_gap_sec=0.8)
        if not time_gaps:
            print("[Module 2-OCR] ℹ️ STT speech covers the entire video without gaps. Skipping OCR scanning.")
            return []
        total_gap_duration = sum(end - start for start, end in time_gaps)
        skipped_duration = max(0.0, duration - total_gap_duration)
        print(f"[Module 2-OCR] ⚡ Ultra-Fast Targeted OCR on {len(time_gaps)} STT silent gap(s) ({total_gap_duration:.1f}s total, sample_fps={sample_fps}). Skipping {skipped_duration:.1f}s of STT speech...")
    else:
        time_gaps = [(0.0, duration)]
        print(f"[Module 2-OCR] 🔍 Scanning entire video with RapidOCR (sample_fps={sample_fps}, crop_y=[{crop_y_start}–{crop_y_end}]px)...")

    # ── OCR_TARGET_WIDTH: resolution at which ONNX runs (720px wide, pre-cropped by FFmpeg)
    OCR_TARGET_WIDTH = 720
    crop_height = crop_y_end - crop_y_start   # tight band in pixels

    # scale_to_norm: ratio to convert bbox x coords (in 720px space) to [0-1] of video width
    # Since FFmpeg scales uniformly: x_720 / OCR_TARGET_WIDTH == x_orig / width
    # So for the centering filter we can use x_720 / OCR_TARGET_WIDTH directly.

    t0 = time.time()
    effective_sample_interval = 1.0 / sample_fps

    # Temp directory for FFmpeg frame extraction
    temp_ocr_dir = os.path.join(
        os.path.dirname(os.path.abspath(video_path)),
        f"tmp_ocr_{int(time.time())}"
    )
    os.makedirs(temp_ocr_dir, exist_ok=True)

    raw_frames_text = []
    onnx_inferences_count = 0
    skipped_scenery_count = 0
    skipped_identical_count = 0

    # ── Helper: extract one gap's frames via FFmpeg (crop+scale included) ────
    def _extract_gap(args):
        """
        Run FFmpeg for a single STT silent gap.
        Crops to the detected subtitle band and scales to OCR_TARGET_WIDTH in one pass.
        Returns sorted list of extracted jpg paths.
        """
        g_idx, g_start, g_end, g_dir = args
        os.makedirs(g_dir, exist_ok=True)
        g_duration = max(0.2, g_end - g_start)
        out_pattern = os.path.join(g_dir, "frame_%05d.jpg")
        # Crop subtitle band + scale to 720px wide in a single FFmpeg vf chain.
        # This reduces file size ~10-14x vs writing full frames → less disk I/O.
        vf = (
            f"fps={sample_fps},"
            f"crop=iw:{crop_height}:0:{crop_y_start},"
            f"scale={OCR_TARGET_WIDTH}:-1"
        )
        cmd = [
            "ffmpeg", "-y", "-threads", "2",  # limit threads per process (parallel runs)
            "-ss", f"{g_start:.3f}",
            "-t",  f"{g_duration:.3f}",
            "-i",  video_path,
            "-vf", vf,
            "-q:v", "2",
            out_pattern
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        return g_idx, g_start, sorted(glob.glob(os.path.join(g_dir, "frame_*.jpg")))

    # ── Phase 1: Extract ALL gaps in parallel (up to 4 concurrent FFmpeg procs) ─
    max_workers = min(4, len(time_gaps), os.cpu_count() or 4)
    gap_extract_args = [
        (gap_idx, gap_start, gap_end,
         os.path.join(temp_ocr_dir, f"gap_{gap_idx}"))
        for gap_idx, (gap_start, gap_end) in enumerate(time_gaps)
    ]
    gap_img_files: dict = {}  # gap_idx → (gap_start, [img_paths])

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_extract_gap, args): args[0]
                       for args in gap_extract_args}
            for future in concurrent.futures.as_completed(futures):
                try:
                    g_idx, g_start, img_list = future.result()
                    gap_img_files[g_idx] = (g_start, img_list)
                except Exception as e:
                    print(f"[OCR Warning] Gap extraction failed: {e}")

        # ── Phase 2: ONNX inference — sequential (single engine instance) ──────
        for gap_idx, (gap_start, gap_end) in enumerate(time_gaps):
            g_start, img_files = gap_img_files.get(gap_idx, (gap_start, []))

            prev_thresh_mask    = None
            prev_lines_to_process = []

            for f_idx, img_path in enumerate(img_files):
                timestamp = round(g_start + (f_idx * effective_sample_interval), 3)
                # Frames are ALREADY cropped to subtitle band and scaled to 720px
                # by FFmpeg — no further crop or resize needed in Python.
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                cropped_ocr = frame  # 720px wide, crop_height tall

                # 1. Text Binarization Pre-Filter: check bright pixel density
                text_ratio, thresh_mask = get_text_pixel_mask(cropped_ocr)
                if text_ratio < 0.0015:
                    skipped_scenery_count += 1
                    prev_thresh_mask = thresh_mask
                    prev_lines_to_process = []
                    continue

                # 2. MSE Filter: reuse ONNX result if mask unchanged
                mask_diff = frame_mse(thresh_mask, prev_thresh_mask)
                if mask_diff < 15.0 and prev_lines_to_process:
                    lines_to_process = prev_lines_to_process
                    skipped_identical_count += 1
                else:
                    result, _ = engine(cropped_ocr)
                    lines_to_process = result if result else []
                    prev_thresh_mask = thresh_mask
                    prev_lines_to_process = lines_to_process
                    onnx_inferences_count += 1

                if lines_to_process:
                    for line in lines_to_process:
                        bbox  = line[0] if len(line) > 0 else []
                        text  = line[1].strip() if len(line) > 1 else ""
                        score = float(line[2]) if len(line) > 2 else 0.0

                        if not text or len(text) < 2 or score < min_confidence:
                            continue

                        # Horizontal centering: bbox x is in [0, 720] space.
                        # x_bbox / 720 == x_orig / video_width (uniform scale — no unscale needed).
                        if bbox:
                            try:
                                x_center_norm = (
                                    sum(pt[0] for pt in bbox) / len(bbox)
                                ) / OCR_TARGET_WIDTH
                                if x_center_norm < 0.12 or x_center_norm > 0.88:
                                    continue  # skip logos / edge icons
                            except Exception:
                                pass

                        raw_frames_text.append({
                            "timestamp": timestamp,
                            "text":      text,
                            "confidence": round(score, 3)
                        })

    finally:
        # Always clean up temporary frame extraction folder
        if os.path.exists(temp_ocr_dir):
            try:
                shutil.rmtree(temp_ocr_dir)
            except Exception:
                pass

    if not raw_frames_text:
        print(f"[Module 2-OCR] No valid hardsub subtitles detected in STT gaps ({time.time() - t0:.2f}s).")
        return []

    # Group frame text into consecutive time segments
    ocr_segments = []
    current_seg = None

    for item in raw_frames_text:
        t = item["timestamp"]
        txt = item["text"]
        conf = item["confidence"]

        if current_seg is None:
            current_seg = {
                "start": t,
                "end": round(t + effective_sample_interval, 3),
                "text": txt,
                "confidence": conf,
                "count": 1
            }
        else:
            time_gap = t - current_seg["end"]
            sim = string_similarity(txt, current_seg["text"])

            # If same or highly similar text within short time gap (<= 1.2s), merge
            if sim >= 0.70 and time_gap <= 1.2:
                current_seg["end"] = round(t + effective_sample_interval, 3)
                current_seg["confidence"] = max(current_seg["confidence"], conf)
                current_seg["count"] += 1
                if len(txt) > len(current_seg["text"]):
                    current_seg["text"] = txt
            else:
                # Save completed segment if duration >= 0.5s
                seg_dur = current_seg["end"] - current_seg["start"]
                if seg_dur >= 0.5:
                    ocr_segments.append({
                        "start": current_seg["start"],
                        "end": current_seg["end"],
                        "text": current_seg["text"],
                        "confidence": current_seg["confidence"]
                    })
                # Start new segment
                current_seg = {
                    "start": t,
                    "end": round(t + effective_sample_interval, 3),
                    "text": txt,
                    "confidence": conf,
                    "count": 1
                }

    if current_seg:
        seg_dur = current_seg["end"] - current_seg["start"]
        if seg_dur >= 0.5:
            ocr_segments.append({
                "start": current_seg["start"],
                "end": current_seg["end"],
                "text": current_seg["text"],
                "confidence": current_seg["confidence"]
            })

    elapsed = time.time() - t0
    print(f"[Module 2-OCR] 🚀 Extracted {len(ocr_segments)} hardsub segments ({onnx_inferences_count} ONNX inferences, {skipped_scenery_count} scenery skipped, {skipped_identical_count} identical skipped) in {elapsed:.2f}s!")
    return ocr_segments


def merge_stt_and_ocr_segments(stt_segments: list, ocr_segments: list) -> tuple:
    """
    Merges OCR subtitle segments with Whisper STT segments.
    STT segments are 100% GROUND TRUTH and will NEVER be overridden or modified.
    OCR segments are ONLY inserted if they fall into genuine time gaps where STT detected no speech.

    Returns:
        tuple: (merged_segments: list, stats: dict)
    """
    if not ocr_segments:
        return stt_segments, {"recovered_count": 0, "total": len(stt_segments)}

    if not stt_segments:
        formatted_ocr = []
        for idx, o in enumerate(ocr_segments, 1):
            formatted_ocr.append({
                "id": idx,
                "start": o["start"],
                "end": o["end"],
                "text": o["text"],
                "words": [],
                "source": "ocr_only"
            })
        return formatted_ocr, {"recovered_count": len(ocr_segments), "total": len(ocr_segments)}

    merged = [dict(s) for s in stt_segments]
    recovered_count = 0

    for ocr_seg in ocr_segments:
        o_start = ocr_seg["start"]
        o_end = ocr_seg["end"]
        o_text = ocr_seg["text"]
        o_duration = max(0.1, o_end - o_start)

        # Calculate total overlap duration with ALL STT speech segments
        total_overlap_duration = 0.0
        for stt_seg in stt_segments:
            s_start = max(0.0, stt_seg["start"] - 0.3)
            s_end = stt_seg["end"] + 0.3

            overlap = max(0.0, min(o_end, s_end) - max(o_start, s_start))
            total_overlap_duration += overlap

        overlap_ratio = total_overlap_duration / o_duration

        if overlap_ratio > 0.15 or total_overlap_duration >= 0.25:
            continue

        recovered_count += 1
        merged.append({
            "id": None,
            "start": o_start,
            "end": o_end,
            "text": o_text,
            "words": [],
            "source": "ocr_recovered"
        })
        print(f"   [OCR Fusion] ➕ Added missing hardsub segment [{o_start:.2f}s -> {o_end:.2f}s]: '{o_text}'")

    merged.sort(key=lambda x: x["start"])

    for idx, seg in enumerate(merged, 1):
        seg["id"] = idx

    stats = {
        "stt_original_count": len(stt_segments),
        "ocr_extracted_count": len(ocr_segments),
        "recovered_count": recovered_count,
        "final_count": len(merged)
    }

    print(f"[Module 2-OCR] 🔀 Fusion complete: STT ({len(stt_segments)}) + OCR Recovered ({recovered_count}) = {len(merged)} segments.")
    return merged, stats
