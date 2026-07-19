import os
import cv2
import time
import difflib
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
    """
    Calculate similarity ratio between two strings.
    """
    if not s1 or not s2:
        return 0.0
    s1_clean = "".join(s1.split()).lower()
    s2_clean = "".join(s2.split()).lower()
    if s1_clean == s2_clean:
        return 1.0
    return difflib.SequenceMatcher(None, s1_clean, s2_clean).ratio()


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


def extract_subtitles_from_video(
    video_path: str,
    stt_segments: list = None,
    crop_bottom_ratio: float = OCR_CROP_BOTTOM_RATIO,
    sample_fps: float = OCR_SAMPLE_FPS,
    min_confidence: float = OCR_MIN_CONFIDENCE,
) -> list:
    """
    Extracts embedded/hardsub subtitle lines from video frames using RapidOCR.
    HIGH-SPEED OPTIMIZATION:
    1. If stt_segments is provided, ONLY scans frames inside STT silent gaps (skips 100% of speech frames).
    2. Seeks ONLY ONCE per gap (cap.set POS_FRAMES) and reads frames sequentially to eliminate 99% of OpenCV seek latency.

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

    if fps <= 0 or height <= 0 or width <= 0:
        cap.release()
        return []

    # Calculate target time gaps to scan with OCR
    if stt_segments is not None and len(stt_segments) > 0:
        time_gaps = compute_stt_time_gaps(stt_segments, duration, min_gap_sec=0.8)
        if not time_gaps:
            print("[Module 2-OCR] ℹ️ STT speech covers the entire video without gaps. Skipping OCR scanning.")
            cap.release()
            return []
        total_gap_duration = sum(end - start for start, end in time_gaps)
        skipped_duration = max(0.0, duration - total_gap_duration)
        print(f"[Module 2-OCR] 🎯 Targeted High-Speed OCR on {len(time_gaps)} STT silent gap(s) ({total_gap_duration:.1f}s total). Skipping {skipped_duration:.1f}s of STT speech...")
    else:
        time_gaps = [(0.0, duration)]
        print(f"[Module 2-OCR] 🔍 Scanning entire video with RapidOCR (sample_fps={sample_fps}, crop_ratio={crop_bottom_ratio})...")

    t0 = time.time()
    crop_y_start = int(height * (1.0 - crop_bottom_ratio))
    frame_step = max(1, int(fps / sample_fps))
    effective_sample_interval = frame_step / fps

    raw_frames_text = []

    for gap_start, gap_end in time_gaps:
        start_frame_idx = int(gap_start * fps)
        end_frame_idx = min(total_frames - 1, int(gap_end * fps))

        if start_frame_idx > end_frame_idx:
            continue

        # Seek ONCE to the start of the gap (eliminates loop seek latency)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

        for curr_frame_idx in range(start_frame_idx, end_frame_idx + 1):
            ret, frame = cap.read()
            if not ret:
                break

            # Read frame sequentially, but only run OCR every frame_step frames
            if (curr_frame_idx - start_frame_idx) % frame_step == 0:
                timestamp = round(curr_frame_idx / fps, 3)

                # Crop bottom subtitle region
                cropped = frame[crop_y_start:height, 0:width]

                # Optimization: Downscale crop to max width 720px for 3x-5x faster ONNX inference
                crop_h, crop_w = cropped.shape[:2]
                if crop_w > 720:
                    scale = 720.0 / crop_w
                    new_w = 720
                    new_h = int(crop_h * scale)
                    cropped_ocr = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    cropped_ocr = cropped
                    scale = 1.0

                # Run RapidOCR on downscaled cropped region
                result, _ = engine(cropped_ocr)

                if result:
                    for line in result:
                        bbox = line[0] if len(line) > 0 else []
                        text = line[1].strip() if len(line) > 1 else ""
                        score = float(line[2]) if len(line) > 2 else 0.0

                        if not text or len(text) < 2 or score < min_confidence:
                            continue

                        # Bounding box horizontal center filter (must be within 12% - 88% of width)
                        if bbox:
                            try:
                                # Scale bbox back to original image coordinate space
                                if scale != 1.0:
                                    bbox = [[pt[0] / scale, pt[1] / scale] for pt in bbox]
                                x_center = sum(pt[0] for pt in bbox) / len(bbox)
                                x_ratio = x_center / width
                                if x_ratio < 0.12 or x_ratio > 0.88:
                                    continue  # Skip edge icons/logos
                            except Exception:
                                pass

                        raw_frames_text.append({
                            "timestamp": timestamp,
                            "text": text,
                            "confidence": round(score, 3)
                        })

    cap.release()

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
                # Save completed segment if it stayed for at least 2 sampled frames & duration >= 0.6s
                seg_dur = current_seg["end"] - current_seg["start"]
                if current_seg["count"] >= 2 and seg_dur >= 0.6:
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
        if current_seg["count"] >= 2 and seg_dur >= 0.6:
            ocr_segments.append({
                "start": current_seg["start"],
                "end": current_seg["end"],
                "text": current_seg["text"],
                "confidence": current_seg["confidence"]
            })

    elapsed = time.time() - t0
    print(f"[Module 2-OCR] ⚡ Extracted {len(ocr_segments)} hardsub segments from STT gaps in {elapsed:.2f}s!")
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
