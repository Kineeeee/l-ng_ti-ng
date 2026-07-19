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


def extract_subtitles_from_video(
    video_path: str,
    crop_bottom_ratio: float = OCR_CROP_BOTTOM_RATIO,
    sample_fps: float = OCR_SAMPLE_FPS,
    min_confidence: float = OCR_MIN_CONFIDENCE,
) -> list:
    """
    Extracts embedded/hardsub subtitle lines from video frames using RapidOCR.
    Strictly filters out single-frame noise, edge logos, and short flickers.

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

    print(f"[Module 2-OCR] 🔍 Extracting video hardsubs with RapidOCR (sample_fps={sample_fps}, crop_ratio={crop_bottom_ratio})...")
    t0 = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[OCR Warning] Could not open video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or height <= 0 or width <= 0:
        cap.release()
        return []

    frame_step = max(1, int(fps / sample_fps))
    crop_y_start = int(height * (1.0 - crop_bottom_ratio))

    raw_frames_text = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp = frame_idx / fps

            # Crop bottom subtitle region
            cropped = frame[crop_y_start:height, 0:width]

            # Run RapidOCR on cropped region
            result, _ = engine(cropped)

            if result:
                # result format: [[bbox, text, score], ...]
                for line in result:
                    bbox = line[0] if len(line) > 0 else []
                    text = line[1].strip() if len(line) > 1 else ""
                    score = float(line[2]) if len(line) > 2 else 0.0

                    if not text or len(text) < 2 or score < min_confidence:
                        continue

                    # Bounding box horizontal center filter (must be within 12% - 88% of width)
                    if bbox:
                        try:
                            x_center = sum(pt[0] for pt in bbox) / len(bbox)
                            x_ratio = x_center / width
                            if x_ratio < 0.12 or x_ratio > 0.88:
                                continue  # Skip edge icons/logos
                        except Exception:
                            pass

                    raw_frames_text.append({
                        "timestamp": round(timestamp, 3),
                        "text": text,
                        "confidence": round(score, 3)
                    })

        frame_idx += 1

    cap.release()

    if not raw_frames_text:
        print(f"[Module 2-OCR] No valid hardsub subtitles detected in {time.time() - t0:.2f}s.")
        return []

    # Group frame text into consecutive time segments
    frame_duration = 1.0 / sample_fps
    ocr_segments = []
    current_seg = None

    for item in raw_frames_text:
        t = item["timestamp"]
        txt = item["text"]
        conf = item["confidence"]

        if current_seg is None:
            current_seg = {
                "start": t,
                "end": round(t + frame_duration, 3),
                "text": txt,
                "confidence": conf,
                "count": 1
            }
        else:
            time_gap = t - current_seg["end"]
            sim = string_similarity(txt, current_seg["text"])

            # If same or highly similar text within short time gap (<= 1.2s), merge
            if sim >= 0.70 and time_gap <= 1.2:
                current_seg["end"] = round(t + frame_duration, 3)
                current_seg["confidence"] = max(current_seg["confidence"], conf)
                current_seg["count"] += 1
                if len(txt) > len(current_seg["text"]):
                    current_seg["text"] = txt
            else:
                # Save completed segment if it stayed for at least 3 sampled frames & duration >= 0.8s
                duration = current_seg["end"] - current_seg["start"]
                if current_seg["count"] >= 3 and duration >= 0.8:
                    ocr_segments.append({
                        "start": current_seg["start"],
                        "end": current_seg["end"],
                        "text": current_seg["text"],
                        "confidence": current_seg["confidence"]
                    })
                # Start new segment
                current_seg = {
                    "start": t,
                    "end": round(t + frame_duration, 3),
                    "text": txt,
                    "confidence": conf,
                    "count": 1
                }

    if current_seg:
        duration = current_seg["end"] - current_seg["start"]
        if current_seg["count"] >= 3 and duration >= 0.8:
            ocr_segments.append({
                "start": current_seg["start"],
                "end": current_seg["end"],
                "text": current_seg["text"],
                "confidence": current_seg["confidence"]
            })

    elapsed = time.time() - t0
    print(f"[Module 2-OCR] ✅ Filtered & Extracted {len(ocr_segments)} stable hardsub segments via OCR in {elapsed:.2f}s.")
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
        # If STT produced nothing at all, fallback to OCR segments entirely
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

    # Deep copy STT segments to guarantee zero modification to STT outputs
    merged = [dict(s) for s in stt_segments]
    recovered_count = 0

    for ocr_seg in ocr_segments:
        o_start = ocr_seg["start"]
        o_end = ocr_seg["end"]
        o_text = ocr_seg["text"]
        o_duration = max(0.1, o_end - o_start)

        # Calculate total overlap duration with ALL STT speech segments (including 0.3s safety buffer)
        total_overlap_duration = 0.0
        for stt_seg in stt_segments:
            s_start = max(0.0, stt_seg["start"] - 0.3)
            s_end = stt_seg["end"] + 0.3

            overlap = max(0.0, min(o_end, s_end) - max(o_start, s_start))
            total_overlap_duration += overlap

        overlap_ratio = total_overlap_duration / o_duration

        # If OCR segment overlaps with STT speech timeline (> 15% ratio or >= 0.25s overlap), IGNORE IT!
        if overlap_ratio > 0.15 or total_overlap_duration >= 0.25:
            continue

        # OCR segment is in a genuine STT silent gap! Insert it as a recovered segment.
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

    # Sort all segments chronologically by start time
    merged.sort(key=lambda x: x["start"])

    # Re-index IDs strictly from 1 to N
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
