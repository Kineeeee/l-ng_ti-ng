import sys
import os
import glob
import argparse
import requests
import json
import uuid
import time
import shutil

# Add parent directory to path so we can import backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.config import (
    LM_STUDIO_BASE_URL,
    TRANSLATION_PROVIDER,
    TRANSLATION_STYLE,
    MASK_OLD_SUBS,
    MASK_SUB_Y_RATIO,
    MASK_SUB_COLOR,
    LOGO_PATH,
    LOGO_POSITION,
    MIRROR_VIDEO,
    WATERMARK_TEXT,
    WATERMARK_POSITION,
    MASK_TOP_TEXT,
    MASK_TOP_Y_RATIO,
    MASK_TOP_COLOR,
    TOP_TEXT,
    MIX_ORIGINAL_AUDIO,
    DUCK_VOLUME_DB,
    ORIGINAL_VOLUME_DB,
    TTS_VOLUME_DB,
    AUDIO_SYNC_OFFSET_MS,
    PITCH_METHOD,
    PITCH_MAX_SHIFT_SEMITONES,
    PITCH_F0_BLEND_RATIO,
    TTS_VOICE,
    TTS_ENGINE,
    WHISPER_DEVICE,
)
from backend.app.pipeline.download import download_video
from backend.app.pipeline.transcribe import transcribe_audio
from backend.app.pipeline.translate import translate_segments
from backend.app.pipeline.summarize import summarize_video_content, format_summary_text
from backend.app.pipeline.tts import generate_tts_for_segments
from backend.app.pipeline.pitch import apply_pitch_to_all_segments
from backend.app.pipeline.align import align_and_merge_audio
from backend.app.pipeline.subtitle import generate_subtitles
from backend.app.pipeline.render import render_final_video, _get_video_info

# --- Checkpoint System and Verification Helpers ---

STEPS = ["download", "transcribe", "translate", "summarize", "tts", "pitch", "align", "subtitle", "render"]

_STEP_INDEX = {step: i for i, step in enumerate(STEPS)}


def _should_run(step: str, start_step: str) -> bool:
    """Return True if this step should execute given the resume start point."""
    return _STEP_INDEX[step] >= _STEP_INDEX[start_step]


def get_job_dir(output_dir: str, job_id: str) -> str:
    return os.path.join(output_dir, job_id)

def get_checkpoint_path(output_dir: str, job_id: str, step_name: str) -> str:
    return os.path.join(get_job_dir(output_dir, job_id), f"checkpoint_{step_name}.json")

def save_checkpoint(output_dir: str, job_id: str, step_name: str, data: dict):
    job_dir = get_job_dir(output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    path = get_checkpoint_path(output_dir, job_id, step_name)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)
    print(f"[Checkpoint] Saved step '{step_name}' to {path}")

def load_checkpoint(output_dir: str, job_id: str, step_name: str) -> dict:
    path = get_checkpoint_path(output_dir, job_id, step_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load checkpoint '{step_name}' at {path}: {e}")
        return None

def backup_checkpoint(path: str):
    if os.path.exists(path):
        bak_path = f"{path}.bak"
        try:
            shutil.copy2(path, bak_path)
            print(f"[Backup] Created backup: {bak_path}")
        except Exception as e:
            print(f"[Warning] Failed to create backup of {path}: {e}")

def get_furthest_completed_step(output_dir: str, job_id: str) -> str:
    for step in reversed(STEPS):
        if load_checkpoint(output_dir, job_id, step) is not None:
            if step == "render":
                return "render"
            idx = STEPS.index(step)
            return STEPS[idx + 1]
    return "download"

def initialize_resumed_state(output_dir: str, job_id: str, resume_step: str) -> dict:
    state = {
        "job_id": job_id,
        "video_path": None,
        "audio_path": None,
        "duration": None,
        "detected_lang": None,
        "segments": None,
        "dubbed_audio_path": None,
        "srt_path": None,
        "ass_path": None,
    }
    
    dl_info = load_checkpoint(output_dir, job_id, "download")
    if dl_info:
        state["video_path"] = dl_info.get("video_path")
        state["audio_path"] = dl_info.get("audio_path")
        state["duration"] = dl_info.get("duration_sec")
        
    for step in ["align", "pitch", "tts", "summarize", "translate", "transcribe"]:
        cp = load_checkpoint(output_dir, job_id, step)
        if cp and "segments" in cp:
            state["segments"] = cp["segments"]
            if "detected_language" in cp:
                state["detected_lang"] = cp["detected_language"]
            elif "source_language" in cp:
                state["detected_lang"] = cp["source_language"]
            break
            
    align_info = load_checkpoint(output_dir, job_id, "align")
    if align_info:
        state["dubbed_audio_path"] = align_info.get("dubbed_audio_path")
        
    sub_info = load_checkpoint(output_dir, job_id, "subtitle")
    if sub_info:
        state["srt_path"] = sub_info.get("srt_path")
        state["ass_path"] = sub_info.get("ass_path")
        
    return state

def validate_transcription_json(data: dict, original_data=None) -> tuple:
    if not isinstance(data, dict):
        return False, "Root element must be a JSON object"
    if "segments" not in data:
        return False, "Missing 'segments' key"
    if not isinstance(data["segments"], list):
        return False, "'segments' must be a list"
    for idx, seg in enumerate(data["segments"]):
        if not isinstance(seg, dict):
            return False, f"Segment at index {idx} is not an object"
        if "id" not in seg or "start" not in seg or "end" not in seg or "text" not in seg:
            return False, f"Segment at index {idx} is missing one or more required keys ('id', 'start', 'end', 'text')"
        try:
            float(seg["start"])
            float(seg["end"])
        except ValueError:
            return False, f"Segment ID {seg.get('id')} has invalid start/end timestamps (must be numbers)"
    return True, ""

def validate_translation_json(data: dict, original_segments: list = None) -> tuple:
    if not isinstance(data, dict):
        return False, "Root element must be a JSON object"
    if "segments" not in data:
        return False, "Missing 'segments' key"
    if not isinstance(data["segments"], list):
        return False, "'segments' must be a list"
    
    if original_segments is not None:
        if len(data["segments"]) != len(original_segments):
            return False, f"Segment count mismatch! Expected {len(original_segments)} segments, but found {len(data['segments'])}"
            
    for idx, seg in enumerate(data["segments"]):
        if not isinstance(seg, dict):
            return False, f"Segment at index {idx} is not an object"
        if "id" not in seg or "start" not in seg or "end" not in seg or "text" not in seg or "translated_text" not in seg:
            return False, f"Segment at index {idx} is missing one or more required keys ('id', 'start', 'end', 'text', 'translated_text')"
        
        if original_segments is not None:
            expected_id = original_segments[idx]["id"]
            if seg["id"] != expected_id:
                return False, f"Segment ID mismatch at index {idx}! Expected ID {expected_id}, but found {seg['id']}"
                
        try:
            float(seg["start"])
            float(seg["end"])
        except ValueError:
            return False, f"Segment ID {seg.get('id')} has invalid start/end timestamps (must be numbers)"
            
    return True, ""

def pause_for_verification(output_dir: str, job_id: str, step_name: str, file_to_check: str, validate_func, original_data=None) -> dict:
    path = os.path.join(get_job_dir(output_dir, job_id), file_to_check)
    
    print("\n" + "="*60)
    print(f"⏸  [Interactive] Step '{step_name}' verification needed!")
    print(f"Please inspect/edit the file: {path}")
    print("="*60)
    
    while True:
        choice = input("Press [Enter] to reload and continue, 'r' to re-run this step, or 'q' to quit: ").strip().lower()
        if choice == 'q':
            print("Exiting pipeline. You can resume later using --resume.")
            sys.exit(0)
        elif choice == 'r':
            print(f"Re-running step '{step_name}'...")
            return {"action": "rerun"}
            
        if not os.path.exists(path):
            print(f"❌ Error: File not found at {path}. Please restore or recreate the file.")
            continue
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as je:
            print(f"❌ JSON Syntax Error: {je}")
            print("Please fix the JSON syntax and try again.")
            continue
            
        is_valid, err_msg = validate_func(data, original_data) if original_data is not None else validate_func(data)
        if not is_valid:
            print(f"❌ Validation Error: {err_msg}")
            print("Please correct the file and try again.")
            continue
            
        backup_checkpoint(path)
        print("✅ Checkpoint validated and loaded successfully.")
        return {"action": "continue", "data": data}

def generate_manual_translation_markdown(output_dir: str, job_id: str, segments: list, source_lang: str):
    job_dir = get_job_dir(output_dir, job_id)
    md_path = os.path.join(job_dir, "manual_translation.md")
    
    if os.path.exists(md_path):
        print(f"[Manual Translation] Markdown file already exists at: {md_path}")
        return
        
    lines = [
        "# AutoDub VN - Manual Translation File",
        f"# Job ID: {job_id}",
        f"# Source Language: {source_lang}",
        "",
        "Hãy dịch các đoạn hội thoại dưới đây sang tiếng Việt.",
        "Điền bản dịch của bạn ngay sau dòng \"Translate: \".",
        "Hãy giữ nguyên tiêu đề \"## Segment <ID>\" và các mốc thời gian.",
        ""
    ]
    
    for seg in segments:
        start_min = int(seg['start'] // 60)
        start_sec = seg['start'] % 60
        end_min = int(seg['end'] // 60)
        end_sec = seg['end'] % 60
        
        time_str = f"{start_min:02d}:{start_sec:05.2f} -> {end_min:02d}:{end_sec:05.2f}"
        
        existing_trans = seg.get("translated_text", "").strip()
        trans_val = existing_trans if existing_trans else "[Điền bản dịch của bạn vào đây]"
        
        lines.extend([
            "---",
            f"## Segment {seg['id']} [{time_str}]",
            f"Original: {seg.get('text', '').strip()}",
            f"Translate: {trans_val}",
            ""
        ])
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"[Manual Translation] Created Markdown translation template at: {md_path}")

def parse_manual_translation_markdown(output_dir: str, job_id: str) -> dict:
    job_dir = get_job_dir(output_dir, job_id)
    md_path = os.path.join(job_dir, "manual_translation.md")
    
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown manual translation file not found at: {md_path}")
        
    translations = {}
    current_seg_id = None
    
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("## Segment"):
                parts = line_str.split()
                if len(parts) >= 3:
                    try:
                        current_seg_id = int(parts[2])
                    except ValueError:
                        current_seg_id = None
            elif line_str.startswith("Translate:") and current_seg_id is not None:
                val = line[len("Translate:"):].strip()
                if "[Điền bản dịch" in val or "[Dịch câu này" in val:
                    val = ""
                translations[current_seg_id] = val
                
    return translations

def cleanup_temp_files(output_dir: str, job_id: str, video_path: str, audio_path: str):
    job_dir = get_job_dir(output_dir, job_id)
    
    if os.path.exists(job_dir):
        try:
            shutil.rmtree(job_dir)
            print(f"[Cleanup] Removed temporary job folder: {job_dir}")
        except Exception as e:
            print(f"[Warning] Failed to remove job folder {job_dir}: {e}")
            
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            print(f"[Cleanup] Removed temporary audio file: {audio_path}")
        except Exception as e:
            print(f"[Warning] Failed to remove audio file {audio_path}: {e}")
            
    if video_path and os.path.exists(video_path):
        try:
            os.remove(video_path)
            print(f"[Cleanup] Removed temporary video file: {video_path}")
        except Exception as e:
            print(f"[Warning] Failed to remove video file {video_path}: {e}")

def check_lm_studio_health():
    if TRANSLATION_PROVIDER.lower() != "lm_studio":
        print(f"[Health Check] Translation provider is '{TRANSLATION_PROVIDER}', skipping LM Studio check.")
        return
        
    print("[Health Check] Checking LM Studio connection...")
    try:
        # Check /v1/models endpoint
        res = requests.get(f"{LM_STUDIO_BASE_URL}/models", timeout=5)
        if res.status_code == 200:
            models = res.json().get("data", [])
            if not models:
                print("⚠️  Warning: LM Studio is running, but no models are loaded. Translation may fail.")
            else:
                model_id = models[0].get("id", "Unknown")
                print(f"✅ LM Studio is ready. Loaded model: {model_id}")
            return True
        else:
            print(f"❌ LM Studio returned status {res.status_code}.")
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Error connecting to LM Studio.")
        print(f"Details: {e}")
        print("Please ensure LM Studio is running and the local server is started at the correct port.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AutoDub VN CLI Pipeline")
    parser.add_argument("video_url", nargs="?", default=None, help="URL of the video to process")
    parser.add_argument("--job-id", "--resume", dest="resume_job_id", default=None, help="Job ID to resume processing (e.g. 7f2a1b9c)")
    parser.add_argument("--resume-from", default=None, choices=["download", "transcribe", "translate", "summarize", "tts", "pitch", "align", "subtitle", "render"],
                        help="Step to start processing from. If not specified when resuming, starts from the next uncompleted step.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear data cache for the resumed step and subsequent steps")
    parser.add_argument("-i", "--interactive", action="store_true", help="Enable interactive verification at critical stages (Transcribe, Translate, Subtitle)")
    parser.add_argument("--manual-translate", action="store_true", help="Manually translate segments instead of using LLM machine translation")
    parser.add_argument("--auto-audio-tags", action="store_true", help="Auto insert emotional audio tags during translation (optimized for ElevenLabs)")
    parser.add_argument("--translation-style", "--style", dest="translation_style", default=TRANSLATION_STYLE,
                        choices=["standard", "humorous", "storyteller"],
                        help="Phong cách dịch thuật: 'standard'=Thuyết minh chuẩn | 'humorous'=Hài hước/Bình luận Fair Use | 'storyteller'=Giật tít kịch tính TikTok")
    parser.add_argument("--skip-summarize", action="store_true", help="Skip summarizing video content and generating recommended titles")
    
    parser.add_argument("--lang", default="auto", help="Source language (e.g., en, zh, auto)")
    parser.add_argument("--watermark", "--logo", dest="watermark", default=LOGO_PATH, help="Path to watermark/logo image (optional)")
    parser.add_argument("--logo-pos", "--watermark-pos", default=LOGO_POSITION,
                        choices=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
                        help="Position of the logo/watermark image (default: bottom-right)")
    parser.add_argument("--mirror", action="store_true", help="Mirror the video horizontally before adding subtitles")
    parser.add_argument("--watermark-text", default=WATERMARK_TEXT, help="Watermark text to show on the screen")
    parser.add_argument("--watermark-text-pos", default=WATERMARK_POSITION,
                        choices=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
                        help="Position of the watermark text (default: center)")
    parser.add_argument("--top-text", default=TOP_TEXT, help="Text to display in the top black box (optional)")
    parser.add_argument("--no-subs", action="store_true", help="Disable subtitles in output video")
    parser.add_argument("--no-mask-subs", action="store_true", help="Disable masking of old subtitles")
    parser.add_argument("--mask-sub-y-ratio", type=float, default=MASK_SUB_Y_RATIO, help="Height ratio of the subtitle mask from bottom (default: 0.15)")
    parser.add_argument("--mask-sub-color", default=MASK_SUB_COLOR, help="Color of the subtitle mask (default: black)")
    parser.add_argument("--mask-top-text", action="store_true", help="Enable masking of original text at the top of the video")
    parser.add_argument("--mask-top-y-ratio", type=float, default=MASK_TOP_Y_RATIO, help="Height ratio of the top mask from top (default: 0.10)")
    parser.add_argument("--mask-top-color", default=MASK_TOP_COLOR, help="Color of the top mask (default: black)")
    
    # Audio mixing and sync arguments
    parser.add_argument("--no-mix-audio", action="store_true", default=not MIX_ORIGINAL_AUDIO, help="Disable mixing of original background audio (output dubbed voice only)")
    parser.add_argument("--duck-vol", type=float, default=DUCK_VOLUME_DB, help="Volume of background audio during speech in dB (default: -18.0)")
    parser.add_argument("--original-vol", type=float, default=ORIGINAL_VOLUME_DB, help="Overall volume adjustment of original audio in dB (default: 0.0)")
    parser.add_argument("--tts-vol", type=float, default=TTS_VOLUME_DB, help="Volume boost of dubbed TTS voice in dB (default: 2.0)")
    parser.add_argument("--sync-offset", type=float, default=AUDIO_SYNC_OFFSET_MS, help="Global audio sync offset in milliseconds (default: 0.0)")

    # Pitch processing arguments
    parser.add_argument("--pitch", default=PITCH_METHOD,
                        choices=["none", "shift", "clone", "hoat_ngon"],
                        help="Pitch method: 'none'=tắt | 'shift'=Ph.1 đơn giản | 'clone'=Ph.2 F0 contour | 'hoat_ngon'=Giọng Cô gái hoạt ngôn CapCut (+3.5 semitones)")
    parser.add_argument("--pitch-max-shift", type=float, default=PITCH_MAX_SHIFT_SEMITONES,
                        help="Giới hạn shift semitones cho method=shift (default: 6.0)")
    parser.add_argument("--pitch-blend", type=float, default=PITCH_F0_BLEND_RATIO,
                        help="Tỷ lệ blend F0 cho method=clone, 0.0-1.0 (default: 0.7)")

    # Transcribe backend selection
    parser.add_argument(
        "--transcribe-device",
        default=WHISPER_DEVICE,
        choices=["auto", "mlx", "cpu", "groq"],
        help=(
            "Transcribe backend: "
            "'auto' (mặc định, tự chọn MLX trên Apple Silicon) | "
            "'mlx' (Apple GPU, nhanh nhất) | "
            "'cpu' (faster-whisper) | "
            "'groq' (Groq Cloud API, gần như tức thì)"
        ),
    )
    
    # OCR hardsub subtitle arguments
    parser.add_argument("--no-ocr", action="store_true", help="Disable Video OCR hardsub subtitle extraction")
    
    # TTS engine selection
    parser.add_argument("--tts-engine", default=None, choices=["edge-tts", "gemini", "elevenlabs"], help="TTS Engine (overrides .env TTS_ENGINE)")
    parser.add_argument("--tts-voice", default=None, help="TTS Voice (overrides .env default voices)")
    
    args = parser.parse_args()
    
    if args.no_ocr:
        import backend.app.config
        backend.app.config.ENABLE_OCR_SUBTITLE = False

    print("="*50)
    print(" AutoDub VN - CLI Pipeline Started ")
    print("="*50)
    
    check_lm_studio_health()
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track timing for each step
    timings = {}
    pipeline_start = time.time()
    
    # Determine job_id and start step
    job_id = None
    start_step = "download"
    video_path = None
    audio_path = None
    duration = 0.0
    detected_lang = None
    segments = None
    dubbed_audio_path = None
    srt_path = None
    ass_path = None
    
    if args.resume_job_id:
        job_id = args.resume_job_id
        job_output_dir = os.path.join(output_dir, job_id)
        if not os.path.exists(job_output_dir):
            print(f"❌ Error: Job directory does not exist at {job_output_dir}")
            sys.exit(1)
            
        if args.resume_from:
            start_step = args.resume_from
        else:
            start_step = get_furthest_completed_step(output_dir, job_id)
            print(f"[Resume] Autodetected next step to run: '{start_step}'")
            
        print(f"[Resume] Resuming job '{job_id}' from step '{start_step}'...")
        
        if getattr(args, "clear_cache", False):
            print(f"[Resume] Clearing data cache for step '{start_step}' and subsequent steps...")
            start_idx = _STEP_INDEX[start_step]
            # Delete checkpoints
            for step_to_clear in STEPS[start_idx:]:
                cp_path = get_checkpoint_path(output_dir, job_id, step_to_clear)
                if os.path.exists(cp_path):
                    try: os.remove(cp_path)
                    except Exception: pass
                bak_path = f"{cp_path}.bak"
                if os.path.exists(bak_path):
                    try: os.remove(bak_path)
                    except Exception: pass
            
            # Delete related output files
            if start_idx <= _STEP_INDEX["tts"]:
                for f in glob.glob(os.path.join(job_output_dir, "tts_*.wav")):
                    try: os.remove(f)
                    except Exception: pass
            elif start_idx == _STEP_INDEX["pitch"]:
                for f in glob.glob(os.path.join(job_output_dir, "tts_*_p*.wav")):
                    try: os.remove(f)
                    except Exception: pass
            
            if start_idx <= _STEP_INDEX["align"]:
                for f in glob.glob(os.path.join(job_output_dir, "dubbed_*.wav")):
                    try: os.remove(f)
                    except Exception: pass
            if start_idx <= _STEP_INDEX["subtitle"]:
                for ext in ["*.srt", "*.ass"]:
                    for f in glob.glob(os.path.join(job_output_dir, ext)):
                        try: os.remove(f)
                        except Exception: pass
            if start_idx <= _STEP_INDEX["render"]:
                for f in glob.glob(os.path.join(output_dir, f"final_{job_id}*.mp4")):
                    try: os.remove(f)
                    except Exception: pass

        # Load and validate resumed state
        if start_step != "download":
            state = initialize_resumed_state(output_dir, job_id, start_step)
            video_path = state["video_path"]
            audio_path = state["audio_path"]
            duration = state["duration"]
            segments = state["segments"]
            detected_lang = state["detected_lang"]
            dubbed_audio_path = state["dubbed_audio_path"]
            srt_path = state["srt_path"]
            ass_path = state["ass_path"]
            
            # Upstream state verification
            if start_step in ["transcribe", "translate", "summarize", "tts", "pitch", "align", "subtitle", "render"] and not video_path:
                print(f"❌ Error: Cannot resume from '{start_step}' because download info is missing. Please restart from 'download'.")
                sys.exit(1)
            if start_step in ["translate", "summarize", "tts", "pitch", "align", "subtitle"] and not segments:
                print(f"❌ Error: Cannot resume from '{start_step}' because transcription/segment checkpoint is missing.")
                sys.exit(1)
            if start_step == "render" and not dubbed_audio_path:
                print(f"❌ Error: Cannot resume from '{start_step}' because align checkpoint (dubbed audio) is missing.")
                sys.exit(1)
    else:
        if not args.video_url:
            parser.error("You must specify a video_url or --resume/--job-id to resume a job.")

    # 1. Download
    step = "download"
    run_step = _should_run(step, start_step)
    if run_step:
        print("\n--- [Step 1] Download ---")
        t0 = time.time()
        dl_res = download_video(args.video_url, output_dir=output_dir, job_id=job_id)
        video_path = dl_res["video_path"]
        audio_path = dl_res["audio_path"]
        duration = dl_res["duration_sec"]
        job_id = dl_res["job_id"]
        job_output_dir = os.path.join(output_dir, job_id)
        
        save_checkpoint(output_dir, job_id, step, {
            "job_id": job_id,
            "video_url": args.video_url,
            "video_path": video_path,
            "audio_path": audio_path,
            "duration_sec": duration,
            "resolution": dl_res["resolution"]
        })
        timings["1_Download"] = time.time() - t0
    else:
        print(f"\n--- [Step 1] Download (Skipped - Loaded from checkpoint) ---")
        print(f"  Job ID: {job_id}")
        print(f"  Video Path: {video_path}")

    # 2. Transcribe
    step = "transcribe"
    run_step = _should_run(step, start_step)
    if run_step:
        while True:
            print("\n--- [Step 2] Transcribe ---")
            t0 = time.time()
            lang_hint = None if args.lang == "auto" else args.lang
            segments = transcribe_audio(audio_path, language_hint=lang_hint, device_override=args.transcribe_device, video_path=video_path)
            
            if not segments:
                print("No speech detected. Exiting.")
                sys.exit(0)
                
            detected_lang = segments[0]["detected_language"]
            timings["2_Transcribe"] = time.time() - t0
            
            checkpoint_data = {
                "detected_language": detected_lang,
                "segments": segments
            }
            save_checkpoint(output_dir, job_id, step, checkpoint_data)
            
            if args.interactive:
                verify_res = pause_for_verification(
                    output_dir, job_id, step, "checkpoint_transcribe.json", validate_transcription_json
                )
                if verify_res["action"] == "rerun":
                    continue
                else:
                    segments = verify_res["data"]["segments"]
                    detected_lang = verify_res["data"]["detected_language"]
            break
    else:
        print(f"\n--- [Step 2] Transcribe (Skipped - Loaded from checkpoint) ---")
        print(f"  Detected Language: {detected_lang}")
        print(f"  Number of Segments: {len(segments)}")

    # 3. Translate
    step = "translate"
    run_step = _should_run(step, start_step)
    if run_step:
        while True:
            print("\n--- [Step 3] Translate ---")
            t0 = time.time()
            if args.manual_translate:
                generate_manual_translation_markdown(output_dir, job_id, segments, detected_lang)
                
                print("\n" + "="*60)
                print(f"⏸  [Interactive] Step '{step}' (Manual Translation) verification needed!")
                print(f"Please open output/{job_id}/manual_translation.md in your editor and fill in the translations.")
                print("="*60)
                
                md_success = False
                while not md_success:
                    choice = input("Press [Enter] to validate and continue, 'r' to re-create the md template, or 'q' to quit: ").strip().lower()
                    if choice == 'q':
                        print("Exiting pipeline. You can resume later.")
                        sys.exit(0)
                    elif choice == 'r':
                        md_path = os.path.join(get_job_dir(output_dir, job_id), "manual_translation.md")
                        if os.path.exists(md_path):
                            os.remove(md_path)
                        generate_manual_translation_markdown(output_dir, job_id, segments, detected_lang)
                        continue
                        
                    try:
                        md_translations = parse_manual_translation_markdown(output_dir, job_id)
                        missing_ids = []
                        for seg in segments:
                            if seg["id"] not in md_translations:
                                missing_ids.append(seg["id"])
                        if missing_ids:
                            print(f"❌ Error: The following segment IDs are missing in the Markdown file: {missing_ids}")
                            print("Please ensure you did not delete the '## Segment <ID>' headers.")
                            continue
                            
                        # Backup the manual translation file
                        md_path = os.path.join(get_job_dir(output_dir, job_id), "manual_translation.md")
                        backup_checkpoint(md_path)
                        
                        for seg in segments:
                            seg["translated_text"] = md_translations[seg["id"]]
                        print("✅ Manual translation Markdown parsed successfully.")
                        md_success = True
                    except Exception as e:
                        print(f"❌ Parsing Error: {e}")
                        print("Please correct the file formatting and try again.")
                        continue
            else:
                auto_tags = args.auto_audio_tags or (args.tts_engine == "elevenlabs" or TTS_ENGINE == "elevenlabs")
                segments = translate_segments(
                    segments,
                    detected_lang,
                    auto_audio_tags=auto_tags,
                    translation_style=args.translation_style
                )
            timings["3_Translate"] = time.time() - t0
            
            # Save standard translation checkpoint (useful if resuming from next steps like TTS)
            checkpoint_data = {
                "source_language": detected_lang,
                "segments": segments
            }
            save_checkpoint(output_dir, job_id, step, checkpoint_data)
            
            # For standard LLM translation, we run the JSON verification prompt if --interactive is enabled
            if args.interactive and not args.manual_translate:
                tx_checkpoint = load_checkpoint(output_dir, job_id, "transcribe")
                orig_segments = tx_checkpoint["segments"] if tx_checkpoint else None
                
                verify_res = pause_for_verification(
                    output_dir, job_id, step, "checkpoint_translate.json", validate_translation_json, orig_segments
                )
                if verify_res["action"] == "rerun":
                    continue
                else:
                    segments = verify_res["data"]["segments"]
            break
    else:
        print(f"\n--- [Step 3] Translate (Skipped - Loaded from checkpoint) ---")

    # 3.5. Summarize & Suggest Titles
    step = "summarize"
    run_step = (not args.skip_summarize) and _should_run(step, start_step)
    if run_step:
        print("\n--- [Step 3.5] Summarize & Suggest Titles ---")
        t0 = time.time()
        summary_data = summarize_video_content(segments, job_dir=job_output_dir)
        save_checkpoint(output_dir, job_id, step, summary_data)
        timings["3.5_Summarize"] = time.time() - t0

        txt_content = format_summary_text(summary_data)
        print("\n" + txt_content)
    elif args.skip_summarize:
        print("\n--- [Step 3.5] Summarize & Suggest Titles (Skipped - CLI flag --skip-summarize) ---")
    else:
        print("\n--- [Step 3.5] Summarize & Suggest Titles (Skipped - Loaded from checkpoint) ---")

    # 4. Text-to-Speech
    step = "tts"
    run_step = _should_run(step, start_step)
    if run_step:
        while True:
            print("\n--- [Step 4] Text-to-Speech ---")
            t0 = time.time()
            segments = generate_tts_for_segments(
                segments, 
                job_output_dir, 
                engine=args.tts_engine, 
                voice=args.tts_voice
            )
            timings["4_TTS"] = time.time() - t0
            
            checkpoint_data = {
                "engine": args.tts_engine,
                "voice": args.tts_voice,
                "segments": segments
            }
            save_checkpoint(output_dir, job_id, step, checkpoint_data)
            
            if args.interactive:
                print("\n" + "="*60)
                print(f"⏸  [Interactive] Step '{step}' (TTS) verification needed!")
                print(f"Individual WAV segments have been generated in: {job_output_dir}")
                print("Feel free to listen to the wav files.")
                print("If you want to modify translation.json and regenerate specific audios,")
                print("simply edit translation.json and select 'r'. The system will automatically")
                print("detect changes and re-generate only the modified segments.")
                print("="*60)
                
                choice = input("Press [Enter] to continue, 'r' to re-run TTS, or 'q' to quit: ").strip().lower()
                if choice == 'q':
                    print("Exiting pipeline. You can resume later.")
                    sys.exit(0)
                elif choice == 'r':
                    trans_cp = load_checkpoint(output_dir, job_id, "translate")
                    if trans_cp:
                        segments = trans_cp["segments"]
                    continue
            break
    else:
        print(f"\n--- [Step 4] Text-to-Speech (Skipped - Loaded from checkpoint) ---")

    # 4.5. Pitch Processing (optional)
    step = "pitch"
    run_step = (args.pitch != "none" and _should_run(step, start_step))
    if run_step:
        print(f"\n--- [Step 4.5] Pitch Processing ({args.pitch}) ---")
        t0 = time.time()
        segments = apply_pitch_to_all_segments(
            segments=segments,
            orig_audio_path=audio_path,
            method=args.pitch,
            max_shift_semitones=args.pitch_max_shift,
            f0_blend_ratio=args.pitch_blend,
        )
        checkpoint_data = {
            "pitch_method": args.pitch,
            "segments": segments
        }
        save_checkpoint(output_dir, job_id, step, checkpoint_data)
        timings["4.5_Pitch"] = time.time() - t0
    elif args.pitch != "none":
        print(f"\n--- [Step 4.5] Pitch Processing ({args.pitch}) (Skipped - Loaded from checkpoint) ---")
    else:
        print("\n--- [Step 4.5] Pitch Processing (tắt) ---")
    
    # 5. Align & Merge Audio
    step = "align"
    run_step = _should_run(step, start_step)
    if run_step:
        print("\n--- [Step 5] Align Audio ---")
        t0 = time.time()
        mix_original = not args.no_mix_audio
        dubbed_audio_path = align_and_merge_audio(
            segments=segments,
            video_duration_sec=duration,
            output_dir=job_output_dir,
            job_id=job_id,
            original_audio_path=audio_path,
            mix_original_audio=mix_original,
            duck_volume_db=args.duck_vol,
            original_volume_db=args.original_vol,
            tts_volume_db=args.tts_vol,
            sync_offset_ms=args.sync_offset
        )
        checkpoint_data = {
            "dubbed_audio_path": dubbed_audio_path,
            "segments": segments
        }
        save_checkpoint(output_dir, job_id, step, checkpoint_data)
        timings["5_Align"] = time.time() - t0
    else:
        print(f"\n--- [Step 5] Align Audio (Skipped - Loaded from checkpoint) ---")
        print(f"  Dubbed Audio Path: {dubbed_audio_path}")
    
    # 6. Subtitles
    step = "subtitle"
    run_step = _should_run(step, start_step)
    if run_step:
        print("\n--- [Step 6] Generate Subtitles ---")
        t0 = time.time()
        # Probe video resolution để ASS subtitle scale đúng cho mọi tỷ lệ màn hình
        _vinfo = _get_video_info(video_path) if video_path else {}
        _vw = _vinfo.get("width") or 1920
        _vh = _vinfo.get("height") or 1080
        sub_res = generate_subtitles(segments, job_output_dir, job_id,
                                     video_width=_vw, video_height=_vh)
        ass_path = sub_res["ass_path"]
        srt_path = sub_res["srt_path"]
        
        checkpoint_data = {
            "srt_path": srt_path,
            "ass_path": ass_path,
            "segments": segments
        }
        save_checkpoint(output_dir, job_id, step, checkpoint_data)
        timings["6_Subtitle"] = time.time() - t0
        
        if args.interactive:
            print("\n" + "="*60)
            print(f"⏸  [Interactive] Step '{step}' (Subtitle) verification needed!")
            print(f"Subtitles have been generated at:")
            print(f"  SRT: {srt_path}")
            print(f"  ASS: {ass_path}")
            print("You can open these files in an editor and tweak the styling, margins, or timing.")
            print("Any edits you save to these files will be burn-in directly during the next Step (Render).")
            print("="*60)
            choice = input("Press [Enter] to continue to Render, or 'q' to quit: ").strip().lower()
            if choice == 'q':
                print("Exiting pipeline. You can resume later.")
                sys.exit(0)
    else:
        print(f"\n--- [Step 6] Generate Subtitles (Skipped - Loaded from checkpoint) ---")
        print(f"  SRT Path: {srt_path}")
        print(f"  ASS Path: {ass_path}")
    
    # 7. Render
    step = "render"
    run_step = _should_run(step, start_step)
    if run_step:
        print("\n--- [Step 7] Render Final Video ---")
        t0 = time.time()
        watermark_enabled = bool(args.watermark)
        subtitle_enabled = not args.no_subs
        
        mask_old_subs = False if args.no_mask_subs else MASK_OLD_SUBS
        
        final_video = render_final_video(
            video_path=video_path,
            dubbed_audio_path=dubbed_audio_path,
            output_dir=output_dir,
            job_id=job_id,
            subtitle_srt_path=srt_path,
            subtitle_ass_path=ass_path,
            watermark_path=args.watermark,
            subtitle_enabled=subtitle_enabled,
            watermark_enabled=watermark_enabled,
            mask_old_subs=mask_old_subs,
            mask_sub_y_ratio=args.mask_sub_y_ratio,
            mask_sub_color=args.mask_sub_color,
            logo_position=args.logo_pos,
            mirror_enabled=args.mirror or MIRROR_VIDEO,
            watermark_text=args.watermark_text,
            watermark_position=args.watermark_text_pos,
            mask_top_text=args.mask_top_text or MASK_TOP_TEXT,
            mask_top_y_ratio=args.mask_top_y_ratio,
            mask_top_color=args.mask_top_color,
            top_text=args.top_text
        )
        
        if final_video:
            save_checkpoint(output_dir, job_id, step, {
                "final_video_path": final_video
            })
        timings["7_Render"] = time.time() - t0
    else:
        print(f"\n--- [Step 7] Render Final Video (Skipped - Loaded from checkpoint) ---")
        # Load final video path from checkpoint
        render_cp = load_checkpoint(output_dir, job_id, "render")
        final_video = render_cp.get("final_video_path") if render_cp else None
    
    total_time = time.time() - pipeline_start
    
    # ---- Summary ----
    print("\n" + "="*50)
    if final_video:
        print(f"✅ Pipeline complete! Final video: {final_video}")
    else:
        print("❌ Pipeline failed during render step.")
    
    print("\n⏱  Timing Breakdown:")
    print("-"*40)
    for step_name, elapsed in timings.items():
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {step_name:<16} {elapsed:>6.1f}s  {bar} {pct:>4.1f}%")
    print("-"*40)
    print(f"  {'TOTAL':<16} {total_time:>6.1f}s")
    
    if total_time > 0 and duration > 0:
        ratio = total_time / duration
        print(f"\n  Video duration: {duration:.0f}s | Processing: {total_time:.0f}s | Ratio: {ratio:.1f}x realtime")
    print("="*50)

    # ---- Confirmation and Cleanup ----
    if final_video and args.interactive:
        print("\n" + "="*60)
        print(f"🎉  [Render Complete] Final video generated at: {final_video}")
        print("Please check the output video.")
        print("="*60)
        
        confirm = input("Confirm the final video is good and run cleanup? (y/n) [default: y]: ").strip().lower()
        if confirm != 'n':
            cleanup_temp_files(output_dir, job_id, video_path, audio_path)
        else:
            print("[Cleanup] Keeping all temporary files and checkpoints so you can resume/tweak.")

if __name__ == "__main__":
    main()
