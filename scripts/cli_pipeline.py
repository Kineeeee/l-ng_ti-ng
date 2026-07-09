import sys
import os
import argparse
import requests
import json
import uuid
import time

# Add parent directory to path so we can import backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.config import (
    LM_STUDIO_BASE_URL,
    TRANSLATION_PROVIDER,
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
    AUDIO_SYNC_OFFSET_MS
)
from backend.app.pipeline.download import download_video
from backend.app.pipeline.transcribe import transcribe_audio
from backend.app.pipeline.translate import translate_segments
from backend.app.pipeline.tts import generate_tts_for_segments
from backend.app.pipeline.align import align_and_merge_audio
from backend.app.pipeline.subtitle import generate_subtitles
from backend.app.pipeline.render import render_final_video

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
    parser.add_argument("video_url", help="URL of the video to process")
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
    
    args = parser.parse_args()
    
    print("="*50)
    print(" AutoDub VN - CLI Pipeline Started ")
    print("="*50)
    
    check_lm_studio_health()
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track timing for each step
    timings = {}
    pipeline_start = time.time()
    
    # 1. Download
    print("\n--- [Step 1] Download ---")
    t0 = time.time()
    dl_res = download_video(args.video_url, output_dir=output_dir)
    video_path = dl_res["video_path"]
    audio_path = dl_res["audio_path"]
    duration = dl_res["duration_sec"]
    job_id = os.path.basename(video_path).replace(".mp4", "")
    timings["1_Download"] = time.time() - t0
    
    # 2. Transcribe
    print("\n--- [Step 2] Transcribe ---")
    t0 = time.time()
    lang_hint = None if args.lang == "auto" else args.lang
    segments = transcribe_audio(audio_path, language_hint=lang_hint)
    timings["2_Transcribe"] = time.time() - t0
    
    if not segments:
        print("No speech detected. Exiting.")
        sys.exit(0)
        
    detected_lang = segments[0]["detected_language"]
    
    # 3. Translate
    print("\n--- [Step 3] Translate ---")
    t0 = time.time()
    segments = translate_segments(segments, detected_lang)
    timings["3_Translate"] = time.time() - t0
    
    # 4. Text-to-Speech
    print("\n--- [Step 4] Text-to-Speech ---")
    t0 = time.time()
    job_output_dir = os.path.join(output_dir, job_id)
    segments = generate_tts_for_segments(segments, job_output_dir)
    timings["4_TTS"] = time.time() - t0
    
    # 5. Align & Merge Audio
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
    timings["5_Align"] = time.time() - t0
    
    # 6. Subtitles
    print("\n--- [Step 6] Generate Subtitles ---")
    t0 = time.time()
    sub_res = generate_subtitles(segments, job_output_dir, job_id)
    ass_path = sub_res["ass_path"]
    srt_path = sub_res["srt_path"]
    timings["6_Subtitle"] = time.time() - t0
    
    # 7. Render
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
    timings["7_Render"] = time.time() - t0
    
    total_time = time.time() - pipeline_start
    
    # ---- Summary ----
    print("\n" + "="*50)
    if final_video:
        print(f"✅ Pipeline complete! Final video: {final_video}")
    else:
        print("❌ Pipeline failed during render step.")
    
    print("\n⏱  Timing Breakdown:")
    print("-"*40)
    for step, elapsed in timings.items():
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {step:<16} {elapsed:>6.1f}s  {bar} {pct:>4.1f}%")
    print("-"*40)
    print(f"  {'TOTAL':<16} {total_time:>6.1f}s")
    
    if total_time > 0 and duration > 0:
        ratio = total_time / duration
        print(f"\n  Video duration: {duration:.0f}s | Processing: {total_time:.0f}s | Ratio: {ratio:.1f}x realtime")
    print("="*50)

if __name__ == "__main__":
    main()
