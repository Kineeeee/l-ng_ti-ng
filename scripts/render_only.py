import sys
import os
import argparse

# Add parent directory to path so we can import backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.config import (
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
    TOP_TEXT
)
from backend.app.pipeline.render import render_final_video

def main():
    parser = argparse.ArgumentParser(description="AutoDub VN - Render Only Utility")
    parser.add_argument("job_id", help="Job ID to render (e.g., a0f4c32c)")
    parser.add_argument("--watermark", "--logo", dest="watermark", default=LOGO_PATH, help="Path to logo/watermark image (optional)")
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
    
    args = parser.parse_args()
    
    output_dir = "output"
    video_path = os.path.join(output_dir, f"{args.job_id}.mp4")
    dubbed_audio_path = os.path.join(output_dir, args.job_id, f"dubbed_{args.job_id}.wav")
    srt_path = os.path.join(output_dir, args.job_id, f"sub_{args.job_id}.srt")
    ass_path = os.path.join(output_dir, args.job_id, f"sub_{args.job_id}.ass")
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"❌ Input video not found at: {video_path}")
        sys.exit(1)
    if not os.path.exists(dubbed_audio_path):
        print(f"❌ Dubbed audio not found at: {dubbed_audio_path}")
        sys.exit(1)
        
    watermark_enabled = bool(args.watermark)
    subtitle_enabled = not args.no_subs
    mask_old_subs = False if args.no_mask_subs else MASK_OLD_SUBS
    
    final_video = render_final_video(
        video_path=video_path,
        dubbed_audio_path=dubbed_audio_path,
        output_dir=output_dir,
        job_id=args.job_id,
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
        print(f"\n✅ Rendering complete: {final_video}")
    else:
        print("\n❌ Rendering failed.")

if __name__ == "__main__":
    main()
