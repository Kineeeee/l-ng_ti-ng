import os
import math

def format_time_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def format_time_ass(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100)) # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def generate_subtitles(segments: list, output_dir: str, job_id: str) -> dict:
    """
    Generates .srt and .ass files from the segments.
    Returns { 'srt_path': str, 'ass_path': str }
    """
    print(f"[Module 6] Generating subtitles for {len(segments)} segments...")
    os.makedirs(output_dir, exist_ok=True)
    
    srt_path = os.path.join(output_dir, f"sub_{job_id}.srt")
    ass_path = os.path.join(output_dir, f"sub_{job_id}.ass")
    
    # 1. Generate SRT
    with open(srt_path, "w", encoding="utf-8") as f_srt:
        for idx, segment in enumerate(segments, start=1):
            start_str = format_time_srt(segment["start"])
            end_str = format_time_srt(segment["end"])
            text = segment.get("translated_text", "").strip()
            
            f_srt.write(f"{idx}\n")
            f_srt.write(f"{start_str} --> {end_str}\n")
            f_srt.write(f"{text}\n\n")
            
    # 2. Generate ASS
    # Basic ASS header for standard styling
    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,2,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w", encoding="utf-8") as f_ass:
        f_ass.write(ass_header)
        for segment in segments:
            start_str = format_time_ass(segment["start"])
            end_str = format_time_ass(segment["end"])
            text = segment.get("translated_text", "").strip()
            # ASS line
            f_ass.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n")
            
    print(f"[Module 6] Subtitles saved to {srt_path} and {ass_path}")
    return {
        "srt_path": srt_path,
        "ass_path": ass_path
    }

if __name__ == "__main__":
    pass
