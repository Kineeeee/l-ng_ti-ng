import os
import subprocess
import re
import time
import platform

from backend.app.config import (
    MASK_OLD_SUBS,
    MASK_SUB_Y_RATIO,
    MASK_SUB_COLOR,
    LOGO_POSITION,
    MIRROR_VIDEO,
    WATERMARK_TEXT,
    MASK_TOP_TEXT,
    MASK_TOP_Y_RATIO,
    MASK_TOP_COLOR,
    WATERMARK_POSITION,
    TOP_TEXT,
    TOP_TEXT_FONT_PATH,
    TOP_TEXT_BOLD_FONT_PATH
)



import re

def parse_highlight_text(text: str) -> list:
    """Parse text to split segments inside single quotes for highlighting."""
    pattern = r"'([^']*)'"
    parts = []
    last_end = 0
    for match in re.finditer(pattern, text):
        start, end = match.span()
        if start > last_end:
            parts.append({
                "text": text[last_end:start],
                "highlight": False
            })
        parts.append({
            "text": match.group(1),
            "highlight": True
        })
        last_end = end
    if last_end < len(text):
        parts.append({
            "text": text[last_end:],
            "highlight": False
        })
    return parts


def estimate_text_width(text: str, fontsize: int, bold: bool = False, font_path: str = None) -> float:
    """Estimate horizontal width of text in pixels using PIL or fallback metrics."""
    if not text:
        return 0.0
    if font_path and os.path.exists(font_path):
        try:
            from PIL import ImageFont
            font = ImageFont.truetype(font_path, fontsize)
            # Use getlength for precise layout width (handles kerning & ligatures natively)
            return font.getlength(text)
        except Exception:
            pass
            
    # Fallback to standard metric estimation
    width_map = {
        "m": 0.778, "w": 0.722, "M": 0.833, "W": 0.944,
        "i": 0.222, "l": 0.222, "t": 0.278, "f": 0.278, "j": 0.222, "r": 0.333,
        "I": 0.278, "1": 0.500, " ": 0.278, "!": 0.278, ".": 0.278, ",": 0.278,
        ":": 0.278, ";": 0.278, "-": 0.333, "/": 0.278, "\\": 0.278,
        "(": 0.333, ")": 0.333, "[": 0.333, "]": 0.333, "{": 0.389, "}": 0.389,
        "'": 0.190, "\"": 0.355, "`": 0.222, "^": 0.469, "*": 0.389, "+": 0.584,
        "=": 0.584, "<": 0.584, ">": 0.584, "%": 0.833, "$": 0.556, "@": 1.010,
        "&": 0.667, "?": 0.556, "#": 0.556, "_": 0.500, "~": 0.584
    }
    total_width = 0
    for char in text:
        if char in width_map:
            w = width_map[char]
        elif char.isupper():
            w = 0.667
        else:
            w = 0.556
            
        if bold:
            w *= 1.1
        total_width += w * fontsize
    return total_width



def _get_video_info(video_path: str) -> dict:
    """Probe video for duration, resolution, and codec info using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        
        info = {"duration": 0, "width": 0, "height": 0, "vcodec": "", "has_audio": False}
        
        # Duration from format
        if "format" in data:
            info["duration"] = float(data["format"].get("duration", 0))
        
        for stream in data.get("streams", []):
            if stream["codec_type"] == "video":
                info["width"] = int(stream.get("width", 0))
                info["height"] = int(stream.get("height", 0))
                info["vcodec"] = stream.get("codec_name", "")
            elif stream["codec_type"] == "audio":
                info["has_audio"] = True
        
        return info
    except Exception:
        return {"duration": 0, "width": 0, "height": 0, "vcodec": "", "has_audio": False}


def _check_hw_encoder_available() -> bool:
    """Check if h264_videotoolbox is available on this system."""
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        return "h264_videotoolbox" in result.stdout
    except Exception:
        return False


def _run_ffmpeg_with_progress(cmd: list, total_duration: float, label: str = "Rendering"):
    """Run FFmpeg command and display real-time progress."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    time_pattern = re.compile(r"time=(\d+):(\d+):(\d+)\.(\d+)")
    speed_pattern = re.compile(r"speed=\s*([\d.]+)x")
    last_print_time = 0
    
    for line in process.stderr:
        match = time_pattern.search(line)
        if match:
            h, m, s, cs = match.groups()
            current_time = int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100
            
            now = time.time()
            # Print progress at most every 2 seconds to avoid spam
            if now - last_print_time >= 2:
                if total_duration > 0:
                    progress = min(current_time / total_duration * 100, 100)
                    speed_match = speed_pattern.search(line)
                    speed_str = f" @ {speed_match.group(1)}x" if speed_match else ""
                    
                    # Estimate remaining time
                    if current_time > 0 and speed_match:
                        speed_val = float(speed_match.group(1))
                        if speed_val > 0:
                            remaining = (total_duration - current_time) / speed_val
                            remaining_str = f" | ETA: {int(remaining)}s"
                        else:
                            remaining_str = ""
                    else:
                        remaining_str = ""
                    
                    print(f"           [{label}] {progress:5.1f}%{speed_str}{remaining_str}", flush=True)
                last_print_time = now
    
    process.wait()
    return process.returncode


def render_final_video(
    video_path: str,
    dubbed_audio_path: str,
    output_dir: str,
    job_id: str,
    subtitle_srt_path: str = None,
    subtitle_ass_path: str = None,
    watermark_path: str = None,
    subtitle_enabled: bool = True,
    watermark_enabled: bool = False,
    use_hw_accel: bool = True,
    quality: str = "balanced",
    mask_old_subs: bool = MASK_OLD_SUBS,
    mask_sub_y_ratio: float = MASK_SUB_Y_RATIO,
    mask_sub_color: str = MASK_SUB_COLOR,
    logo_position: str = LOGO_POSITION,
    mirror_enabled: bool = MIRROR_VIDEO,
    watermark_text: str = WATERMARK_TEXT,
    watermark_position: str = WATERMARK_POSITION,
    mask_top_text: bool = MASK_TOP_TEXT,
    mask_top_y_ratio: float = MASK_TOP_Y_RATIO,
    mask_top_color: str = MASK_TOP_COLOR,
    top_text: str = TOP_TEXT,
    top_text_font_path: str = TOP_TEXT_FONT_PATH,
    top_text_bold_font_path: str = TOP_TEXT_BOLD_FONT_PATH
) -> str:
    """
    Renders the final video using FFmpeg with hardware acceleration.
    
    Optimization strategy:
    - Uses VideoToolbox (h264_videotoolbox) on macOS for 5-10x faster encoding
    - Burns ASS subtitles directly into video for better quality & compatibility
    - Falls back to libx264 if hardware encoding is unavailable
    - Shows real-time progress with ETA
    
    Args:
        quality: "fast" (lower quality, fastest), "balanced" (good quality, fast), "high" (best quality, slower)
    """
    print("[Module 7] Rendering final video with FFmpeg...")
    os.makedirs(output_dir, exist_ok=True)
    
    final_video_path = os.path.join(output_dir, f"final_{job_id}.mp4")
    
    # Probe video info for progress tracking and smart encoding decisions
    video_info = _get_video_info(video_path)
    total_duration = video_info["duration"]
    print(f"           Source: {video_info['width']}x{video_info['height']}, "
          f"{video_info['vcodec']}, {total_duration:.1f}s")
    
    # Determine encoder
    hw_available = use_hw_accel and _check_hw_encoder_available()
    encoder = "h264_videotoolbox" if hw_available else "libx264"
    encoder_label = "VideoToolbox HW" if hw_available else "libx264 SW"
    print(f"           Encoder: {encoder_label}")
    
    # ----- Build FFmpeg command -----
    cmd = ["ffmpeg", "-y"]
    
    # Input files
    cmd.extend(["-i", video_path, "-i", dubbed_audio_path])
    
    # ----- Build video filter complex -----
    needs_video_encode = False
    y_min, y_max = None, None
    
    # Subtitle burn-in (prefer ASS over SRT for better styling)
    subtitle_file = None
    if subtitle_enabled:
        # Prefer ASS file if available (better styling support)
        if subtitle_ass_path and os.path.exists(subtitle_ass_path):
            subtitle_file = subtitle_ass_path
        elif subtitle_srt_path and os.path.exists(subtitle_srt_path):
            subtitle_file = subtitle_srt_path

    # Search for standard and bold font files supporting unicode (Vietnamese)
    font_file = None
    bold_font_file = None
    
    # Check if custom font files are configured and exist
    if top_text_font_path:
        abs_font = os.path.abspath(top_text_font_path)
        if os.path.exists(abs_font):
            font_file = abs_font
    if top_text_bold_font_path:
        abs_bold = os.path.abspath(top_text_bold_font_path)
        if os.path.exists(abs_bold):
            bold_font_file = abs_bold
        
    if not font_file:
        if platform.system() == "Darwin":
            possible_fonts = [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/HelveticaNeue.ttc",
                "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Arial.ttf"
            ]
            for f in possible_fonts:
                if os.path.exists(f):
                    font_file = f
                    break
        elif platform.system() == "Windows":
            possible_fonts = [
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\calibri.ttf",
                "C:\\Windows\\Fonts\\trebuc.ttf"
            ]
            for f in possible_fonts:
                if os.path.exists(f):
                    font_file = f
                    break
        else:  # Linux/other
            possible_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
            ]
            for f in possible_fonts:
                if os.path.exists(f):
                    font_file = f
                    break
                    
    if not bold_font_file:
        if platform.system() == "Darwin":
            possible_bold_fonts = [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Trebuchet MS Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Arial Bold.ttf"
            ]
            for f in possible_bold_fonts:
                if os.path.exists(f):
                    bold_font_file = f
                    break
        elif platform.system() == "Windows":
            possible_bold_fonts = [
                "C:\\Windows\\Fonts\\arialbd.ttf",
                "C:\\Windows\\Fonts\\calibrib.ttf",
                "C:\\Windows\\Fonts\\trebucbd.ttf"
            ]
            for f in possible_bold_fonts:
                if os.path.exists(f):
                    bold_font_file = f
                    break
        else:  # Linux/other
            possible_bold_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
            ]
            for f in possible_bold_fonts:
                if os.path.exists(f):
                    bold_font_file = f
                    break

    # Construct linear filters for the main video stream
    linear_filters = []
    
    # 1. Mirror video (hflip)
    if mirror_enabled:
        linear_filters.append("hflip")
        needs_video_encode = True
        
    # 2. Mask old subtitles
    if subtitle_file and mask_old_subs:
        # Try to dynamically detect subtitle bounding box coordinates
        y_min, y_max = detect_subtitle_bounding_box(
            video_path=video_path,
            subtitle_srt_path=subtitle_srt_path,
            subtitle_ass_path=subtitle_ass_path
        )
        
        if y_min is not None and y_max is not None:
            mask_filter = f"drawbox=x=0:y=ih*{y_min:.4f}:w=iw:h=ih*{(y_max - y_min):.4f}:color={mask_sub_color}:t=fill"
        else:
            mask_filter = f"drawbox=x=0:y=ih-ih*{mask_sub_y_ratio}:w=iw:h=ih*{mask_sub_y_ratio}:color={mask_sub_color}:t=fill"
            print(f"           Subtitle Mask (Static Fallback): Enabled (y=ih-ih*{mask_sub_y_ratio}, h=ih*{mask_sub_y_ratio}, color={mask_sub_color})")
        linear_filters.append(mask_filter)
        needs_video_encode = True
        
    # 2b. Mask top text (original subtitles / watermarks at the top)
    if top_text and not mask_top_text:
        mask_top_text = True

    if mask_top_text:
        top_mask_filter = f"drawbox=x=0:y=0:w=iw:h=ih*{mask_top_y_ratio:.4f}:color={mask_top_color}:t=fill"
        linear_filters.append(top_mask_filter)
        needs_video_encode = True

        if top_text:
            video_width = video_info.get("width") or 1280
            video_height = video_info.get("height") or 720
            box_height = int(video_height * mask_top_y_ratio)

            # Check if logo overlay is enabled and placed in top corners to prevent overlap
            logo_enabled = watermark_enabled and watermark_path and os.path.exists(watermark_path)
            left_margin = 20
            right_margin = 20
            if logo_enabled:
                logo_w = int(video_width * 0.1)
                if logo_position == "top-left":
                    left_margin = logo_w + 40
                elif logo_position == "top-right":
                    right_margin = logo_w + 40

            w_avail = video_width - left_margin - right_margin

            # Parse text into highlight/normal segments
            top_segments = parse_highlight_text(top_text)

            # Calculate top text font size to fit box_height and w_avail
            max_top_fontsize = int(box_height * 0.6)
            top_fontsize = max_top_fontsize
            
            while top_fontsize > 12:
                total_width = 0
                for seg in top_segments:
                    seg_font = bold_font_file if (seg["highlight"] and bold_font_file) else font_file
                    seg["width"] = estimate_text_width(seg["text"], top_fontsize, bold=seg["highlight"], font_path=seg_font)
                    total_width += seg["width"]
                if total_width <= w_avail:
                    break
                top_fontsize -= 2
            top_fontsize = max(12, top_fontsize)

            # Recalculate widths and compute starting position for centering
            total_width = 0
            for seg in top_segments:
                seg_font = bold_font_file if (seg["highlight"] and bold_font_file) else font_file
                seg["width"] = estimate_text_width(seg["text"], top_fontsize, bold=seg["highlight"], font_path=seg_font)
                total_width += seg["width"]

            x_start = left_margin + (w_avail - total_width) / 2.0
            
            # Calculate a constant y_val using font metrics to align baselines perfectly
            y_val = 0
            if font_file and os.path.exists(font_file):
                try:
                    from PIL import ImageFont
                    font_metric = ImageFont.truetype(font_file, top_fontsize)
                    ascent, descent = font_metric.getmetrics()
                    y_val = int((box_height - (ascent + descent)) / 2.0)
                except Exception:
                    y_val = int((box_height - top_fontsize * 1.15) / 2.0)
            else:
                y_val = int((box_height - top_fontsize * 1.15) / 2.0)
            y_val = max(0, y_val)
            
            current_x = x_start
            for seg in top_segments:
                if not seg["text"]:
                    continue
                
                escaped_seg_text = seg["text"].replace("'", "\\'").replace(":", "\\:").replace(" ", "\\ ")
                seg_font = bold_font_file if (seg["highlight"] and bold_font_file) else font_file
                seg_color = "yellow" if seg["highlight"] else "white"
                x_val = int(current_x)
                
                top_drawtext_opts = [
                    f"text='{escaped_seg_text}'",
                    f"x={x_val}",
                    f"y={y_val}",
                    f"fontsize={top_fontsize}",
                    f"fontcolor={seg_color}",
                    "shadowcolor=black@0.4",
                    "shadowx=1",
                    "shadowy=1"
                ]
                if seg_font:
                    escaped_font = seg_font.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
                    top_drawtext_opts.append(f"fontfile='{escaped_font}'")
                    
                top_drawtext_filter = f"drawtext={':'.join(top_drawtext_opts)}"
                linear_filters.append(top_drawtext_filter)
                
                current_x += seg["width"]
        
    # 3. New subtitles burn-in
    if subtitle_file:
        escaped_path = subtitle_file.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        if subtitle_file.endswith(".ass"):
            linear_filters.append(f"ass='{escaped_path}'")
        else:
            linear_filters.append(f"subtitles='{escaped_path}'")
        needs_video_encode = True
        
    # 4. Logo & Watermark positioning & conflict resolution
    logo_enabled = watermark_enabled and watermark_path and os.path.exists(watermark_path)
    
    video_width = video_info.get("width") or 1280
    video_height = video_info.get("height") or 720
    
    # Scale logo to 10% of video width to fit nicely
    logo_w = int(video_width * 0.1)
    
    # 4a. Default Logo positioning
    logo_x_expr = "20"
    logo_y_expr = "20"
    if logo_position == "top-right":
        logo_x_expr = "W-w-20"
        logo_y_expr = "20"
    elif logo_position == "bottom-left":
        logo_x_expr = "20"
        logo_y_expr = "H-h-20"
    elif logo_position == "bottom-right":
        logo_x_expr = "W-w-20"
        logo_y_expr = "H-h-20"
    elif logo_position == "center":
        logo_x_expr = "(W-w)/2"
        logo_y_expr = "(H-h)/2"

    # 4b. Default Watermark positioning
    top_offset = 20
    if mask_top_text:
        top_offset = int(video_height * mask_top_y_ratio) + 20
        
    bottom_offset_expr = "h-text_h-20"
    if subtitle_file and mask_old_subs:
        if y_min is not None:
            bottom_y_boundary = int(video_height * y_min)
        else:
            bottom_y_boundary = int(video_height * (1.0 - mask_sub_y_ratio))
        bottom_offset_expr = f"{bottom_y_boundary}-text_h-20"
        
    # Standard watermark fontsize
    fontsize = int(video_height * 0.05)

    watermark_pos_map = {
        "top-left": ("20", f"{top_offset}"),
        "top-right": ("w-text_w-20", f"{top_offset}"),
        "bottom-left": ("20", bottom_offset_expr),
        "bottom-right": ("w-text_w-20", bottom_offset_expr),
        "center": ("(w-text_w)/2", "(h-text_h)/2")
    }

    # 4c. Conflict resolution: if both are enabled and at the same position
    if logo_enabled and watermark_text and (logo_position == watermark_position):
        # We need the estimated width of the watermark text
        est_watermark_w = estimate_text_width(watermark_text, fontsize, bold=False, font_path=font_file)
        
        if logo_position == "top-left":
            # Logo at 20:20 (edge). Watermark to the right of it.
            watermark_pos_map["top-left"] = (f"{logo_w + 40}", f"{top_offset}")
        elif logo_position == "top-right":
            # Logo at W-w-20 (edge). Watermark to the left of it.
            watermark_pos_map["top-right"] = (f"w-{logo_w + 40}-text_w", f"{top_offset}")
        elif logo_position == "bottom-left":
            # Logo at 20:H-h-20 (edge). Watermark to the right of it.
            watermark_pos_map["bottom-left"] = (f"{logo_w + 40}", bottom_offset_expr)
        elif logo_position == "bottom-right":
            # Logo at W-w-20 (edge). Watermark to the left of it.
            watermark_pos_map["bottom-right"] = (f"w-{logo_w + 40}-text_w", bottom_offset_expr)
        elif logo_position == "center":
            # Both centered horizontally. Logo on the left, Watermark on the right.
            logo_x_val = int((video_width - logo_w - 20 - est_watermark_w) / 2.0)
            logo_x_expr = f"{logo_x_val}"
            
            watermark_x_val = int(logo_x_val + logo_w + 20)
            watermark_pos_map["center"] = (f"{watermark_x_val}", "(h-text_h)/2")

    x_expr, y_expr = watermark_pos_map.get(watermark_position, watermark_pos_map["center"])

    # 4d. Draw Watermark text
    if watermark_text:
        # Escaping for drawtext
        escaped_text = watermark_text.replace("'", "\\'").replace(":", "\\:")
        
        drawtext_opts = [
            f"text='{escaped_text}'",
            f"x='{x_expr}'",
            f"y='{y_expr}'",
            f"fontsize={fontsize}",
            "fontcolor=white@0.3",
            "shadowcolor=black@0.2",
            "shadowx=2",
            "shadowy=2"
        ]
        if font_file:
            escaped_font = font_file.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            drawtext_opts.append(f"fontfile='{escaped_font}'")
            
        drawtext_filter = f"drawtext={':'.join(drawtext_opts)}"
        linear_filters.append(drawtext_filter)
        needs_video_encode = True

    # 5. Logo image overlay (using filter_complex)
    logo_enabled = watermark_enabled and watermark_path and os.path.exists(watermark_path)
    
    # We will build the filter complex parts
    filter_parts = []
    video_out = "[0:v]"
    
    if linear_filters:
        filter_parts.append(f"[0:v]{','.join(linear_filters)}[v_filtered]")
        video_out = "[v_filtered]"
        
    if logo_enabled:
        cmd.extend(["-i", watermark_path])
        logo_idx = 2  # 0=video, 1=audio, 2=logo
        
        pos_expr = f"{logo_x_expr}:{logo_y_expr}"
        
        filter_parts.append(f"[{logo_idx}:v]scale={logo_w}:-1[scaled_logo]")
        filter_parts.append(f"{video_out}[scaled_logo]overlay={pos_expr}[vout]")
        video_out = "[vout]"
        needs_video_encode = True
        
    if needs_video_encode:
        if filter_parts:
            cmd.extend(["-filter_complex", ";".join(filter_parts)])
            cmd.extend(["-map", video_out, "-map", "1:a"])
        else:
            cmd.extend(["-map", "0:v", "-map", "1:a"])
    else:
        # No filters needed — copy video stream directly
        cmd.extend(["-c:v", "copy", "-map", "0:v", "-map", "1:a"])
        print("           Mode: Stream copy (no re-encoding needed — fastest)")
    
    # ----- Encoder settings -----
    if needs_video_encode:

        if hw_available:
            # VideoToolbox hardware encoding settings
            # Quality mapping: VT uses bitrate-based or quality-based encoding
            quality_map = {
                "fast": {"q": "65", "profile": "main"},
                "balanced": {"q": "55", "profile": "high"},
                "high": {"q": "45", "profile": "high"},
            }
            q_settings = quality_map.get(quality, quality_map["balanced"])
            
            cmd.extend([
                "-c:v", "h264_videotoolbox",
                "-profile:v", q_settings["profile"],
                "-q:v", q_settings["q"],        # Quality-based VBR (lower = better, 1-100)
                "-prio_speed", "true",           # Prioritize encoding speed
            ])
        else:
            # Software fallback with optimized settings
            quality_map = {
                "fast": {"crf": "26", "preset": "veryfast"},
                "balanced": {"crf": "22", "preset": "fast"},
                "high": {"crf": "18", "preset": "medium"},
            }
            q_settings = quality_map.get(quality, quality_map["balanced"])
            
            cmd.extend([
                "-c:v", "libx264",
                "-crf", q_settings["crf"],
                "-preset", q_settings["preset"],
                "-threads", "0",  # Auto-detect optimal thread count
            ])
    else:
        # No filters needed — copy video stream directly (instant!)
        cmd.extend(["-c:v", "copy"])
        print("           Mode: Stream copy (no re-encoding needed — fastest)")
    
    # Audio encoding
    cmd.extend([
        "-c:a", "aac",
        "-b:a", "192k",        # Slightly higher bitrate for dubbed voice quality
        "-ac", "2",             # Stereo
    ])
    
    # Global flags
    cmd.extend([
        "-movflags", "+faststart",  # Enable streaming (moov atom at start)
        "-shortest",                # End when shortest input ends
        final_video_path
    ])
    
    # Pretty-print command for debugging
    cmd_str = " ".join(cmd)
    print(f"           Command: {cmd_str}")
    
    # ----- Execute -----
    start_time = time.time()
    
    # Try hardware encoder first, fallback to software if it fails
    returncode = _run_ffmpeg_with_progress(cmd, total_duration, encoder_label)
    
    if returncode != 0 and hw_available:
        print(f"           ⚠️  Hardware encoder failed (code {returncode}), falling back to libx264...")
        
        # Rebuild command with software encoder
        fallback_cmd = [c for c in cmd]
        try:
            vt_idx = fallback_cmd.index("h264_videotoolbox")
            fallback_cmd[vt_idx] = "libx264"
            # Remove VT-specific args and replace with CRF
            for arg in ["-q:v", "-prio_speed"]:
                if arg in fallback_cmd:
                    idx = fallback_cmd.index(arg)
                    fallback_cmd.pop(idx)  # Remove flag
                    fallback_cmd.pop(idx)  # Remove value
            # Remove hwaccel
            if "-hwaccel" in fallback_cmd:
                idx = fallback_cmd.index("-hwaccel")
                fallback_cmd.pop(idx)  # Remove flag
                fallback_cmd.pop(idx)  # Remove value
            # Add CRF
            enc_idx = fallback_cmd.index("libx264")
            fallback_cmd.insert(enc_idx + 1, "-crf")
            fallback_cmd.insert(enc_idx + 2, "22")
            fallback_cmd.insert(enc_idx + 3, "-preset")
            fallback_cmd.insert(enc_idx + 4, "fast")
            fallback_cmd.insert(enc_idx + 5, "-threads")
            fallback_cmd.insert(enc_idx + 6, "0")
            
            returncode = _run_ffmpeg_with_progress(fallback_cmd, total_duration, "libx264 SW (fallback)")
        except (ValueError, IndexError):
            # If fallback reconstruction fails, just report original error
            pass
    
    elapsed = time.time() - start_time
    
    if returncode == 0:
        # Get output file size
        output_size = os.path.getsize(final_video_path) / (1024 * 1024)
        
        # Calculate speed ratio
        if elapsed > 0 and total_duration > 0:
            speed_ratio = total_duration / elapsed
            print(f"[Module 7] ✅ Render complete in {elapsed:.1f}s ({speed_ratio:.1f}x realtime)")
        else:
            print(f"[Module 7] ✅ Render complete in {elapsed:.1f}s")
        
        print(f"           Output: {final_video_path} ({output_size:.1f} MB)")
        return final_video_path
    else:
        print(f"[Module 7] ❌ FFmpeg failed with exit code {returncode}")
        return None


def _parse_srt_timestamps(srt_path: str) -> list:
    """Parse start/end timestamps from SRT file."""
    timestamps = []
    if not srt_path or not os.path.exists(srt_path):
        return timestamps
    try:
        import re
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Matches formats like: 00:01:23,456 --> 00:01:25,789
        pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
        matches = re.findall(pattern, content)
        for m in matches:
            h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m)
            start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
            end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
            timestamps.append((start, end))
    except Exception as e:
        print(f"           ⚠️ Failed to parse SRT timestamps: {e}")
    return timestamps


def _parse_ass_timestamps(ass_path: str) -> list:
    """Parse start/end timestamps from ASS file."""
    timestamps = []
    if not ass_path or not os.path.exists(ass_path):
        return timestamps
    try:
        with open(ass_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Text
        # Pattern to capture Start and End times
        for line in lines:
            if line.startswith("Dialogue:"):
                parts = line.split(",")
                if len(parts) >= 3:
                    start_str = parts[1].strip()
                    end_str = parts[2].strip()
                    
                    def time_to_sec(t_str):
                        h, m, s_cs = t_str.split(":")
                        s, cs = s_cs.split(".")
                        return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0
                    
                    start = time_to_sec(start_str)
                    end = time_to_sec(end_str)
                    timestamps.append((start, end))
    except Exception as e:
        print(f"           ⚠️ Failed to parse ASS timestamps: {e}")
    return timestamps


def detect_subtitle_bounding_box(
    video_path: str,
    subtitle_srt_path: str = None,
    subtitle_ass_path: str = None
) -> tuple:
    """
    Detect the exact vertical coordinates (y_min_ratio, y_max_ratio) of burned-in subtitles in the video.
    Returns (y_min_ratio, y_max_ratio) or (None, None) if detection fails or cv2 is not installed.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("           ⚠️ OpenCV (opencv-python) is not installed. Skipping dynamic subtitle detection.")
        return None, None

    # 1. Get sample timestamps from subtitle files
    dialogue_times = []
    if subtitle_srt_path:
        dialogue_times = _parse_srt_timestamps(subtitle_srt_path)
    if not dialogue_times and subtitle_ass_path:
        dialogue_times = _parse_ass_timestamps(subtitle_ass_path)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"           ⚠️ Failed to open video file for subtitle detection: {video_path}")
        return None, None
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    if width == 0 or height == 0:
        cap.release()
        return None, None

    # 2. Pick sample points (midpoints of speech segments)
    sample_times = []
    if dialogue_times:
        # Sample midpoints of segments, up to 10 points
        for start, end in dialogue_times:
            mid = (start + end) / 2.0
            if mid < duration:
                sample_times.append(mid)
        # Limit to at most 10 segments evenly distributed
        if len(sample_times) > 10:
            indices = np.linspace(0, len(sample_times) - 1, 10, dtype=int)
            sample_times = [sample_times[i] for i in indices]
    else:
        # Fallback: if no subtitle timings are available, sample every 5s in the middle 50% of the video
        if duration > 10:
            start_s = duration * 0.25
            end_s = duration * 0.75
            sample_times = list(np.linspace(start_s, end_s, 8))
        else:
            sample_times = [duration * 0.5]

    print(f"           Detecting subtitle position using OpenCV (analyzing {len(sample_times)} frames)...")
    detected_y_ranges = []
    
    # We only search in the bottom 30% of the video height (where subtitles are expected)
    y_crop_start = int(height * 0.70)
    
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        crop = frame[y_crop_start:, :]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Binary threshold to capture white/yellow subtitle text
        # Usually subtitles have high luminance (value > 200)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Dilate horizontally to group text characters into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Subtitle detection heuristic filters:
            # - Bounding box width must be at least 8% of video width
            # - Bounding box height must be between 1.5% and 12% of video height
            # - Bounding box must be relatively centered horizontally (e.g. center x within 15% to 85% of width)
            is_wide = w > int(width * 0.08)
            is_valid_height = int(height * 0.015) < h < int(height * 0.12)
            is_centered = int(width * 0.15) < (x + w/2) < int(width * 0.85)
            
            if is_wide and is_valid_height and is_centered:
                y_top_orig = y_crop_start + y
                y_bottom_orig = y_crop_start + y + h
                detected_y_ranges.append((y_top_orig, y_bottom_orig))
                
    cap.release()
    
    if not detected_y_ranges:
        print("           ⚠️ No subtitles detected in sample frames. Using configuration fallback.")
        return None, None

    # Group detected coordinates and find the median bounding box
    y_tops = [r[0] for r in detected_y_ranges]
    y_bottoms = [r[1] for r in detected_y_ranges]
    
    median_top = np.median(y_tops)
    median_bottom = np.median(y_bottoms)
    
    # Check if height is reasonable
    detected_height = median_bottom - median_top
    if detected_height < int(height * 0.01) or detected_height > int(height * 0.15):
        print("           ⚠️ Detected vertical range is unrealistic. Using configuration fallback.")
        return None, None

    # Add padding (approx 1% of video height or at least 8 pixels)
    padding = max(8, int(height * 0.01))
    final_top = max(0, int(median_top - padding))
    # To prevent "hở bottom", extend the mask box all the way to the bottom of the video frame
    final_bottom = height
    
    y_min_ratio = final_top / height
    y_max_ratio = 1.0
    
    print(f"           ✅ Detected subtitle region: y_min={y_min_ratio:.3f} ({final_top}px), "
          f"y_max={y_max_ratio:.3f} ({final_bottom}px), height={final_bottom - final_top}px")
          
    return y_min_ratio, y_max_ratio


if __name__ == "__main__":
    pass
