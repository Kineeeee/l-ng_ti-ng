import os
import re
from backend.app.config import (
    SUBTITLE_MAX_CHARS_PER_LINE,
    SUBTITLE_MAX_LINES_BEFORE_SPLIT,
    SUBTITLE_MIN_SEGMENT_DURATION
)

# ─── Cấu hình chia câu ──────────────────────────────────────────────────────
MAX_CHARS_PER_LINE = SUBTITLE_MAX_CHARS_PER_LINE
MAX_LINES_BEFORE_SPLIT = SUBTITLE_MAX_LINES_BEFORE_SPLIT
MIN_SEGMENT_DURATION = SUBTITLE_MIN_SEGMENT_DURATION

# Các điểm chia ngữ nghĩa, theo thứ tự ưu tiên (penalty ngữ nghĩa tăng dần)
# Mỗi phần tử: (regex_pattern, mô tả, semantic_penalty)
_SEMANTIC_SPLIT_PATTERNS = [
    # Dấu câu mạnh: dấu phẩy, chấm phẩy, hai chấm — chia SAU dấu câu đó
    (r'[,;:]\s+', "punctuation", 0),
    # Liên từ tiếng Việt — chia TRƯỚC liên từ
    (r'\s+(?:nhưng|mà|tuy nhiên|tuy vậy|dù vậy|dù thế|song|bởi vì|vì|bởi|'
     r'nên|vậy nên|do đó|vì vậy|vì thế|do vậy|mặc dù|dẫu vậy|'
     r'và|hoặc|hay|thì|nếu|còn|cũng|rồi)\s+', "conjunction_vi", 15),
    # Liên từ tiếng Anh — chia TRƯỚC liên từ
    (r'\s+(?:but|however|although|though|yet|while|whereas|'
     r'and|or|so|because|since|if|then|also|still)\s+', "conjunction_en", 25),
    # Chia ở khoảng trắng — fallback (penalty cao nhất)
    (r'\s+', "whitespace", 50),
]


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
    cs = int(round((seconds - int(seconds)) * 100))  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def find_best_split_point(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> int:
    """
    Tìm vị trí chia tốt nhất trong text để phần đầu ≤ max_chars ký tự.

    Ưu tiên điểm chia NGỮNGHĨA gần với GIỮA vùng hợp lệ nhất.
    Điểm chia phải ở trong [min_start, max_chars] để cân bằng 2 dòng.
    
    Trả về index (exclusive) của điểm chia, hoặc -1 nếu không tìm được.
    """
    if len(text) <= max_chars:
        return -1  # Không cần chia

    # Vùng hợp lệ: phần đầu phải từ min_start đến max_chars ký tự
    min_start = max(3, max_chars // 3)   # Tránh chia quá sớm (>1/3 của max)
    # Điểm lý tưởng: giữa vùng hợp lệ để 2 dòng cân bằng nhất
    ideal_pos = (min_start + max_chars) // 2

    best_pos = -1
    best_score = float('inf')  # Score thấp hơn = tốt hơn

    for pattern, desc, semantic_penalty in _SEMANTIC_SPLIT_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            # Dấu câu: chia SAU dấu (bao gồm cả space sau dấu)
            # Liên từ/whitespace: chia TRƯỚC từ (bỏ space trước từ)
            if desc == "punctuation":
                split_pos = m.end()
            else:
                split_pos = m.start()

            # Loại bỏ điểm chia ngoài vùng hợp lệ
            if not (min_start <= split_pos <= max_chars):
                continue

            # Score = khoảng cách đến điểm lý tưởng + penalty ngữ nghĩa
            distance = abs(split_pos - ideal_pos)
            score = distance + semantic_penalty

            if score < best_score:
                best_score = score
                best_pos = split_pos

    return best_pos


def _split_text_into_lines(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> list:
    """
    Chia text thành danh sách các dòng, mỗi dòng ≤ max_chars ký tự.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    lines = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            lines.append(remaining.strip())
            break

        split_pos = find_best_split_point(remaining, max_chars)
        if split_pos == -1 or split_pos >= len(remaining):
            # Không tìm được điểm chia tốt → chia cứng tại khoảng trắng gần nhất
            pos = remaining.rfind(' ', 0, max_chars + 1)
            if pos == -1:
                pos = max_chars
            lines.append(remaining[:pos].strip())
            remaining = remaining[pos:].strip()
        else:
            lines.append(remaining[:split_pos].strip())
            remaining = remaining[split_pos:].strip()

    return [ln for ln in lines if ln]


def split_long_subtitle(text: str, start: float, end: float,
                        max_chars: int = MAX_CHARS_PER_LINE,
                        max_lines: int = MAX_LINES_BEFORE_SPLIT) -> list:
    """
    Chia một segment subtitle dài thành danh sách sub-segments.

    Chiến lược:
    - text ≤ max_chars: 1 segment nguyên vẹn
    - text chia thành ≤ max_lines dòng, mỗi dòng ≤ max_chars:
        → 1 segment với text = "dòng1\\Ndòng2" (ASS inline newline)
    - Cần nhiều hơn max_lines dòng:
        → Nhóm mỗi max_lines dòng thành 1 entry, timestamps tỷ lệ theo số ký tự

    Returns:
        List of dict: [{'text': str, 'start': float, 'end': float}, ...]
    """
    text = text.strip()
    if not text:
        return []

    total_duration = max(0.0, end - start)

    # Chia text thành các dòng đơn ≤ max_chars mỗi dòng
    lines = _split_text_into_lines(text, max_chars)

    if len(lines) == 1:
        # Câu ngắn — không cần chia
        return [{'text': text, 'start': start, 'end': end}]

    # Nhóm các dòng thành entries, mỗi entry tối đa max_lines dòng.
    # Với 3 dòng và max 2/entry: ưu tiên (1 dòng) + (2 dòng) để
    # entry cuối có đủ ngữ nghĩa, tránh để 1 dòng lẻ.
    groups = []
    n = len(lines)
    if n <= max_lines:
        groups.append(r"\N".join(lines))
    elif n == max_lines + 1 and max_lines == 2:
        # 3 dòng, max 2/entry → phân bố (1) + (2) cho cân bằng
        groups.append(lines[0])
        groups.append(r"\N".join(lines[1:]))
    else:
        for i in range(0, n, max_lines):
            groups.append(r"\N".join(lines[i:i + max_lines]))

    if len(groups) == 1:
        # Tất cả vừa trong 1 entry — timestamps gốc
        return [{'text': groups[0], 'start': start, 'end': end}]

    # Nhiều groups → tính timestamps tỷ lệ theo số ký tự thuần
    plain_groups = [g.replace(r"\N", "") for g in groups]
    total_chars = sum(len(g) for g in plain_groups)

    segments = []
    current_time = start

    for i, (group_text, plain_group) in enumerate(zip(groups, plain_groups)):
        char_ratio = len(plain_group) / total_chars if total_chars > 0 else 1.0 / len(groups)
        duration = total_duration * char_ratio

        if i == len(groups) - 1:
            seg_end = end  # Segment cuối khớp chính xác với end gốc
        else:
            seg_end = round(current_time + duration, 3)
            min_end = current_time + MIN_SEGMENT_DURATION
            max_end = end - MIN_SEGMENT_DURATION * (len(groups) - 1 - i)
            seg_end = min(max(seg_end, min_end), max_end)

        segments.append({
            'text': group_text,
            'start': round(current_time, 3),
            'end': round(seg_end, 3)
        })
        current_time = seg_end

    return segments


def generate_subtitles(
    segments: list,
    output_dir: str,
    job_id: str,
    video_width: int = 1920,
    video_height: int = 1080
) -> dict:
    """
    Generates .srt and .ass subtitle files from the segments.

    Câu dài hơn MAX_CHARS_PER_LINE ký tự được tự động chia nhỏ thành
    nhiều entries với timestamps tỷ lệ theo độ dài, đảm bảo khớp với
    nhịp đọc của giọng nói.

    Args:
        video_width: Chiều rộng video (px) — dùng cho ASS PlayResX.
        video_height: Chiều cao video (px) — dùng cho ASS PlayResY,
                      tự động scale fontsize và MarginV tỷ lệ.
    Returns { 'srt_path': str, 'ass_path': str }
    """
    print(f"[Module 6] Generating subtitles for {len(segments)} segments...")
    print(f"           Video resolution: {video_width}x{video_height}")
    os.makedirs(output_dir, exist_ok=True)

    srt_path = os.path.join(output_dir, f"sub_{job_id}.srt")
    ass_path = os.path.join(output_dir, f"sub_{job_id}.ass")

    # Bước 1: Expand tất cả segments — chia câu dài thành sub-segments
    expanded = []
    split_count = 0
    for seg in segments:
        text = seg.get("translated_text", "").strip()
        
        # Lọc bỏ audio tags (ví dụ: [happy], [laughs])
        text = re.sub(r'\[.*?\]\s*', '', text).strip()
        
        start = seg["start"]
        end = seg["end"]

        sub_segs = split_long_subtitle(text, start, end)
        expanded.extend(sub_segs)

        if len(sub_segs) > 1:
            split_count += 1

    if split_count > 0:
        print(f"           ✂️  Chia nhỏ {split_count} câu dài → tổng {len(expanded)} "
              f"sub-segments (từ {len(segments)} segments gốc)")

    # Bước 2: Ghi SRT
    with open(srt_path, "w", encoding="utf-8") as f_srt:
        for idx, sub in enumerate(expanded, start=1):
            start_str = format_time_srt(sub["start"])
            end_str = format_time_srt(sub["end"])
            # Chuyển ASS inline newline \N → newline thật trong SRT
            srt_text = sub["text"].replace(r"\N", "\n")
            f_srt.write(f"{idx}\n")
            f_srt.write(f"{start_str} --> {end_str}\n")
            f_srt.write(f"{srt_text}\n\n")

    # Bước 3: Ghi ASS
    # PlayResX/Y khớp với kích thước video thực tế để ASS renderer scale đúng
    # Fontsize và MarginV tự scale theo chiều cao video (không hardcode 1080)
    #   - YouTube 1080p  (height=1080): fontsize≈59,  MarginV≈49
    #   - YouTube 720p   (height=720):  fontsize≈39,  MarginV≈33
    #   - TikTok portrait(height=1920): fontsize≈80,  MarginV≈88  (capped)
    base_fontsize = max(36, min(80, int(video_height * 0.055)))
    base_margin_v = max(20, int(video_height * 0.046))

    # WrapStyle: 1 = smart wrap (tự xuống dòng nếu text dài)
    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{base_fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,2,2,10,10,{base_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w", encoding="utf-8") as f_ass:
        f_ass.write(ass_header)
        for sub in expanded:
            start_str = format_time_ass(sub["start"])
            end_str = format_time_ass(sub["end"])
            text = sub["text"]  # ASS hiểu \N là inline newline
            f_ass.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n")

    print(f"[Module 6] ✅ Subtitles saved: {srt_path} and {ass_path}")
    return {
        "srt_path": srt_path,
        "ass_path": ass_path
    }


if __name__ == "__main__":
    # Test nhanh với dữ liệu mẫu
    test_segments = [
        {
            "start": 0.0,
            "end": 5.96,
            "translated_text": "Hôm nay, cuộc đời mà bạn sẽ trải nghiệm đó là một kiếp người an phận ở mãi một thị trấn nhỏ, không bon chen, không cần phải phấn đấu gì cả."
        },
        {
            "start": 5.96,
            "end": 7.72,
            "translated_text": "Năm bạn mười tám tuổi, mùa hè ấy."
        },
        {
            "start": 7.72,
            "end": 11.28,
            "translated_text": "Ve sầu bám trên thân cây, kêu rát cả cổ họng."
        },
        {
            "start": 11.28,
            "end": 15.88,
            "translated_text": "Gió thổi vào nhà, mang theo mùi nhựa đường bị nung chảy dưới nắng hè."
        },
    ]
    result = generate_subtitles(test_segments, "/tmp/test_sub", "test001")

    # In nội dung SRT để verify
    with open(result['srt_path'], 'r', encoding='utf-8') as f:
        print("\n--- SRT OUTPUT ---")
        print(f.read())
    
    # Kiểm tra từng dòng không vượt max_chars
    print("--- LINE LENGTH CHECK ---")
    for seg in test_segments:
        text = seg["translated_text"]
        lines = _split_text_into_lines(text, MAX_CHARS_PER_LINE)
        status = "✅" if all(len(l) <= MAX_CHARS_PER_LINE for l in lines) else "⚠️ "
        print(f"{status} '{text[:40]}...' ({len(text)} chars) → {len(lines)} lines")
        for l in lines:
            ok = "✅" if len(l) <= MAX_CHARS_PER_LINE else "❌ OVERFLOW"
            print(f"     {ok} [{len(l):2d}] {l}")
