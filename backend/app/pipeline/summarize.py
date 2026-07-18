import os
import json
import time
from openai import OpenAI
from google import genai
from backend.app.config import (
    LM_STUDIO_BASE_URL,
    TRANSLATION_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


def _build_summarize_prompt(transcript_text: str) -> str:
    return f"""Bạn là một chuyên gia sáng tạo nội dung video ngắn và biên tập viên social media (TikTok, YouTube, Facebook Reels).
Dưới đây là toàn bộ nội dung kịch bản (bản dịch tiếng Việt) của một video:

--- BẮT ĐẦU NỘI DUNG KỊCH BẢN ---
{transcript_text}
--- KẾT THÚC NỘI DUNG KỊCH BẢN ---

Hãy phân tích nội dung trên và thực hiện 3 yêu cầu sau:

1. TÓM TẮT NỘI DUNG (summary): Tóm tắt 3-5 ý chính nổi bật nhất của video ngắn gọn, súc tích.
2. ĐỀ XUẤT TIÊU ĐỀ (recommended_titles): Đề xuất danh sách các tiêu đề thu hút người xem, chia thành 3 nhóm:
   - "question_comparison": Tiêu đề dạng CÂU HỎI SO SÁNH hoặc đặt vấn đề (Ví dụ: "Liệu X có tốt hơn Y?", "So sánh A và B: Cái nào đáng tiền hơn?", "Tại sao người ta lại chọn X thay vì Y?").
   - "clickbait_viral": Tiêu đề dạng GIẬT TÍT, gây tò mò, kích thích tương tác (Ví dụ: "Sự thật giật mình về X!", "Đừng mua Y nếu chưa biết điều này!", "Bí mật chưa từng tiết lộ...").
   - "seo_standard": Tiêu đề chuẩn SEO, ngắn gọn, mô tả đúng nội dung chính.
3. HASHTAGS (suggested_hashtags): Gợi ý 5-10 hashtags phổ biến liên quan đến nội dung video.

Yêu cầu định dạng đầu ra:
Trả về duy nhất 1 JSON object hợp lệ với cấu trúc chuẩn sau (KHÔNG thêm bất kỳ lời giải thích hay định dạng markdown fences nào outside JSON):

{{
  "summary": [
    "Ý chính 1...",
    "Ý chính 2...",
    "Ý chính 3..."
  ],
  "recommended_titles": {{
    "question_comparison": [
      "Tiêu đề câu hỏi / so sánh 1",
      "Tiêu đề câu hỏi / so sánh 2"
    ],
    "clickbait_viral": [
      "Tiêu đề giật tít 1",
      "Tiêu đề giật tít 2"
    ],
    "seo_standard": [
      "Tiêu đề SEO 1",
      "Tiêu đề SEO 2"
    ]
  }},
  "suggested_hashtags": [
    "#Hashtag1",
    "#Hashtag2",
    "#Hashtag3"
  ]
}}
"""


def _call_gemini_summarize(prompt: str, max_retries: int = 5) -> str:
    """Call Gemini API for summary generation."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY chưa được cấu hình trong file .env. "
            "Vui lòng lấy API Key từ https://aistudio.google.com/app/apikey"
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    from google.genai import types

    full_prompt = (
        "You are a professional video editor assistant. Respond ONLY with valid JSON.\n\n"
        + prompt
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1 and ("503" in str(e) or "429" in str(e)):
                wait = 5 * (attempt + 1)
                print(
                    f"[Summarize] ⟳ Gemini API busy. Retrying in {wait}s (Attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
            else:
                raise


def _call_lm_studio_summarize(prompt: str) -> str:
    """Call LM Studio local model for summary generation."""
    client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed")

    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that only outputs valid JSON objects. Do NOT include markdown fences, preambles, or extra text.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    reply = response.choices[0].message.content.strip()

    # Clean markdown code blocks if present
    if reply.startswith("```json"):
        reply = reply[7:]
    if reply.startswith("```"):
        reply = reply[3:]
    if reply.endswith("```"):
        reply = reply[:-3]
    reply = reply.strip()

    # Clean up non-JSON prefixes
    if not reply.startswith("{"):
        start = reply.find("{")
        end = reply.rfind("}")
        if start >= 0 and end > start:
            reply = reply[start : end + 1]

    return reply


def _parse_summarize_response(raw_reply: str) -> dict:
    """Parse raw LLM JSON output into a structured dictionary."""
    try:
        data = json.loads(raw_reply)
        summary = data.get("summary", [])
        titles = data.get("recommended_titles", {})
        hashtags = data.get("suggested_hashtags", [])

        if not isinstance(summary, list):
            summary = [str(summary)]
        if not isinstance(titles, dict):
            titles = {"clickbait_viral": [str(titles)]}
        if not isinstance(hashtags, list):
            hashtags = [str(hashtags)]

        return {
            "summary": summary,
            "recommended_titles": {
                "question_comparison": titles.get("question_comparison", []),
                "clickbait_viral": titles.get("clickbait_viral", []),
                "seo_standard": titles.get("seo_standard", []),
            },
            "suggested_hashtags": hashtags,
        }
    except Exception as e:
        print(f"[Summarize] Warning: Failed to parse LLM JSON response: {e}")
        return {
            "summary": [raw_reply[:300]],
            "recommended_titles": {
                "question_comparison": [],
                "clickbait_viral": [],
                "seo_standard": [],
            },
            "suggested_hashtags": [],
        }


def format_summary_text(summary_data: dict) -> str:
    """Format summary output into a human-readable text document."""
    lines = []
    lines.append("=" * 60)
    lines.append("        TỔNG HỢP NỘI DUNG & ĐỀ XUẤT TIÊU ĐỀ VIDEO")
    lines.append("=" * 60)
    lines.append("")

    # Section 1: Summary
    lines.append("📌 1. TÓM TẮT NỘI DUNG CHÍNH:")
    summary_list = summary_data.get("summary", [])
    if summary_list:
        for item in summary_list:
            lines.append(f"  • {item}")
    else:
        lines.append("  (Không có dữ liệu tóm tắt)")
    lines.append("")

    # Section 2: Titles
    lines.append("🎬 2. ĐỀ XUẤT TIÊU ĐỀ THU HÚT:")
    titles = summary_data.get("recommended_titles", {})

    q_titles = titles.get("question_comparison", [])
    if q_titles:
        lines.append("  [❓ Dạng Câu hỏi & So sánh]:")
        for i, t in enumerate(q_titles, 1):
            lines.append(f"    {i}. {t}")
        lines.append("")

    c_titles = titles.get("clickbait_viral", [])
    if c_titles:
        lines.append("  [🔥 Dạng Giật tít / Viral]:")
        for i, t in enumerate(c_titles, 1):
            lines.append(f"    {i}. {t}")
        lines.append("")

    s_titles = titles.get("seo_standard", [])
    if s_titles:
        lines.append("  [🎯 Dạng SEO / Chuẩn chuẩn]:")
        for i, t in enumerate(s_titles, 1):
            lines.append(f"    {i}. {t}")
        lines.append("")

    # Section 3: Hashtags
    hashtags = summary_data.get("suggested_hashtags", [])
    lines.append("🏷️ 3. HASHTAGS GỢI Ý:")
    if hashtags:
        lines.append("  " + " ".join(hashtags))
    else:
        lines.append("  (Không có hashtag)")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def summarize_video_content(
    segments: list, job_dir: str = None, provider: str = None
) -> dict:
    """
    Summarize video script and suggest catchy titles / hashtags.

    Args:
        segments: List of dicts containing 'text' and 'translated_text'
        job_dir: Optional output directory to write summary.txt and summary.json
        provider: Provider to use ('gemini' or 'lm_studio'). Defaults to config TRANSLATION_PROVIDER.

    Returns:
        dict containing 'summary', 'recommended_titles', 'suggested_hashtags'
    """
    print("[Summarize] Bắt đầu tổng hợp nội dung video và đề xuất tiêu đề...")

    if not segments:
        print("[Summarize] Warning: Segments list is empty.")
        return {
            "summary": [],
            "recommended_titles": {
                "question_comparison": [],
                "clickbait_viral": [],
                "seo_standard": [],
            },
            "suggested_hashtags": [],
        }

    # Extract full Vietnamese translated transcript
    transcript_lines = []
    for s in segments:
        txt = s.get("translated_text", s.get("text", "")).strip()
        if txt:
            start_ts = s.get("start", 0.0)
            mins = int(start_ts // 60)
            secs = int(start_ts % 60)
            transcript_lines.append(f"[{mins:02d}:{secs:02d}] {txt}")

    transcript_text = "\n".join(transcript_lines)

    # Build prompt
    prompt = _build_summarize_prompt(transcript_text)

    # Determine provider
    active_provider = provider or TRANSLATION_PROVIDER

    print(f"[Summarize] Đang gọi LLM ({active_provider})...")
    if active_provider.lower() == "gemini":
        raw_reply = _call_gemini_summarize(prompt)
    else:
        raw_reply = _call_lm_studio_summarize(prompt)

    summary_data = _parse_summarize_response(raw_reply)

    # Write output files if job_dir is provided
    if job_dir:
        os.makedirs(job_dir, exist_ok=True)

        # Save summary.json
        json_path = os.path.join(job_dir, "summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"[Summarize] Saved JSON summary to: {json_path}")

        # Save summary.txt
        txt_content = format_summary_text(summary_data)
        txt_path = os.path.join(job_dir, "summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        print(f"[Summarize] Saved formatted text summary to: {txt_path}")

    return summary_data
