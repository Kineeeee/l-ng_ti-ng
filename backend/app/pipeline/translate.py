import json
import time
from openai import OpenAI
from google import genai
from backend.app.config import (
    LM_STUDIO_BASE_URL, TRANSLATION_PROVIDER,
    GEMINI_API_KEY, GEMINI_MODEL, TRANSLATION_BATCH_SIZE
)


def _build_prompt(
    segments_to_translate: list,
    source_language: str,
    history_context: list = None,
    future_context: list = None
) -> str:
    """Build translation prompt for a batch of segments with context support."""
    # Build history context string
    if history_context:
        history_lines = []
        for s in history_context:
            orig = s.get("text", "").strip()
            trans = s.get("translated_text", "").strip()
            history_lines.append(f"- ID {s['id']}: '{orig}' -> '{trans}'")
        history_str = "\n".join(history_lines)
    else:
        history_str = "(Không có ngữ cảnh trước đó)"

    # Build future context string
    if future_context:
        future_lines = []
        for s in future_context:
            orig = s.get("text", "").strip()
            future_lines.append(f"- ID {s['id']}: '{orig}'")
        future_str = "\n".join(future_lines)
    else:
        future_str = "(Không có ngữ cảnh tiếp theo)"

    input_json_str = json.dumps(segments_to_translate, ensure_ascii=False)

    return f"""Bạn là chuyên gia dịch thuật lồng tiếng và thuyết minh video chuyên nghiệp. Hãy dịch các câu được yêu cầu dưới đây từ {source_language} sang tiếng Việt với phong cách tự nhiên, sinh động (phù hợp để lồng tiếng nói), duy trì đại từ nhân xưng nhất quán và giữ đúng ngữ cảnh.

ĐẶC BIỆT LƯU Ý KHI DỊCH LỒNG TIẾNG:
1. BẢO TOÀN SỰ NHẤN MẠNH VÀ LẶP TỪ: Nếu câu gốc lặp từ để nhấn mạnh cường độ hoặc cảm xúc (ví dụ: 'really really really long', 'very very slow', 'go go go'), bạn CẦN dịch lặp lại tương ứng trong tiếng Việt (ví dụ: 'rất rất rất dài', 'rất rất chậm', 'đi đi đi') thay vì dịch tóm gọn ('rất dài', 'chậm', 'đi'). Không được lược bỏ các từ biểu thị sắc thái cảm xúc hoặc mức độ.
2. TƯƠNG ĐỒNG ĐỘ DÀI/NHỊP ĐIỆU: Đảm bảo độ dài và nhịp điệu của bản dịch tiếng Việt tương đương với câu gốc để người nói lồng tiếng có thể đọc khớp thời gian với video gốc. Tránh dịch quá ngắn gọn làm mất nhịp điệu.
3. VĂN PHONG NÓI TỰ NHIÊN: Sử dụng từ ngữ khẩu ngữ tự nhiên, tránh dịch word-by-word hoặc văn viết khô khan.

--- NGỮ CẢNH HỘI THOẠI ĐÃ DỊCH TRƯỚC ĐÓ (Dùng để tham khảo cách xưng hô và mạch truyện, KHÔNG dịch lại):
{history_str}

--- CÁC CÂU CẦN DỊCH BÂY GIỜ (Dịch ĐÚNG các câu này, trả về dưới dạng JSON list có "id" và "translated_text"):
{input_json_str}

--- NGỮ CẢNH TIẾP THEO (Dùng để hiểu hướng tiếp theo của câu chuyện, KHÔNG dịch):
{future_str}

Hãy dịch CÁC CÂU CẦN DỊCH BÂY GIỜ sang tiếng Việt.
Yêu cầu bắt buộc:
1. Trả về ĐÚNG định dạng JSON list, giữ nguyên số lượng segment, KHÔNG gộp/tách câu.
2. Mỗi phần tử trong list JSON kết quả phải có "id" (giữ nguyên) và "translated_text" (bản dịch tiếng Việt).
3. KHÔNG giải thích gì thêm, CHỈ trả về JSON.
"""



def _call_gemini(prompt: str, max_retries: int = 5) -> str:
    """Call Gemini API with retries."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set in .env. "
            "Vui lòng lấy API Key từ https://aistudio.google.com/app/apikey và cập nhật vào file .env"
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    from google.genai import types

    full_prompt = "You are a helpful translation assistant that only outputs valid JSON arrays.\n\n" + prompt

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
                print(f"[Module 3]   ⟳ Gemini API busy. Retrying in {wait}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def _call_lm_studio(prompt: str) -> str:
    """Call LM Studio local model."""
    client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed")

    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "You are a helpful translation assistant that only outputs valid JSON arrays. Do NOT include any explanation, markdown, or extra text. Output ONLY the JSON array."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    reply = response.choices[0].message.content.strip()

    # Clean markdown fences if present
    if reply.startswith("```json"):
        reply = reply[7:]
    if reply.startswith("```"):
        reply = reply[3:]
    if reply.endswith("```"):
        reply = reply[:-3]
    reply = reply.strip()

    # Additional cleanup: sometimes models wrap in <think> tags or add preamble
    # Try to extract the JSON array from the response
    if not reply.startswith("["):
        # Find the first '[' and last ']'
        start = reply.find("[")
        end = reply.rfind("]")
        if start >= 0 and end > start:
            reply = reply[start:end + 1]

    return reply


def _translate_via_provider(prompt: str) -> str:
    """Route to the configured translation provider."""
    if TRANSLATION_PROVIDER.lower() == "gemini":
        return _call_gemini(prompt)
    else:
        return _call_lm_studio(prompt)


def _parse_translation_response(reply_content: str, expected_ids: list) -> dict:
    """
    Parse translation JSON response and return a dict of {id: translated_text}.
    Raises ValueError if parsing fails.
    """
    translated_data = json.loads(reply_content)

    if not isinstance(translated_data, list):
        raise ValueError(f"Expected JSON array, got {type(translated_data).__name__}")

    result = {}
    for item in translated_data:
        if isinstance(item, dict) and "id" in item and "translated_text" in item:
            result[item["id"]] = item["translated_text"]

    return result


def _translate_batch(
    batch_segments: list,
    source_language: str,
    batch_num: int,
    total_batches: int,
    history_context: list = None,
    future_context: list = None
) -> dict:
    """
    Translate a batch of segments. Returns dict of {id: translated_text}.
    Implements retry logic with progressively simpler prompts.
    If the batch returns incomplete translations, it translates the missing segments 1-by-1.
    """
    input_for_llm = [{"id": s["id"], "text": s["text"]} for s in batch_segments]
    expected_ids = [s["id"] for s in batch_segments]

    result = {}
    batch_success = False

    # Attempt batch translation
    for attempt in range(3):
        try:
            prompt = _build_prompt(
                input_for_llm,
                source_language,
                history_context=history_context,
                future_context=future_context
            )
            reply = _translate_via_provider(prompt)
            parsed = _parse_translation_response(reply, expected_ids)

            if len(parsed) > 0:
                result = parsed
                batch_success = True
                if len(result) < len(expected_ids):
                    print(f"[Module 3]   ⚠ Batch {batch_num}/{total_batches}: got {len(result)}/{len(expected_ids)} translations")
                break
        except Exception as e:
            print(f"[Module 3]   ⟳ Batch {batch_num}/{total_batches} attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    # If the batch failed completely or returned incomplete translations,
    # fill in the missing segments 1-by-1
    missing_segs = [s for s in batch_segments if s["id"] not in result or not result[s["id"]]]
    
    if missing_segs:
        if not batch_success:
            print(f"[Module 3]   ↓ Batch {batch_num} failed completely, falling back to per-segment translation...")
        else:
            print(f"[Module 3]   ↓ Batch {batch_num} is missing {len(missing_segs)} translations, resolving 1-by-1...")
        
        for seg in missing_segs:
            # Build combined history context (previous batch + current batch so far)
            completed_in_batch = []
            for s in batch_segments:
                if s["id"] in result and result[s["id"]]:
                    completed_in_batch.append({
                        "id": s["id"],
                        "text": s["text"],
                        "translated_text": result[s["id"]]
                    })
            combined_history = (history_context or []) + completed_in_batch
            combined_history = combined_history[-4:]  # Limit context to last 4 items
            
            single_result = _translate_single_segment(seg, source_language, history_context=combined_history)
            if single_result:
                result[seg["id"]] = single_result

    return result


def _translate_single_segment(segment: dict, source_language: str, history_context: list = None) -> str:
    """Translate a single segment with context support. Returns translated text or None."""
    text = segment["text"].strip()
    if not text:
        return ""

    # Build history context string
    if history_context:
        history_lines = []
        for s in history_context:
            orig = s.get("text", "").strip()
            trans = s.get("translated_text", "").strip()
            history_lines.append(f"- '{orig}' -> '{trans}'")
        history_str = "\n".join(history_lines)
    else:
        history_str = "(Không có ngữ cảnh trước đó)"

    simple_prompt = f"""Bạn là chuyên gia dịch thuật lồng tiếng và thuyết minh video chuyên nghiệp. Dịch câu được yêu cầu dưới đây sang tiếng Việt tự nhiên, đúng ngữ cảnh và giữ nguyên sắc thái biểu cảm.

ĐẶC BIỆT LƯU Ý KHI DỊCH LỒNG TIẾNG:
1. BẢO TOÀN SỰ NHẤN MẠNH VÀ LẶP TỪ: Nếu câu gốc lặp từ để nhấn mạnh cường độ hoặc cảm xúc (ví dụ: 'really really really long', 'very very slow', 'go go go'), bạn CẦN dịch lặp lại tương ứng trong tiếng Việt (ví dụ: 'rất rất rất dài', 'rất rất chậm', 'đi đi đi') thay vì dịch tóm gọn. Không được lược bỏ các từ biểu thị sắc thái cảm xúc hoặc mức độ.
2. TƯƠNG ĐỒNG ĐỘ DÀI/NHỊP ĐIỆU: Đảm bảo độ dài và nhịp điệu của bản dịch tiếng Việt tương đương với câu gốc để người nói lồng tiếng có thể đọc khớp thời gian với video gốc. Tránh dịch quá ngắn gọn làm mất nhịp điệu.
3. VĂN PHONG NÓI TỰ NHIÊN: Sử dụng từ ngữ khẩu ngữ tự nhiên, phù hợp để lồng tiếng nói.

--- NGỮ CẢNH CÁC CÂU ĐÃ NÓI TRƯỚC ĐÓ (Để tham khảo cách xưng hô):
{history_str}

--- CÂU CẦN DỊCH BÂY GIỜ:
Câu gốc ({source_language}): {text}

Yêu cầu:
1. Chỉ trả về duy nhất bản dịch tiếng Việt của câu trên.
2. KHÔNG giải thích, KHÔNG thêm bớt thông tin ngoài bản dịch.
3. Đảm bảo đại từ nhân xưng thống nhất với ngữ cảnh trước đó.
"""

    for attempt in range(2):
        try:
            if TRANSLATION_PROVIDER.lower() == "gemini":
                gemini_prompt = f"""You are a professional video dubbing and voiceover translator. Translate the target text to Vietnamese naturally, keeping pronouns consistent with the context.

CRITICAL REQUIREMENTS FOR DUBBING TRANSLATION:
1. PRESERVE EMPHASIS & REPETITIONS: If the source text repeats words for emphasis (e.g. "really really really long", "very very slow", "go go go"), you MUST repeat them in Vietnamese (e.g. "rất rất rất dài", "rất rất chậm", "đi đi đi") to match the character's speaking length and emotional intensity. Do not summarize or shorten them.
2. NATURAL SPOKEN STYLE: Use a natural, conversational spoken Vietnamese style suitable for voiceover recording, not dry written text.
3. MATCH PACING: Keep the length/rhythm of the translation reasonably aligned with the source text.

Context of previous lines:
{history_str}

Target text to translate ({source_language}): {text}

Output only the translated text as a JSON string or JSON array with one element."""
                reply = _call_gemini(gemini_prompt)
                # Gemini with response_mime_type might wrap in JSON, try to extract
                try:
                    data = json.loads(reply)
                    if isinstance(data, str):
                        return data.strip()
                    elif isinstance(data, list) and data:
                        return str(data[0]).strip()
                except json.JSONDecodeError:
                    return reply.strip()
            else:
                client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed")
                response = client.chat.completions.create(
                    model="local-model",
                    messages=[
                        {"role": "system", "content": "You are a translator. Output ONLY the Vietnamese translation, nothing else. No explanation, no markdown."},
                        {"role": "user", "content": simple_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                reply = response.choices[0].message.content.strip()
                # Clean up: remove quotes, markdown, think tags
                if reply.startswith('"') and reply.endswith('"'):
                    reply = reply[1:-1]
                # Remove <think>...</think> blocks
                import re
                reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
                if reply:
                    return reply

        except Exception as e:
            print(f"[Module 3]     ⟳ Single segment {segment['id']} attempt {attempt + 1} failed: {e}")
            if attempt < 1:
                time.sleep(2)

    print(f"[Module 3]     ✗ Could not translate segment {segment['id']}, keeping original")
    return None


def translate_segments(segments: list, source_language: str) -> list:
    """
    Translates a list of transcription segments into Vietnamese.
    Uses batched approach: splits segments into batches of TRANSLATION_BATCH_SIZE,
    translates each batch, with fallback to per-segment translation for failed batches.
    
    Input segments: [{id, start, end, text, detected_language}, ...]
    Output segments: same but with added `translated_text` field.
    """
    if not segments:
        return []

    # If source is already Vietnamese, just copy text
    if source_language.lower() in ("vi", "vie", "vietnamese"):
        print("[Module 3] Source language is Vietnamese, skipping translation.")
        for seg in segments:
            seg["translated_text"] = seg["text"]
        return segments

    print(f"[Module 3] Translating {len(segments)} segments from {source_language} to Vietnamese...")
    print(f"[Module 3] Provider: {TRANSLATION_PROVIDER}, Batch size: {TRANSLATION_BATCH_SIZE}")

    # Split into batches
    batches = []
    for i in range(0, len(segments), TRANSLATION_BATCH_SIZE):
        batches.append(segments[i:i + TRANSLATION_BATCH_SIZE])

    print(f"[Module 3] Split into {len(batches)} batch(es)")

    # Translate each batch
    all_translations = {}  # id -> translated_text
    for batch_idx, batch in enumerate(batches, 1):
        print(f"[Module 3] Translating batch {batch_idx}/{len(batches)} ({len(batch)} segments)...")
        
        # Calculate indices in the original segments list
        start_idx = (batch_idx - 1) * TRANSLATION_BATCH_SIZE
        end_idx = start_idx + len(batch)
        
        # Build history context (last 4 translated segments)
        history_context = []
        if start_idx > 0:
            hist_segs = segments[max(0, start_idx - 4):start_idx]
            for hs in hist_segs:
                translated_txt = all_translations.get(hs["id"], "")
                history_context.append({
                    "id": hs["id"],
                    "text": hs["text"],
                    "translated_text": translated_txt
                })
                
        # Build future context (next 2 segments)
        future_context = []
        if end_idx < len(segments):
            fut_segs = segments[end_idx:end_idx + 2]
            for fs in fut_segs:
                future_context.append({
                    "id": fs["id"],
                    "text": fs["text"]
                })
                
        batch_result = _translate_batch(
            batch,
            source_language,
            batch_idx,
            len(batches),
            history_context=history_context,
            future_context=future_context
        )
        all_translations.update(batch_result)

    # Merge translations back into segments
    success_count = 0
    fallback_count = 0
    for segment in segments:
        if segment["id"] in all_translations and all_translations[segment["id"]]:
            segment["translated_text"] = all_translations[segment["id"]]
            success_count += 1
        else:
            # Last resort: keep original text (should rarely happen now)
            segment["translated_text"] = segment["text"]
            fallback_count += 1

    print(f"[Module 3] Translation complete: {success_count} translated, {fallback_count} kept original")
    if fallback_count > 0:
        print(f"[Module 3] ⚠ {fallback_count} segments could not be translated and kept source text")

    return segments


if __name__ == "__main__":
    # Test block
    sample_segments = [
        {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello everyone.", "detected_language": "en"},
        {"id": 2, "start": 2.0, "end": 4.0, "text": "Welcome to my channel.", "detected_language": "en"}
    ]
    res = translate_segments(sample_segments, "en")
    import pprint
    pprint.pprint(res)
