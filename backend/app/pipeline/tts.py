import os
import re
import asyncio
import time
import wave
import hashlib
import edge_tts
from pydub import AudioSegment
from backend.app.config import (
    TTS_ENGINE, TTS_VOICE, TTS_CONCURRENCY, TTS_SPEED,
    GEMINI_API_KEY, GEMINI_TTS_MODEL,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL, ELEVENLABS_CONCURRENCY
)
from backend.app.pipeline.audio_utils import (
    get_wav_duration_fast as _get_wav_duration_fast,
    is_valid_wav as _is_valid_wav,
    trim_silence as _trim_silence,
    create_silence_wav as _create_silence_wav,
)

# --- Constants ---
MAX_RETRIES = 4
RETRY_DELAY = 2.0


def _sanitize_text(text: str) -> str:
    """Clean up text before sending to edge-tts."""
    if not text:
        return ""
    # Remove characters that edge-tts can't pronounce, but keep % for percentage
    cleaned = re.sub(r'[^\w\s.,!?;:\-–—\'\"()…%\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # If only punctuation remain → not speakable (numbers are speakable)
    if cleaned and not re.search(r'[a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF\u0400-\u04FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]', cleaned):
        return ""
    return cleaned


def _sanitize_text_elevenlabs(text: str) -> str:
    """Clean up text before sending to ElevenLabs, preserving audio tags like [laughs], [sighs]."""
    if not text:
        return ""
    # Pattern to match ElevenLabs audio tags: [word] or [multi word]
    tag_pattern = re.compile(r'\[([a-zA-Z][a-zA-Z\s]*?)\]')

    result_parts = []
    pos = 0
    for match in tag_pattern.finditer(text):
        # Sanitize text before this tag
        before = text[pos:match.start()]
        if before:
            cleaned_before = re.sub(r'[^\w\s.,!?;:\-–—\'\"()…%\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F]', ' ', before)
            cleaned_before = re.sub(r'\s+', ' ', cleaned_before).strip()
            if cleaned_before:
                result_parts.append(cleaned_before)
        # Preserve the audio tag as-is
        result_parts.append(match.group(0))
        pos = match.end()

    # Remaining text after last tag
    remaining = text[pos:]
    if remaining:
        cleaned_remaining = re.sub(r'[^\w\s.,!?;:\-–—\'\"()…%\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F]', ' ', remaining)
        cleaned_remaining = re.sub(r'\s+', ' ', cleaned_remaining).strip()
        if cleaned_remaining:
            result_parts.append(cleaned_remaining)

    result = ' '.join(result_parts)
    result = re.sub(r'\s+', ' ', result).strip()
    # Speakability check (tags alone are still okay for ElevenLabs)
    if result and not tag_pattern.search(result) and not re.search(
        r'[a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF\u0400-\u04FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]',
        result
    ):
        return ""
    return result


def _has_vietnamese_chars(text: str) -> bool:
    """Check if text contains Vietnamese-specific characters."""
    # Vietnamese has unique diacritics not shared with other Latin languages
    vn_pattern = re.compile(r'[\u00C0-\u00C3\u00C8-\u00CA\u00CC-\u00CD\u00D2-\u00D5\u00D9-\u00DA\u00DD'
                           r'\u00E0-\u00E3\u00E8-\u00EA\u00EC-\u00ED\u00F2-\u00F5\u00F9-\u00FA\u00FD'
                           r'\u0102-\u0103\u0110-\u0111\u0128-\u0129\u0168-\u0169\u01A0-\u01B0'
                           r'\u1EA0-\u1EF9]')
    return bool(vn_pattern.search(text))




async def _generate_single_tts(text: str, voice: str, output_path: str, segment_id: int,
                                semaphore: asyncio.Semaphore) -> dict:
    """
    Generate TTS for a single segment with retry logic.
    Returns dict with {success: bool, segment_id: int, duration: float} to maintain correct mapping.
    """
    async with semaphore:
        # Add small pacing delay to avoid simultaneous WebSocket connection bursts
        await asyncio.sleep(0.2)
        
        for attempt in range(MAX_RETRIES):
            try:
                communicate = edge_tts.Communicate(text, voice, rate=TTS_SPEED)
                # Use edge_tts save to a temporary mp3, then convert to wav
                tmp_mp3 = output_path.replace(".wav", ".tmp.mp3")

                await communicate.save(tmp_mp3)

                # Check if file was created and has content
                if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 100:
                    # Convert mp3 to 24kHz mono wav (match native edge-tts quality) and trim silence
                    audio = AudioSegment.from_file(tmp_mp3, format="mp3")
                    audio = audio.set_frame_rate(24000).set_channels(1)
                    trimmed_audio = _trim_silence(audio)
                    trimmed_audio.export(output_path, format="wav")

                    # Clean up temp file
                    os.remove(tmp_mp3)

                    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                        # Get duration using fast wave reader
                        duration = _get_wav_duration_fast(output_path)
                        return {"success": True, "segment_id": segment_id, "duration": duration}
                    else:
                        raise ValueError("WAV file created but is invalid or empty")
                else:
                    raise ValueError("Temporary MP3 file was empty or not created")

            except Exception as e:
                # Clean up any temp files
                tmp_mp3 = output_path.replace(".wav", ".tmp.mp3")
                if os.path.exists(tmp_mp3):
                    try:
                        os.remove(tmp_mp3)
                    except OSError:
                        pass

                backoff = RETRY_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    print(f"[Module 4]   ⟳ Segment {segment_id} attempt {attempt + 1}/{MAX_RETRIES} failed, retrying in {backoff:.1f}s... (Error: {e})")
                    await asyncio.sleep(backoff)
                else:
                    print(f"[Module 4]   ✗ Segment {segment_id} failed after {MAX_RETRIES} retries: {e}")

        return {"success": False, "segment_id": segment_id, "duration": 0.5}


def _generate_gemini_tts_bytes(text: str, voice: str) -> bytes:
    """Synchronous call to Gemini API to generate TTS audio bytes in PCM format."""
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured in .env")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Map edge-tts or invalid voices to valid Gemini voices
    valid_gemini_voices = {"Aoede", "Charon", "Fenrir", "Kore", "Puck"}
    cleaned_voice = voice.strip()
    if cleaned_voice not in valid_gemini_voices:
        if "HoaiMy" in voice or "female" in voice.lower():
            cleaned_voice = "Aoede"
        elif "NamMinh" in voice or "male" in voice.lower():
            cleaned_voice = "Puck"
        else:
            cleaned_voice = "Aoede"

    # Instruct Gemini to read aloud with highly natural emotion and pacing
    prompt = f"Hãy đọc đoạn văn sau bằng tiếng Việt với giọng kể chuyện cực kỳ truyền cảm, tự nhiên, biểu cảm sắc thái cảm xúc phù hợp với nội dung câu: {text}"

    response = client.models.generate_content(
        model=GEMINI_TTS_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=cleaned_voice
                    )
                )
            )
        )
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data

    raise ValueError("No audio content returned from Gemini API response")


async def _generate_single_gemini_tts(text: str, voice: str, output_path: str, segment_id: int,
                                      semaphore: asyncio.Semaphore) -> dict:
    """
    Generate TTS for a single segment using Gemini API with retry logic.
    Returns dict with {success: bool, segment_id: int, duration: float}.
    """
    async with semaphore:
        # Add small pacing delay to avoid hitting rate limits
        await asyncio.sleep(0.5)

        for attempt in range(MAX_RETRIES):
            try:
                # Call the synchronous SDK method in a thread pool
                pcm_bytes = await asyncio.to_thread(_generate_gemini_tts_bytes, text, voice)

                # Convert the raw PCM bytes (16-bit, 24000Hz, mono) to standard WAV format
                tmp_wav = output_path.replace(".wav", ".tmp.wav")

                with wave.open(tmp_wav, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit (2 bytes per sample)
                    wf.setframerate(24000)
                    wf.writeframes(pcm_bytes)

                if os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 100:
                    # Convert to AudioSegment to trim silence
                    audio = AudioSegment.from_file(tmp_wav, format="wav")
                    trimmed_audio = _trim_silence(audio)
                    trimmed_audio.export(output_path, format="wav")

                    # Clean up temp file
                    try:
                        os.remove(tmp_wav)
                    except OSError:
                        pass

                    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                        duration = _get_wav_duration_fast(output_path)
                        return {"success": True, "segment_id": segment_id, "duration": duration}
                    else:
                        raise ValueError("WAV file created but is invalid or empty after trimming")
                else:
                    raise ValueError("Temporary WAV file was empty or not created")

            except Exception as e:
                # Clean up any temp files
                tmp_wav = output_path.replace(".wav", ".tmp.wav")
                if os.path.exists(tmp_wav):
                    try:
                        os.remove(tmp_wav)
                    except OSError:
                        pass

                backoff = RETRY_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    print(f"[Module 4]   ⟳ Segment {segment_id} Gemini attempt {attempt + 1}/{MAX_RETRIES} failed, retrying in {backoff:.1f}s... (Error: {e})")
                    await asyncio.sleep(backoff)
                else:
                    print(f"[Module 4]   ✗ Segment {segment_id} Gemini failed after {MAX_RETRIES} retries: {e}")

        return {"success": False, "segment_id": segment_id, "duration": 0.5}


def _generate_elevenlabs_tts_bytes(text: str, voice_id: str, model_id: str) -> bytes:
    """Synchronous HTTP call to ElevenLabs API to generate TTS audio bytes (MP3)."""
    import requests

    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY is not configured in .env")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.45,          # Lower = more expressive/emotional
            "similarity_boost": 0.75,
            "style": 0.5,               # Style exaggeration (0-1)
            "use_speaker_boost": True
        }
    }
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    if response.status_code == 401:
        raise ValueError("ElevenLabs API key không hợp lệ hoặc hết hạn.")
    if response.status_code == 422:
        raise ValueError(f"ElevenLabs từ chối yêu cầu (422): {response.text[:200]}")
    response.raise_for_status()
    return response.content


async def _generate_single_elevenlabs_tts(text: str, output_path: str, segment_id: int,
                                           semaphore: asyncio.Semaphore, voice_id: str) -> dict:
    """
    Generate TTS for a single segment using ElevenLabs API with retry logic.
    Supports emotional audio tags: [laughs], [sighs], [whispers], [excited], etc.
    Returns dict with {success: bool, segment_id: int, duration: float}.
    """
    async with semaphore:
        await asyncio.sleep(0.3)  # Gentle pacing to avoid rate limits

        for attempt in range(MAX_RETRIES):
            try:
                mp3_bytes = await asyncio.to_thread(
                    _generate_elevenlabs_tts_bytes, text, voice_id, ELEVENLABS_MODEL
                )

                tmp_mp3 = output_path.replace(".wav", ".tmp.mp3")
                with open(tmp_mp3, "wb") as f:
                    f.write(mp3_bytes)

                if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 100:
                    audio = AudioSegment.from_file(tmp_mp3, format="mp3")
                    audio = audio.set_frame_rate(24000).set_channels(1)
                    trimmed_audio = _trim_silence(audio)
                    trimmed_audio.export(output_path, format="wav")

                    try:
                        os.remove(tmp_mp3)
                    except OSError:
                        pass

                    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                        duration = _get_wav_duration_fast(output_path)
                        return {"success": True, "segment_id": segment_id, "duration": duration}
                    else:
                        raise ValueError("WAV file created but is invalid or empty")
                else:
                    raise ValueError("ElevenLabs trả về response rỗng")

            except Exception as e:
                tmp_mp3 = output_path.replace(".wav", ".tmp.mp3")
                if os.path.exists(tmp_mp3):
                    try:
                        os.remove(tmp_mp3)
                    except OSError:
                        pass

                backoff = RETRY_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    print(f"[Module 4]   ⟳ Segment {segment_id} ElevenLabs attempt {attempt + 1}/{MAX_RETRIES} failed, retrying in {backoff:.1f}s... (Error: {e})")
                    await asyncio.sleep(backoff)
                else:
                    print(f"[Module 4]   ✗ Segment {segment_id} ElevenLabs failed after {MAX_RETRIES} retries: {e}")

        return {"success": False, "segment_id": segment_id, "duration": 0.5}


async def _generate_tts_all_segments(segments: list, output_dir: str, engine: str, voice: str):

    """
    Generate TTS for all segments with concurrency control.
    Uses asyncio.gather to preserve correct ordering (fixes as_completed bug).
    """
    # Use separate concurrency limit for ElevenLabs (free tier is sensitive to bursts)
    if engine == "elevenlabs":
        semaphore = asyncio.Semaphore(ELEVENLABS_CONCURRENCY)
    else:
        semaphore = asyncio.Semaphore(TTS_CONCURRENCY)

    success_count = 0
    skip_count = 0
    fail_count = 0
    cache_count = 0
    lang_warn_count = 0
    total = len(segments)

    # Prepare tasks — build segment_map for result mapping by segment_id
    tasks = []
    segment_map = {}  # segment_id -> segment dict

    for idx, segment in enumerate(segments):
        # CRITICAL USER REQUIREMENT: OCR hardsub segments (on-screen text notes) are translated for SUBTITLE display ONLY.
        # They MUST NOT be spoken by TTS voiceover!
        seg_source = str(segment.get("source", ""))
        if seg_source in ("ocr_recovered", "ocr_only") or seg_source.startswith("ocr"):
            print(f"[Module 4] ℹ️ Skipping TTS voice generation for OCR Hardsub Segment #{segment['id']} (Subtitle display only).")
            segment["tts_audio_path"] = None
            segment["tts_duration"] = 0.0
            skip_count += 1
            continue

        raw_text = segment.get("translated_text", "").strip()
        # ElevenLabs uses its own sanitizer to preserve [audio tags]
        if engine == "elevenlabs":
            cleaned = _sanitize_text_elevenlabs(raw_text)
        else:
            cleaned = _sanitize_text(raw_text)
        wav_path = os.path.join(output_dir, f"tts_{segment['id']}.wav")

        if not cleaned:
            # Empty/unspeakable → silence immediately
            _create_silence_wav(wav_path, duration_sec=0.3)
            segment["tts_audio_path"] = wav_path
            segment["tts_duration"] = 0.3
            skip_count += 1
            continue

        hash_input = f"{raw_text}|{engine}|{voice}"
        text_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
        hash_path = wav_path + ".hash"

        # Cache: skip generation if valid WAV already exists and hash matches
        if _is_valid_wav(wav_path) and os.path.exists(hash_path):
            try:
                with open(hash_path, "r", encoding="utf-8") as f:
                    saved_hash = f.read().strip()
                if saved_hash == text_hash:
                    duration = _get_wav_duration_fast(wav_path)
                    segment["tts_audio_path"] = wav_path
                    segment["tts_duration"] = duration
                    cache_count += 1
                    continue
            except Exception:
                pass

        # Language check warning
        if not _has_vietnamese_chars(cleaned) and len(cleaned) > 5:
            lang_warn_count += 1

        segment_map[segment["id"]] = segment
        if engine == "gemini":
            tasks.append(_generate_single_gemini_tts(cleaned, voice, wav_path, segment["id"], semaphore))
        elif engine == "elevenlabs":
            tasks.append(_generate_single_elevenlabs_tts(cleaned, wav_path, segment["id"], semaphore, voice))
        else:
            tasks.append(_generate_single_tts(cleaned, voice, wav_path, segment["id"], semaphore))

    if lang_warn_count > 0:
        print(f"[Module 4] ⚠ {lang_warn_count} segments may not be Vietnamese text — TTS quality may be affected")

    if cache_count > 0:
        print(f"[Module 4] ♻ {cache_count} segments loaded from cache")

    if not tasks:
        print(f"[Module 4] All segments resolved (cache/silence), no TTS calls needed")
        return

    # Run all TTS tasks with asyncio.gather — preserves order!
    _active_concurrency = ELEVENLABS_CONCURRENCY if engine == "elevenlabs" else TTS_CONCURRENCY
    print(f"[Module 4] Processing {len(tasks)} segments (concurrency={_active_concurrency})...")
    t0 = time.time()

    # Use gather instead of as_completed to maintain correct ordering
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - t0

    # Map results back to segments using segment_id (safe regardless of order)
    for result in results:
        if isinstance(result, Exception):
            fail_count += 1
            continue

        seg_id = result["segment_id"]
        segment = segment_map.get(seg_id)
        if not segment:
            continue

        wav_path = os.path.join(output_dir, f"tts_{seg_id}.wav")
        hash_path = wav_path + ".hash"
        
        raw_text = segment.get("translated_text", "").strip()
        hash_input = f"{raw_text}|{engine}|{voice}"
        text_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()

        if result["success"]:
            segment["tts_audio_path"] = wav_path
            segment["tts_duration"] = result["duration"]
            try:
                with open(hash_path, "w", encoding="utf-8") as f:
                    f.write(text_hash)
            except Exception:
                pass
            success_count += 1
        else:
            _create_silence_wav(wav_path, duration_sec=0.5)
            segment["tts_audio_path"] = wav_path
            segment["tts_duration"] = 0.5
            if os.path.exists(hash_path):
                try: os.remove(hash_path)
                except Exception: pass
            fail_count += 1

    print(f"[Module 4] Results: {success_count} success, {cache_count} cached, "
          f"{skip_count} skipped, {fail_count} failed | {elapsed:.1f}s")


def generate_tts_for_segments(segments: list, output_dir: str, engine: str = None, voice: str = None) -> list:
    """
    Generates TTS audio for each segment's translated_text.
    Supports multiple engines: 'edge-tts' (default), 'gemini', 'elevenlabs'.
    Uses per-segment async concurrency with caching to avoid re-generation.

    Output: updates each segment in-place with 'tts_audio_path' and 'tts_duration'.
    """
    if not segments:
        return []

    if engine is None:
        engine = TTS_ENGINE
        
    if not voice:
        if engine == "elevenlabs":
            voice = ELEVENLABS_VOICE_ID
        elif engine == "gemini":
            voice = "Aoede"
        else:
            voice = TTS_VOICE

    print(f"[Module 4] Generating TTS for {len(segments)} segments using engine '{engine}' and voice '{voice}'...")
    print(f"[Module 4] Strategy: Per-segment with {TTS_CONCURRENCY if engine != 'elevenlabs' else ELEVENLABS_CONCURRENCY} concurrent requests")
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Check if there is an existing event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import threading
        def _run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(_generate_tts_all_segments(segments, output_dir, engine, voice))
            new_loop.close()

        t = threading.Thread(target=_run_in_thread)
        t.start()
        t.join()
    else:
        asyncio.run(_generate_tts_all_segments(segments, output_dir, engine, voice))

    elapsed = time.time() - t0
    print(f"[Module 4] TTS generation complete in {elapsed:.1f}s")
    return segments


if __name__ == "__main__":
    # Test block
    sample_segments = [
        {"id": 1, "translated_text": "Xin chào mọi người."},
        {"id": 2, "translated_text": "Chào mừng đến với kênh của tôi."},
        {"id": 3, "translated_text": ""},       # empty text test
        {"id": 4, "translated_text": "123..."},  # numbers-only test
        {"id": 5, "translated_text": "Hôm nay chúng ta sẽ tìm hiểu về trí tuệ nhân tạo."},
    ]
    res = generate_tts_for_segments(sample_segments, "output/test_tts")
    for s in res:
        print(f"  Segment {s['id']}: path={s.get('tts_audio_path')}, duration={s.get('tts_duration', 0):.2f}s")
