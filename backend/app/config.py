import os
from dotenv import load_dotenv

load_dotenv(override=True)

LM_STUDIO_BASE_URL_DEFAULT = "http://localhost:1234/v1"
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", LM_STUDIO_BASE_URL_DEFAULT)
TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER", "lm_studio") # 'lm_studio' or 'gemini'
TRANSLATION_STYLE = os.getenv("TRANSLATION_STYLE", "humorous") # 'standard', 'humorous', or 'storyteller'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")

# ElevenLabs TTS configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Default: Adam
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_v3")  # eleven_v3 supports audio tags
ELEVENLABS_CONCURRENCY = int(os.getenv("ELEVENLABS_CONCURRENCY", "3"))  # Keep low for free tier



WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
# Transcribe backend: 'auto' (default) | 'mlx' (Apple GPU) | 'cpu' | 'groq' (cloud API)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "vi-VN-HoaiMyNeural")
TTS_CONCURRENCY = int(os.getenv("TTS_CONCURRENCY", "2"))
TRANSLATION_BATCH_SIZE = int(os.getenv("TRANSLATION_BATCH_SIZE", "12"))
TTS_SPEED = os.getenv("TTS_SPEED", "+10%")
TTS_DYNAMIC_SPEEDUP = os.getenv("TTS_DYNAMIC_SPEEDUP", "True").lower() == "true"
ENABLE_VIDEO_RETIMING = os.getenv("ENABLE_VIDEO_RETIMING", "True").lower() == "true"

MASK_OLD_SUBS = os.getenv("MASK_OLD_SUBS", "True").lower() == "true"
MASK_SUB_Y_RATIO = float(os.getenv("MASK_SUB_Y_RATIO", "0.15"))
MASK_SUB_COLOR = os.getenv("MASK_SUB_COLOR", "blur")

# Logo and Watermark settings
LOGO_PATH = os.getenv("LOGO_PATH", "")
LOGO_POSITION = os.getenv("LOGO_POSITION", "bottom-right")
MIRROR_VIDEO = os.getenv("MIRROR_VIDEO", "False").lower() == "true"
WATERMARK_TEXT = os.getenv("WATERMARK_TEXT", "")
WATERMARK_POSITION = os.getenv("WATERMARK_POSITION", "center")

# Top text masking settings
MASK_TOP_TEXT = os.getenv("MASK_TOP_TEXT", "False").lower() == "true"
MASK_TOP_Y_RATIO = float(os.getenv("MASK_TOP_Y_RATIO", "0.10"))
MASK_TOP_COLOR = os.getenv("MASK_TOP_COLOR", "black")
TOP_TEXT = os.getenv("TOP_TEXT", "")
TOP_TEXT_FONT_PATH = os.getenv("TOP_TEXT_FONT_PATH", "")
TOP_TEXT_BOLD_FONT_PATH = os.getenv("TOP_TEXT_BOLD_FONT_PATH", "")

# Audio mixing and synchronization settings
MIX_ORIGINAL_AUDIO = os.getenv("MIX_ORIGINAL_AUDIO", "True").lower() == "true"
DUCK_VOLUME_DB = float(os.getenv("DUCK_VOLUME_DB", "-24.0"))
ORIGINAL_VOLUME_DB = float(os.getenv("ORIGINAL_VOLUME_DB", "-3.0"))
TTS_VOLUME_DB = float(os.getenv("TTS_VOLUME_DB", "2.0"))
AUDIO_SYNC_OFFSET_MS = float(os.getenv("AUDIO_SYNC_OFFSET_MS", "0.0"))

# Pitch Processing settings
# method: "none" | "shift" (Phương án 1) | "clone" (Phương án 2)
PITCH_METHOD = os.getenv("PITCH_METHOD", "none")
PITCH_MAX_SHIFT_SEMITONES = float(os.getenv("PITCH_MAX_SHIFT_SEMITONES", "6.0"))
PITCH_F0_BLEND_RATIO = float(os.getenv("PITCH_F0_BLEND_RATIO", "0.7"))

# Subtitle settings
SUBTITLE_MAX_CHARS_PER_LINE = int(os.getenv("SUBTITLE_MAX_CHARS_PER_LINE", "42"))
SUBTITLE_MAX_LINES_BEFORE_SPLIT = int(os.getenv("SUBTITLE_MAX_LINES_BEFORE_SPLIT", "1"))
SUBTITLE_MIN_SEGMENT_DURATION = float(os.getenv("SUBTITLE_MIN_SEGMENT_DURATION", "0.5"))

# OCR Subtitle settings
ENABLE_OCR_SUBTITLE = os.getenv("ENABLE_OCR_SUBTITLE", "True").lower() == "true"
OCR_CROP_BOTTOM_RATIO = float(os.getenv("OCR_CROP_BOTTOM_RATIO", "0.25"))
OCR_SAMPLE_FPS = float(os.getenv("OCR_SAMPLE_FPS", "1.0"))
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.5"))

# Download & Video Speed settings
VIDEO_SPEED_FACTOR = float(os.getenv("VIDEO_SPEED_FACTOR", "1.0"))


