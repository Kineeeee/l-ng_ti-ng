import os
from dotenv import load_dotenv

load_dotenv()

LM_STUDIO_BASE_URL_DEFAULT = "http://localhost:1234/v1"
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", LM_STUDIO_BASE_URL_DEFAULT)
TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER", "lm_studio") # 'lm_studio' or 'gemini'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")



WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "vi-VN-HoaiMyNeural")
TTS_CONCURRENCY = int(os.getenv("TTS_CONCURRENCY", "2"))
TRANSLATION_BATCH_SIZE = int(os.getenv("TRANSLATION_BATCH_SIZE", "12"))
TTS_SPEED = os.getenv("TTS_SPEED", "+0%")
TTS_DYNAMIC_SPEEDUP = os.getenv("TTS_DYNAMIC_SPEEDUP", "True").lower() == "true"

MASK_OLD_SUBS = os.getenv("MASK_OLD_SUBS", "True").lower() == "true"
MASK_SUB_Y_RATIO = float(os.getenv("MASK_SUB_Y_RATIO", "0.15"))
MASK_SUB_COLOR = os.getenv("MASK_SUB_COLOR", "black")

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
DUCK_VOLUME_DB = float(os.getenv("DUCK_VOLUME_DB", "-18.0"))
ORIGINAL_VOLUME_DB = float(os.getenv("ORIGINAL_VOLUME_DB", "0.0"))
TTS_VOLUME_DB = float(os.getenv("TTS_VOLUME_DB", "2.0"))
AUDIO_SYNC_OFFSET_MS = float(os.getenv("AUDIO_SYNC_OFFSET_MS", "0.0"))




