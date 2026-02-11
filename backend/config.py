"""
Configuration file for AWS Transcribe + Sarvam TTS system
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── AWS Configuration ───────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

# ─── Agent Configuration ─────────────────────────────────────────────────────
AGENT_TRIGGER_NAME = os.getenv("AGENT_TRIGGER_NAME", "Cupcake")
MAX_BUFFER_SIZE = int(os.getenv("MAX_BUFFER_SIZE", "100"))
UTTERANCE_SILENCE_TIMEOUT = float(os.getenv("UTTERANCE_SILENCE_TIMEOUT", "2.0"))

# ─── Sarvam TTS Configuration ────────────────────────────────────────────────
SARVAM_TTS_API_KEY = os.getenv("SARVAM_TTS_API_KEY", "YOUR_SARVAM_API_KEY_HERE")
SARVAM_TTS_ENDPOINT = os.getenv("SARVAM_TTS_ENDPOINT", "https://api.sarvam.ai/text-to-speech")

# TTS Model Configuration (Valid: bulbul:v2, bulbul:v3-beta, bulbul:v3)
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")

# TTS Voice Configuration
SARVAM_TTS_VOICE = os.getenv("SARVAM_TTS_VOICE", "abhilash")  # Default voice/speaker
SARVAM_TTS_LANGUAGE = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN")  # en-IN for Indian English
SARVAM_TTS_SPEED = float(os.getenv("SARVAM_TTS_SPEED", "1.0"))  # Speech pace (0.5 - 2.0)
SARVAM_TTS_PITCH = float(os.getenv("SARVAM_TTS_PITCH", "0"))  # Pitch adjustment (-20 to 20)
SARVAM_TTS_SAMPLE_RATE = int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "16000"))  # 8000, 16000, 24000
SARVAM_TTS_LOUDNESS = float(os.getenv("SARVAM_TTS_LOUDNESS", "1.5"))  # Volume level

# Audio format configuration
SARVAM_TTS_AUDIO_FORMAT = os.getenv("SARVAM_TTS_AUDIO_FORMAT", "wav")  # wav, mp3, or pcm

# TTS Text Preprocessing
SARVAM_TTS_PREPROCESS_TEXT = os.getenv("SARVAM_TTS_PREPROCESS_TEXT", "true").lower() == "true"

# ─── STT Configuration ───────────────────────────────────────────────────────
STT_ENDPOINT = os.getenv("STT_ENDPOINT", "")  # Placeholder for custom STT endpoint if needed

# ─── Memory Configuration ────────────────────────────────────────────────────
AGENTCORE_MEMORY_ID = os.getenv("AGENTCORE_MEMORY_ID", "")
AGENTCORE_ACTOR_ID = "podcast_agent_actor"
RECENT_CONTEXT_WINDOW = int(os.getenv("RECENT_CONTEXT_WINDOW", "20"))
MEMORY_FLUSH_INTERVAL = int(os.getenv("MEMORY_FLUSH_INTERVAL", "10"))

# STM (Short-Term Memory) Configuration
STM_MAX_MESSAGES = int(os.getenv("STM_MAX_MESSAGES", "50"))
STM_TIME_WINDOW_MINUTES = int(os.getenv("STM_TIME_WINDOW_MINUTES", "30"))

# LTM (Long-Term Memory) Configuration
LTM_FLUSH_THRESHOLD = int(os.getenv("LTM_FLUSH_THRESHOLD", "15"))
LTM_SUMMARY_ENABLED = os.getenv("LTM_SUMMARY_ENABLED", "true").lower() == "true"
