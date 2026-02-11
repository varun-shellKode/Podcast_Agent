"""
Sarvam TTS Client for Text-to-Speech conversion
"""
import logging
import asyncio
import json
import base64
import aiohttp
from typing import Optional, AsyncGenerator
from config import (
    SARVAM_TTS_API_KEY,
    SARVAM_TTS_ENDPOINT,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_VOICE,
    SARVAM_TTS_LANGUAGE,
    SARVAM_TTS_SPEED,
    SARVAM_TTS_PITCH,
    SARVAM_TTS_SAMPLE_RATE,
    SARVAM_TTS_LOUDNESS,
    SARVAM_TTS_AUDIO_FORMAT,
)
from tts_utils import preprocess_text_for_tts

logger = logging.getLogger("podcast_agent")


class SarvamTTSClient:
    """
    Client for Sarvam AI Text-to-Speech API
    Converts agent responses to speech audio
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.api_key = api_key or SARVAM_TTS_API_KEY
        self.endpoint = endpoint or SARVAM_TTS_ENDPOINT
        self.model = SARVAM_TTS_MODEL
        self.voice = voice or SARVAM_TTS_VOICE
        self.language = language or SARVAM_TTS_LANGUAGE
        self.speed = SARVAM_TTS_SPEED
        self.pitch = SARVAM_TTS_PITCH
        self.loudness = SARVAM_TTS_LOUDNESS
        self.sample_rate = SARVAM_TTS_SAMPLE_RATE
        self.audio_format = SARVAM_TTS_AUDIO_FORMAT

        # Validate configuration
        if not self.api_key or self.api_key == "YOUR_SARVAM_API_KEY_HERE":
            logger.warning("⚠️  Sarvam TTS API key not configured")

        logger.info(
            f"🔊 Sarvam TTS Client initialized\n"
            f"   Endpoint: {self.endpoint}\n"
            f"   Model: {self.model}\n"
            f"   Voice: {self.voice}\n"
            f"   Language: {self.language}\n"
            f"   Sample Rate: {self.sample_rate}Hz"
        )

    async def synthesize_speech(self, text: str, preprocess: bool = True) -> Optional[bytes]:
        """
        Convert text to speech using Sarvam TTS API

        Args:
            text: Text to convert to speech
            preprocess: Whether to preprocess text for better pronunciation (default: True)

        Returns:
            Audio bytes in the configured format, or None if failed
        """
        if not text or not text.strip():
            logger.warning("🔊 [TTS] Empty text provided, skipping synthesis")
            return None

        if not self.api_key or self.api_key == "YOUR_SARVAM_API_KEY_HERE":
            logger.error("🔊 [TTS] API key not configured, cannot synthesize")
            return None

        try:
            # Preprocess text for better pronunciation
            if preprocess:
                original_text = text
                text = preprocess_text_for_tts(text)
                logger.info(f"🔊 [TTS] Text preprocessing:")
                logger.info(f"   Original: {original_text}")
                logger.info(f"   Preprocessed: {text}")
                if text != original_text:
                    logger.info(f"   ✅ Text was modified by preprocessing")
                else:
                    logger.info(f"   ℹ️  No changes made by preprocessing")

            logger.info(f"🔊 [TTS] Synthesizing speech: {text[:100]}...")

            # Prepare request payload
            payload = {
                "text": text,
                "target_language_code": self.language,
                "speaker": self.voice,
                "pitch": self.pitch,
                "pace": self.speed,
                "loudness": self.loudness,
                "speech_sample_rate": self.sample_rate,
                "enable_preprocessing": True,
                "model": self.model,
            }

            # Log the full request for debugging
            logger.debug(f"🔊 [TTS] Request payload: {payload}")

            # Prepare headers
            headers = {
                "API-Subscription-Key": self.api_key,
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        # Log response headers to see what format we're getting
                        content_type = response.headers.get('Content-Type', 'unknown')
                        logger.info(f"🔊 [TTS] Response Content-Type: {content_type}")

                        response_data = await response.read()

                        # Check if response is JSON (Sarvam returns JSON with base64 audio)
                        if content_type and 'application/json' in content_type:
                            logger.info("🔊 [TTS] Response is JSON, extracting audio data...")

                            try:
                                json_response = json.loads(response_data.decode('utf-8'))
                                logger.info(f"🔊 [TTS] JSON response keys: {json_response.keys()}")

                                # Sarvam returns base64 audio in 'audios' field
                                if 'audios' in json_response and len(json_response['audios']) > 0:
                                    audio_base64 = json_response['audios'][0]
                                    logger.info(f"🔊 [TTS] Found base64 audio in 'audios' field")
                                    audio_data = base64.b64decode(audio_base64)
                                else:
                                    logger.error(f"❌ [TTS] Unexpected JSON structure: {json_response}")
                                    return None

                            except Exception as e:
                                logger.error(f"❌ [TTS] Failed to parse JSON response: {e}")
                                return None
                        else:
                            # Response is raw audio
                            logger.info("🔊 [TTS] Response is raw audio data")
                            audio_data = response_data

                        # Log first few bytes to check audio format
                        if len(audio_data) > 12:
                            header_bytes = audio_data[:12]
                            header_str = ''.join([f'{b:02x}' for b in header_bytes])
                            logger.info(f"🔊 [TTS] Audio header (first 12 bytes): {header_str}")

                            # Check if it's a valid WAV file (RIFF header)
                            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                                logger.info("✅ [TTS] Valid WAV file detected")
                            else:
                                logger.warning(f"⚠️  [TTS] Not a standard WAV file format. Header: {audio_data[:4]}")

                        logger.info(
                            f"✅ [TTS] Speech synthesized successfully "
                            f"({len(audio_data)} bytes)"
                        )
                        return audio_data
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"❌ [TTS] API error {response.status}: {error_text}"
                        )
                        return None

        except asyncio.TimeoutError:
            logger.error("❌ [TTS] Request timeout")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"❌ [TTS] HTTP client error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ [TTS] Unexpected error: {e}", exc_info=True)
            return None

    async def synthesize_speech_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized speech in chunks (if API supports streaming)

        Args:
            text: Text to convert to speech

        Yields:
            Audio chunks as bytes
        """
        try:
            logger.info(f"🔊 [TTS] Streaming speech synthesis: {text[:100]}...")

            payload = {
                "text": text,
                "target_language_code": self.language,
                "speaker": self.voice,
                "pitch": self.pitch,
                "pace": self.speed,
                "loudness": self.loudness,
                "speech_sample_rate": self.sample_rate,
                "enable_preprocessing": True,
                "model": self.model,
            }

            headers = {
                "API-Subscription-Key": self.api_key,
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        content_type = response.headers.get('Content-Type', 'unknown')

                        # If JSON response, need to get full response first
                        if 'application/json' in content_type:
                            response_data = await response.read()
                            try:
                                json_response = json.loads(response_data.decode('utf-8'))
                                if 'audios' in json_response and len(json_response['audios']) > 0:
                                    audio_base64 = json_response['audios'][0]
                                    audio_data = base64.b64decode(audio_base64)

                                    # Yield in chunks
                                    chunk_size = 4096
                                    for i in range(0, len(audio_data), chunk_size):
                                        yield audio_data[i:i + chunk_size]

                                    logger.info(f"✅ [TTS] Streaming complete")
                                else:
                                    logger.error("❌ [TTS] No audio in JSON response")
                            except Exception as e:
                                logger.error(f"❌ [TTS] Streaming JSON parse error: {e}")
                        else:
                            # Stream raw audio response
                            chunk_count = 0
                            async for chunk in response.content.iter_chunked(4096):
                                chunk_count += 1
                                yield chunk

                            logger.info(
                                f"✅ [TTS] Streaming complete ({chunk_count} chunks)"
                            )
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"❌ [TTS] Streaming error {response.status}: {error_text}"
                        )

        except Exception as e:
            logger.error(f"❌ [TTS] Streaming error: {e}", exc_info=True)

    def get_audio_format(self) -> str:
        """Get the configured audio format"""
        return self.audio_format

    def get_sample_rate(self) -> int:
        """Get the configured sample rate"""
        return self.sample_rate

    def get_voice_info(self) -> dict:
        """Get current voice configuration"""
        return {
            "model": self.model,
            "voice": self.voice,
            "language": self.language,
            "speed": self.speed,
            "pitch": self.pitch,
            "loudness": self.loudness,
            "sample_rate": self.sample_rate,
            "format": self.audio_format,
        }


# ─── Convenience Functions ────────────────────────────────────────────────────


async def text_to_speech(text: str, voice: Optional[str] = None, preprocess: bool = True) -> Optional[bytes]:
    """
    Convenience function to quickly convert text to speech

    Args:
        text: Text to convert
        voice: Optional voice override
        preprocess: Whether to preprocess text for better pronunciation

    Returns:
        Audio bytes or None if failed
    """
    client = SarvamTTSClient(voice=voice)
    return await client.synthesize_speech(text, preprocess=preprocess)


# ─── Voice Presets ────────────────────────────────────────────────────────────

# Common Sarvam TTS voice options
VOICE_PRESETS = {
    "meera": {"name": "Meera", "language": "en-IN", "description": "Indian English Female"},
    "arvind": {"name": "Arvind", "language": "en-IN", "description": "Indian English Male"},
    "bulbul:v1": {"name": "Bulbul v1", "language": "en-IN", "description": "Sarvam AI Voice v1"},
    "bulbul:v2": {"name": "Bulbul v2", "language": "en-IN", "description": "Sarvam AI Voice v2"},
    "bulbul:v3": {"name": "Bulbul v3", "language": "en-IN", "description": "Sarvam AI Voice v3 (Recommended)"},
    "bulbul:v3-beta": {"name": "Bulbul v3 Beta", "language": "en-IN", "description": "Sarvam AI Voice v3 Beta"},
}


def get_available_voices() -> dict:
    """Get available voice presets"""
    return VOICE_PRESETS
