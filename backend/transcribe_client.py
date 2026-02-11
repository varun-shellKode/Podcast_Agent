import asyncio
import logging
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

logger = logging.getLogger("podcast_agent")


class TranscriptHandler(TranscriptResultStreamHandler):
    """
    Handler for processing transcript events with speaker labels
    """
    def __init__(self, output_queue: asyncio.Queue, stream):
        super().__init__(stream)
        self.output_queue = output_queue
        self.current_speaker = None
        self.current_text_buffer = []

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """
        Process transcript events and extract speaker labels
        """
        results = transcript_event.transcript.results
        
        for result in results:
            # Only process final results (not partial)
            if result.is_partial:
                continue
            
            for alternative in result.alternatives:
                transcript = alternative.transcript.strip()
                
                if not transcript:
                    continue
                
                # Extract speaker label from items
                speaker_id = self._extract_speaker_from_items(alternative.items)
                
                # Debug: Log what we found
                logger.debug(f"[Transcribe Debug] Items count: {len(alternative.items)}")
                if alternative.items:
                    first_item = alternative.items[0]
                    logger.debug(f"[Transcribe Debug] First item attributes: {dir(first_item)}")
                    logger.debug(f"[Transcribe Debug] First item type: {getattr(first_item, 'type', 'N/A')}")
                
                logger.info(f"[Transcribe] Speaker={speaker_id}, Text={transcript}")
                
                await self.output_queue.put({
                    "speaker_id": str(speaker_id),
                    "text": transcript
                })
    
    def _extract_speaker_from_items(self, items):
        """
        Extract speaker label from transcript items
        
        With ShowSpeakerLabel=True, AWS Transcribe adds speaker information
        to each item in the alternatives. The speaker label is typically
        found in the item's attributes.
        """
        if not items:
            return "spk_0"
        
        # Try to find speaker label in items
        for item in items:
            # The item might have different attribute names depending on the SDK version
            # Try all possible variations
            
            # Check if item has 'speaker' attribute (most common)
            if hasattr(item, 'speaker') and item.speaker is not None:
                return f"spk_{item.speaker}"
            
            # Check for 'speaker_label'
            if hasattr(item, 'speaker_label') and item.speaker_label is not None:
                return item.speaker_label
            
            # Check for 'attendee' (alternative name)
            if hasattr(item, 'attendee') and item.attendee is not None:
                return f"spk_{item.attendee}"
            
            # Try accessing as dict if it's dict-like
            if hasattr(item, '__getitem__'):
                try:
                    if 'speaker' in item:
                        return f"spk_{item['speaker']}"
                    if 'speaker_label' in item:
                        return item['speaker_label']
                except (KeyError, TypeError):
                    pass
        
        # If no speaker found, return default
        return "spk_0"


class TranscribeClient:
    """
    AWS Transcribe Streaming client with speaker diarization support
    """
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.client = TranscribeStreamingClient(region=region)
        self.output_queue = asyncio.Queue()

    async def start_transcription(self, audio_generator):
        """
        Start streaming transcription with speaker diarization
        Uses AWS recommended settings for optimal speaker identification
        
        Args:
            audio_generator: Async generator yielding audio bytes (PCM 16-bit, 16kHz, mono)
        """
        try:
            logger.info("[Transcribe] Starting stream transcription with AWS recommended settings...")
            
            # Start stream transcription with AWS recommended parameters
            # Note: number_of_channels is only used with enable_channel_identification
            # For speaker diarization, we use show_speaker_label instead
            stream = await self.client.start_stream_transcription(
                language_code="en-US",
                media_sample_rate_hz=16000,
                media_encoding="pcm",
                
                # Speaker Diarization Settings (AWS Recommended)
                show_speaker_label=True,          # Enable speaker identification
                
                # Optional: Improve accuracy (uncomment if needed)
                # enable_partial_results_stabilization=True,  # More stable partial results
                # partial_results_stability="high",           # High stability for partials
                
                # Optional: Custom vocabulary (if you have domain-specific terms)
                # vocabulary_name="your-vocabulary-name",
                
                # Optional: Content filtering
                # vocabulary_filter_method="mask",  # mask, remove, or tag
            )

            logger.info("[Transcribe] Stream started successfully with speaker diarization enabled")

            # Create handler for processing results
            handler = TranscriptHandler(self.output_queue, stream.output_stream)
            
            # Run audio sender and result handler concurrently
            await asyncio.gather(
                self._send_audio(audio_generator, stream.input_stream),
                handler.handle_events()
            )
            
        except Exception as e:
            logger.error(f"Transcribe error: {e}", exc_info=True)
        finally:
            await self.output_queue.put(None)  # Signal end

    async def _send_audio(self, audio_generator, input_stream):
        """
        Send audio chunks to the transcription stream
        AWS recommends consistent chunk delivery for optimal speaker diarization
        """
        chunk_count = 0
        total_bytes = 0
        
        try:
            async for chunk in audio_generator:
                # Validate chunk size (AWS recommends 100 bytes to 1 MB)
                chunk_size = len(chunk)
                if chunk_size < 100:
                    logger.warning(f"[Transcribe] Chunk too small: {chunk_size} bytes (min 100)")
                    continue
                if chunk_size > 1048576:  # 1 MB
                    logger.warning(f"[Transcribe] Chunk too large: {chunk_size} bytes (max 1 MB)")
                    continue
                
                await input_stream.send_audio_event(audio_chunk=chunk)
                chunk_count += 1
                total_bytes += chunk_size
                
                # Log progress every 50 chunks (~3.2 seconds at 1024 samples/chunk)
                if chunk_count % 50 == 0:
                    duration_seconds = (total_bytes / 2) / 16000  # bytes/2 = samples, /16000 = seconds
                    logger.debug(
                        f"[Transcribe] Sent {chunk_count} chunks, "
                        f"{total_bytes} bytes, ~{duration_seconds:.1f}s of audio"
                    )
            
            logger.info(
                f"[Transcribe] Finished sending audio. "
                f"Total: {chunk_count} chunks, {total_bytes} bytes, "
                f"~{(total_bytes / 2) / 16000:.1f}s of audio"
            )
            await input_stream.end_stream()
            
        except Exception as e:
            logger.error(f"Error sending audio to Transcribe: {e}", exc_info=True)

    async def get_results(self):
        """
        Async generator to yield transcription results
        
        Yields:
            dict: {"speaker_id": str, "text": str}
        """
        while True:
            result = await self.output_queue.get()
            if result is None:
                break
            yield result
