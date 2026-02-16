import os
import asyncio
import json
import base64
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("podcast_agent")

from transcribe_client import TranscribeClient
from tts_client import SarvamTTSClient
from podcast_orchestrator import PodcastOrchestrator, PodcastState
from speaker_personas import get_persona, get_speaker_name
from strands.models import BedrockModel
from tts_preprocessor import preprocess_for_tts, format_question_for_speech

# Optional: Old agent_logic for backward compatibility with /ws/podcast endpoint
try:
    from agent_logic import (
        init_agent,
        detect_trigger,
        strip_trigger_from_query,
        invoke_agent,
        conversation_buffer,
        memory_flusher,
        PERSONAS,
        start_new_session,
        end_current_session,
        get_session_info,
    )
    OLD_AGENT_AVAILABLE = True
except ImportError:
    OLD_AGENT_AVAILABLE = False
    logger.warning("⚠️  agent_logic not available - old /ws/podcast endpoint disabled")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Server starting...")
    if OLD_AGENT_AVAILABLE:
        await asyncio.to_thread(init_agent)
    yield
    # Shutdown
    logger.info("🛑 Server shutting down...")
    if OLD_AGENT_AVAILABLE:
        await memory_flusher.flush_to_memory()

app = FastAPI(title="Podcast AI Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "running", "agent": os.getenv("AGENT_TRIGGER_NAME", "Cupcake")}

@app.get("/session/info")
async def session_info():
    """Get current session information"""
    if not OLD_AGENT_AVAILABLE:
        return {"error": "Old agent system not available", "available": False}
    return get_session_info()

@app.post("/session/start")
async def start_session():
    """Start a new podcast session"""
    if not OLD_AGENT_AVAILABLE:
        return {"error": "Old agent system not available", "available": False}
    session_id = start_new_session()
    return {"session_id": session_id, "status": "started"}

@app.post("/session/end")
async def end_session():
    """End current session"""
    if not OLD_AGENT_AVAILABLE:
        return {"error": "Old agent system not available", "available": False}
    end_current_session()
    return {"status": "ended"}

@app.post("/podcast/start")
async def start_podcast(topic: str, max_turns: int = 20):
    """Start an orchestrated podcast session"""
    return {
        "status": "ready",
        "topic": topic,
        "max_turns": max_turns,
        "message": "Connect to /ws/orchestrated-podcast to start"
    }

@app.get("/podcast/info")
async def podcast_info():
    """Get information about available role templates and active participants"""
    from speaker_personas import ROLE_TEMPLATES, get_all_participants
    return {
        "role_templates": ROLE_TEMPLATES,
        "active_participants": get_all_participants(),
        "total_roles": len(ROLE_TEMPLATES),
        "total_participants": len(get_all_participants())
    }

@app.get("/tts/info")
async def tts_info():
    """Get TTS configuration info"""
    from tts_client import get_available_voices
    client = SarvamTTSClient()
    return {
        "configured": client.api_key != "YOUR_SARVAM_API_KEY_HERE",
        "current_voice": client.get_voice_info(),
        "available_voices": get_available_voices(),
    }

@app.post("/tts/test")
async def test_tts(text: str = "Hello, this is a test of the text to speech system."):
    """Test TTS synthesis (returns JSON with base64)"""
    client = SarvamTTSClient()
    audio_data = await client.synthesize_speech(text)

    if audio_data:
        # Save audio file for inspection
        test_file = "test_audio_output.wav"
        with open(test_file, "wb") as f:
            f.write(audio_data)
        logger.info(f"💾 Test audio saved to {test_file}")

        return {
            "status": "success",
            "audio_base64": base64.b64encode(audio_data).decode('utf-8'),
            "audio_format": client.get_audio_format(),
            "sample_rate": client.get_sample_rate(),
            "size_bytes": len(audio_data),
            "saved_to": test_file,
        }
    else:
        return {"status": "failed", "error": "TTS synthesis failed"}

@app.get("/tts/test-audio")
async def test_tts_audio(text: str = "Hello, this is a test"):
    """Test TTS synthesis (returns audio file directly)"""
    client = SarvamTTSClient()
    audio_data = await client.synthesize_speech(text)

    if audio_data:
        # Return audio directly with proper headers
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=test.wav",
                "Accept-Ranges": "bytes"
            }
        )
    else:
        return {"status": "failed", "error": "TTS synthesis failed"}

@app.websocket("/ws/podcast")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    if not OLD_AGENT_AVAILABLE:
        await websocket.send_json({
            "type": "error",
            "message": "Old agent system not available. Use /ws/orchestrated-podcast instead."
        })
        await websocket.close()
        return

    logger.info("🔌 WebSocket connected")

    audio_queue = asyncio.Queue()
    transcribe_client = TranscribeClient(region=os.getenv("AWS_REGION", "us-east-1"))
    tts_client = SarvamTTSClient()
    is_tts_playing = False  # Track TTS playback state

    async def audio_generator():
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield chunk

    # Task to run transcription
    transcribe_task = asyncio.create_task(transcribe_client.start_transcription(audio_generator()))

    # Task to process transcription results
    async def process_results():
        nonlocal is_tts_playing
        async for result in transcribe_client.get_results():
            speaker_id = result["speaker_id"]
            text = result["text"]
            
            persona = PERSONAS.get(speaker_id, {"name": f"Speaker {speaker_id}", "role": "Participant"})
            speaker_name = persona["name"]
            
            # Add to buffer
            conversation_buffer.add(speaker_id, speaker_name, text)
            
            # Send transcript to UI
            await websocket.send_json({
                "type": "transcript",
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "text": text
            })
            
            # Only check for trigger if TTS is NOT playing
            if not is_tts_playing and detect_trigger(text):
                query = strip_trigger_from_query(text)
                # Notify UI that agent is thinking
                await websocket.send_json({"type": "thinking"})

                # Invoke agent
                agent_response = await invoke_agent(speaker_id, query)

                # Send text response to UI
                await websocket.send_json({
                    "type": "agent_response",
                    **agent_response
                })

                # Generate TTS audio from agent response
                try:
                    response_text = agent_response.get("response", "")
                    if response_text:
                        logger.info("🔊 Generating TTS audio for agent response...")

                        # Preprocess for TTS (handle acronyms, remove markdown, etc.)
                        processed_text = preprocess_for_tts(response_text)

                        # Split response into sentences for better streaming
                        import re
                        # Split on sentence boundaries (., !, ?) followed by space or end
                        sentences = re.split(r'(?<=[.!?])\s+', processed_text.strip())
                        sentences = [s.strip() for s in sentences if s.strip()]

                        logger.info(f"🔊 Split response into {len(sentences)} sentences")

                        # Generate and send audio for each sentence
                        for i, sentence in enumerate(sentences, 1):
                            logger.info(f"🔊 Generating audio for sentence {i}/{len(sentences)}: {sentence[:50]}...")
                            audio_data = await tts_client.synthesize_speech(sentence)

                            if audio_data:
                                # Send audio data to UI (base64 encoded for JSON)
                                audio_base64 = base64.b64encode(audio_data).decode('utf-8')

                                await websocket.send_json({
                                    "type": "agent_audio",
                                    "audio_data": audio_base64,
                                    "audio_format": tts_client.get_audio_format(),
                                    "sample_rate": tts_client.get_sample_rate(),
                                    "voice_info": tts_client.get_voice_info(),
                                    "sentence_index": i,
                                    "total_sentences": len(sentences),
                                })
                                logger.info(f"✅ TTS audio {i}/{len(sentences)} sent to client")
                            else:
                                logger.warning(f"⚠️  TTS audio generation failed for sentence {i}")
                        
                        logger.info(f"✅ All {len(sentences)} audio packets sent")
                except Exception as tts_error:
                    logger.error(f"❌ TTS error: {tts_error}", exc_info=True)
                    # Continue even if TTS fails - text response was already sent
            elif is_tts_playing and detect_trigger(text):
                logger.info(f"🔇 Ignoring trigger during TTS playback: {text}")
            
            # Check if flush needed
            if memory_flusher.on_utterance_committed():
                asyncio.create_task(memory_flusher.flush_to_memory())

    results_task = asyncio.create_task(process_results())

    try:
        while True:
            data = await websocket.receive()
            
            # Handle text messages (TTS state updates)
            if 'text' in data:
                try:
                    message = json.loads(data['text'])
                    if message.get('type') == 'tts_started':
                        is_tts_playing = True
                        logger.info("🔇 TTS playback started - ignoring triggers")
                    elif message.get('type') == 'tts_ended':
                        is_tts_playing = False
                        logger.info("🔊 TTS playback ended - resuming trigger detection")
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON message")
            # Handle binary messages (audio data)
            elif 'bytes' in data:
                await audio_queue.put(data['bytes'])
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await audio_queue.put(None)
        await transcribe_task
        await results_task
        logger.info("Cleanup complete")

@app.websocket("/ws/orchestrated-podcast")
async def orchestrated_podcast_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for orchestrated podcast mode

    Expected messages from client:
    - {"type": "start", "topic": "...", "max_turns": 20}
    - {"type": "speaker_response", "speaker_id": "1", "text": "..."}
    - {"type": "audio_chunk", "data": <binary>}  # For transcription
    """
    await websocket.accept()
    logger.info("🔌 Orchestrated Podcast WebSocket connected")

    orchestrator: Optional[PodcastOrchestrator] = None
    audio_queue = asyncio.Queue()
    transcribe_client = TranscribeClient(region=os.getenv("AWS_REGION", "us-east-1"))
    tts_client = SarvamTTSClient()
    is_tts_playing = False

    async def audio_generator():
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield chunk

    # Start transcription task
    transcribe_task = asyncio.create_task(transcribe_client.start_transcription(audio_generator()))

    # Callback for orchestrator to send messages
    async def send_orchestrator_message(message: dict):
        """Callback for orchestrator to send messages to UI"""
        try:
            await websocket.send_json(message)

            # If it's a question, generate TTS
            if message.get("type") in ["orchestrator_question", "orchestrator_redirect", "orchestrator_intro", "orchestrator_wrapup"]:
                text_to_speak = message.get("question") or message.get("text") or message.get("message")
                if text_to_speak:
                    await generate_and_send_tts(text_to_speak, "orchestrator")

        except Exception as e:
            logger.error(f"❌ Failed to send orchestrator message: {e}")

    async def generate_and_send_tts(text: str, speaker_type: str):
        """Generate and send TTS audio"""
        try:
            # Preprocess text for TTS (handle acronyms, remove markdown, etc.)
            processed_text = preprocess_for_tts(text)
            logger.info(f"🔊 Generating TTS for {speaker_type}: {processed_text[:50]}...")

            # Split into sentences
            import re
            sentences = re.split(r'(?<=[.!?])\s+', processed_text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]

            for i, sentence in enumerate(sentences, 1):
                audio_data = await tts_client.synthesize_speech(sentence)

                if audio_data:
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

                    await websocket.send_json({
                        "type": f"{speaker_type}_audio",
                        "audio_data": audio_base64,
                        "audio_format": tts_client.get_audio_format(),
                        "sample_rate": tts_client.get_sample_rate(),
                        "sentence_index": i,
                        "total_sentences": len(sentences),
                    })
                    logger.info(f"✅ TTS audio {i}/{len(sentences)} sent")

        except Exception as tts_error:
            logger.error(f"❌ TTS error: {tts_error}", exc_info=True)

    # Response accumulation for orchestrator
    accumulated_responses = {}  # {speaker_id: {"text": str, "last_update": float}}
    response_timeout = 3.0  # seconds of silence before considering response complete

    def normalize_speaker_id(speaker_id: str) -> str:
        """Normalize speaker ID: spk_0 -> 0, spk_1 -> 1, etc."""
        if speaker_id.startswith("spk_"):
            return speaker_id.replace("spk_", "")
        return speaker_id

    async def check_response_completion():
        """Check if accumulated response is complete and should be processed"""
        nonlocal accumulated_responses
        import time

        while True:
            await asyncio.sleep(0.5)  # Check every 0.5 seconds

            if not orchestrator or not orchestrator.awaiting_response:
                continue

            current_time = time.time()
            expected_speaker = orchestrator.current_speaker_id

            # Check for completion signals in ANY recent speech (not just expected speaker)
            completion_signals = [
                "let's begin",
                "let's start",
                "shall we get started",
                "shall we start",
                "you can ask",
                "you could ask",
                "ask me questions",
                "ask me any questions",
                "i'm here to answer",
                "here to answer",
                "that's it",
                "go ahead",
                "ready",
                "proceed",
                "let's get started"
            ]

            if accumulated_responses:
                # 1. First priority: Check ALL speakers for completion signals
                # This handles cases where AWS Transcribe shifts the speaker ID mid-turn (e.g. 1 -> 2)
                best_speaker = None
                
                # Check expected speaker first
                if expected_speaker in accumulated_responses:
                    text = accumulated_responses[expected_speaker]["text"].strip().lower()
                    for signal in completion_signals:
                        if text.endswith(signal) or any(text.endswith(signal + p) for p in [".", "!", "?"]):
                            best_speaker = expected_speaker
                            break
                
                # If expected speaker didn't have a signal, check others
                if not best_speaker:
                    for sid, data in accumulated_responses.items():
                        if sid == expected_speaker: continue
                        text = data["text"].strip().lower()
                        for signal in completion_signals:
                            if text.endswith(signal) or any(text.endswith(signal + p) for p in [".", "!", "?"]):
                                logger.info(f"🔔 Completion signal from speaker {sid} (expected: {expected_speaker})")
                                best_speaker = sid
                                break
                        if best_speaker: break

                # 2. Second priority: Check for timeouts on MOST ACTIVE speaker
                has_timeout = False
                timeout_speaker = None
                
                # Find most recently active speaker
                latest_update = 0
                for sid, data in accumulated_responses.items():
                    if data["last_update"] > latest_update:
                        latest_update = data["last_update"]
                        timeout_speaker = sid
                
                if timeout_speaker and (current_time - latest_update >= response_timeout):
                    # Only timeout if they actually said something substantial (> 10 chars)
                    if len(accumulated_responses[timeout_speaker]["text"].strip()) > 10:
                        has_timeout = True
                        logger.info(f"⏰ Response timeout for speaker {timeout_speaker} (expected: {expected_speaker})")

                # If we have a winner, process it
                if best_speaker or has_timeout:
                    winner_sid = best_speaker or timeout_speaker
                    full_response = accumulated_responses[winner_sid]["text"].strip()
                    
                    logger.info(f"✅ Processing response from speaker {winner_sid}: {full_response[:100]}...")

                    # Clear ALL accumulated responses (reset for next turn)
                    accumulated_responses.clear()

                    # Process the response
                    await orchestrator.handle_speaker_response(winner_sid, full_response)

    # Start response completion checker
    completion_checker_task = asyncio.create_task(check_response_completion())

    # Task to process transcription results
    async def process_transcription_results():
        nonlocal is_tts_playing, orchestrator, accumulated_responses
        import time

        async for result in transcribe_client.get_results():
            speaker_id = result["speaker_id"]
            text = result["text"]

            # Filter out very short, meaningless utterances (noise)
            text_stripped = text.strip()
            if len(text_stripped) <= 2 and text_stripped.lower() in ['i', 'a', 'o', 'uh', 'um', 'ah']:
                logger.debug(f"🔇 Filtered noise: '{text_stripped}'")
                continue

            # Normalize speaker ID (spk_0 -> 0, etc.)
            normalized_id = normalize_speaker_id(speaker_id)

            speaker_name = get_speaker_name(normalized_id)

            # Send transcript to UI
            await websocket.send_json({
                "type": "transcript",
                "speaker_id": normalized_id,
                "speaker_name": speaker_name,
                "text": text
            })

            # Debug: Log orchestrator state
            if orchestrator:
                logger.debug(f"🔍 Orchestrator state: awaiting={orchestrator.awaiting_response}, tts_playing={is_tts_playing}, expected_speaker={orchestrator.current_speaker_id}")

            # If orchestrator is active and awaiting response, accumulate it
            if orchestrator and orchestrator.awaiting_response and not is_tts_playing:
                # Accumulate ALL speakers' responses (not just expected one)
                # This allows completion signals from any speaker
                if normalized_id not in accumulated_responses:
                    accumulated_responses[normalized_id] = {
                        "text": text,
                        "last_update": time.time()
                    }
                    logger.info(f"📝 Started accumulating response from speaker {normalized_id} (expected: {orchestrator.current_speaker_id})")
                else:
                    # Append to existing response
                    accumulated_responses[normalized_id]["text"] += " " + text
                    accumulated_responses[normalized_id]["last_update"] = time.time()
                    logger.info(f"📝 Accumulated response from speaker {normalized_id}: ...{text}")
            elif orchestrator and not orchestrator.awaiting_response:
                logger.info(f"⏸️  Ignoring transcript - orchestrator not awaiting response")
            elif orchestrator and is_tts_playing:
                logger.info(f"⏸️  Ignoring transcript - TTS is playing: '{text[:50]}...'")

    results_task = asyncio.create_task(process_transcription_results())

    try:
        while True:
            data = await websocket.receive()

            # Handle text messages (commands)
            if 'text' in data:
                try:
                    message = json.loads(data['text'])
                    msg_type = message.get('type')

                    if msg_type == 'start':
                        # Start orchestrated podcast
                        topic = message.get('topic', 'Technology and Innovation')
                        max_turns = message.get('max_turns', 20)

                        logger.info(f"🎬 Starting orchestrated podcast: {topic}")

                        # Initialize Bedrock model
                        model = BedrockModel(
                            model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
                            region_name=os.getenv("AWS_REGION", "us-east-1"),
                            temperature=0.7
                        )

                        # Create orchestrator
                        orchestrator = PodcastOrchestrator(
                            model=model,
                            topic=topic,
                            max_turns=max_turns
                        )
                        orchestrator.send_message_callback = send_orchestrator_message

                        # Start podcast
                        await orchestrator.start_podcast()

                    elif msg_type == 'speaker_response':
                        # Manual speaker response (for testing/manual mode)
                        if orchestrator:
                            speaker_id = message.get('speaker_id')
                            response_text = message.get('text', '')
                            await orchestrator.handle_speaker_response(speaker_id, response_text)

                    elif msg_type == 'get_state':
                        # Get orchestrator state
                        if orchestrator:
                            state = orchestrator.get_state_info()
                            await websocket.send_json({
                                "type": "state_info",
                                "state": state
                            })

                    elif msg_type == 'tts_started':
                        is_tts_playing = True
                        logger.info("🔇 TTS playback started - pausing transcript accumulation")

                    elif msg_type == 'tts_ended':
                        is_tts_playing = False
                        logger.info("🔊 TTS playback ended - resuming transcript accumulation")
                        # Log current orchestrator state after TTS ends
                        if orchestrator:
                            logger.info(f"📍 After TTS: awaiting_response={orchestrator.awaiting_response}, expected_speaker={orchestrator.current_speaker_id}")

                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON message")

            # Handle binary messages (audio data for transcription)
            elif 'bytes' in data:
                await audio_queue.put(data['bytes'])

    except WebSocketDisconnect:
        logger.info("🔌 Orchestrated Podcast WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if orchestrator:
            orchestrator.is_active = False
        await audio_queue.put(None)
        completion_checker_task.cancel()
        await transcribe_task
        await results_task
        logger.info("Orchestrated podcast cleanup complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
