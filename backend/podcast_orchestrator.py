"""
Podcast Orchestrator - Central controller for AI-moderated podcast with 5 speakers
"""
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Awaitable
from datetime import datetime
from enum import Enum

from strands.models import BedrockModel
from strands import Agent

from speaker_personas import (
    get_persona,
    get_speaker_name,
    set_speaker_name,
    get_all_participants,
    add_participant,
    detect_role_from_introduction,
    increment_turn_count,
    reset_participants
)
from response_evaluator import ResponseEvaluator
from question_generator import QuestionGenerator
from speaker_selector import SpeakerSelector

logger = logging.getLogger("orchestrator")

class PodcastState(Enum):
    """Current state of the podcast"""
    IDLE = "idle"
    INTRO = "intro"
    QUESTIONING = "questioning"
    LISTENING = "listening"
    INTRO_WAITING = "intro_waiting"
    EVALUATING = "evaluating"
    INTERRUPTING = "interrupting"
    WRAPPING_UP = "wrapping_up"
    ENDED = "ended"

class PodcastOrchestrator:
    """
    Central orchestrator that controls the podcast flow:
    - Generates questions based on topic and speaker persona
    - Listens to speaker responses
    - Evaluates responses for relevance
    - Interrupts politely when needed
    - Maintains conversation context
    """

    def __init__(
        self,
        model: BedrockModel,
        topic: str,
        max_turns: int = 20,
        min_turns_per_speaker: int = 2
    ):
        self.model = model
        self.topic = topic
        self.max_turns = max_turns
        self.min_turns_per_speaker = min_turns_per_speaker

        # State management
        self.state = PodcastState.IDLE
        self.turn_count = 0
        self.current_speaker_id: Optional[str] = None
        self.current_question: Optional[str] = None
        self.awaiting_response = False
        self.conversation_history: List[Dict] = []
        self.is_active = True  # Flag to track if the session is still active

        # Reset participants for this session
        reset_participants()

        # Components
        self.evaluator = ResponseEvaluator(model)
        self.question_generator = QuestionGenerator(model)
        self.speaker_selector = SpeakerSelector(model)

        # Agent for orchestrator's own responses
        self.agent = Agent(
            model=model,
            system_prompt=self._build_orchestrator_prompt()
        )

        # Callback for sending messages (set by main.py)
        self.send_message_callback: Optional[Callable] = None

        # Topics coverage tracking
        self.covered_topics = []
        self.uncovered_topics = []

        logger.info(
            f"🎙️  Podcast Orchestrator initialized\n"
            f"   Topic: {topic}\n"
            f"   Max turns: {max_turns}\n"
            f"   Mode: Dynamic participants"
        )

    def _build_orchestrator_prompt(self) -> str:
        return f"""You are the AI orchestrator for a technical podcast on the topic: "{self.topic}"

ROLE:
You are an intelligent moderator for a dynamic podcast where participants join and introduce themselves.

Participants may have various roles:
- Host (you may act as this)
- AWS Expert
- Developer
- Tech Expert
- Business stakeholder
- General participant

RESPONSIBILITIES:
1. Welcome participants as they introduce themselves
2. Identify their expertise based on their introduction
3. Generate contextual questions tailored to each participant's expertise
4. Listen to responses and evaluate relevance
5. Politely redirect if speakers go off-track
6. Keep conversation flowing and engaging
7. Ensure all participants contribute meaningfully
8. Cover the topic comprehensively

BEHAVIOR:
- Be professional yet conversational
- Show curiosity and interest
- Redirect diplomatically when needed
- Acknowledge good points before moving on
- Keep the energy and pace appropriate
- When a new participant introduces themselves, note their name and expertise
- Address participants by their actual names

IMPORTANT - TEXT-TO-SPEECH OUTPUT:
Your output will be converted to speech. Follow these rules:
- NO markdown formatting (no **, *, __, `, #, etc.)
- NO hashtags
- NO special characters or symbols
- Write naturally as you would speak
- Use plain, conversational English
- Questions should naturally address the speaker by name
"""

    async def start_podcast(self):
        """Start the podcast with an introduction"""
        self.state = PodcastState.INTRO
        logger.info("🎬 Starting podcast...")

        # Generate introduction
        intro = await self._generate_introduction()

        # Send intro to UI
        if self.send_message_callback:
            await self.send_message_callback({
                "type": "orchestrator_intro",
                "text": intro,
                "topic": self.topic,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Add to history
        self._add_to_history("orchestrator", "Orchestrator", intro)

        # Transition to waiting for introductions
        logger.info("⏳ Waiting for human participant introductions...")
        self.state = PodcastState.INTRO_WAITING
        self.awaiting_response = True
        self.current_speaker_id = None # Accept any speaker initially

    async def _generate_introduction(self) -> str:
        """Generate podcast introduction"""
        prompt = f"""Generate a brief, engaging podcast introduction and invite participants to introduce themselves.

TOPIC: {self.topic}

Instructions:
1. Welcome everyone to the podcast and state the topic clearly.
2. Invite participants to introduce themselves (name and background).
3. Keep it to 2-3 sentences max.
4. Be warm and inviting.
5. Write as natural speech (this will be spoken aloud via text-to-speech).
6. NO markdown, hashtags, or special formatting.

Return ONLY the introduction, no preamble or formatting."""

        try:
            intro = str(self.agent(prompt)).strip()
            # Clean up any quotes that might wrap the response
            intro = intro.strip('"').strip("'")
            return intro
        except Exception as e:
            logger.error(f"❌ Introduction generation failed: {e}")
            return f"Welcome to our podcast on {self.topic}. I'm excited to have everyone here today. Please introduce yourselves, your name and a bit about your background!"

    async def _ask_next_question(self):
        """Generate and ask the next question"""
        if self.turn_count >= self.max_turns:
            await self._wrap_up_podcast()
            return

        # Select next speaker
        logger.info("🧠 Orchestrator is thinking about who to call next...")
        next_speaker_id = await self._select_next_speaker()
        if next_speaker_id is None:
            await self._wrap_up_podcast()
            return

        persona = get_persona(next_speaker_id)

        # Check if this is the speaker's first turn (for introduction prompt)
        is_first_turn = persona.get("turn_count", 0) == 0

        # Generate question
        question = await self.question_generator.generate_question(
            topic=self.topic,
            speaker_id=next_speaker_id,
            speaker_persona=persona,
            conversation_history=self.conversation_history,
            uncovered_areas=self.uncovered_topics,
            ask_for_introduction=is_first_turn
        )

        # Update state
        self.current_speaker_id = next_speaker_id
        self.current_question = question
        self.awaiting_response = True
        self.state = PodcastState.LISTENING

        # Get display name
        display_name = get_speaker_name(next_speaker_id)

        # Send question to UI
        if self.send_message_callback:
            await self.send_message_callback({
                "type": "orchestrator_question",
                "speaker_id": next_speaker_id,
                "speaker_name": display_name,
                "speaker_role": persona["role"],
                "question": question,
                "turn_number": self.turn_count + 1,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Add to history
        self._add_to_history("orchestrator", "Orchestrator", f"[To {display_name}] {question}")

        self.turn_count += 1
        increment_turn_count(next_speaker_id, self.turn_count)

        logger.info(
            f"❓ Turn {self.turn_count}: Asked {display_name} ({persona['role']})\n"
            f"   Question: {question}"
        )

    async def _select_next_speaker(self) -> Optional[str]:
        """
        Select next speaker to ask using the SpeakerSelector agent
        """
        # Get all active participants
        participants = get_all_participants()

        if not participants:
            logger.warning("⚠️ No active participants to select from")
            return None

        # Build turn counts from participants
        turn_counts = {pid: p["turn_count"] for pid, p in participants.items()}
        last_asked = {pid: p["last_turn"] for pid, p in participants.items()}

        # Get decision from speaker selector
        next_speaker_id = await self.speaker_selector.select_next_speaker(
            topic=self.topic,
            history=self.conversation_history,
            personas=participants,
            turn_counts=turn_counts,
            last_asked=last_asked,
            consecutive_ai_turns=0  # No AI turns anymore
        )

        return next_speaker_id


    async def handle_speaker_response(self, speaker_id: str, response_text: str):
        """
        Handle a speaker's response:
        1. Verify it's the expected speaker
        2. Evaluate response quality and relevance
        3. Decide: continue, redirect, interrupt, or move on
        """
        # Reset awaiting state immediately so it can be set again by redirects/interrupts
        self.awaiting_response = False

        if speaker_id != self.current_speaker_id and self.state != PodcastState.INTRO_WAITING:
            logger.warning(
                f"⚠️  Expected response from {self.current_speaker_id}, "
                f"got response from {speaker_id}"
            )
            # Still process it but note the discrepancy
            # In a real system, you might want to handle this differently

        persona = get_persona(speaker_id)
        speaker_name = get_speaker_name(speaker_id)

        logger.info(f"👂 Received response from {speaker_name}: {response_text[:100]}...")

        # Add to history
        self._add_to_history(speaker_id, speaker_name, response_text)

        # Special handling for Introduction phase
        if self.state == PodcastState.INTRO_WAITING:
            logger.info(f"✨ Intro phase: received introduction from speaker {speaker_id}")

            # Check if this is a new participant
            participant = get_persona(speaker_id)
            if not participant or participant.get("name", "").startswith("Unknown"):
                # New participant - extract name and role
                extracted_name = self._extract_name_from_introduction(response_text)
                detected_role = detect_role_from_introduction(response_text)

                if not extracted_name:
                    extracted_name = f"Participant {speaker_id}"

                # Add them to the active participants
                actual_speaker_id = add_participant(
                    name=extracted_name,
                    role=detected_role,
                    voice_id="default"
                )

                # Update the speaker_id reference if needed
                speaker_id = actual_speaker_id
                speaker_name = extracted_name

                logger.info(f"✅ Registered new participant: {extracted_name} as {detected_role}")

                # Send notification to UI
                if self.send_message_callback:
                    await self.send_message_callback({
                        "type": "participant_registered",
                        "speaker_id": speaker_id,
                        "speaker_name": extracted_name,
                        "speaker_role": detected_role,
                        "timestamp": datetime.utcnow().isoformat()
                    })

            # Check for "start" intent
            start_keywords = ["start", "begin", "go ahead", "let's go", "ready", "diving in", "dive in", "let's get started"]
            text_lower = response_text.lower()
            if any(kw in text_lower for kw in start_keywords):
                logger.info("🎬 Participant signalled to start the show!")
                self.state = PodcastState.QUESTIONING
                await self._ask_next_question()
                return

            # Otherwise, keep waiting for others or more introductions
            self.awaiting_response = True
            return

        # Evaluate response
        self.state = PodcastState.EVALUATING
        evaluation = await self.evaluator.evaluate_response(
            question=self.current_question or self.topic,
            response=response_text,
            topic=self.topic,
            speaker_role=persona["role"],
            history=self.conversation_history
        )

        logger.info(
            f"📊 Evaluation: score={evaluation['score']:.2f}, "
            f"on_track={evaluation['is_on_track']}, "
            f"intervention_level={evaluation['intervention_level']}"
        )

        # Send evaluation to UI (for debugging/monitoring)
        if self.send_message_callback:
            await self.send_message_callback({
                "type": "response_evaluation",
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "evaluation": evaluation,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Decide on next action based on evaluation
        intervention_level = evaluation["intervention_level"]

        if intervention_level == 0:
            # Good response, continue
            await self._acknowledge_and_continue()

        elif intervention_level in [1, 2]:
            # Soft or medium redirect
            await self._handle_redirect(
                speaker_id,
                response_text,
                evaluation["intervention_message"]
            )

        elif intervention_level == 3:
            # Polite interrupt
            await self._handle_interrupt(
                speaker_id,
                response_text,
                evaluation["intervention_message"]
            )

        elif intervention_level == 4:
            # Move to different speaker/topic
            await self._handle_topic_switch(
                speaker_id,
                evaluation["intervention_message"]
            )


    async def _acknowledge_and_continue(self):
        """Acknowledge good response and move to next question"""
        # Optional: Send brief acknowledgment
        # For now, just move to next question
        await asyncio.sleep(1)  # Brief pause
        self.state = PodcastState.QUESTIONING
        await self._ask_next_question()

    async def _handle_redirect(self, speaker_id: str, response: str, message: str):
        """Handle soft redirect for partially off-track response"""
        self.state = PodcastState.INTERRUPTING

        # Send redirect message
        if self.send_message_callback:
            await self.send_message_callback({
                "type": "orchestrator_redirect",
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            })

        self._add_to_history("orchestrator", "Orchestrator", message)

        # Generate redirect question
        persona = get_persona(speaker_id)
        redirect_question = await self.question_generator.generate_redirect_question(
            original_question=self.current_question,
            off_track_response=response,
            topic=self.topic,
            speaker_persona=persona
        )

        # Ask redirect question to same speaker
        self.current_question = redirect_question
        self.awaiting_response = True
        self.state = PodcastState.LISTENING

        if self.send_message_callback:
            await self.send_message_callback({
                "type": "orchestrator_question",
                "speaker_id": speaker_id,
                "speaker_name": persona["name"],
                "speaker_role": persona["role"],
                "question": redirect_question,
                "is_redirect": True,
                "turn_number": self.turn_count,
                "timestamp": datetime.utcnow().isoformat()
            })

        self._add_to_history("orchestrator", "Orchestrator", f"[To {persona['name']}] {redirect_question}")

    async def _handle_interrupt(self, speaker_id: str, response: str, message: str):
        """Handle polite interrupt for rambling/off-track response"""
        self.state = PodcastState.INTERRUPTING

        # Send interrupt message
        if self.send_message_callback:
            await self.send_message_callback({
                "type": "orchestrator_interrupt",
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            })

        self._add_to_history("orchestrator", "Orchestrator", message)

        # Move to next speaker
        await asyncio.sleep(1)
        self.state = PodcastState.QUESTIONING
        await self._ask_next_question()

    async def _handle_topic_switch(self, speaker_id: str, message: str):
        """Handle topic switch when response is completely off-track"""
        self.state = PodcastState.INTERRUPTING

        # Send switch message
        if self.send_message_callback:
            await self.send_message_callback({
                "type": "orchestrator_topic_switch",
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            })

        self._add_to_history("orchestrator", "Orchestrator", message)

        # Move to next speaker
        await asyncio.sleep(1)
        self.state = PodcastState.QUESTIONING
        await self._ask_next_question()

    async def _wrap_up_podcast(self):
        """Generate wrap-up and end podcast"""
        self.state = PodcastState.WRAPPING_UP
        logger.info("🎬 Wrapping up podcast...")

        # Generate wrap-up
        wrap_up = await self._generate_wrap_up()

        # Send wrap-up to UI
        if self.send_message_callback:
            await self.send_message_callback({
                "type": "orchestrator_wrapup",
                "text": wrap_up,
                "total_turns": self.turn_count,
                "speaker_turns": self.speaker_turn_counts,
                "timestamp": datetime.utcnow().isoformat()
            })

        self._add_to_history("orchestrator", "Orchestrator", wrap_up)

        self.state = PodcastState.ENDED
        logger.info("✅ Podcast ended")

    async def _generate_wrap_up(self) -> str:
        """Generate podcast wrap-up"""
        # Summarize conversation
        recent_context = self._format_conversation_for_summary()

        prompt = f"""Generate a brief podcast wrap-up.

TOPIC: {self.topic}

CONVERSATION SUMMARY:
{recent_context}

Generate a warm closing (2-3 sentences) that:
1. Thanks the speakers by name
2. Summarizes key takeaways
3. Closes the discussion professionally
4. Write as natural speech (this will be spoken aloud).
5. NO markdown, hashtags, or special formatting.

Return ONLY the wrap-up, no preamble or formatting."""

        try:
            wrap_up = str(self.agent(prompt)).strip()
            wrap_up = wrap_up.strip('"').strip("'")
            return wrap_up
        except Exception as e:
            logger.error(f"❌ Wrap-up generation failed: {e}")
            return f"Thank you to all our speakers for this engaging discussion on {self.topic}. We covered a lot of ground today. Until next time!"

    def _format_conversation_for_summary(self) -> str:
        """Format conversation history for wrap-up summary"""
        # Take key entries, not all
        if len(self.conversation_history) > 10:
            # Take first 3, last 5, and some middle
            summary_entries = (
                self.conversation_history[:3] +
                ["..."] +
                self.conversation_history[-5:]
            )
        else:
            summary_entries = self.conversation_history

        lines = []
        for entry in summary_entries:
            if entry == "...":
                lines.append("...")
            else:
                lines.append(f"[{entry['speaker_name']}]: {entry['text'][:100]}...")

        return "\n".join(lines)

    def _extract_name_from_introduction(self, text: str) -> Optional[str]:
        """
        Try to extract speaker name from their introduction
        Looks for patterns like "I'm X", "My name is X", "This is X"
        """
        import re

        patterns = [
            r"(?:I'm|I am|my name is|this is|call me|i'm|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"^([A-Z][a-z]+)\s+(?:here|speaking|from)",
            r"([A-Z][a-z]+)\s+from\s+(?:AWS|ShellKode|the team|Amazon)",
            r"(?:hi|hello|hey),?\s+(?:this is|I'm|I am)\s+([A-Z][a-z]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Validate it's likely a name (not a common word)
                common_words = ['hello', 'thanks', 'great', 'yes', 'sure', 'okay', 'thank', 'good']
                name_first_word = name.split()[0].lower()
                if name_first_word not in common_words and len(name) > 2:
                    # Capitalize properly
                    return ' '.join(word.capitalize() for word in name.split())

        return None

    def _add_to_history(self, speaker_id: str, speaker_name: str, text: str):
        """Add entry to conversation history"""
        entry = {
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
            "turn": self.turn_count
        }
        self.conversation_history.append(entry)

    def get_state_info(self) -> Dict:
        """Get current orchestrator state"""
        participants = get_all_participants()
        turn_counts = {pid: p["turn_count"] for pid, p in participants.items()}

        return {
            "state": self.state.value,
            "topic": self.topic,
            "turn_count": self.turn_count,
            "max_turns": self.max_turns,
            "current_speaker_id": self.current_speaker_id,
            "current_question": self.current_question,
            "awaiting_response": self.awaiting_response,
            "active_participants": len(participants),
            "participant_turn_counts": turn_counts,
            "conversation_length": len(self.conversation_history)
        }
