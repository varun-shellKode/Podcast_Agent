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

from speaker_personas import SPEAKER_PERSONAS, get_persona, get_speaker_name, set_speaker_name
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
        self.consecutive_ai_turns = 0

        # Speaker tracking
        self.speaker_turn_counts = {sid: 0 for sid in SPEAKER_PERSONAS.keys()}
        self.speaker_last_asked = {sid: None for sid in SPEAKER_PERSONAS.keys()}

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
            f"   Speakers: {len(SPEAKER_PERSONAS)}"
        )

    def _build_orchestrator_prompt(self) -> str:
        return f"""You are the AI orchestrator for a technical podcast on the topic: "{self.topic}"

ROLE:
You are an intelligent moderator controlling a 5-speaker podcast:
- Speaker 0: Host
- Speaker 1: AWS Expert
- Speaker 2: ShellKode Developer
- Speaker 3: Tech Generalist
- Speaker 4: Client

Note: Speakers will introduce themselves by name. Address them by their role initially, then use their actual names once introduced.

RESPONSIBILITIES:
1. Generate contextual questions tailored to each speaker's expertise
2. Listen to responses and evaluate relevance
3. Politely interrupt if speakers go off-track
4. Keep conversation flowing and engaging
5. Ensure all speakers contribute meaningfully
6. Cover the topic comprehensively

BEHAVIOR:
- Be professional yet conversational
- Show curiosity and interest
- Redirect diplomatically when needed
- Acknowledge good points before moving on
- Keep the energy and pace appropriate
- When asking first question to a speaker, encourage them to introduce themselves
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

        # Start with host (speaker 0)
        self.state = PodcastState.QUESTIONING
        await self._ask_next_question()

    async def _generate_introduction(self) -> str:
        """Generate podcast introduction"""
        prompt = f"""Generate a brief, engaging podcast introduction.

TOPIC: {self.topic}

Generate a concise, professional introduction (1-2 sentences maximum) that:
1. Welcomes listeners and introduces the topic

Keep it short and punchy. Return ONLY the introduction, no preamble."""

        try:
            intro = str(self.agent(prompt)).strip()
            return intro
        except Exception as e:
            logger.error(f"❌ Introduction generation failed: {e}")
            return f"Welcome to our podcast on {self.topic}. Today we have experts from different backgrounds discussing this topic in depth. Let's dive in!"

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
        is_first_turn = self.speaker_turn_counts[next_speaker_id] == 0

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
        
        # If this is an AI speaker, we don't wait for audio; we generate the response
        if persona.get("is_ai", False):
            self.consecutive_ai_turns += 1
            logger.info(f"🤖 Speaker {next_speaker_id} is AI (Turn {self.consecutive_ai_turns}) - generating response...")
            self.awaiting_response = False
            self.state = PodcastState.LISTENING
            # Use gather or create_task to not block the WebSocket loop too long
            asyncio.create_task(self._handle_ai_speaker_turn(next_speaker_id, question))
        else:
            self.consecutive_ai_turns = 0
            self.awaiting_response = True
            self.state = PodcastState.LISTENING

        # Get display name (role if name not known yet)
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
        self.speaker_turn_counts[next_speaker_id] += 1
        self.speaker_last_asked[next_speaker_id] = self.turn_count

        logger.info(
            f"❓ Turn {self.turn_count}: Asked {display_name} ({persona['role']})\n"
            f"   Question: {question}"
        )

    async def _select_next_speaker(self) -> Optional[str]:
        """
        Select next speaker to ask using the SpeakerSelector agent
        """
        # Force host every 3rd turn if AI has been dominating
        if self.consecutive_ai_turns >= 2:
            logger.info("✋ AI turn limit reached - forcing Host/User participation.")
            return "0"

        # Get decision from speaker selector
        next_speaker_id = await self.speaker_selector.select_next_speaker(
            topic=self.topic,
            history=self.conversation_history,
            personas=SPEAKER_PERSONAS,
            turn_counts=self.speaker_turn_counts,
            last_asked=self.speaker_last_asked,
            consecutive_ai_turns=self.consecutive_ai_turns
        )
        
        return next_speaker_id

    async def _handle_ai_speaker_turn(self, speaker_id: str, question: str):
        """Generate and process a response from an AI guest"""
        if not self.is_active:
            return

        persona = get_persona(speaker_id)
        
        # Build prompt for the guest agent
        prompt = f"""You are {persona['name']}, a {persona['role']} participating in a podcast about "{self.topic}".
        
EXPERTISE: {", ".join(persona.get('expertise', []))}
TRAITS: {", ".join(persona.get('traits', []))}

RECENT CONVERSATION:
{self.question_generator._format_conversation_history(self.conversation_history, last_n=3)}

THE HOST ASKED YOU: "{question}"

Instructions:
1. Provide a substantive, expert response as your persona.
2. Be conversational but stay on topic.
3. Keep it to 3-4 sentences.
4. Don't use preamble like "Certainly" or "As an AWS expert". Just speak.

Your response:"""

        try:
            # Pacing: wait for the question's TTS to finish (roughly)
            # In a more advanced system, we'd wait for a "tts_ended" signal for the question
            # For now, let's wait a bit longer to simulate natural flow
            await asyncio.sleep(8)
            
            if not self.is_active:
                return

            # Use orchestrator agent to generate response (re-sharing the model)
            response_text = str(self.agent(prompt)).strip()
            
            if not self.is_active:
                return

            logger.info(f"✅ AI Guest {speaker_id} ({persona['name']}) responded: {response_text[:50]}...")
            
            # Process this as a normal response
            await self.handle_speaker_response(speaker_id, response_text)
            
        except Exception as e:
            logger.error(f"❌ AI Guest response failed: {e}")
            if self.is_active:
                self.awaiting_response = False
                await self._ask_next_question()

    async def handle_speaker_response(self, speaker_id: str, response_text: str):
        """
        Handle a speaker's response:
        1. Verify it's the expected speaker
        2. Evaluate response quality and relevance
        3. Decide: continue, redirect, interrupt, or move on
        """
        # Reset awaiting state immediately so it can be set again by redirects/interrupts
        self.awaiting_response = False

        if speaker_id != self.current_speaker_id:
            logger.warning(
                f"⚠️  Expected response from {self.current_speaker_id}, "
                f"got response from {speaker_id}"
            )
            # Still process it but note the discrepancy
            # In a real system, you might want to handle this differently

        persona = get_persona(speaker_id)
        speaker_name = get_speaker_name(speaker_id)

        logger.info(f"👂 Received response from {speaker_name}: {response_text[:100]}...")

        # Try to extract name from first response if not yet known
        if persona.get("name") is None and self.speaker_turn_counts[speaker_id] == 1:
            extracted_name = self._extract_name_from_introduction(response_text)
            if extracted_name:
                set_speaker_name(speaker_id, extracted_name)
                speaker_name = extracted_name
                logger.info(f"✅ Learned speaker name: {speaker_id} = {extracted_name}")

        # Add to history
        self._add_to_history(speaker_id, speaker_name, response_text)

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
1. Thanks the speakers
2. Summarizes key takeaways
3. Closes the discussion professionally

Return ONLY the wrap-up, no preamble."""

        try:
            wrap_up = str(self.agent(prompt)).strip()
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
            r"(?:I'm|I am|my name is|this is|call me)\s+([A-Z][a-z]+)",
            r"^([A-Z][a-z]+)\s+(?:here|speaking)",
            r"([A-Z][a-z]+)\s+from\s+(?:AWS|ShellKode|the team)",
        ]

        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Validate it's likely a name (not a common word)
                common_words = ['hello', 'thanks', 'great', 'yes', 'sure', 'okay']
                if name.lower() not in common_words and len(name) > 2:
                    return name.capitalize()

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
        return {
            "state": self.state.value,
            "topic": self.topic,
            "turn_count": self.turn_count,
            "max_turns": self.max_turns,
            "current_speaker_id": self.current_speaker_id,
            "current_question": self.current_question,
            "awaiting_response": self.awaiting_response,
            "speaker_turn_counts": self.speaker_turn_counts,
            "conversation_length": len(self.conversation_history)
        }
