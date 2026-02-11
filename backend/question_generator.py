"""
Question generator for creating contextual questions based on topic and persona
"""
import logging
from typing import List, Dict, Optional
from strands.models import BedrockModel
from strands import Agent

logger = logging.getLogger("question_generator")

class QuestionGenerator:
    """
    Generates contextual questions for podcast speakers based on:
    - Main topic
    - Speaker persona
    - Conversation history
    - Uncovered areas
    """

    def __init__(self, model: BedrockModel):
        self.model = model
        self.agent = Agent(
            model=model,
            system_prompt=self._build_system_prompt()
        )

    def _build_system_prompt(self) -> str:
        return """You are a professional podcast host question generator.

Your role is to create engaging, contextual questions that:
1. Are tailored to the speaker's expertise and role
2. Keep the conversation focused on the main topic
3. Explore different angles and perspectives
4. Build on previous discussion points
5. Are clear, concise, and open-ended

Generate questions that encourage substantive responses while keeping the podcast flowing naturally."""

    async def generate_question(
        self,
        topic: str,
        speaker_id: str,
        speaker_persona: Dict,
        conversation_history: List[Dict],
        uncovered_areas: List[str] = None,
        ask_for_introduction: bool = False
    ) -> str:
        """
        Generate a contextual question for the speaker

        Args:
            topic: Main podcast topic
            speaker_id: ID of speaker to ask
            speaker_persona: Persona dict with role, expertise, etc.
            conversation_history: Recent conversation entries
            uncovered_areas: Topics not yet covered
            ask_for_introduction: If True, prompt speaker to introduce themselves

        Returns:
            Generated question string
        """
        # Build context from conversation history
        recent_context = self._format_conversation_history(conversation_history, last_n=5)

        # Build persona context
        persona_context = self._format_persona_context(speaker_persona)

        # Build uncovered areas context
        uncovered_context = ""
        if uncovered_areas:
            uncovered_context = f"\n\nAreas not yet covered:\n" + "\n".join(f"- {area}" for area in uncovered_areas)

        # Get speaker display name (role if name not known)
        from speaker_personas import get_speaker_name
        speaker_display = get_speaker_name(speaker_id)

        # Add introduction prompt if this is first question
        intro_instruction = ""
        if ask_for_introduction:
            intro_instruction = "\n6. Ask them to briefly introduce themselves (name and background) before answering"

        prompt = f"""Generate a question for the podcast.

MAIN TOPIC: {topic}

SPEAKER TO ASK: {speaker_display} ({speaker_persona['role']})
{persona_context}

RECENT CONVERSATION:
{recent_context}{uncovered_context}

Generate ONE clear, engaging question that:
1. Is tailored to the {speaker_persona['role']}'s expertise
2. Relates to the main topic: {topic}
3. Builds on or explores a new angle from the recent conversation
4. Is open-ended and encourages a substantive response
5. Is concise (1-2 sentences maximum){intro_instruction}

Return ONLY the question, no preamble or explanation."""

        try:
            result = str(self.agent(prompt)).strip()
            # Clean up any extra formatting
            question = result.strip('"').strip("'").strip()

            logger.info(f"✨ Generated question for {speaker_persona['name']}: {question}")
            return question

        except Exception as e:
            logger.error(f"❌ Question generation failed: {e}", exc_info=True)
            # Fallback to generic question
            return self._generate_fallback_question(topic, speaker_persona)

    async def generate_followup_question(
        self,
        original_question: str,
        response: str,
        topic: str,
        speaker_persona: Dict
    ) -> str:
        """
        Generate a follow-up question based on a previous response
        """
        prompt = f"""Generate a follow-up question for the podcast.

MAIN TOPIC: {topic}

SPEAKER: {speaker_persona['name']} ({speaker_persona['role']})

PREVIOUS QUESTION: {original_question}

SPEAKER'S RESPONSE: {response}

Generate ONE concise follow-up question that:
1. Digs deeper into their response
2. Clarifies or expands on an interesting point they raised
3. Stays relevant to {topic}
4. Is appropriate for {speaker_persona['role']}

Return ONLY the question, no preamble."""

        try:
            result = str(self.agent(prompt)).strip()
            question = result.strip('"').strip("'").strip()

            logger.info(f"✨ Generated follow-up for {speaker_persona['name']}: {question}")
            return question

        except Exception as e:
            logger.error(f"❌ Follow-up generation failed: {e}", exc_info=True)
            return f"Could you elaborate more on that aspect of {topic}?"

    async def generate_redirect_question(
        self,
        original_question: str,
        off_track_response: str,
        topic: str,
        speaker_persona: Dict
    ) -> str:
        """
        Generate a redirect question to bring conversation back on track
        """
        prompt = f"""The speaker went off-track. Generate a polite redirect question.

MAIN TOPIC: {topic}

SPEAKER: {speaker_persona['name']} ({speaker_persona['role']})

ORIGINAL QUESTION: {original_question}

OFF-TRACK RESPONSE: {off_track_response}

Generate a redirect question that:
1. Politely acknowledges their point
2. Refocuses on the original question or {topic}
3. Is diplomatic and maintains good conversation flow
4. Gives them a chance to address the core topic

Return ONLY the redirect question, no preamble."""

        try:
            result = str(self.agent(prompt)).strip()
            question = result.strip('"').strip("'").strip()

            logger.info(f"✨ Generated redirect for {speaker_persona['name']}: {question}")
            return question

        except Exception as e:
            logger.error(f"❌ Redirect generation failed: {e}", exc_info=True)
            return f"Let me refocus the question: {original_question}"

    def _format_conversation_history(self, history: List[Dict], last_n: int = 5) -> str:
        """Format recent conversation history for context"""
        if not history:
            return "(Beginning of conversation)"

        recent = history[-last_n:] if len(history) > last_n else history
        lines = []
        for entry in recent:
            speaker_name = entry.get("speaker_name", "Unknown")
            text = entry.get("text", "")
            lines.append(f"[{speaker_name}]: {text}")

        return "\n".join(lines)

    def _format_persona_context(self, persona: Dict) -> str:
        """Format persona details for context"""
        lines = []

        if "expertise" in persona:
            expertise = ", ".join(persona["expertise"][:3])  # Top 3
            lines.append(f"Expertise: {expertise}")

        if "traits" in persona:
            traits = ", ".join(persona["traits"][:2])  # Top 2
            lines.append(f"Traits: {traits}")

        if "question_style" in persona:
            lines.append(f"Question style: {persona['question_style']}")

        return "\n".join(lines)

    def _generate_fallback_question(self, topic: str, persona: Dict) -> str:
        """Generate a simple fallback question if AI generation fails"""
        role = persona.get("role", "expert")

        fallback_questions = [
            f"What's your perspective on {topic}?",
            f"How would you approach {topic} in your work?",
            f"What challenges do you see with {topic}?",
            f"Can you share your experience with {topic}?",
        ]

        # Simple rotation based on persona id
        persona_id = int(persona.get("id", "0"))
        return fallback_questions[persona_id % len(fallback_questions)]
