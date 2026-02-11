"""
Speaker selector for choosing the most relevant speaker for the next turn
"""
import logging
import json
from typing import List, Dict, Optional
from strands.models import BedrockModel
from strands import Agent

logger = logging.getLogger("speaker_selector")

class SpeakerSelector:
    """
    Selects the next speaker based on:
    - Conversation history
    - Persona expertise
    - Topic flow
    """

    def __init__(self, model: BedrockModel):
        self.model = model
        self.agent = Agent(
            model=model,
            system_prompt=self._build_system_prompt()
        )

    def _build_system_prompt(self) -> str:
        return """You are a podcast director responsible for selecting the next speaker to invite into the conversation.

Your goal is to choose the speaker who can provide the most relevant expertise or perspective based on the current state of the discussion.

Guidelines:
1. Choose someone whose expertise directly addresses the most recent point or question.
2. If the current speaker is diverging, choose someone who can bring it back (often the Host).
3. **User Participation**: Prioritize the Host (Speaker 0) and the Client (Speaker 4) as they represent the human user.
4. If AI agents (1, 2, 3) have spoken recently, it is CRITICAL to involve a human (Speaker 0 or 4) next to maintain a dialogue.
5. If someone hasn't spoken in a while, involve them.

Return ONLY a JSON object with the following format:
{
  "selected_speaker_id": "ID_STRING",
  "reasoning": "Brief explanation of why this speaker was chosen"
}"""

    async def select_next_speaker(
        self,
        topic: str,
        history: List[Dict],
        personas: Dict[str, Dict],
        turn_counts: Dict[str, int],
        last_asked: Dict[str, Optional[int]],
        consecutive_ai_turns: int = 0
    ) -> str:
        """
        Choose the best speaker for the next turn
        """
        # Format context
        formatted_history = self._format_history(history)
        formatted_personas = self._format_personas(personas, turn_counts)
        
        prompt = f"""TOPIC: {topic}

CONSECUTIVE AI TURNS: {consecutive_ai_turns}
(If this is 2 or more, you MUST choose a human: Speaker 0 or 4)

RECENT CONVERSATION:
{formatted_history}

AVAILABLE SPEAKERS:
{formatted_personas}

Which speaker should be invited to talk next? 
Consider who hasn't spoken in a while, but prioritize relevance to the topic.
"""

        try:
            # Get decision from LLM
            # Using __call__ for the agent
            result_str = str(self.agent(prompt)).strip()
            
            # Extract JSON
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0].strip()
            elif "{" in result_str:
                result_str = result_str[result_str.find("{"):result_str.rfind("}")+1]

            decision = json.loads(result_str)
            selected_id = decision.get("selected_speaker_id")
            reasoning = decision.get("reasoning", "No reasoning provided")

            if selected_id and str(selected_id) in personas:
                logger.info(f"🧠 Speaker Selector thinking: {reasoning}")
                logger.info(f"👉 Selected speaker: {selected_id} ({personas[str(selected_id)]['role']})")
                return str(selected_id)
            
            # Fallback to rotation if invalid ID returned
            return self._fallback_choice(personas, turn_counts)

        except Exception as e:
            logger.error(f"❌ Speaker selection failed: {e}", exc_info=True)
            return self._fallback_choice(personas, turn_counts)

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "(Show just started)"
        
        recent = history[-5:]
        lines = []
        for entry in recent:
            name = entry.get("speaker_name", "Unknown")
            text = entry.get("text", "")
            lines.append(f"[{name}]: {text[:200]}...")
        return "\n".join(lines)

    def _format_personas(self, personas: Dict[str, Dict], turn_counts: Dict[str, int]) -> str:
        lines = []
        for sid, p in personas.items():
            expertise = ", ".join(p.get("expertise", []))
            lines.append(f"ID {sid}: {p['role']} (Expertise: {expertise}) - Turns so far: {turn_counts.get(sid, 0)}")
        return "\n".join(lines)

    def _fallback_choice(self, personas: Dict[str, Dict], turn_counts: Dict[str, int]) -> str:
        # Simple rotation fallback
        min_turns = min(turn_counts.values()) if turn_counts else 0
        candidates = [sid for sid, count in turn_counts.items() if count == min_turns]
        return candidates[0] if candidates else list(personas.keys())[0]
