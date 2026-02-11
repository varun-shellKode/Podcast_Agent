"""
Response evaluator for detecting off-track responses and quality issues
"""
import logging
import re
from typing import Dict, List, Optional
from strands.models import BedrockModel

logger = logging.getLogger("response_evaluator")

class ResponseEvaluator:
    """
    Evaluates speaker responses for relevance, quality, and alignment with questions
    """

    # Thresholds for response evaluation
    RELEVANCE_THRESHOLD = 0.4  # Lowered from 0.6 to be more lenient
    MIN_SUBSTANCE_LENGTH = 20  # Minimum characters for substantial response
    MAX_RAMBLE_RATIO = 3.0  # Max response length / question length ratio

    def __init__(self, model: BedrockModel):
        self.model = model

    async def evaluate_response(
        self,
        question: str,
        response: str,
        topic: str,
        speaker_role: str,
        history: List[Dict] = None
    ) -> Dict:
        """
        Evaluate a speaker's response for quality and relevance
        """
        issues = []
        score = 1.0

        # 1. Length checks
        response_length = len(response.strip())
        question_length = len(question.strip())

        if response_length < self.MIN_SUBSTANCE_LENGTH:
            issues.append("too_short")
            score -= 0.3

        if response_length > question_length * self.MAX_RAMBLE_RATIO:
            issues.append("too_long")
            score -= 0.2

        # 2. Semantic relevance check (using LLM)
        relevance_score = await self._check_semantic_relevance(
            question, response, topic, speaker_role, history
        )

        if relevance_score < 0.2:
            issues.append("completely_off_track")
            score -= 0.5
        elif relevance_score < self.RELEVANCE_THRESHOLD:
            issues.append("partially_off_track")
            score -= 0.3

        # 3. Quality checks
        quality_issues = self._check_response_quality(response)
        issues.extend(quality_issues)
        score -= len(quality_issues) * 0.1

        # Clamp score to 0-1
        score = max(0.0, min(1.0, score))

        # Determine if on track
        is_on_track = score >= self.RELEVANCE_THRESHOLD and "completely_off_track" not in issues

        # Determine intervention level
        intervention_level = self._determine_intervention_level(score, issues)
        intervention_message = self._generate_intervention_message(
            intervention_level, question, response, issues, topic
        )

        logger.info(
            f"📊 Response evaluation: score={score:.2f}, "
            f"on_track={is_on_track}, issues={issues}, "
            f"intervention_level={intervention_level}"
        )

        return {
            "score": score,
            "relevance_score": relevance_score,
            "is_on_track": is_on_track,
            "issues": issues,
            "intervention_level": intervention_level,
            "intervention_message": intervention_message
        }

    async def _check_semantic_relevance(
        self,
        question: str,
        response: str,
        topic: str,
        speaker_role: str,
        history: List[Dict] = None
    ) -> float:
        """
        Use LLM to check semantic relevance of response to question
        Returns: 0.0 (irrelevant) to 1.0 (highly relevant)
        """
        # Format history for context
        history_context = ""
        if history:
            recent = history[-3:]
            history_lines = []
            for entry in recent:
                name = entry.get("speaker_name", "Unknown")
                text = entry.get("text", "")
                history_lines.append(f"[{name}]: {text[:150]}...")
            history_context = "\n".join(history_lines)

        prompt = f"""You are evaluating a podcast response for relevance.
        
PODCAST TOPIC: {topic}
SPEAKER ROLE: {speaker_role}

RECENT CONVERSATION:
{history_context}

QUESTION ASKED: {question}
SPEAKER RESPONSE: {response}

Rate how relevant and on-topic the response is to the question asked, CONSIDERING THE CONTEXT of the recent conversation. 
Sometimes a response might address a point from the conversation rather than the direct question—that's okay as long as it's not a complete derailment.

Respond with ONLY a number between 0.0 and 1.0:
- 1.0 = Perfectly relevant, directly answers question
- 0.8 = Mostly relevant, good contextual answer
- 0.5 = Partially relevant, major tangent but still somewhat related
- 0.2 = Barely relevant, mostly off-track
- 0.0 = Completely irrelevant

Your score (0.0-1.0):"""

        try:
            from strands import Agent
            agent = Agent(model=self.model, system_prompt="You are a response evaluator.")
            result = str(agent(prompt)).strip()

            # Extract number from result
            match = re.search(r'(\d+\.?\d*)', result)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5

        except Exception as e:
            logger.error(f"❌ Relevance check failed: {e}")
            return 0.5

    def _check_response_quality(self, response: str) -> List[str]:
        issues = []
        response_lower = response.lower()

        # Check for vague/generic responses
        vague_indicators = ["i think maybe", "it depends", "not sure", "probably"]
        vague_count = sum(1 for indicator in vague_indicators if indicator in response_lower)
        if vague_count >= 2:
            issues.append("too_vague")

        # Check for repetitive content
        words = response.split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                issues.append("repetitive")

        # Check for filler words (excessive)
        filler_words = ["um", "uh", "like", "you know"]
        filler_count = sum(response_lower.count(filler) for filler in filler_words)
        if filler_count > len(words) * 0.15: # 15% threshold
            issues.append("excessive_filler")

        return issues

    def _determine_intervention_level(self, score: float, issues: List[str]) -> int:
        """
        Determine intervention level based on score and issues
        """
        # Level 0: No intervention (score >= 0.6)
        if score >= 0.6 and not issues:
            return 0

        # Level 4: Complete off track
        if "completely_off_track" in issues or score < 0.2:
            return 4

        # Level 3: Long/Ramble
        if "too_long" in issues or "repetitive" in issues:
            return 3

        # Level 2: Redirect (score < 0.4)
        if score < 0.4:
            return 2

        # Level 1: Soft nudges (minor stuff)
        if issues and score < 0.6:
            return 1

        return 0

    def _generate_intervention_message(
        self,
        level: int,
        question: str,
        response: str,
        issues: List[str],
        topic: str
    ) -> str:
        if level == 0:
            return None

        if level == 1:
            return f"That's interesting. To build on that, could you elaborate more specifically on the core question?"

        if level == 2:
            tangent = self._extract_tangent_topic(response)
            return (
                f"I appreciate that perspective on {tangent}. "
                f"Let me refocus the question to stay on our topic of {topic}: "
                f"{question}"
            )

        if level == 3:
            return f"Let me pause you there - I want to make sure we cover the key points. Could you give us a more concise take on: {question}"

        if level == 4:
            return f"I think we've explored that angle. Let me bring in another perspective to address {topic} from a different angle."

        return None

    def _extract_tangent_topic(self, response: str) -> str:
        # Recursive stripping of fillers AND intro phrases
        clean_response = response.strip()
        
        # Expanded fillers including "I think", "I would say", etc.
        intro_phrases = [
            r'^(?:um|uh|well|so|thanks|thank you|actually|you know|i mean|basically)(?:[\s,]+)?',
            r'^(?:i (?:think|feel|believe|would say|guess)|it (?:seems|feels)|to be honest|honestly)(?:[\s,]+)?',
            r'^(?:first|always|in the beginning|before starting)(?:[\s,]+)?',
            r'^(?:that\'s a great question|thanks for the intro)(?:[\s,]+)?'
        ]
        
        changed = True
        while changed:
            original = clean_response
            for pattern in intro_phrases:
                clean_response = re.sub(pattern, '', clean_response, count=1, flags=re.IGNORECASE).strip()
            changed = clean_response != original

        words = clean_response.split()
        if len(words) > 5:
            return " ".join(words[:5]) + "..."
        return clean_response[:50] + "..." if len(clean_response) > 50 else clean_response
