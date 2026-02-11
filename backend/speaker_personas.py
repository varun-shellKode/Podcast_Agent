"""
Dynamic speaker persona management for podcast orchestrator
Speakers are discovered dynamically as they join and introduce themselves.
"""
import logging
from typing import Dict, Optional, List

logger = logging.getLogger("speaker_personas")

# Role templates - NOT individual speakers
# These define what expertise/traits we expect from each role
ROLE_TEMPLATES = {
    "Host": {
        "description": "Podcast host who moderates the discussion",
        "traits": ["Asks clarifying questions", "Keeps conversation on track", "Summarizes key points"],
        "expertise": ["Facilitation", "Topic overview"],
        "question_style": "Open-ended questions that encourage discussion"
    },
    "AWS Expert": {
        "description": "Cloud/AWS solutions expert",
        "traits": ["Deep AWS knowledge", "Scalability focus", "Cost optimization mindset"],
        "expertise": ["AWS services", "Cloud architecture", "Serverless", "Lambda"],
        "question_style": "Technical and architecture-focused"
    },
    "Developer": {
        "description": "Software developer or engineer",
        "traits": ["Implementation-focused", "Code quality conscious", "Practical problem solver"],
        "expertise": ["Software development", "Backend systems", "API design"],
        "question_style": "Hands-on and implementation-focused"
    },
    "Tech Expert": {
        "description": "General technology expert",
        "traits": ["Broad tech knowledge", "Full-stack perspective", "Pragmatic"],
        "expertise": ["System architecture", "Technology trends", "Integration"],
        "question_style": "Balanced between technical depth and practical use"
    },
    "Business": {
        "description": "Business stakeholder or client",
        "traits": ["Business value focused", "Timeline conscious", "ROI aware"],
        "expertise": ["Business strategy", "Project management", "Value delivery"],
        "question_style": "Business-oriented and outcome-focused"
    },
    "Participant": {
        "description": "General participant",
        "traits": ["Engaged", "Curious"],
        "expertise": ["General knowledge"],
        "question_style": "Open and exploratory"
    }
}

# Active participants in the current podcast session
# Key: speaker_id (assigned dynamically), Value: participant info
active_participants: Dict[str, Dict] = {}

# Counter for assigning IDs
_next_speaker_id = 1

def reset_participants():
    """Reset all participants (for new podcast session)"""
    global active_participants, _next_speaker_id
    active_participants = {}
    _next_speaker_id = 1
    logger.info("🔄 Reset all participants")

def add_participant(name: str, role: str = "Participant", voice_id: str = "default") -> str:
    """
    Add a new participant to the podcast

    Args:
        name: Speaker's name
        role: Their role (must be a key in ROLE_TEMPLATES)
        voice_id: Voice ID for TTS

    Returns:
        Assigned speaker_id
    """
    global _next_speaker_id

    # Validate role
    if role not in ROLE_TEMPLATES:
        logger.warning(f"⚠️ Unknown role '{role}', defaulting to 'Participant'")
        role = "Participant"

    speaker_id = str(_next_speaker_id)
    _next_speaker_id += 1

    # Get role template
    template = ROLE_TEMPLATES[role]

    # Create participant
    active_participants[speaker_id] = {
        "id": speaker_id,
        "name": name,
        "role": role,
        "voice_id": voice_id,
        "description": template["description"],
        "traits": template["traits"],
        "expertise": template["expertise"],
        "question_style": template["question_style"],
        "turn_count": 0,
        "last_turn": None
    }

    logger.info(f"✅ Added participant: {name} (ID: {speaker_id}, Role: {role})")
    return speaker_id

def get_participant(speaker_id: str) -> Optional[Dict]:
    """Get participant by ID"""
    return active_participants.get(speaker_id)

def get_all_participants() -> Dict[str, Dict]:
    """Get all active participants"""
    return active_participants.copy()

def get_participant_by_name(name: str) -> Optional[Dict]:
    """Get participant by name (case-insensitive)"""
    name_lower = name.lower()
    for participant in active_participants.values():
        if participant["name"].lower() == name_lower:
            return participant
    return None

def update_participant_role(speaker_id: str, new_role: str) -> bool:
    """Update a participant's role"""
    if speaker_id not in active_participants:
        return False

    if new_role not in ROLE_TEMPLATES:
        logger.warning(f"⚠️ Unknown role '{new_role}'")
        return False

    template = ROLE_TEMPLATES[new_role]
    participant = active_participants[speaker_id]

    participant["role"] = new_role
    participant["description"] = template["description"]
    participant["traits"] = template["traits"]
    participant["expertise"] = template["expertise"]
    participant["question_style"] = template["question_style"]

    logger.info(f"✅ Updated {participant['name']}'s role to {new_role}")
    return True

def increment_turn_count(speaker_id: str, turn_number: int):
    """Increment turn count for a participant"""
    if speaker_id in active_participants:
        active_participants[speaker_id]["turn_count"] += 1
        active_participants[speaker_id]["last_turn"] = turn_number

def detect_role_from_introduction(text: str) -> str:
    """
    Detect role from introduction text

    Examples:
        "I am from AWS" → "AWS Expert"
        "I am a developer" → "Developer"
        "I represent the business side" → "Business"
    """
    text_lower = text.lower()

    # AWS related
    if any(keyword in text_lower for keyword in ["aws", "amazon web services", "cloud architect"]):
        return "AWS Expert"

    # Developer related
    if any(keyword in text_lower for keyword in ["developer", "engineer", "programmer", "coder", "shellkode"]):
        return "Developer"

    # Business related
    if any(keyword in text_lower for keyword in ["business", "client", "stakeholder", "manager", "product owner"]):
        return "Business"

    # Tech expert
    if any(keyword in text_lower for keyword in ["tech", "technology", "architect", "consultant", "specialist", "expert"]):
        return "Tech Expert"

    # Host (usually won't need this as host is predefined)
    if any(keyword in text_lower for keyword in ["host", "moderator", "facilitator"]):
        return "Host"

    # Default
    return "Participant"

def get_speaker_name(speaker_id: str) -> str:
    """Get speaker name by ID"""
    participant = get_participant(speaker_id)
    if participant:
        return participant["name"]
    return f"Speaker {speaker_id}"

def get_speaker_role(speaker_id: str) -> str:
    """Get speaker role by ID"""
    participant = get_participant(speaker_id)
    if participant:
        return participant["role"]
    return "Participant"

# Backward compatibility aliases
def get_persona(speaker_id: str) -> Dict:
    """Get participant (backward compatibility)"""
    participant = get_participant(speaker_id)
    if participant:
        return participant
    # Return empty participant for unknown IDs
    return {
        "id": speaker_id,
        "name": f"Unknown {speaker_id}",
        "role": "Participant",
        "traits": [],
        "expertise": []
    }

def set_speaker_name(speaker_id: str, name: str):
    """Set speaker name (backward compatibility)"""
    if speaker_id in active_participants:
        active_participants[speaker_id]["name"] = name
        logger.info(f"✅ Updated speaker {speaker_id} name to {name}")
        return True
    return False
