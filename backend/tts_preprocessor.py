"""
Text preprocessing for TTS to ensure natural speech output
"""
import re
import logging

logger = logging.getLogger("tts_preprocessor")

def preprocess_for_tts(text: str) -> str:
    """
    Preprocess text to make it TTS-friendly

    Transformations:
    1. Uppercase acronyms → Letter-by-letter (AWS → A W S)
    2. Remove hashtags
    3. Remove markdown formatting
    4. Clean up special characters
    5. Normalize spacing

    Args:
        text: Raw text from orchestrator/agent

    Returns:
        TTS-friendly text
    """
    if not text:
        return ""

    original_text = text

    # 1. Convert uppercase acronyms to letter-by-letter
    # Matches 2+ consecutive uppercase letters (optionally followed by 's')
    # AWS, API, AI, JSON, REST, etc.
    def replace_acronym(match):
        acronym = match.group(0)
        # Check if it ends with 's' (plural)
        if acronym.endswith('s') and len(acronym) > 2 and acronym[-2].isupper():
            # Handle plural: APIs → A P Is
            base = acronym[:-1]
            letters = ' '.join(base)
            return f"{letters}s"
        else:
            # Regular acronym: AWS → A W S
            letters = ' '.join(acronym)
            return letters

    # Match uppercase acronyms (2+ letters)
    text = re.sub(r'\b[A-Z]{2,}s?\b', replace_acronym, text)

    # 2. Remove hashtags but keep the word
    # #AWS → AWS (which will already be converted to A W S above)
    text = re.sub(r'#(\w+)', r'\1', text)

    # 3. Remove markdown formatting
    # **bold** → bold
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # *italic* → italic
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # __underline__ → underline
    text = re.sub(r'__(.+?)__', r'\1', text)
    # `code` → code
    text = re.sub(r'`(.+?)`', r'\1', text)

    # 4. Remove markdown headers
    # ## Header → Header
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # 5. Remove markdown lists
    # - item → item
    # * item → item
    # 1. item → item
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # 6. Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # 7. Clean up special characters that don't sound good
    # Keep: . , ! ? ' " - ()
    # Remove: @ # $ % ^ & * _ + = [ ] { } | \ / < > ~
    text = re.sub(r'[_+=\[\]{}|\\/<>~@$%^&*]', ' ', text)

    # 8. Normalize spacing
    # Multiple spaces → single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Log if significant changes were made
    if text != original_text:
        logger.debug(f"TTS preprocessed:\n  Before: {original_text[:100]}\n  After:  {text[:100]}")

    return text


def format_question_for_speech(speaker_name: str, question: str, speaker_role: str = None) -> str:
    """
    Format a question to include the speaker's name naturally in speech

    Args:
        speaker_name: Name of the person being asked
        question: The question text
        speaker_role: Optional role (e.g., "AWS Expert")

    Returns:
        Speech-friendly question addressing the speaker

    Examples:
        format_question_for_speech("Varun", "What is the best practice?", "AWS Expert")
        → "Varun, as an AWS expert, what is the best practice?"

        format_question_for_speech("Varun", "What do you think?")
        → "Varun, what do you think?"
    """
    question = question.strip()

    # Remove any [To Name] prefix if present
    question = re.sub(r'^\[To\s+[^\]]+\]\s*', '', question, flags=re.IGNORECASE)

    # If question already starts with the name, return as-is
    if question.lower().startswith(speaker_name.lower()):
        return preprocess_for_tts(question)

    # Build the addressed question
    if speaker_role:
        # Clean up role for speech
        role_lower = speaker_role.lower()

        # Special handling for common roles
        if "expert" in role_lower:
            prefix = f"{speaker_name}, as an {speaker_role.lower()}"
        elif "developer" in role_lower or "host" in role_lower:
            prefix = f"{speaker_name}, as a {speaker_role.lower()}"
        else:
            prefix = f"{speaker_name}"
    else:
        prefix = speaker_name

    # Combine prefix with question
    # Make sure question starts with lowercase unless it's a proper noun
    if question and question[0].isupper() and not question.split()[0][0].isupper():
        question = question[0].lower() + question[1:]

    addressed_question = f"{prefix}, {question}"

    return preprocess_for_tts(addressed_question)


# Common acronyms that should be spelled out (for reference/testing)
COMMON_ACRONYMS = [
    "AWS", "API", "AI", "ML", "DL", "NLP", "SQL", "NoSQL",
    "REST", "HTTP", "HTTPS", "JSON", "XML", "YAML",
    "CI", "CD", "DevOps", "IoT", "SaaS", "PaaS", "IaaS",
    "VPC", "EC2", "S3", "RDS", "IAM", "ARN",
    "SDK", "CLI", "UI", "UX", "CSS", "HTML", "JS",
    "CPU", "GPU", "RAM", "SSD", "HDD",
    "OS", "VM", "VPN", "DNS", "CDN", "SSL", "TLS"
]


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "AWS is great for serverless",
        "The API uses REST and JSON",
        "Configure your VPC and EC2 instances",
        "**Important**: Use #AWS best practices",
        "APIs are scalable",
        "The AI/ML pipeline uses S3",
        "[To Varun] What's your view on AWS Lambda?",
        "Visit https://aws.amazon.com for more info",
        "## Section 1\n- Point 1\n- Point 2",
    ]

    print("TTS Preprocessing Tests:\n")
    for test in test_cases:
        result = preprocess_for_tts(test)
        print(f"Input:  {test}")
        print(f"Output: {result}")
        print()

    print("\nQuestion Formatting Tests:\n")
    question_tests = [
        ("Varun", "What is the best AWS practice?", "AWS Expert"),
        ("Varun", "How do you approach API design?", "Developer"),
        ("Varun", "What do you think?", None),
    ]

    for name, question, role in question_tests:
        result = format_question_for_speech(name, question, role)
        print(f"Name: {name}, Role: {role}")
        print(f"Question: {question}")
        print(f"Speech:   {result}")
        print()
