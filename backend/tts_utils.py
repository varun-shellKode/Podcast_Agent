"""
Utility functions for TTS text preprocessing
"""
import re
import logging

logger = logging.getLogger("podcast_agent")


def preprocess_text_for_tts(text: str) -> str:
    """
    Preprocess text before sending to TTS to improve pronunciation.

    Handles:
    - Acronyms (AI → A I, API → A P I)
    - URLs (preserve them)
    - Mixed case words (preserve them)
    - Numbers and special characters

    Args:
        text: Raw text from agent

    Returns:
        Preprocessed text optimized for TTS
    """
    if not text:
        return text

    logger.info(f"[TTS Preprocessing] Input: {text}")

    # Split text into words while preserving punctuation
    words = text.split()

    processed_words = []

    for word in words:
        # Separate punctuation from the word
        match = re.match(r'^(.*?)([.,!?;:"\']*)$', word)
        if match:
            word_part = match.group(1)
            punct_part = match.group(2)
        else:
            word_part = word
            punct_part = ""

        # Skip empty words
        if not word_part:
            if punct_part:
                processed_words.append(punct_part)
            continue

        # Check if it's a URL (preserve URLs)
        if re.match(r'https?://', word_part, re.IGNORECASE):
            processed_words.append(word_part + punct_part)
            continue

        # Check if it's an email (preserve emails)
        if '@' in word_part:
            processed_words.append(word_part + punct_part)
            continue

        # Check if it's all uppercase and length > 1 (likely an acronym)
        if word_part.isupper() and len(word_part) > 1:
            # Special cases: Don't split common words that happen to be uppercase
            common_uppercase = {'I', 'A', 'OK'}
            if word_part not in common_uppercase:
                # Split into letters with spaces: AI → A I
                spaced_letters = ' '.join(list(word_part))
                processed_words.append(spaced_letters + punct_part)
                logger.info(f"[TTS Preprocessing] Acronym: {word_part} → {spaced_letters}")
                continue

        # Check if it's camelCase or PascalCase (preserve them)
        if re.match(r'^[a-z]+[A-Z]', word_part) or re.match(r'^[A-Z][a-z]+[A-Z]', word_part):
            processed_words.append(word_part + punct_part)
            continue

        # Check if it's a number with letters (like "3D", "4K")
        if re.match(r'^\d+[A-Z]+$', word_part):
            # Split letters only: 4K → 4 K
            num_match = re.match(r'^(\d+)([A-Z]+)$', word_part)
            if num_match:
                number = num_match.group(1)
                letters = num_match.group(2)
                spaced = number + ' ' + ' '.join(list(letters))
                processed_words.append(spaced + punct_part)
                logger.info(f"[TTS Preprocessing] Number+Letters: {word_part} → {spaced}")
                continue

        # Default: keep the word as is
        processed_words.append(word_part + punct_part)

    result = ' '.join(processed_words)

    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()

    logger.info(f"[TTS Preprocessing] Output: {result}")

    return result


# Test cases
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_cases = [
        "Hey! I'm Cupcake, your AI co-host.",
        "Let's talk about API design and AWS services.",
        "The CEO of the company uses 4K displays.",
        "Visit https://example.com for more info.",
        "Contact us at info@example.com",
        "I love JavaScript and TypeScript!",
        "The USA and UK have different spellings.",
        "This is a test of TTS preprocessing.",
        "NASA launched a new rocket today.",
        "iPhone and iPad are Apple products.",
    ]

    print("=" * 60)
    print("TTS Text Preprocessing Test")
    print("=" * 60)

    for test in test_cases:
        result = preprocess_text_for_tts(test)
        print(f"\nInput:  {test}")
        print(f"Output: {result}")

    print("\n" + "=" * 60)
