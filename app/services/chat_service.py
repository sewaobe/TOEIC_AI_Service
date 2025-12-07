"""
Business logic for chat/speaking conversation processing.
Separated from routers for better maintainability and testability.
"""

from difflib import SequenceMatcher
from typing import Optional, Tuple

from app.schemas.chat_schema import ChatMessage, SpeakingConfig


def calculate_transcript_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two transcripts using SequenceMatcher.

    Args:
        text1: First transcript
        text2: Second transcript

    Returns:
        Similarity ratio from 0.0 to 1.0
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def build_system_message_from_config(config: SpeakingConfig) -> ChatMessage:
    """
    Build a system message from speaking configuration.

    Args:
        config: Speaking session configuration

    Returns:
        ChatMessage with role="system" containing configuration context
    """
    system_parts: list[str] = [
        "You are an English speaking partner helping the learner practice conversation.",
    ]

    if config.scenario:
        system_parts.append(f"Scenario: {config.scenario}.")
    if config.level:
        system_parts.append(f"Learner level: {config.level}.")
    if config.userRole:
        system_parts.append(f"Learner role: {config.userRole}.")
    if config.botTone:
        system_parts.append(f"Your tone: {config.botTone}.")
    if config.goal:
        system_parts.append(f"Conversation goal: {config.goal}.")
    if config.durationMinutes is not None:
        system_parts.append(f"Target duration: {config.durationMinutes} minutes.")
    if config.botSpeed:
        system_parts.append(f"Your speaking speed should be: {config.botSpeed}.")

    return ChatMessage(role="system", content=" ".join(system_parts))


def get_tts_voice_from_tone(bot_tone: str) -> str:
    """
    Map bot tone to appropriate Azure TTS voice.

    Args:
        bot_tone: One of the predefined bot tones

    Returns:
        Azure TTS voice name
    """
    tone_to_voice = {
        "Friendly & Encouraging": "en-US-JennyNeural",  # warm, friendly female
        "Professional & Formal": "en-US-GuyNeural",  # professional male
        "Strict & Correction-focused": "en-US-DavisNeural",  # authoritative male
        "Funny & Casual": "en-US-JasonNeural",  # casual, energetic male
        "Fast Native Speaker": "en-US-SaraNeural",  # clear, fast-paced female
    }

    return tone_to_voice.get(bot_tone, "en-US-AriaNeural")  # default: friendly female


def get_speaking_rate_from_config(config: SpeakingConfig) -> float:
    """
    Calculate speaking rate from configuration.

    Args:
        config: Speaking session configuration

    Returns:
        Speaking rate multiplier (0.8 for slow, 1.0 for normal, 1.2 for fast)
    """
    if not config or not config.botSpeed:
        return 1.0

    speed_to_rate = {
        "slow": 0.8,
        "normal": 1.0,
        "fast": 1.2,
    }

    base_rate = speed_to_rate.get(config.botSpeed, 1.0)

    # Boost rate for "Fast Native Speaker" tone
    if config.botTone == "Fast Native Speaker":
        base_rate = max(base_rate, 1.1)

    return base_rate


def get_tts_parameters(config: Optional[SpeakingConfig]) -> Tuple[str, float]:
    """
    Get TTS voice and speaking rate from configuration.

    Args:
        config: Speaking session configuration (optional)

    Returns:
        Tuple of (voice_name, speaking_rate)
    """
    if not config:
        return "en-US-AriaNeural", 1.0

    voice = (
        get_tts_voice_from_tone(config.botTone)
        if config.botTone
        else "en-US-AriaNeural"
    )
    speaking_rate = get_speaking_rate_from_config(config)

    return voice, speaking_rate
