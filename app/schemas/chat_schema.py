from typing import Literal, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SpeakingConfig(BaseModel):
    scenario: Optional[str] = None
    level: Optional[str] = None
    userRole: Optional[str] = None
    botTone: Optional[str] = None
    goal: Optional[str] = None
    durationMinutes: Optional[int] = None
    botSpeed: Optional[Literal["slow", "normal", "fast"]] = None


class ChatTurnRequest(BaseModel):
    # Full conversation context, if you want LLM to see history
    context: list[ChatMessage]
    # Base64-encoded audio for this turn (e.g. wav)
    audio_base64: str
    # Optional transcript from client; will be validated against STT
    user_transcript: Optional[str] = None
    # Optional speaking configuration from the Node service
    config: Optional[SpeakingConfig] = None


class Mistake(BaseModel):
    original: str
    correction: str
    type: Literal["grammar", "vocabulary", "pronunciation"]
    explanation: str


class Feedback(BaseModel):
    pronunciation_score: float
    fluency_score: float
    intonation_score: float
    grammar_score: float
    total_score: float
    improvement_tip: str
    mistakes: list[Mistake]


class TurnResponse(BaseModel):
    feedback: Feedback
    bot_text: str
    bot_translation: str
    user_transcript: str
    user_translation: Optional[str] = None
    is_unintelligible: bool = False
    # Optional WAV audio (base64) synthesized for bot_text
    bot_audio_base64: Optional[str] = None


class ChatTurnResponse(BaseModel):
    turn: TurnResponse
