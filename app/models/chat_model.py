import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.schemas.chat_schema import ChatInitConfig, ChatMessage


@dataclass
class ChatState:
    session_id: str
    title: str
    type: str
    config: ChatInitConfig
    messages: List[ChatMessage] = field(default_factory=list)
    user_name: Optional[str] = None


class InMemoryChatStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, ChatState] = {}

    def create_session(
        self, title: str, type_: str, config: ChatInitConfig
    ) -> ChatState:
        session_id = str(uuid.uuid4())
        state = ChatState(session_id=session_id, title=title, type=type_, config=config)

        system_parts: list[str] = []
        if config.scenario:
            system_parts.append(f"Scenario: {config.scenario}.")
        if config.level:
            system_parts.append(f"Learner level: {config.level}.")
        if config.user_role:
            system_parts.append(f"User role: {config.user_role}.")
        if config.bot_tone:
            system_parts.append(f"Bot tone: {config.bot_tone}.")
        if config.goal:
            system_parts.append(f"Goal: {config.goal}.")

        base_instruction = (
            "You are an English speaking practice partner. "
            "Keep responses concise and conversational. "
            "Always respond in English, and encourage the learner."
        )

        system_content = base_instruction + " " + " ".join(system_parts)
        state.messages.append(ChatMessage(role="system", content=system_content))

        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[ChatState]:
        return self._sessions.get(session_id)


chat_store = InMemoryChatStore()
