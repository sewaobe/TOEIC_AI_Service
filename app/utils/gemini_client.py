import os
from typing import List, Tuple

import google.generativeai as genai

from app.core.config import settings
from app.schemas.chat_schema import ChatMessage, Mistake

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)


GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-flash-latest",
]


def build_prompt_from_messages(messages: List[ChatMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        if msg.role == "system":
            prefix = "System"
        elif msg.role == "user":
            prefix = "User"
        else:
            prefix = "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    return "\n".join(lines) + "\nAssistant:"


def generate_gemini_response(
    messages: List[ChatMessage], target_lang: str = "vi"
) -> Tuple[str, str]:
    """Generate a conversational bot reply and its translation.

    Returns (bot_text_en, bot_translation_target_lang).
    """

    # Extract system message if present (from chat_router.py)
    system_context = ""
    conversation_messages = messages
    if messages and messages[0].role == "system":
        system_context = messages[0].content
        conversation_messages = messages[1:]

    conversation_prompt = build_prompt_from_messages(conversation_messages)

    # Build a strong system instruction that enforces the scenario
    system_instruction = (
        "You are a friendly English conversation partner for language learners. "
    )

    if system_context:
        system_instruction += (
            f"\n\nIMPORTANT CONTEXT AND CONSTRAINTS:\n{system_context}\n\n"
            "You MUST stay within the defined scenario and role. "
            "If the learner tries to change the topic or scenario, "
            "politely redirect them back to the current scenario. "
            "For example: 'Let's focus on [current scenario] for now. How can I help you with that?'\n\n"
        )

    system_instruction += (
        "Continue the conversation naturally in English based on the dialogue above. "
        "After generating the reply in English, also provide a translation of your reply "
        f"into the target language (language code: {target_lang}). "
        "Return a strict JSON object only, no extra text, in this exact format: "
        '{"bot_text": string, "bot_translation": string}. '
        "Do NOT use markdown or code fences. Respond with a plain JSON object only."
    )

    prompt = system_instruction + "\n\nConversation so far:\n" + conversation_prompt

    last_error: Exception | None = None
    raw_text = "{}"
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            raw_text = response.text or "{}"
            break
        except Exception as e:
            # Log and try next model
            print(f"[generate_gemini_response] Error with model {model_name}:", e)
            last_error = e

    if last_error and raw_text == "{}":
        print("[generate_gemini_response] All Gemini models failed.", last_error)
        return (
            "I'm having trouble responding right now. Please try again.",
            "Tôi đang gặp sự cố khi trả lời. Vui lòng thử lại sau.",
        )

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Extract JSON object from response (handle cases where Gemini adds extra text)
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        cleaned = cleaned[json_start : json_end + 1]

    import json

    try:
        data = json.loads(cleaned)
        bot_text = str(data.get("bot_text", "")).strip()
        bot_translation = str(data.get("bot_translation", "")).strip()
        if not bot_text:
            bot_text = "I'm having trouble responding right now. Please try again."
        if not bot_translation:
            bot_translation = "Tôi đang gặp sự cố khi trả lời. Vui lòng thử lại sau."
        return bot_text, bot_translation
    except Exception as parse_error:
        print("[generate_gemini_response] JSON parse error:", parse_error)
        print("[generate_gemini_response] raw response:", raw_text)
        return (
            "I'm having trouble responding right now. Please try again.",
            "Tôi đang gặp sự cố khi trả lời. Vui lòng thử lại sau.",
        )


def grammar_feedback_from_gemini(
    transcript: str,
    source_lang: str = "en",
    target_lang: str = "vi",
) -> Tuple[float, str, List[Mistake], str]:
    """Ask Gemini to grade grammar, list mistakes, and provide a user-side translation.

    Returns (grammar_score_0_100, improvement_tip, mistakes_list, user_translation).
    If parsing fails, falls back to neutral values.
    """

    if not transcript.strip():
        return 0.0, "No transcript provided for grammar evaluation.", [], ""

    system_instruction = (
        "You are an English speaking and grammar coach. "
        "The student sentence comes from SPOKEN English, not written text. "
        "Evaluate only errors that would be clear when listening to speech: grammar (tense, word order, missing auxiliaries), "
        "word choice, and naturalness for oral communication. "
        "Do NOT penalize or mention purely written punctuation issues such as commas, question marks, or capitalization "
        "if the sentence is otherwise clear and natural when spoken. "
        "Also translate the student's sentence into the target language. "
        "Return a strict JSON object only, no extra text, in this exact format: "
        '{"grammar_score": number (0-100), '
        '"overall_feedback": string, '
        '"mistakes": ['
        '{"original": string, "correction": string, "type": "grammar"|"vocabulary"|"pronunciation", "explanation": string}'
        "], "
        '"user_translation": string'
        "}. "
        "Do NOT use markdown or code fences. Respond with a plain JSON object only."
    )

    prompt = (
        system_instruction
        + "\n\nStudent sentence: "
        + transcript
        + f"\nSource language code: {source_lang}"
        + f"\nTarget language code: {target_lang}"
        + "\n\nRemember: respond with JSON only."
    )

    try:
        raw_text = "{}"

        last_error: Exception | None = None
        for model_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                raw_text = response.text or "{}"
                break
            except Exception as e:
                print(
                    f"[grammar_feedback_from_gemini] Error with model {model_name}:", e
                )
                last_error = e

        if last_error and (raw_text == "{}"):
            # All models failed completely
            print(
                "[grammar_feedback_from_gemini] All Gemini models failed.", last_error
            )
            return 0.0, "No grammar feedback due to a model error.", [], ""

        # Handle Gemini sometimes wrapping JSON in markdown code fences ```json ... ```
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # drop first line (``` or ```json)
            lines = lines[1:]
            # drop last line if it's a closing ```
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Extract JSON object from response (handle cases where Gemini adds extra text)
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            cleaned = cleaned[json_start : json_end + 1]

        import json

        try:
            data = json.loads(cleaned)
        except Exception as parse_error:
            # Log raw response for debugging when JSON is invalid
            print("[grammar_feedback_from_gemini] JSON parse error:", parse_error)
            print("[grammar_feedback_from_gemini] raw response:", raw_text)
            # Fallback: neutral score with textual feedback
            return 50.0, "I could not reliably parse the grammar feedback.", [], ""

        grammar_score = float(data.get("grammar_score", 0.0))
        overall_feedback = str(data.get("overall_feedback", ""))
        user_translation = str(data.get("user_translation", ""))

        mistakes_raw = data.get("mistakes", []) or []
        mistakes: List[Mistake] = []
        for m in mistakes_raw:
            try:
                mistakes.append(
                    Mistake(
                        original=str(m.get("original", "")),
                        correction=str(m.get("correction", "")),
                        type=str(m.get("type", "grammar")),
                        explanation=str(m.get("explanation", "")),
                    )
                )
            except Exception:
                # Skip malformed entries
                continue

        # Clamp grammar_score to 0-100
        grammar_score = max(0.0, min(100.0, grammar_score))

        return (
            grammar_score,
            overall_feedback,
            mistakes,
            user_translation,
        )
    except Exception as e:
        # Fallback if LLM call fails completely (network, auth, etc.)
        print("[grammar_feedback_from_gemini] LLM call error:", e)
        return 0.0, "No grammar feedback due to a model error.", [], ""
