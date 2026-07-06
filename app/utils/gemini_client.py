import json
from typing import Any, List, Tuple

import google.generativeai as genai

from app.core.config import settings
from app.schemas.chat_schema import (
    ChatMessage,
    GrammarBreakdownItem,
    Mistake,
    VocabSuggestion,
)

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

DEFAULT_BOT_TEXT = "I'm having trouble responding right now. Please try again."
DEFAULT_BOT_TRANSLATION = "Tôi đang gặp sự cố khi trả lời. Vui lòng thử lại sau."
DEFAULT_GRAMMAR_ERROR = "No grammar feedback due to a model error."


def _clean_json_text(raw_text: str) -> str:
    cleaned = (raw_text or "{}").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        cleaned = cleaned[json_start : json_end + 1]

    return cleaned


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    data = json.loads(_clean_json_text(raw_text))
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object.")
    return data


def _generate_deepseek_json(prompt: str, caller: str, max_tokens: int) -> dict[str, Any]:
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY.")

    try:
        from openai import OpenAI
    except Exception as import_error:
        raise RuntimeError("Python package 'openai' is not installed.") from import_error

    client = OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model=settings.DEEPSEEK_MODEL or "deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "Return one valid JSON object only. Do not use markdown, "
                    "code fences, comments, or explanatory text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=max_tokens,
    )
    raw_text = response.choices[0].message.content or "{}"
    print(f"[{caller}] DeepSeek succeeded with model {settings.DEEPSEEK_MODEL}.")
    return _parse_json_object(raw_text)


def _conversation_reply_from_data(data: dict[str, Any]) -> Tuple[str, str]:
    bot_text = str(data.get("bot_text", "")).strip()
    bot_translation = str(data.get("bot_translation", "")).strip()
    if not bot_text:
        bot_text = DEFAULT_BOT_TEXT
    if not bot_translation:
        bot_translation = DEFAULT_BOT_TRANSLATION
    return bot_text, bot_translation


def _grammar_feedback_from_data(
    data: dict[str, Any],
) -> Tuple[
    float, str, List[Mistake], str, List[VocabSuggestion], List[GrammarBreakdownItem]
]:
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
            continue

    vocab_raw = data.get("vocab_suggestions", []) or []
    vocab_suggestions: List[VocabSuggestion] = []
    for v in vocab_raw:
        try:
            alternatives = v.get("alternatives", [])
            if not isinstance(alternatives, list):
                alternatives = [str(alternatives)]
            vocab_suggestions.append(
                VocabSuggestion(
                    word=str(v.get("word", "")),
                    context=str(v.get("context", "")),
                    alternatives=[str(item) for item in alternatives],
                )
            )
        except Exception:
            continue

    grammar_breakdown_raw = data.get("grammar_breakdown", []) or []
    grammar_breakdown: List[GrammarBreakdownItem] = []
    for g in grammar_breakdown_raw:
        try:
            status = str(g.get("status", "Needs Improvement"))
            if status not in ["Correct", "Needs Improvement"]:
                status = "Needs Improvement"
            grammar_breakdown.append(
                GrammarBreakdownItem(
                    structure=str(g.get("structure", "")),
                    example=str(g.get("example", "")),
                    advice=str(g.get("advice", "")),
                    status=status,
                )
            )
        except Exception:
            continue

    grammar_score = max(0.0, min(100.0, grammar_score))
    return (
        grammar_score,
        overall_feedback,
        mistakes,
        user_translation,
        vocab_suggestions,
        grammar_breakdown,
    )


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

    system_context = ""
    conversation_messages = messages
    if messages and messages[0].role == "system":
        system_context = messages[0].content
        conversation_messages = messages[1:]

    conversation_prompt = build_prompt_from_messages(conversation_messages)

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
            print(f"[generate_gemini_response] Error with model {model_name}:", e)
            last_error = e

    if last_error and raw_text == "{}":
        print("[generate_gemini_response] All Gemini models failed.", last_error)
        try:
            data = _generate_deepseek_json(
                prompt,
                caller="generate_gemini_response",
                max_tokens=1200,
            )
            return _conversation_reply_from_data(data)
        except Exception as deepseek_error:
            print("[generate_gemini_response] DeepSeek fallback failed.", deepseek_error)
            return DEFAULT_BOT_TEXT, DEFAULT_BOT_TRANSLATION

    try:
        data = _parse_json_object(raw_text)
        return _conversation_reply_from_data(data)
    except Exception as parse_error:
        print("[generate_gemini_response] JSON parse error:", parse_error)
        print("[generate_gemini_response] raw response:", raw_text)
        try:
            data = _generate_deepseek_json(
                prompt,
                caller="generate_gemini_response",
                max_tokens=1200,
            )
            return _conversation_reply_from_data(data)
        except Exception as deepseek_error:
            print("[generate_gemini_response] DeepSeek fallback failed.", deepseek_error)
            return DEFAULT_BOT_TEXT, DEFAULT_BOT_TRANSLATION

def grammar_feedback_from_gemini(
    transcript: str,
    source_lang: str = "en",
    target_lang: str = "vi",
) -> Tuple[
    float, str, List[Mistake], str, List[VocabSuggestion], List[GrammarBreakdownItem]
]:
    """Ask Gemini to grade grammar, list mistakes, provide vocabulary suggestions,
    grammar breakdown, and a user-side translation.

    Returns (grammar_score_0_100, improvement_tip, mistakes_list, user_translation,
             vocab_suggestions, grammar_breakdown).
    If parsing fails, falls back to neutral values.
    """

    if not transcript.strip():
        return 0.0, "No transcript provided for grammar evaluation.", [], "", [], []

    system_instruction = (
        "You are an English speaking and grammar coach. "
        "The student sentence comes from SPOKEN English, not written text. "
        "Evaluate only errors that would be clear when listening to speech: grammar (tense, word order, missing auxiliaries), "
        "word choice, and naturalness for oral communication. "
        "Do NOT penalize or mention purely written punctuation issues such as commas, question marks, or capitalization "
        "if the sentence is otherwise clear and natural when spoken. "
        "Also translate the student's sentence into the target language. "
        "Additionally, provide vocabulary suggestions to help the student diversify their word choices, "
        "and analyze any grammar structures they used (correctly or incorrectly). "
        "Return a strict JSON object only, no extra text, in this exact format: "
        '{"grammar_score": number (0-100), '
        '"overall_feedback": string, '
        '"mistakes": ['
        '{"original": string, "correction": string, "type": "grammar"|"vocabulary"|"pronunciation", "explanation": string}'
        "], "
        '"user_translation": string, '
        '"vocab_suggestions": ['
        '{"word": string (the word/phrase student used), "context": string (how they used it), "alternatives": [string] (better/diverse alternatives)}'
        "], "
        '"grammar_breakdown": ['
        '{"structure": string (grammar structure name like "Present Perfect", "Conditional"), "example": string (student sentence using this structure), "advice": string (improvement advice), "status": "Correct"|"Needs Improvement"}'
        "]"
        "}. "
        "For vocab_suggestions: identify common or repetitive words the student used and suggest better alternatives. "
        "For grammar_breakdown: identify 1-3 main grammar structures the student attempted and evaluate them. "
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
                    f"[grammar_feedback_from_gemini] Error with model {model_name}:",
                    e,
                )
                last_error = e

        if last_error and raw_text == "{}":
            print("[grammar_feedback_from_gemini] All Gemini models failed.", last_error)
            try:
                data = _generate_deepseek_json(
                    prompt,
                    caller="grammar_feedback_from_gemini",
                    max_tokens=2400,
                )
                return _grammar_feedback_from_data(data)
            except Exception as deepseek_error:
                print(
                    "[grammar_feedback_from_gemini] DeepSeek fallback failed.",
                    deepseek_error,
                )
                return 0.0, DEFAULT_GRAMMAR_ERROR, [], "", [], []

        try:
            data = _parse_json_object(raw_text)
        except Exception as parse_error:
            print("[grammar_feedback_from_gemini] JSON parse error:", parse_error)
            print("[grammar_feedback_from_gemini] raw response:", raw_text)
            try:
                data = _generate_deepseek_json(
                    prompt,
                    caller="grammar_feedback_from_gemini",
                    max_tokens=2400,
                )
                return _grammar_feedback_from_data(data)
            except Exception as deepseek_error:
                print(
                    "[grammar_feedback_from_gemini] DeepSeek fallback failed.",
                    deepseek_error,
                )
                return (
                    50.0,
                    "I could not reliably parse the grammar feedback.",
                    [],
                    "",
                    [],
                    [],
                )

        return _grammar_feedback_from_data(data)
    except Exception as e:
        print("[grammar_feedback_from_gemini] LLM call error:", e)
        return 0.0, DEFAULT_GRAMMAR_ERROR, [], "", [], []
