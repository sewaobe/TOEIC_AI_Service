from fastapi import APIRouter

from app.schemas.chat_schema import (
    ChatMessage,
    ChatTurnRequest,
    ChatTurnResponse,
    Feedback,
    Mistake,
    TurnResponse,
)
from app.services import (
    build_system_message_from_config,
    calculate_transcript_similarity,
    get_tts_parameters,
)
from app.utils.azure_speech import (
    decode_base64_to_wav,
    pronunciation_assessment_from_file,
    stt_from_audio_file,
    tts_to_wav_base64,
)
from app.utils.gemini_client import (
    generate_gemini_response,
    grammar_feedback_from_gemini,
)
from app.utils.intonation_utils import calculate_intonation_score, pitch_per_word

router = APIRouter(prefix="/chat", tags=["Chat / Speaking Conversation"])


@router.post("/turn", response_model=ChatTurnResponse)
async def process_chat_turn(req: ChatTurnRequest) -> ChatTurnResponse:
    # 1. Decode audio and run Azure STT
    audio_path = decode_base64_to_wav(req.audio_base64)
    azure_transcript = stt_from_audio_file(audio_path, language="en-US").strip()

    # 2. Choose transcript to use, based on similarity with user_transcript
    user_txt = (req.user_transcript or "").strip()
    chosen_transcript = azure_transcript
    is_unintelligible = False

    if user_txt:
        sim = calculate_transcript_similarity(user_txt, azure_transcript)
        if sim < 0.9:
            # low match -> mark unintelligible, skip heavy scoring
            is_unintelligible = True
        else:
            chosen_transcript = azure_transcript

    # 3. If unintelligible -> return quick feedback, no PA/intonation/LLM
    if is_unintelligible or not chosen_transcript:
        feedback = Feedback(
            pronunciation_score=0,
            fluency_score=0,
            intonation_score=0,
            grammar_score=0,
            total_score=0,
            improvement_tip="I couldn't clearly understand this sentence. Please speak a bit slower and more clearly.",
            mistakes=[],
        )
        turn = TurnResponse(
            feedback=feedback,
            bot_text="I had trouble understanding you. Could you repeat that more slowly?",
            bot_translation="Tôi hơi khó nghe rõ. Bạn có thể nói lại chậm hơn không?",
            user_transcript=azure_transcript or user_txt,
            user_translation=None,
            is_unintelligible=True,
        )
        return ChatTurnResponse(turn=turn)

    # 4. Pronunciation Assessment using Azure
    accuracy, completeness, fluency, per_word_eval, final_words = (
        pronunciation_assessment_from_file(
            filename=audio_path,
            language="en-US",
            reference_text=chosen_transcript,
        )
    )

    # 5. Calculate intonation score based on pitch range and variation (native comparison)
    intonation_score = calculate_intonation_score(audio_path)

    # 6. Build LLM (Gemini) feedback prompt based on speaking_evaluation.py style
    # 6a. Ask Gemini for grammar feedback and user translation on the chosen transcript.
    (
        grammar_score,
        grammar_tip,
        mistakes,
        user_translation,
    ) = grammar_feedback_from_gemini(
        chosen_transcript,
        source_lang="en",
        target_lang="vi",
    )

    # 6b. Get conversational bot reply from Gemini using full context, config, plus its translation.
    messages = list(req.context)
    messages.append(ChatMessage(role="user", content=chosen_transcript))

    # Prepend a system message synthesized from config if provided
    if req.config:
        system_message = build_system_message_from_config(req.config)
        messages.insert(0, system_message)

    bot_text, bot_translation = generate_gemini_response(messages, target_lang="vi")

    # 7. Map to Feedback structure: Azure scores + grammar + normalized intonation.
    # Total score is a simple average of pronunciation, fluency, grammar, and intonation.
    pronunciation_score = round(float(accuracy), 1)
    fluency_score = round(float(fluency), 1)
    grammar_score = round(float(grammar_score), 1)
    intonation_score = round(float(intonation_score), 1)

    total_score = round(
        (pronunciation_score + fluency_score + grammar_score + intonation_score) / 4.0,
        1,
    )

    feedback = Feedback(
        pronunciation_score=pronunciation_score,
        fluency_score=fluency_score,
        intonation_score=intonation_score,
        grammar_score=grammar_score,
        total_score=total_score,
        improvement_tip=grammar_tip or bot_text,
        mistakes=mistakes,
    )

    # 8. Generate Azure TTS audio for the bot's reply (WAV -> base64)
    try:
        voice, speaking_rate = get_tts_parameters(req.config)
        bot_audio_base64 = tts_to_wav_base64(
            bot_text, voice=voice, speaking_rate=speaking_rate
        )
    except Exception as e:
        print("[process_chat_turn] TTS error:", e)
        bot_audio_base64 = None

    turn = TurnResponse(
        feedback=feedback,
        bot_text=bot_text,
        bot_translation=bot_translation,
        user_transcript=chosen_transcript,
        user_translation=user_translation,
        is_unintelligible=False,
        bot_audio_base64=bot_audio_base64,
    )

    return ChatTurnResponse(turn=turn)
