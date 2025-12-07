import base64
from pathlib import Path
from typing import Tuple

import azure.cognitiveservices.speech as speechsdk

from app.core.config import settings


def _get_speech_config() -> speechsdk.SpeechConfig:
    if not settings.AZURE_SPEECH_KEY or not settings.AZURE_SPEECH_REGION:
        raise RuntimeError("AZURE_SPEECH_KEY or AZURE_SPEECH_REGION not set")
    speech_config = speechsdk.SpeechConfig(
        settings.AZURE_SPEECH_KEY, settings.AZURE_SPEECH_REGION
    )
    speech_config.speech_recognition_language = "en-US"
    return speech_config


def stt_from_audio_file(path: str, language: str = "en-US") -> str:
    speech_config = _get_speech_config()
    speech_config.speech_recognition_language = language
    audio_config = speechsdk.audio.AudioConfig(filename=path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    raise RuntimeError(f"STT failed: {result.reason}")


def decode_base64_to_wav(audio_base64: str, out_dir: str = "tmp_audio") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    audio_bytes = base64.b64decode(audio_base64)
    # simple unique name
    path = Path(out_dir) / "speaking_turn.wav"
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return str(path)


def pronunciation_assessment_from_file(
    filename: str,
    language: str,
    reference_text: str,
) -> Tuple[float, float, float, list[str], list]:
    """Wrapper around Azure PronunciationAssessment like pronounce_assessment_file.pronunciation_assessment_continuous_from_file.
    Returns (accuracy, completeness, fluency, per_word_eval, final_words).
    """
    import difflib
    import json
    import string
    import time

    speech_config = speechsdk.SpeechConfig(
        subscription=settings.AZURE_SPEECH_KEY, region=settings.AZURE_SPEECH_REGION
    )
    audio_config = speechsdk.audio.AudioConfig(filename=filename)

    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True,
    )

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        language=language,
        audio_config=audio_config,
    )
    pronunciation_config.apply_to(recognizer)

    done = False
    recognized_words = []
    fluency_scores = []
    durations = []

    def stop_cb(evt):
        nonlocal done
        done = True

    def recognized(evt):
        nonlocal recognized_words, fluency_scores, durations
        pronunciation_result = speechsdk.PronunciationAssessmentResult(evt.result)
        recognized_words += pronunciation_result.words
        fluency_scores.append(pronunciation_result.fluency_score)
        json_result = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceResponse_JsonResult
        )
        jo = json.loads(json_result)
        nb = jo["NBest"][0]
        durations.append(sum(int(w["Duration"]) for w in nb["Words"]))

    recognizer.recognized.connect(recognized)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)

    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.5)
    recognizer.stop_continuous_recognition()

    reference_words = [
        w.strip(string.punctuation) for w in reference_text.lower().split()
    ]
    diff = difflib.SequenceMatcher(
        None, reference_words, [x.word.lower() for x in recognized_words]
    )
    final_words = []
    for tag, i1, i2, j1, j2 in diff.get_opcodes():
        if tag in ["insert", "replace"]:
            for word in recognized_words[j1:j2]:
                if word.error_type == "None":
                    word._error_type = "Insertion"
                final_words.append(word)
        if tag in ["delete", "replace"]:
            for word_text in reference_words[i1:i2]:
                word = speechsdk.PronunciationAssessmentWordResult(
                    {
                        "Word": word_text,
                        "PronunciationAssessment": {"ErrorType": "Omission"},
                    }
                )
                final_words.append(word)
        if tag == "equal":
            final_words += recognized_words[j1:j2]

    final_accuracy_scores = [
        word.accuracy_score for word in final_words if word.error_type != "Insertion"
    ]
    accuracy_score = sum(final_accuracy_scores) / len(final_accuracy_scores)
    fluency_score = sum(x * y for x, y in zip(fluency_scores, durations)) / sum(
        durations
    )
    completeness_score = (
        len([w for w in recognized_words if w.error_type == "None"])
        / len(reference_words)
        * 100
    )
    completeness_score = completeness_score if completeness_score <= 100 else 100

    per_word_eval: list[str] = []
    for idx, word in enumerate(final_words):
        per_word_eval.append(
            f"word {idx + 1}: {word.word}, accuracy: {word.accuracy_score} error type: {word.error_type}"
        )

    return accuracy_score, completeness_score, fluency_score, per_word_eval, final_words


import base64
import io
import os
import time

import azure.cognitiveservices.speech as speechsdk


def tts_to_wav_base64(
    text: str, voice: str = "en-US-AriaNeural", speaking_rate: float = 1.0
) -> str:
    """
    Synthesize text to WAV audio and return as base64.
    Saves to tmp_audio folder to avoid playing on server speaker.
    """
    print("[tts_to_wav_base64] START, len(text) =", len(text))

    speech_config = _get_speech_config()
    speech_config.speech_synthesis_voice_name = voice

    # Lưu vào folder tmp_audio với tên unique
    output_dir = Path("tmp_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_dir / f"tts_{int(time.time() * 1000)}.wav"

    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(tmp_path))
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    if abs(speaking_rate - 1.0) < 1e-3:
        result = synthesizer.speak_text_async(text).get()
    else:
        ssml = f"""
        <speak version="1.0" xml:lang="en-US">
          <voice name="{voice}">
            <prosody rate="{speaking_rate:.2f}">
              {text}
            </prosody>
          </voice>
        </speak>
        """
        result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Đọc file WAV vừa tạo
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        print("[tts_to_wav_base64] SUCCESS, audio bytes =", len(audio_bytes))

        # # Xóa file sau khi đọc xong (optional, có thể bỏ để debug)
        # try:
        #     os.remove(tmp_path)
        # except Exception as e:
        #     print(f"[tts_to_wav_base64] Warning: Could not delete {tmp_path}: {e}")

        return base64.b64encode(audio_bytes).decode("utf-8")

    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(
            f"[tts_to_wav_base64] CANCELED: {details.reason}, {details.error_details}"
        )
        raise RuntimeError(f"TTS canceled: {details.reason}, {details.error_details}")

    raise RuntimeError(f"TTS failed with reason: {result.reason}")
