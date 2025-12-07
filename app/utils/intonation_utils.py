from typing import List, Tuple

import numpy as np
import parselmouth
from parselmouth.praat import call


def pitch_per_word(words: List[str], audio_path: str) -> tuple[list[str], float]:
    """Compute simple pitch per word and overall average, similar to intonation.pitch."""
    per_word_pitch: list[str] = []
    sound = parselmouth.Sound(audio_path)
    pitch_obj = sound.to_pitch()
    values = pitch_obj.selected_array["frequency"]
    non_zero = [x for x in values if x != 0]

    for i, w in enumerate(words):
        if i < len(non_zero):
            per_word_pitch.append(f"{w}: {non_zero[i]} Hz")
        else:
            per_word_pitch.append(f"{w}: 0 Hz")

    overall = float(np.mean(non_zero)) if non_zero else 0.0
    return per_word_pitch, overall


def calculate_intonation_score(audio_path: str) -> float:
    """
    Calculate intonation score (0-100) based on pitch range and variation
    compared to native English speaker reference values.

    Uses research-based native reference:
    - Mean F0: 130 Hz
    - Range: 70 Hz (max - min)
    - Std: 30 Hz

    Ideal scores:
    - Pitch range: 80-120% of native
    - Pitch std: 70-130% of native

    Returns:
        float: Intonation score from 0 to 100
    """
    try:
        sound = parselmouth.Sound(audio_path)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)

        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        min_f0 = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        max_f0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        std_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")

        user_range = max_f0 - min_f0

        # Native English speaker reference values (from research)
        native_mean = 130.0  # Hz
        native_range = 70.0  # Hz (max - min)
        native_std = 30.0  # Hz

        # Calculate percentages
        pitch_range_percent = (
            (user_range / native_range) * 100 if native_range > 0 else 0
        )
        pitch_std_percent = (std_f0 / native_std) * 100 if native_std > 0 else 0

        # Convert percentage to 0-100 score
        def _score_from_percent(
            percent: float, ideal_min: float = 80, ideal_max: float = 120
        ) -> float:
            """
            Convert percentage to score with penalties for being outside ideal range.

            Args:
                percent: Percentage value to convert
                ideal_min: Minimum ideal percentage (default 80)
                ideal_max: Maximum ideal percentage (default 120)

            Returns:
                Score from 0 to 100
            """
            if ideal_min <= percent <= ideal_max:
                return 100.0
            elif percent < ideal_min:
                # Too flat: linear penalty
                return max(0, 100 * (percent / ideal_min))
            else:
                # Too varied: gentle penalty
                over = percent - ideal_max
                penalty = min(50, over / 2)  # max 50 point penalty
                return max(0, 100 - penalty)

        range_score = _score_from_percent(pitch_range_percent, 80, 120)
        std_score = _score_from_percent(pitch_std_percent, 70, 130)

        # Weighted average: range is more important than std
        intonation_score = range_score * 0.7 + std_score * 0.3

        return round(intonation_score, 1)

    except Exception as e:
        print(f"[calculate_intonation_score] Error: {e}")
        # Fallback: simple pitch-based scoring
        try:
            sound = parselmouth.Sound(audio_path)
            pitch_obj = sound.to_pitch()
            values = pitch_obj.selected_array["frequency"]
            non_zero = [x for x in values if x != 0]
            overall_pitch = float(np.mean(non_zero)) if non_zero else 0.0

            # Simple normalization
            if overall_pitch <= 0:
                return 0.0
            min_pitch, max_pitch = 80.0, 300.0
            ratio = (overall_pitch - min_pitch) / (max_pitch - min_pitch)
            ratio = max(0.0, min(1.0, ratio))
            return round(ratio * 100.0, 1)
        except Exception:
            return 0.0
