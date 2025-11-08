from sentence_transformers import SentenceTransformer, util

from app.core.config import settings


class SentenceEvaluator:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def evaluate(self, student_def: str, correct_def: str):
        emb1 = self.model.encode(student_def, convert_to_tensor=True)
        emb2 = self.model.encode(correct_def, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()

        if score >= 0.85:
            feedback = "Excellent! Your definition captures the full meaning."
        elif score >= 0.65:
            feedback = (
                "Good attempt! You understand the main idea, but refine the details."
            )
        else:
            feedback = "Not quite right. Try focusing on the core concept of the word."

        return {"similarity": round(score, 3), "feedback": feedback}
