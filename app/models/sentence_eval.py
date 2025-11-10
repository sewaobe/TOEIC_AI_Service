import re

from sentence_transformers import SentenceTransformer, util

from app.core.config import settings


class SentenceEvaluator:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def normalize(self, text: str):
        text = text.lower().strip()
        text = re.sub(r"[^a-zA-ZÀ-ỹ0-9,\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def split_definition(self, definition: str):
        if "||" in definition:
            parts = [part.strip() for part in definition.split("||") if part.strip()]
        else:
            # Fallback: tách theo dấu câu thông thường (trường hợp dữ liệu cũ)
            parts = re.split(r"[.,;:/\-]", definition)
            parts = [p.strip() for p in parts if p.strip()]
        return parts

    def cosine(self, vec1, vec2):
        emb1 = self.model.encode(vec1, convert_to_tensor=True)
        emb2 = self.model.encode(vec2, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

    def evaluate(self, student_def: str, correct_def: str):
        student_text = self.normalize(student_def)
        correct_texts = [
            self.normalize(part) for part in self.split_definition(correct_def)
        ]

        scores = [
            self.cosine(student_text, correct_text) for correct_text in correct_texts
        ]

        best_score = max(scores) if scores else 0.0

        if best_score >= 0.85:
            feedback = "Rất xuất sắc! Định nghĩa của bạn thể hiện đầy đủ và chính xác ý nghĩa của từ."
        elif best_score >= 0.65:
            feedback = "Nỗ lực tốt! Bạn đã hiểu ý chính, nhưng hãy chỉnh sửa lại một số chi tiết."
        else:
            feedback = "Chưa chính xác lắm. Hãy tập trung vào khái niệm cốt lõi của từ."

        return {
            "similarity": round(best_score, 3),
            "feedback": feedback,
            "sub_scores": [
                {
                    "definition_part": correct_text,
                    "similarity": round(score, 3),
                }
                for correct_text, score in zip(correct_texts, scores)
            ],
        }
