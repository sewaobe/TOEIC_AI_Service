from pydantic import BaseModel


class SentenceEvalRequest(BaseModel):
    student_def: str
    correct_def: str


class SubScore(BaseModel):
    definition_part: str
    similarity: float


class SentenceEvalResponse(BaseModel):
    similarity: float
    feedback: str
    sub_scores: list[SubScore]
