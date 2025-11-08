from pydantic import BaseModel


class SentenceEvalRequest(BaseModel):
    student_def: str
    correct_def: str


class SentenceEvalResponse(BaseModel):
    similarity: float
    feedback: str
