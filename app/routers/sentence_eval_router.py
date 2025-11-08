from fastapi import APIRouter

from app.models import SentenceEvaluator
from app.schemas import SentenceEvalRequest, SentenceEvalResponse

router = APIRouter(prefix="/sentence-eval", tags=["Sentence Evaluation"])
model = SentenceEvaluator()


@router.post("/", response_model=SentenceEvalResponse)
async def evaluate_sentence(req: SentenceEvalRequest):
    result = model.evaluate(req.student_def, req.correct_def)
    return SentenceEvalResponse(**result)
