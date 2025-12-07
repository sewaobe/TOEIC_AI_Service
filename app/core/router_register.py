from fastapi import FastAPI

from app.routers import chat_router, sentence_eval_router


def register_routers(app: FastAPI):
    app.include_router(sentence_eval_router)
    app.include_router(chat_router)
