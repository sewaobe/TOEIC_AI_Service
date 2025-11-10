from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.router_register import register_routers

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION, root_path="/api/v1")

origins = ["http://localhost:5173", "http://localhost:5174"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_routers(app)


@app.get("/")
def root():
    return {"message": f"Welcome to {settings.APP_NAME}!"}
