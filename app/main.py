from fastapi import FastAPI

from app.core.config import settings
from app.core.router_register import register_routers

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION, root_path="/api/v1")

register_routers(app)


@app.get("/")
def root():
    return {"message": f"Welcome to {settings.APP_NAME}!"}
