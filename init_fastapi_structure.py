import os

folders = [
    "app/core",
    "app/models",
    "app/routers",
    "app/schemas",
    "app/utils",
    "data",
]

files = {
    "app/__init__.py": "",
    "app/core/__init__.py": "",
    "app/models/__init__.py": "",
    "app/routers/__init__.py": "",
    "app/schemas/__init__.py": "",
    "app/utils/__init__.py": "",
    "app/core/config.py": """from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "AI Service Hub"
    VERSION: str = "1.0.0"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

settings = Settings()
""",
    "app/core/router_register.py": """from fastapi import FastAPI
from app.routers import sentence_eval_router

def register_routers(app: FastAPI):
    app.include_router(sentence_eval_router.router)
""",
    "app/main.py": """from fastapi import FastAPI
from app.core.config import settings
from app.core.router_register import register_routers

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

register_routers(app)

@app.get("/")
def root():
    return {"message": f"Welcome to {settings.APP_NAME}!"}
""",
    "run.py": """import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
""",
    "requirements.txt": "fastapi\nuvicorn\nsentence-transformers\nnumpy\n",
}

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ FastAPI project structure created successfully!")
