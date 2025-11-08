from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "AI Service Hub"
    VERSION: str = "1.0.0"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

settings = Settings()
