from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AI Service Hub"
    VERSION: str = "1.0.0"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Azure Speech
    AZURE_SPEECH_KEY: str = ""
    AZURE_SPEECH_REGION: str = ""

    # Gemini
    GEMINI_API_KEY: str = ""

    # DeepSeek (OpenAI-compatible API)
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"

    # Sentry DSN
    SENTRY_DSN: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
