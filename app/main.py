import sentry_sdk

from app.core.config import settings

# Initialize Sentry if DSN is provided
sentry_sdk.init(
    dsn=settings.SENTRY_DSN,
    # Bật tính năng đo lường hiệu suất API (Tracing)
    traces_sample_rate=1.0,
    # Bật tính năng theo dõi xem hàm nào chạy chậm (Profiling)
    profiles_sample_rate=1.0,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.router_register import register_routers

# Initialize FastAPI app with settings
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


@app.get("/healthy")
def health_check():
    return {"status": "healthy"}


@app.get("/debug-sentry")
def debug_sentry():
    error = 1 / 0  # This will raise a ZeroDivisionError and be captured by Sentry
    return {"message": "Debug message sent to Sentry"}
