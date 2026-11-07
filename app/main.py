"""
CheckHadir.Id — Enterprise Face Recognition Attendance System
FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.timezone import WIB, now_wib
from app.core.firebase import initialize_firebase, is_firebase_ready, test_firestore_connection
from app.api.auth import router as auth_router
from app.api.employees import router as employees_router
from app.api.attendance import router as attendance_router
from app.api.face import router as face_router, admin_face_approvals_router
from app.api.settings import router as settings_router
from app.api.reports import router as reports_router
from app.api.leaves import router as leaves_router
from app.api.audit import router as audit_router
from datetime import datetime
import logging

logging.Formatter.converter = lambda *args: datetime.now(WIB).timetuple()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s WIB - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("checkhadir-api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: refuse to boot with insecure production configuration
    problems = settings.validate_production_config()
    if problems:
        for p in problems:
            logger.critical(f"INSECURE PRODUCTION CONFIG: {p}")
        raise RuntimeError("Refusing to start: " + " | ".join(problems))

    # Startup: Initialize Firebase Admin SDK & test connectivity
    logger.info(f"Initializing {settings.APP_NAME} for {settings.ORGANIZATION_NAME}...")
    initialize_firebase()
    if settings.is_production and not is_firebase_ready():
        raise RuntimeError("Refusing to start in production without live Firebase Admin credentials.")
    if is_firebase_ready():
        diag = test_firestore_connection()
        if diag.get("connected"):
            logger.info(
                f"Firebase Admin SDK successfully connected to Cloud Firestore on project '{diag.get('projectId')}'! "
                f"Latency: {diag.get('latencyMs')}ms"
            )
        else:
            logger.warning(f"Firebase initialized but Firestore test reported: {diag.get('error')}")
    else:
        logger.warning(
            "Firebase Admin SDK running in local development mode. "
            "Place serviceAccountKey.json in backend/ to connect to live Firebase."
        )
    yield
    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}...")

# Interactive API docs expose the full attack surface; keep them off in production
app = FastAPI(
    title=f"{settings.APP_NAME} API",
    version=settings.APP_VERSION,
    description=f"Enterprise Attendance System for {settings.ORGANIZATION_NAME} using Server-Side Face Recognition.",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

# Configure Cross-Origin Resource Sharing (CORS) - explicit methods/headers only
allowed_origins = list(set(settings.CORS_ORIGINS + [
    "https://paudlabum-checkhadir.web.app",
    "https://paudlabum-checkhadir.firebaseapp.com",
    "https://employee-attendance-syst-39d40.web.app",
    "https://employee-attendance-syst-39d40.firebaseapp.com",
    "http://localhost:5173",
]))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Total-Count"],
)
# Largest accepted request body (face samples are base64 JPEGs; 5 samples are well under this)
MAX_REQUEST_BODY_BYTES = 12 * 1024 * 1024

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Reject oversized bodies before they are read into memory
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_REQUEST_BODY_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": "Ukuran permintaan terlalu besar."},
        )

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"  # responses contain personal / biometric-derived data
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Global Unhandled Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled system exception at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact the system administrator."}
    )

# Lightweight liveness probe for Cloud Run (no database round-trip)
@app.get("/healthz", tags=["Health"])
async def liveness_probe():
    return {"status": "ok"}

# Health & Connectivity Check Endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    diag = test_firestore_connection()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "organization": settings.ORGANIZATION_NAME,
        "version": settings.APP_VERSION,
        "firebaseReady": is_firebase_ready(),
        "firestoreConnected": diag.get("connected", False),
        "projectId": settings.FIREBASE_PROJECT_ID,
        "firestoreLatencyMs": diag.get("latencyMs"),
        "timezone": settings.DEFAULT_TIMEZONE,
        "serverTimeWib": now_wib().isoformat(),
        "environment": settings.APP_ENV
    }

# Register Routers
app.include_router(auth_router)
app.include_router(employees_router)
app.include_router(attendance_router)
app.include_router(face_router)
app.include_router(admin_face_approvals_router)
app.include_router(settings_router)
app.include_router(reports_router)
app.include_router(leaves_router)
app.include_router(audit_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
