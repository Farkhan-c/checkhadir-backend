"""
Firebase Admin SDK Initialization and Client Management.
Connects FastAPI to Cloud Firestore and Firebase Authentication on project employee-attendance-syst-39d40.
"""

import os
import json
import time
import asyncio
import inspect
import logging
from typing import Optional, Dict, Any, List
import firebase_admin
from firebase_admin import credentials, auth, firestore
from app.core.config import settings

logger = logging.getLogger(__name__)

_firebase_app: Optional[firebase_admin.App] = None
_firestore_db = None
_detected_key_path: Optional[str] = None
_is_connected_to_live: bool = False

def detect_service_account_path() -> Optional[str]:
    """
    Automatically detects the location of serviceAccountKey.json across
    environment variables and standard directory locations.
    """
    candidates = []

    # 1. Check GOOGLE_APPLICATION_CREDENTIALS env var
    google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if google_creds:
        candidates.append(google_creds)
        candidates.append(os.path.join(os.getcwd(), google_creds))
        candidates.append(os.path.join(os.path.dirname(__file__), "..", "..", google_creds))

    # 2. Check settings.FIREBASE_CREDENTIALS_PATH
    if settings.FIREBASE_CREDENTIALS_PATH:
        candidates.append(settings.FIREBASE_CREDENTIALS_PATH)
        candidates.append(os.path.join(os.getcwd(), settings.FIREBASE_CREDENTIALS_PATH))
        candidates.append(os.path.join(os.path.dirname(__file__), "..", "..", settings.FIREBASE_CREDENTIALS_PATH))

    # 3. Standard fallback locations (supports running from either repo root or backend/)
    candidates.extend([
        "serviceAccountKey.json",
        os.path.join("backend", "serviceAccountKey.json"),
        os.path.join(os.getcwd(), "serviceAccountKey.json"),
        os.path.join(os.getcwd(), "backend", "serviceAccountKey.json"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "serviceAccountKey.json")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "serviceAccountKey.json")),
    ])

    for path in candidates:
        if path and os.path.exists(path) and os.path.isfile(path):
            try:
                # Validate it contains valid JSON without logging secret contents
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("type") == "service_account":
                    logger.info(f"Detected valid Firebase service account JSON at: {os.path.abspath(path)}")
                    return os.path.abspath(path)
            except Exception as e:
                logger.warning(f"File at {path} could not be parsed as JSON: {e}")

    return None


def _parse_service_account_env_json() -> Optional[Dict[str, Any]]:
    """
    Parses Firebase Service Account credentials from the `FIREBASE_SERVICE_ACCOUNT_JSON`
    (or `FIREBASE_CREDENTIALS_JSON`) environment variable when a physical `serviceAccountKey.json`
    file is not present in the container/deployment environment.
    Supports both raw JSON strings and base64-encoded JSON payloads.
    """
    import base64

    raw_val = (
        os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        or settings.FIREBASE_SERVICE_ACCOUNT_JSON
        or os.environ.get("FIREBASE_CREDENTIALS_JSON")
        or settings.FIREBASE_CREDENTIALS_JSON
    )
    if not raw_val or not raw_val.strip():
        return None

    candidate = raw_val.strip()
    # Strip surrounding single/double quotes if injected by container env configuration
    if (candidate.startswith("'") and candidate.endswith("'")) or (
        candidate.startswith('"') and candidate.endswith('"')
    ):
        candidate = candidate[1:-1].strip()

    try:
        if not candidate.startswith("{"):
            decoded_bytes = base64.b64decode(candidate)
            candidate = decoded_bytes.decode("utf-8").strip()
        cert_dict = json.loads(candidate)
        if isinstance(cert_dict, dict) and "private_key" in cert_dict and isinstance(cert_dict["private_key"], str):
            cert_dict["private_key"] = cert_dict["private_key"].replace("\\n", "\n")
        return cert_dict
    except Exception as e:
        logger.error(f"Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON environment variable: {e}")
        return None


def initialize_firebase() -> Optional[firebase_admin.App]:
    """
    Initializes the Firebase Admin SDK using:
    1. Physical `serviceAccountKey.json` file (if present locally), OR
    2. `FIREBASE_SERVICE_ACCOUNT_JSON` / `FIREBASE_CREDENTIALS_JSON` environment variable
       (recommended for cloud deployments so `serviceAccountKey.json` is never committed to GitHub), OR
    3. Google Application Default Credentials (Cloud Run / GCP).
    """
    global _firebase_app, _firestore_db, _detected_key_path, _is_connected_to_live

    if _firebase_app is not None and _is_connected_to_live:
        return _firebase_app

    # Reuse existing default Firebase app if already initialized across worker/reload boundary
    if firebase_admin._apps:
        try:
            _firebase_app = firebase_admin.get_app()
            _is_connected_to_live = True
            return _firebase_app
        except Exception as e:
            logger.warning(f"Existing Firebase app lookup warning: {e}")

    app_options = {
        "projectId": settings.FIREBASE_PROJECT_ID or "employee-attendance-syst-39d40",
        "httpTimeout": 20,
    }

    # 1. Check for physical serviceAccountKey.json file first
    key_path = detect_service_account_path()
    if key_path:
        try:
            _detected_key_path = key_path
            cred = credentials.Certificate(key_path)
            _firebase_app = firebase_admin.initialize_app(cred, app_options)
            _is_connected_to_live = True
            logger.info(f"Firebase Admin SDK successfully connected to project: {_firebase_app.project_id}")
            return _firebase_app
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Admin SDK from {key_path}: {e}")

    # 2. Fallback to FIREBASE_SERVICE_ACCOUNT_JSON / FIREBASE_CREDENTIALS_JSON environment variable
    cert_dict = _parse_service_account_env_json()
    if cert_dict:
        try:
            cred = credentials.Certificate(cert_dict)
            _firebase_app = firebase_admin.initialize_app(cred, app_options)
            _is_connected_to_live = True
            logger.info(
                "Initialized Firebase Admin SDK from FIREBASE_SERVICE_ACCOUNT_JSON environment variable"
            )
            return _firebase_app
        except Exception as e:
            logger.error(f"Failed to initialize Firebase from FIREBASE_SERVICE_ACCOUNT_JSON: {e}")

    # 3. Check for GCP Cloud Run / Application Default Credentials
    if os.getenv("K_SERVICE"):
        try:
            cred = credentials.ApplicationDefault()
            _firebase_app = firebase_admin.initialize_app(cred, app_options)
            _is_connected_to_live = True
            logger.info("Initialized Firebase Admin SDK via Google Application Default Credentials")
            return _firebase_app
        except Exception as e:
            logger.warning(f"Error with Google Application Default Credentials: {e}")

    # 4. Fallback dev mode (only when no credential file is present)
    logger.warning(
        "No service account credentials found. Running in local fallback mode. "
        "Place serviceAccountKey.json inside backend/ to connect to live Firebase."
    )
    _is_connected_to_live = False
    return None

class FirebaseServiceUnavailableError(RuntimeError):
    """Raised when Google Firebase Auth or Cloud Firestore cannot be reached due to timeout or network outage."""


def is_firebase_transient_error(exc: Exception) -> bool:
    """Identifies transient network, timeout, DNS, or gRPC/HTTP 503 errors from Firebase / Google Cloud SDKs."""
    cls_name = type(exc).__name__.lower()
    transient_cls_markers = (
        "serviceunavailable",
        "deadlineexceeded",
        "aborted",
        "internalservererror",
        "transporterror",
        "connectionerror",
        "connecttimeout",
        "readtimeout",
        "timeout",
        "unavailableerror",
    )
    if any(m in cls_name for m in transient_cls_markers):
        return True

    msg = str(exc).lower()
    transient_msg_markers = (
        "service unavailable",
        "503",
        "deadline exceeded",
        "deadlineexceeded",
        "timed out",
        "timeout",
        "connection aborted",
        "connection reset",
        "failed to resolve",
        "nameresolutionerror",
        "temporarily unavailable",
        "failed to establish a new connection",
        "socket closed",
    )
    return any(m in msg for m in transient_msg_markers)


async def execute_firebase_with_retry_async(
    operation,
    operation_name: str = "Firebase operation",
    retries: int = 1,
    delay_seconds: float = 0.8,
):
    """
    Executes a Firebase Auth / Cloud Firestore operation with automatic single retry on
    transient connection timeout or 503 ServiceUnavailable, raising FirebaseServiceUnavailableError
    with a clear user-facing diagnostic if Firebase remains unreachable.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:
            if not is_firebase_transient_error(exc):
                raise
            last_exc = exc
            if attempt < retries:
                logger.warning(
                    "Transient Firebase connectivity issue during %s (attempt %d/%d): %s. Retrying in %.1fs...",
                    operation_name,
                    attempt + 1,
                    retries + 1,
                    exc,
                    delay_seconds,
                )
                await _async_sleep(delay_seconds)
            else:
                logger.error(
                    "Firebase remained unreachable during %s after %d attempts: %s",
                    operation_name,
                    retries + 1,
                    exc,
                )
    raise FirebaseServiceUnavailableError(
        f"Layanan Google Firebase sedang tidak dapat dijangkau ({operation_name}): batas waktu koneksi habis "
        "atau terjadi gangguan jaringan sementara. Pastikan koneksi internet server stabil lalu coba kembali."
    ) from last_exc


def is_firebase_ready() -> bool:
    global _firebase_app, _is_connected_to_live
    if _firebase_app is None and firebase_admin._apps:
        initialize_firebase()
    return _firebase_app is not None and _is_connected_to_live

def get_firestore_client():
    global _firestore_db
    if _firebase_app is None or not _is_connected_to_live:
        initialize_firebase()
    if not is_firebase_ready():
        return None
    if _firestore_db is None:
        _firestore_db = firestore.client()
    return _firestore_db

def test_firestore_connection() -> Dict[str, Any]:
    """
    Performs an actual live connectivity test against Cloud Firestore.
    Returns diagnostic details without exposing private keys.
    """
    initialize_firebase()
    if not is_firebase_ready():
        return {
            "connected": False,
            "error": "Firebase Admin SDK is not initialized with live credentials.",
            "detectedKeyPath": _detected_key_path
        }

    try:
        start_time = time.time()
        db = get_firestore_client()
        if db is None:
            return {"connected": False, "error": "Firestore client could not be instantiated."}

        # Real test: list collections or fetch server timestamp
        collections = [col.id for col in db.collections()]
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "connected": True,
            "projectId": settings.FIREBASE_PROJECT_ID or "employee-attendance-syst-39d40",
            "latencyMs": latency_ms,
            "detectedKeyPath": _detected_key_path,
            "existingCollections": collections,
            "authReady": True
        }
    except Exception as e:
        logger.error(f"Firestore live connectivity test failed: {e}")
        return {
            "connected": False,
            "error": str(e),
            "detectedKeyPath": _detected_key_path
        }

_SUPPORTS_CLOCK_SKEW: bool = (
    "clock_skew_seconds" in inspect.signature(auth.verify_id_token).parameters
)
_CLOCK_DRIFT_RETRY_DELAY_SECONDS: float = 1.5


async def _async_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _is_token_used_too_early(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "token used too early" in msg or "used too early" in msg


def _call_verify_id_token(id_token: str, skew_seconds: int) -> Dict[str, Any]:
    clamped_skew = max(0, min(60, int(skew_seconds)))
    if _SUPPORTS_CLOCK_SKEW:
        return auth.verify_id_token(
            id_token,
            check_revoked=True,
            clock_skew_seconds=clamped_skew,
        )
    return auth.verify_id_token(id_token, check_revoked=True)


def verify_firebase_id_token(
    id_token: str,
    clock_skew_seconds: Optional[int] = None,
    _allow_sync_retry: bool = True,
) -> Dict[str, Any]:
    """
    Verifies a Firebase ID token sent from the client with clock-skew tolerance (`clock_skew_seconds=10`)
    and a single 1.5s retry fallback when `InvalidIdTokenError` reports 'Token used too early'.
    Returns decoded token dictionary containing uid, email, custom claims.
    """
    if not is_firebase_ready():
        # Fallback for dev mode (never in production)
        if settings.dev_tokens_allowed and id_token.startswith("dev-token-"):
            parts = id_token.split("-")
            uid = parts[2] if len(parts) > 2 else "dev_user_1"
            role = parts[3] if len(parts) > 3 else "employee"
            return {
                "uid": uid,
                "email": f"{uid}@enterprise.com",
                "name": f"Dev User ({uid})",
                "role": role,
                "is_dev": True
            }
        raise ValueError("Firebase Admin SDK is not configured with service account credentials.")

    # Dev tokens are only honoured outside production and only when explicitly allowed
    if settings.dev_tokens_allowed and settings.DEBUG and id_token.startswith("dev-token-"):
        parts = id_token.split("-")
        uid = parts[2] if len(parts) > 2 else "dev_user_1"
        role = parts[3] if len(parts) > 3 else "employee"
        return {
            "uid": uid,
            "email": f"{uid}@enterprise.com",
            "name": f"Dev User ({uid})",
            "role": role,
            "is_dev": True
        }

    skew = (
        settings.FIREBASE_CLOCK_SKEW_SECONDS
        if clock_skew_seconds is None
        else clock_skew_seconds
    )

    try:
        return _call_verify_id_token(id_token, skew_seconds=skew)
    except auth.RevokedIdTokenError:
        raise ValueError("Firebase ID token has been revoked.")
    except auth.ExpiredIdTokenError:
        raise ValueError("Firebase ID token has expired. Please refresh your session.")
    except auth.InvalidIdTokenError as e:
        if _allow_sync_retry and _is_token_used_too_early(e):
            logger.warning(
                "Clock drift detected on Firebase ID token ('Token used too early'). "
                "Sleeping %.1fs and retrying once...",
                _CLOCK_DRIFT_RETRY_DELAY_SECONDS,
            )
            time.sleep(_CLOCK_DRIFT_RETRY_DELAY_SECONDS)
            try:
                return _call_verify_id_token(id_token, skew_seconds=max(skew, 15))
            except Exception as retry_exc:
                raise ValueError(f"Invalid Firebase ID token: {str(retry_exc)}")
        raise ValueError(f"Invalid Firebase ID token: {str(e)}")
    except Exception as e:
        if _allow_sync_retry and _is_token_used_too_early(e):
            logger.warning(
                "Clock drift detected on Firebase ID token ('Token used too early'). "
                "Sleeping %.1fs and retrying once...",
                _CLOCK_DRIFT_RETRY_DELAY_SECONDS,
            )
            time.sleep(_CLOCK_DRIFT_RETRY_DELAY_SECONDS)
            try:
                return _call_verify_id_token(id_token, skew_seconds=max(skew, 15))
            except Exception as retry_exc:
                raise ValueError(f"Invalid Firebase ID token: {str(retry_exc)}")
        if is_firebase_transient_error(e):
            raise FirebaseServiceUnavailableError(
                "Gagal memverifikasi sesi ke server Google Firebase karena koneksi timeout atau tidak tersedia sementara."
            ) from e
        raise ValueError(f"Invalid Firebase ID token: {str(e)}")


async def verify_firebase_id_token_async(
    id_token: str,
    clock_skew_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Non-blocking async wrapper for FastAPI endpoints and security dependencies.
    Applies `clock_skew_seconds=10` and performs a single `await asyncio.sleep(1.5)` retry
    if Firebase reports 'Token used too early' (or transient cert fetch timeout) before raising.
    """
    skew = (
        settings.FIREBASE_CLOCK_SKEW_SECONDS
        if clock_skew_seconds is None
        else clock_skew_seconds
    )
    try:
        return verify_firebase_id_token(
            id_token,
            clock_skew_seconds=skew,
            _allow_sync_retry=False,
        )
    except FirebaseServiceUnavailableError:
        logger.warning("Transient network failure verifying Firebase ID token; retrying once in 0.8s...")
        await _async_sleep(0.8)
        return verify_firebase_id_token(
            id_token,
            clock_skew_seconds=skew,
            _allow_sync_retry=False,
        )
    except ValueError as err:
        if _is_token_used_too_early(err):
            logger.warning(
                "Async token check detected clock drift ('Token used too early'); "
                "waiting %.1fs before single retry...",
                _CLOCK_DRIFT_RETRY_DELAY_SECONDS,
            )
            await _async_sleep(_CLOCK_DRIFT_RETRY_DELAY_SECONDS)
            return verify_firebase_id_token(
                id_token,
                clock_skew_seconds=max(skew, 15),
                _allow_sync_retry=False,
            )
        raise
