from typing import List, Optional
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Public development-only key that ships in source control. Must never be used in production.
_INSECURE_DEFAULT_ENCRYPTION_KEY = "eFjZ1_4v61_qV17k7_7KjG3mQ9uY0W5zR4tE8pL2oA8="

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "CheckHadir.Id"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = Field(default="development")
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Organization Configuration
    ORGANIZATION_NAME: str = "PAUD LAB UM KOTA BLITAR"

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://localhost:5173",
        "https://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

    # Firebase Admin Configuration
    FIREBASE_CREDENTIALS_PATH: Optional[str] = "serviceAccountKey.json"
    FIREBASE_SERVICE_ACCOUNT_JSON: Optional[str] = None
    FIREBASE_CREDENTIALS_JSON: Optional[str] = None
    FIREBASE_PROJECT_ID: str = "employee-attendance-syst-39d40"
    FIREBASE_USE_EMULATOR: bool = False
    FIREBASE_AUTH_EMULATOR_HOST: Optional[str] = None
    FIRESTORE_EMULATOR_HOST: Optional[str] = None

    # Biometric Encryption Key (Fernet / Fernet (AES-128 + HMAC)-GCM compatible base64 32-byte key)
    BIOMETRIC_ENCRYPTION_KEY: str = Field(
        default=_INSECURE_DEFAULT_ENCRYPTION_KEY,
        description="32-byte URL-safe base64 key for symmetric encryption of face embeddings"
    )
    BIOMETRIC_KEY_VERSION: str = "v1"

    # Default Attendance & Schedule Settings
    DEFAULT_TIMEZONE: str = "Asia/Jakarta"
    DEFAULT_WORKING_DAYS: List[int] = [1, 2, 3, 4, 5]  # 1=Senin .. 5=Jumat, 6=Sabtu, 7=Minggu
    DEFAULT_CHECK_IN_OPEN_TIME: str = "06:30"
    DEFAULT_CHECK_IN_CLOSE_TIME: str = "09:00"
    DEFAULT_SCHEDULE_START_TIME: str = "08:00"
    DEFAULT_GRACE_PERIOD_MINUTES: int = 15
    DEFAULT_CHECK_OUT_OPEN_TIME: str = "16:00"
    DEFAULT_CHECK_OUT_CLOSE_TIME: str = "19:00"
    DEFAULT_SCHEDULE_END_TIME: str = "17:00"
    DEFAULT_GPS_ENABLED: bool = True
    DEFAULT_EMAIL_NOTIFICATION_ENABLED: bool = True

    # Office Geofence Defaults (PAUD Laboratorium UM Kota Blitar - Jl. R.A. Kartini No. 17, Kepanjenlor, Blitar)
    DEFAULT_OFFICE_NAME: str = "PAUD Laboratorium UM Kota Blitar"
    DEFAULT_OFFICE_ADDRESS: str = "Jl. R.A. Kartini No. 17, Kepanjenlor, Kec. Kepanjenkidul, Kota Blitar, Jawa Timur 66117"
    DEFAULT_OFFICE_LATITUDE: float = -8.095469
    DEFAULT_OFFICE_LONGITUDE: float = 112.170750
    DEFAULT_GEOFENCE_RADIUS_METERS: float = 100.0
    MAX_GPS_ACCURACY_METERS: float = 50.0

    # Face Recognition Thresholds
    FACE_SIMILARITY_THRESHOLD: float = 0.637  # Calibrated for OpenCV SFace cosine similarity
    LIVENESS_CHALLENGE_TIMEOUT_SECONDS: int = 60

    # Firebase ID Token Clock Skew Tolerance (seconds, 0..60)
    FIREBASE_CLOCK_SKEW_SECONDS: int = 10

    # Security Hardening
    # Dev tokens ("dev-token-<uid>-<role>") bypass Firebase verification. They are for local
    # automated tests only and are ALWAYS rejected when APP_ENV=production.
    ALLOW_DEV_TOKENS: bool = True
    RATE_LIMIT_FACE_PER_MINUTE: int = 20
    RATE_LIMIT_ATTENDANCE_PER_MINUTE: int = 10
    # Enforce a passed liveness challenge before check-in/check-out (can also be toggled in Admin Settings)
    REQUIRE_LIVENESS_DEFAULT: bool = True

    # Employee Self-Registration Configuration
    SELF_REGISTER_AUTO_ACTIVE: bool = True

    # Automatic Attendance Notification Settings (Email SMTP & WhatsApp Gateway)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_NAME: str = "CheckHadir.Id - PAUD LAB UM KOTA BLITAR"

    WA_GATEWAY_URL: Optional[str] = None
    WA_API_TOKEN: Optional[str] = None

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.strip().lower() == "production"

    @property
    def dev_tokens_allowed(self) -> bool:
        return self.ALLOW_DEV_TOKENS and not self.is_production

    def validate_production_config(self) -> List[str]:
        """Returns a list of fatal misconfigurations when running with APP_ENV=production."""
        problems: List[str] = []
        if not self.is_production:
            return problems
        if self.BIOMETRIC_ENCRYPTION_KEY == _INSECURE_DEFAULT_ENCRYPTION_KEY:
            problems.append(
                "BIOMETRIC_ENCRYPTION_KEY is the public default from source code. "
                "Generate a unique key and store it in Secret Manager."
            )
        if self.DEBUG:
            problems.append("DEBUG must be false in production.")
        if any("localhost" in o or "127.0.0.1" in o for o in self.CORS_ORIGINS):
            problems.append("CORS_ORIGINS still contains localhost entries; set the real Hosting domains.")
        return problems

settings = Settings()
