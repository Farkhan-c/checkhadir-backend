import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from app.models.user import FaceVerificationStatus, UserRole


def normalize_indonesian_phone(raw_phone: str) -> str:
    """
    Validates and normalizes Indonesian phone / WhatsApp numbers to standard '62...' format.
    Accepts '08...', '628...', '+628...' (and Indonesian landlines '021...'/'6221...') with optional spaces/hyphens.
    """
    cleaned = re.sub(r"[\s\-()]", "", (raw_phone or "").strip())
    if cleaned.startswith("+62"):
        cleaned = "62" + cleaned[3:]
    elif cleaned.startswith("0"):
        cleaned = "62" + cleaned[1:]

    if not re.fullmatch(r"62[1-9][0-9]{7,12}", cleaned):
        raise ValueError(
            "Nomor WhatsApp/Telepon tidak valid. Gunakan format standar Indonesia (contoh: 081234567890 atau 6281234567890)."
        )
    return cleaned


class TokenData(BaseModel):
    uid: str
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    role: UserRole = UserRole.EMPLOYEE
    claims: Dict[str, Any] = {}


class VerifyTokenRequest(BaseModel):
    idToken: str
    isLoginAttempt: bool = False


class LoginAuditRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=160)
    status: str = Field(default="failed", pattern=r"^(success|failed)$")
    reason: Optional[str] = Field(default=None, max_length=300)
    idToken: Optional[str] = None


class TokenValidationResponse(BaseModel):
    valid: bool
    uid: str
    email: Optional[str] = None
    name: Optional[str] = None
    role: UserRole
    employeeId: Optional[str] = None
    faceRegistered: bool = False
    face_registered: bool = False
    faceVerificationStatus: FaceVerificationStatus = FaceVerificationStatus.NOT_ENROLLED
    face_verification_status: FaceVerificationStatus = FaceVerificationStatus.NOT_ENROLLED
    faceRejectionReason: Optional[str] = None
    face_rejection_reason: Optional[str] = None


class SelfRegisterRequest(BaseModel):
    """
    Public employee self-registration schema.
    `extra='forbid'` strictly rejects any attempt to supply `role`, `is_active`,
    or privileged attributes from the public registration form.
    """
    model_config = ConfigDict(extra="forbid")

    niy: str = Field(..., min_length=2, max_length=50, description="Nomor Induk Yayasan / NIP")
    name: str = Field(..., min_length=2, max_length=120, description="Nama Lengkap Karyawan")
    email: EmailStr = Field(..., description="Alamat Email Aktif")
    phone: str = Field(..., min_length=9, max_length=25, description="Nomor WhatsApp / Telepon Indonesia")
    password: str = Field(..., min_length=6, max_length=128, description="Password minimal 6 karakter")
    department: str = Field(..., min_length=2, max_length=100, description="Departemen / Unit Kerja")
    position: str = Field(..., min_length=2, max_length=100, description="Posisi / Jabatan")
    idToken: Optional[str] = Field(
        None,
        description="Optional Firebase ID Token when account was created client-side via createUserWithEmailAndPassword",
    )

    @field_validator("niy", "name", "department", "position")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Kolom ini tidak boleh kosong.")
        return cleaned

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, value: str) -> str:
        return normalize_indonesian_phone(value)


class SelfRegisterResponse(BaseModel):
    uid: str
    niy: str
    name: str
    email: str
    phone: str
    department: str
    position: str
    role: UserRole = UserRole.EMPLOYEE
    is_active: bool
    isActive: bool
    face_registered: bool = False
    faceRegistered: bool = False
    face_verification_status: FaceVerificationStatus = FaceVerificationStatus.NOT_ENROLLED
    faceVerificationStatus: FaceVerificationStatus = FaceVerificationStatus.NOT_ENROLLED
    message: str
