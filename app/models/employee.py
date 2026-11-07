"""
Employee Pydantic Schemas (Phase 2 — Employee Management)
Defines Create, Update, and Response models for managing employees in
Firebase Authentication and Cloud Firestore (`users/{uid}`).
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

from app.models.user import DepartmentModel, EmploymentStatus, FaceVerificationStatus, UserRole


def resolve_face_verification_fields(data: dict) -> dict:
    """
    Normalizes biometric verification state from a Firestore `users/{uid}` document.
    Invariant: `face_registered` / `faceRegistered` is True ONLY when status === 'APPROVED'.
    """
    raw_status = (
        data.get("face_verification_status")
        or data.get("faceVerificationStatus")
        or ""
    )
    raw_status_upper = str(raw_status).strip().upper()
    valid_statuses = {s.value for s in FaceVerificationStatus}

    if raw_status_upper in valid_statuses:
        status_val = FaceVerificationStatus(raw_status_upper)
    else:
        legacy_registered = bool(data.get("faceRegistered", data.get("face_registered", False)))
        status_val = (
            FaceVerificationStatus.APPROVED
            if legacy_registered
            else FaceVerificationStatus.NOT_ENROLLED
        )

    is_approved = status_val == FaceVerificationStatus.APPROVED
    rejection_reason = (
        data.get("face_rejection_reason")
        if data.get("face_rejection_reason") is not None
        else data.get("faceRejectionReason")
    )
    if status_val != FaceVerificationStatus.REJECTED:
        rejection_reason = None

    submitted_raw = (
        data.get("face_submitted_at")
        if data.get("face_submitted_at") is not None
        else data.get("faceSubmittedAt")
    )

    return {
        "status": status_val,
        "face_registered": is_approved,
        "rejection_reason": rejection_reason,
        "submitted_raw": submitted_raw,
    }


class EmployeeCreate(BaseModel):
    """Payload for creating a new employee in Firebase Auth + Firestore."""

    employeeId: Optional[str] = Field(
        None,
        min_length=2,
        max_length=50,
        description="Nomor Induk Yayasan / ID unik karyawan (mis. NIY-2026001)",
    )
    niy: Optional[str] = Field(
        None,
        min_length=2,
        max_length=50,
        description="Nomor Induk Yayasan (NIY)",
    )
    name: str = Field(..., min_length=2, max_length=100, description="Nama lengkap karyawan")
    email: EmailStr = Field(..., description="Alamat email resmi untuk login Firebase Auth")
    password: str = Field(
        default="PaudLab2026!",
        min_length=6,
        max_length=128,
        description="Password awal akun Firebase Auth (minimal 6 karakter)",
    )
    phoneNumber: Optional[str] = Field(None, max_length=25)
    department: str = Field(..., min_length=2, max_length=100, description="Unit / Departemen")
    departmentId: Optional[str] = None
    position: str = Field(..., min_length=2, max_length=100, description="Jabatan / Posisi")
    joinDate: Optional[str] = Field(None, description="Tanggal bergabung (YYYY-MM-DD)")
    employmentStatus: EmploymentStatus = EmploymentStatus.TETAP
    role: UserRole = UserRole.EMPLOYEE
    isActive: bool = Field(default=True, description="Status aktif akun karyawan")
    uid: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_niy_and_employee_id(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        emp_id = (values.get("employeeId") or "").strip()
        niy_val = (values.get("niy") or "").strip()
        resolved = niy_val or emp_id
        if not resolved:
            raise ValueError("Field 'niy' atau 'employeeId' wajib diisi.")
        values["employeeId"] = emp_id or resolved
        values["niy"] = niy_val or resolved
        if values.get("password") is None or str(values.get("password")).strip() == "":
            values["password"] = "PaudLab2026!"
        return values


class EmployeeUpdate(BaseModel):
    """Payload for updating employee profile fields or toggling active status."""

    niy: Optional[str] = Field(None, min_length=2, max_length=50)
    employeeId: Optional[str] = Field(None, min_length=2, max_length=50)
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    phoneNumber: Optional[str] = Field(None, max_length=25)
    department: Optional[str] = Field(None, min_length=2, max_length=100)
    departmentId: Optional[str] = None
    position: Optional[str] = Field(None, min_length=2, max_length=100)
    joinDate: Optional[str] = None
    employmentStatus: Optional[EmploymentStatus] = None
    role: Optional[UserRole] = None
    isActive: Optional[bool] = None


class EmployeeResponse(BaseModel):
    """Authoritative employee representation returned from Cloud Firestore."""

    model_config = ConfigDict(from_attributes=True)

    uid: str
    employeeId: str
    niy: str
    name: str
    email: str
    phoneNumber: Optional[str] = None
    department: str
    departmentId: Optional[str] = None
    position: str
    joinDate: Optional[str] = None
    employmentStatus: EmploymentStatus = EmploymentStatus.TETAP
    role: UserRole = UserRole.EMPLOYEE
    isActive: bool = True
    faceRegistered: bool = False
    face_registered: bool = False
    faceVerificationStatus: FaceVerificationStatus = FaceVerificationStatus.NOT_ENROLLED
    face_verification_status: FaceVerificationStatus = FaceVerificationStatus.NOT_ENROLLED
    faceRejectionReason: Optional[str] = None
    face_rejection_reason: Optional[str] = None
    faceSubmittedAt: Optional[datetime] = None
    face_submitted_at: Optional[datetime] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None


__all__ = [
    "EmployeeCreate",
    "EmployeeUpdate",
    "EmployeeResponse",
    "DepartmentModel",
    "EmploymentStatus",
    "FaceVerificationStatus",
    "UserRole",
    "resolve_face_verification_fields",
]
