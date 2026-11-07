"""
Face Recognition, Liveness, and Quality Data Models
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from app.models.user import FaceVerificationStatus

class FaceQualityReport(BaseModel):
    isQualityPassed: bool = Field(..., description="Whether all image quality checks passed")
    blurScore: float = Field(..., description="Laplacian variance sharpness score")
    isBlurry: bool = Field(..., description="True if blurScore < blur threshold")
    brightnessScore: float = Field(..., description="Mean pixel brightness (0-255)")
    isTooDark: bool = Field(..., description="True if brightness < 40")
    isTooBright: bool = Field(..., description="True if brightness > 220")
    faceCount: int = Field(..., description="Number of faces detected in the image")
    message: str = Field(..., description="Human-readable status summary")

class LivenessChallengeStartResponse(BaseModel):
    sessionId: str
    challengeType: str  # e.g. "turn_right", "turn_left", "smile", "blink", "nod"
    instruction: str
    expiresInSeconds: int

class LivenessVerifyRequest(BaseModel):
    sessionId: str
    baselineImage: str = Field(..., description="Base64 encoded neutral face image")
    actionImage: str = Field(..., description="Base64 encoded challenge action image")

class LivenessVerifyResponse(BaseModel):
    passed: bool
    challengeType: str
    confidence: float
    message: str
    passivePassed: bool

class FaceRegisterRequest(BaseModel):
    samples: List[str] = Field(..., min_length=3, max_length=5, description="3-5 base64 encoded face images")
    livenessSessionId: Optional[str] = None
    consentAgreed: bool = Field(..., description="UU PDP consent agreement confirmation")

class FaceRegisterResponse(BaseModel):
    success: bool
    uid: str
    samplesProcessed: int
    algorithm: str
    enrolledAt: datetime
    faceRegistered: bool = False
    face_registered: bool = False
    faceVerificationStatus: FaceVerificationStatus = FaceVerificationStatus.PENDING_APPROVAL
    face_verification_status: FaceVerificationStatus = FaceVerificationStatus.PENDING_APPROVAL
    faceSubmittedAt: Optional[datetime] = None
    face_submitted_at: Optional[datetime] = None
    message: str

class FaceRecognizeRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded webcam image")
    targetUid: Optional[str] = Field(None, description="Optional target UID for 1:1 verification")
    similarityThreshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Custom similarity threshold override")

class FaceRecognizeResponse(BaseModel):
    matched: bool
    similarity: float
    threshold: float
    confidence: str  # "high" | "medium" | "low" | "rejected"
    matchedUid: Optional[str] = None
    matchedEmployeeName: Optional[str] = None
    quality: FaceQualityReport
    processingTimeMs: float

class FaceProfileRecord(BaseModel):
    uid: str
    encryptedEmbedding: str
    keyVersion: str
    algorithm: str
    vectorDimension: int = 128
    sampleCount: int
    previewThumbnail: Optional[str] = None
    verificationStatus: FaceVerificationStatus = FaceVerificationStatus.PENDING_APPROVAL
    enrolledAt: datetime
    lastUpdated: datetime


class FaceApprovalPendingItem(BaseModel):
    uid: str
    name: str
    niy: str
    employeeId: str
    email: Optional[str] = None
    phoneNumber: Optional[str] = None
    department: str
    position: str = "Staf"
    faceVerificationStatus: FaceVerificationStatus = FaceVerificationStatus.PENDING_APPROVAL
    face_verification_status: FaceVerificationStatus = FaceVerificationStatus.PENDING_APPROVAL
    faceSubmittedAt: Optional[datetime] = None
    face_submitted_at: Optional[datetime] = None
    faceSubmittedAtWib: Optional[str] = None
    previewThumbnail: Optional[str] = None
    sampleCount: int = 3
    algorithm: str = "SFace-v1"


class FaceApprovalRejectRequest(BaseModel):
    reason: str = Field(..., min_length=3, max_length=300, description="Alasan penolakan verifikasi wajah")

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if len(cleaned) < 3:
            raise ValueError("Alasan penolakan minimal 3 karakter.")
        return cleaned


class FaceApprovalActionResponse(BaseModel):
    success: bool
    uid: str
    faceVerificationStatus: FaceVerificationStatus
    face_verification_status: FaceVerificationStatus
    faceRegistered: bool
    face_registered: bool
    faceRejectionReason: Optional[str] = None
    face_rejection_reason: Optional[str] = None
    message: str
