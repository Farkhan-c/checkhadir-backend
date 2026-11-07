"""
Leave & Permission Data Models
Supports sick leave, annual leave, official duty, maternity, and personal permissions
with review workflows for PAUD LAB UM KOTA BLITAR.
"""

from typing import Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class LeaveType(str, Enum):
    SICK = "sick"                        # Sakit (Surat Dokter)
    ANNUAL_LEAVE = "annual_leave"        # Cuti Tahunan
    OFFICIAL_DUTY = "official_duty"      # Tugas Dinas Luar / Pelatihan
    IMPORTANT_REASON = "important_reason" # Izin Alasan Penting / Urusan Keluarga
    MATERNITY = "maternity"              # Cuti Melahirkan

class LeaveStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class LeaveRequestCreate(BaseModel):
    leaveType: LeaveType
    startDate: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Format: YYYY-MM-DD")
    endDate: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Format: YYYY-MM-DD")
    reason: str = Field(..., min_length=3, max_length=500)
    attachmentBase64: Optional[str] = Field(None, description="Optional image/document attachment")

class LeaveReviewRequest(BaseModel):
    status: LeaveStatus = Field(..., description="approved or rejected")
    reviewerNotes: Optional[str] = Field(None, max_length=500)

class LeaveRequestResponse(BaseModel):
    leaveId: str
    uid: str
    employeeId: str
    employeeName: str
    department: str
    leaveType: LeaveType
    startDate: str
    endDate: str
    totalDays: int
    reason: str
    hasAttachment: bool = False
    status: LeaveStatus
    reviewerUid: Optional[str] = None
    reviewerName: Optional[str] = None
    reviewerNotes: Optional[str] = None
    reviewedAt: Optional[datetime] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
