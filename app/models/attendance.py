"""
Attendance Data Models and Schema Definitions
"""

from typing import Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class CheckInStatus(str, Enum):
    ON_TIME = "on_time"
    LATE = "late"
    PENDING_REVIEW = "pending_review"

class CheckOutStatus(str, Enum):
    NORMAL = "normal"
    EARLY_DEPARTURE = "early_departure"

class GeoLocation(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    accuracy: float = Field(..., ge=0.0)
    distanceMeters: Optional[float] = None

class AttendanceCheckInRequest(BaseModel):
    image: str = Field(..., description="Base64 webcam live capture for biometric recognition")
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    accuracy: float = Field(default=15.0, ge=0.0)
    notes: Optional[str] = Field(None, max_length=255)
    livenessSessionId: Optional[str] = None

class AttendanceCheckOutRequest(BaseModel):
    image: str = Field(..., description="Base64 webcam live capture for biometric recognition")
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    accuracy: float = Field(default=15.0, ge=0.0)
    notes: Optional[str] = Field(None, max_length=255)
    livenessSessionId: Optional[str] = None

class AttendanceRecordResponse(BaseModel):
    attendanceId: str
    uid: str
    employeeId: str
    employeeName: Optional[str] = None
    department: Optional[str] = None
    workDate: str
    scheduleId: Optional[str] = None
    checkInAt: Optional[datetime] = None
    checkOutAt: Optional[datetime] = None
    checkInStatus: Optional[CheckInStatus] = None
    checkOutStatus: Optional[CheckOutStatus] = None
    lateMinutes: int = 0
    checkInLocation: Optional[GeoLocation] = None
    checkOutLocation: Optional[GeoLocation] = None
    faceVerified: bool = False
    faceSimilarity: Optional[float] = None
    livenessVerified: bool = False
    notes: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

class AttendanceSummaryResponse(BaseModel):
    month: str
    totalDays: int = 0
    attendedDays: int = 0
    onTimeCount: int = 0
    lateCount: int = 0
    totalLateMinutes: int = 0

class TodayAttendanceStatsResponse(BaseModel):
    workDate: str
    totalEmployees: int = 0
    totalAttended: int = 0
    onTimeCount: int = 0
    lateCount: int = 0
    notYetAttended: int = 0
