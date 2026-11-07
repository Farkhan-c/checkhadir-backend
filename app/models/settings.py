import re
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator

HHMM_PATTERN = r"^(?:[01]\d|2[0-3]):[0-5]\d$"


def _hhmm_to_minutes(val: str) -> int:
    parts = val.split(":")
    return int(parts[0]) * 60 + int(parts[1])


class OfficeLocationModel(BaseModel):
    locationId: Optional[str] = None
    officeName: str = Field(..., min_length=2, max_length=100)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    radiusMeters: float = Field(default=100.0, gt=0, le=10000.0)
    maxGpsAccuracyMeters: float = Field(default=50.0, gt=0, le=500.0)
    isActive: bool = True


class WorkScheduleModel(BaseModel):
    scheduleId: Optional[str] = None
    name: str = Field(..., min_length=2, max_length=100)
    startTime: str = Field(..., pattern=HHMM_PATTERN)  # e.g. "08:00"
    endTime: str = Field(..., pattern=HHMM_PATTERN)    # e.g. "17:00"
    gracePeriodMinutes: int = Field(default=15, ge=0, le=180)
    workingDays: List[int] = Field(default=[1, 2, 3, 4, 5])  # 1=Mon .. 7=Sun
    timezone: str = "Asia/Jakarta"
    isActive: bool = True


class AttendanceSettingsModel(BaseModel):
    timezone: str = "Asia/Jakarta"
    defaultScheduleId: Optional[str] = None
    geofenceEnabled: bool = True
    livenessEnabled: bool = True
    faceThreshold: float = Field(default=0.637, ge=0.0, le=1.0)
    updatedAt: Optional[datetime] = None


class DynamicAttendanceSettingsModel(BaseModel):
    """
    Authoritative dynamic attendance & schedule configuration stored at
    Firestore `attendance_settings/default`.
    """
    organizationName: str = Field(default="PAUD LAB UM KOTA BLITAR", min_length=2, max_length=150)
    address: str = Field(
        default="Jl. R.A. Kartini No. 17, Kepanjenlor, Kec. Kepanjenkidul, Kota Blitar, Jawa Timur 66117",
        min_length=5,
        max_length=255,
    )
    timezone: str = "Asia/Jakarta"

    # Active working days: ISO weekday integers 1 (Senin) .. 7 (Minggu)
    workingDays: List[int] = Field(default=[1, 2, 3, 4, 5])

    # Check-In window & official entry schedule
    checkInOpenTime: str = Field(default="06:30", pattern=HHMM_PATTERN)
    checkInCloseTime: str = Field(default="09:00", pattern=HHMM_PATTERN)
    officialStartTime: str = Field(default="08:00", pattern=HHMM_PATTERN)
    gracePeriodMinutes: int = Field(default=15, ge=0, le=180)

    # Check-Out window & official departure schedule
    checkOutOpenTime: str = Field(default="16:00", pattern=HHMM_PATTERN)
    checkOutCloseTime: str = Field(default="19:00", pattern=HHMM_PATTERN)
    officialEndTime: str = Field(default="17:00", pattern=HHMM_PATTERN)

    # Office GPS Geofencing parameters
    latitude: float = Field(default=-8.095469, ge=-90.0, le=90.0)
    longitude: float = Field(default=112.170750, ge=-180.0, le=180.0)
    radiusMeters: float = Field(default=100.0, gt=0, le=10000.0)
    maxGpsAccuracyMeters: float = Field(default=50.0, gt=0, le=500.0)

    # Feature Toggles & Biometric Threshold
    gpsEnabled: bool = True
    livenessEnabled: bool = True
    emailNotificationEnabled: bool = True
    faceThreshold: float = Field(default=0.637, ge=0.0, le=1.0)

    updatedAt: Optional[datetime] = None

    @field_validator("workingDays")
    @classmethod
    def validate_working_days(cls, v: List[int]) -> List[int]:
        cleaned = sorted({int(d) for d in v if 1 <= int(d) <= 7})
        if not cleaned:
            raise ValueError("Minimal pilih satu hari kerja aktif (1=Senin s/d 7=Minggu).")
        return cleaned

    @model_validator(mode="after")
    def validate_time_windows(self) -> "DynamicAttendanceSettingsModel":
        if _hhmm_to_minutes(self.checkInOpenTime) > _hhmm_to_minutes(self.checkInCloseTime):
            raise ValueError("Jam Mulai Check-In tidak boleh melewati Jam Tutup Check-In.")
        if _hhmm_to_minutes(self.checkOutOpenTime) > _hhmm_to_minutes(self.checkOutCloseTime):
            raise ValueError("Jam Mulai Check-Out tidak boleh melewati Jam Tutup Check-Out.")
        return self


class DynamicAttendanceSettingsUpdate(BaseModel):
    organizationName: Optional[str] = Field(None, min_length=2, max_length=150)
    address: Optional[str] = Field(None, min_length=5, max_length=255)
    workingDays: Optional[List[int]] = None
    checkInOpenTime: Optional[str] = Field(None, pattern=HHMM_PATTERN)
    checkInCloseTime: Optional[str] = Field(None, pattern=HHMM_PATTERN)
    officialStartTime: Optional[str] = Field(None, pattern=HHMM_PATTERN)
    gracePeriodMinutes: Optional[int] = Field(None, ge=0, le=180)
    checkOutOpenTime: Optional[str] = Field(None, pattern=HHMM_PATTERN)
    checkOutCloseTime: Optional[str] = Field(None, pattern=HHMM_PATTERN)
    officialEndTime: Optional[str] = Field(None, pattern=HHMM_PATTERN)
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    radiusMeters: Optional[float] = Field(None, gt=0, le=10000.0)
    maxGpsAccuracyMeters: Optional[float] = Field(None, gt=0, le=500.0)
    gpsEnabled: Optional[bool] = None
    livenessEnabled: Optional[bool] = None
    emailNotificationEnabled: Optional[bool] = None
    faceThreshold: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("workingDays")
    @classmethod
    def validate_optional_working_days(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is None:
            return None
        cleaned = sorted({int(d) for d in v if 1 <= int(d) <= 7})
        if not cleaned:
            raise ValueError("Minimal pilih satu hari kerja aktif (1=Senin s/d 7=Minggu).")
        return cleaned


class OrganizationSettingsModel(BaseModel):
    organizationName: str = Field(..., min_length=2, max_length=150)
    address: str = Field(..., min_length=5, max_length=255)
    contactEmail: Optional[str] = None
    contactPhone: Optional[str] = None
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    radiusMeters: float = Field(default=100.0, gt=0, le=10000.0)
    maxGpsAccuracyMeters: float = Field(default=50.0, gt=0, le=500.0)
    startTime: str = Field(default="08:00", pattern=HHMM_PATTERN)
    endTime: str = Field(default="17:00", pattern=HHMM_PATTERN)
    gracePeriodMinutes: int = Field(default=15, ge=0, le=180)
    faceThreshold: float = Field(default=0.637, ge=0.0, le=1.0)
    livenessEnabled: bool = True
    updatedAt: Optional[datetime] = None


class OrganizationSettingsUpdate(BaseModel):
    organizationName: Optional[str] = Field(None, min_length=2, max_length=150)
    address: Optional[str] = Field(None, min_length=5, max_length=255)
    contactEmail: Optional[str] = None
    contactPhone: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    radiusMeters: Optional[float] = Field(None, gt=0, le=10000.0)
    maxGpsAccuracyMeters: Optional[float] = Field(None, gt=0, le=500.0)
    startTime: Optional[str] = Field(None, pattern=HHMM_PATTERN)
    endTime: Optional[str] = Field(None, pattern=HHMM_PATTERN)
    gracePeriodMinutes: Optional[int] = Field(None, ge=0, le=180)
    faceThreshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    livenessEnabled: Optional[bool] = None
