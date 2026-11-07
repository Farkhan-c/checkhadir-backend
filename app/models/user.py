from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict

class UserRole(str, Enum):
    ADMIN = "admin"
    EMPLOYEE = "employee"

class FaceVerificationStatus(str, Enum):
    NOT_ENROLLED = "NOT_ENROLLED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class EmploymentStatus(str, Enum):
    TETAP = "Tetap"
    KONTRAK = "Kontrak"
    HONORER = "Honorer"
    MAGANG = "Magang"

class UserBase(BaseModel):
    employeeId: str = Field(..., description="Unique alphanumeric identifier / NIY e.g. NIY-2026001")
    niy: Optional[str] = Field(None, description="Nomor Induk Yayasan (alias/same as employeeId)")
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phoneNumber: Optional[str] = Field(None, max_length=20)
    department: str = Field(..., min_length=2, max_length=100)
    departmentId: Optional[str] = None
    position: str = Field(..., min_length=2, max_length=100)
    joinDate: Optional[str] = Field(None, description="Format: YYYY-MM-DD")
    employmentStatus: EmploymentStatus = EmploymentStatus.TETAP
    role: UserRole = UserRole.EMPLOYEE
    isActive: bool = True

class UserCreate(UserBase):
    uid: Optional[str] = None
    password: Optional[str] = Field(default="PaudLab2026!", min_length=6)

class UserUpdate(BaseModel):
    name: Optional[str] = None
    phoneNumber: Optional[str] = None
    department: Optional[str] = None
    departmentId: Optional[str] = None
    position: Optional[str] = None
    joinDate: Optional[str] = None
    employmentStatus: Optional[EmploymentStatus] = None
    role: Optional[UserRole] = None
    isActive: Optional[bool] = None

class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    uid: str
    faceRegistered: bool = False
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

class DepartmentModel(BaseModel):
    departmentId: str
    name: str
    description: Optional[str] = None
    employeeCount: int = 0
