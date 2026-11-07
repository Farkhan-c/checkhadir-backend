"""
Leave & Permission API Router
Enables employees to apply for leave (sick, annual, official duty) and administrators to review.
"""

import uuid
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
from firebase_admin import firestore

from app.models.auth import TokenData
from app.models.user import UserRole
from app.models.leave import (
    LeaveRequestCreate,
    LeaveReviewRequest,
    LeaveRequestResponse,
    LeaveStatus,
    LeaveType
)
from app.security.auth import get_current_user, require_admin
from app.core.firebase import get_firestore_client
from app.core.timezone import now_wib
from app.services.audit_logger import (
    ACTION_LEAVE_APPLICATION,
    ACTION_LEAVE_REVIEWED,
    record_audit_event,
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/leaves", tags=["Leaves & Permissions"])

def _format_leave_doc(doc_id: str, data: dict) -> LeaveRequestResponse:
    def _convert_dt(dt):
        if dt is None:
            return None
        if hasattr(dt, "to_datetime"):
            return dt.to_datetime()
        if isinstance(dt, datetime):
            return dt
        return None

    return LeaveRequestResponse(
        leaveId=doc_id,
        uid=data.get("uid", ""),
        employeeId=data.get("employeeId", ""),
        employeeName=data.get("employeeName", "Pegawai"),
        department=data.get("department", "-"),
        leaveType=data.get("leaveType", LeaveType.ANNUAL_LEAVE),
        startDate=data.get("startDate", ""),
        endDate=data.get("endDate", ""),
        totalDays=data.get("totalDays", 1),
        reason=data.get("reason", ""),
        hasAttachment=bool(data.get("hasAttachment", False)),
        status=data.get("status", LeaveStatus.PENDING),
        reviewerUid=data.get("reviewerUid"),
        reviewerName=data.get("reviewerName"),
        reviewerNotes=data.get("reviewerNotes"),
        reviewedAt=_convert_dt(data.get("reviewedAt")),
        createdAt=_convert_dt(data.get("createdAt")),
        updatedAt=_convert_dt(data.get("updatedAt"))
    )

@router.post("", response_model=LeaveRequestResponse)
async def submit_leave_request(
    payload: LeaveRequestCreate,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Submits a new leave or permission request.
    Calculates total days automatically and stores in Cloud Firestore.
    """
    try:
        d1 = datetime.strptime(payload.startDate, "%Y-%m-%d")
        d2 = datetime.strptime(payload.endDate, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Format tanggal harus YYYY-MM-DD.")

    if d2 < d1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tanggal akhir cuti tidak boleh lebih awal dari tanggal mulai."
        )

    total_days = (d2 - d1).days + 1
    db = get_firestore_client()
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database tidak tersedia.")

    # Fetch user data
    user_doc = db.collection("users").document(current_user.uid).get()
    user_data = user_doc.to_dict() if user_doc.exists else {}

    leave_id = f"leave_{uuid.uuid4().hex[:12]}"
    current_wib = now_wib()

    leave_data = {
        "leaveId": leave_id,
        "uid": current_user.uid,
        "employeeId": user_data.get("employeeId", ""),
        "employeeName": user_data.get("name", current_user.name or "Pegawai"),
        "department": user_data.get("department", "-"),
        "leaveType": payload.leaveType.value,
        "startDate": payload.startDate,
        "endDate": payload.endDate,
        "totalDays": total_days,
        "reason": payload.reason,
        "hasAttachment": bool(payload.attachmentBase64),
        "status": LeaveStatus.PENDING.value,
        "reviewerUid": None,
        "reviewerName": None,
        "reviewerNotes": None,
        "reviewedAt": None,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP
    }

    db.collection("leaves").document(leave_id).set(leave_data)

    # Audit log (LEAVE_APPLICATION)
    emp_name = user_data.get("name", current_user.name or "Pegawai")
    emp_niy = user_data.get("niy") or user_data.get("employeeId") or "-"
    record_audit_event(
        db,
        action=ACTION_LEAVE_APPLICATION,
        actor_uid=current_user.uid,
        target_id=leave_id,
        employee_name=emp_name,
        niy=emp_niy,
        department=user_data.get("department", "-"),
        status="success",
        details=(
            f"Pengajuan izin/cuti ({payload.leaveType.value}) tanggal {payload.startDate} s/d "
            f"{payload.endDate} ({total_days} hari): {payload.reason}"
        ),
        request=request,
        metadata={
            "leaveType": payload.leaveType.value,
            "startDate": payload.startDate,
            "endDate": payload.endDate,
            "totalDays": total_days,
        },
    )

    return _format_leave_doc(leave_id, {**leave_data, "createdAt": current_wib, "updatedAt": current_wib})

@router.get("/me", response_model=List[LeaveRequestResponse])
async def get_my_leaves(current_user: TokenData = Depends(get_current_user)):
    """
    Returns personal leave requests submitted by the signed-in employee.
    """
    db = get_firestore_client()
    if db is None:
        return []

    docs = db.collection("leaves").where("uid", "==", current_user.uid).stream()
    leaves = []
    for doc in docs:
        leaves.append(_format_leave_doc(doc.id, doc.to_dict() or {}))

    leaves.sort(key=lambda x: x.startDate, reverse=True)
    return leaves

@router.get("", response_model=List[LeaveRequestResponse])
async def list_company_leaves(
    status: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    admin_user: TokenData = Depends(require_admin)
):
    """
    Admin-only: Retrieves all employee leave requests with filtering.
    """
    db = get_firestore_client()
    if db is None:
        return []

    query = db.collection("leaves")
    if status:
        query = query.where("status", "==", status)
    if department:
        query = query.where("department", "==", department)

    docs = query.limit(100).stream()
    leaves = []
    for doc in docs:
        leaves.append(_format_leave_doc(doc.id, doc.to_dict() or {}))

    leaves.sort(key=lambda x: x.startDate, reverse=True)
    return leaves

@router.patch("/{leave_id}/review", response_model=LeaveRequestResponse)
async def review_leave_request(
    leave_id: str,
    payload: LeaveReviewRequest,
    request: Request,
    admin_user: TokenData = Depends(require_admin)
):
    """
    Admin-only: Approves or rejects an employee's leave request.
    """
    db = get_firestore_client()
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database tidak tersedia.")

    leave_ref = db.collection("leaves").document(leave_id)
    doc = leave_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permohonan cuti tidak ditemukan.")

    existing_leave = doc.to_dict() or {}
    current_wib = now_wib()
    update_data = {
        "status": payload.status.value,
        "reviewerUid": admin_user.uid,
        "reviewerName": admin_user.name or "Administrator",
        "reviewerNotes": payload.reviewerNotes,
        "reviewedAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP
    }

    leave_ref.update(update_data)

    # Audit log
    decision_label = "disetujui" if payload.status.value == "approved" else "ditolak"
    record_audit_event(
        db,
        action=ACTION_LEAVE_REVIEWED,
        actor_uid=admin_user.uid,
        target_id=leave_id,
        employee_name=existing_leave.get("employeeName") or admin_user.name or "Administrator",
        niy=existing_leave.get("employeeId") or "-",
        department=existing_leave.get("department") or "-",
        status="success",
        details=(
            f"Permohonan izin/cuti {existing_leave.get('employeeName') or leave_id} "
            f"({existing_leave.get('startDate')} s/d {existing_leave.get('endDate')}) {decision_label} oleh Admin."
        ),
        request=request,
    )

    merged = {**existing_leave, **update_data, "reviewedAt": current_wib, "updatedAt": current_wib}
    return _format_leave_doc(leave_id, merged)
