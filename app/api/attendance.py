"""
Attendance API Router
Handles server-side biometric face verification, GPS geofencing,
check-in, check-out, late-arrival calculation, and personal/organization attendance ledgers.
"""

from typing import List, Optional
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status, Query
from firebase_admin import firestore
from google.api_core.exceptions import AlreadyExists

from app.models.auth import TokenData
from app.models.employee import resolve_face_verification_fields
from app.models.user import FaceVerificationStatus, UserRole
from app.models.attendance import (
    AttendanceCheckInRequest,
    AttendanceCheckOutRequest,
    AttendanceRecordResponse,
    AttendanceSummaryResponse,
    TodayAttendanceStatsResponse,
    CheckInStatus,
    CheckOutStatus,
    GeoLocation
)
from app.security.auth import get_current_user, require_admin
from app.security.encryption import biometric_crypto
from app.security.rate_limit import rate_limit
from app.core.firebase import get_firestore_client
from app.core.config import settings
from app.core.geofence import validate_office_geofence, calculate_haversine_distance
from app.core.org_settings import get_effective_org_settings
from app.face_recognition.engine import face_engine, FaceQualityError
from app.face_recognition.liveness import liveness_service
from app.services.notification import dispatch_attendance_notifications
from app.services.audit_logger import (
    ACTION_ATTENDANCE_CHECKIN,
    ACTION_ATTENDANCE_CHECKOUT,
    ACTION_ATTENDANCE_REJECTED,
    ACTION_SPOOF_ATTEMPT,
    record_audit_event,
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/attendance", tags=["Attendance"])

# Throttles the two biometric write endpoints per authenticated user
attendance_rate_limit = rate_limit("attendance", lambda: settings.RATE_LIMIT_ATTENDANCE_PER_MINUTE)

# Timezone Definition: Waktu Indonesia Barat (UTC+7)
WIB = timezone(timedelta(hours=7), name="WIB")

def _get_current_wib() -> datetime:
    return datetime.now(WIB)


def _at_hhmm(base_dt: datetime, hhmm: str) -> datetime:
    hour, minute = [int(v) for v in hhmm.split(":")]
    return base_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _validate_checkin_schedule_window(now_wib: datetime, org: dict) -> None:
    """
    Validates server time (Asia/Jakarta WIB) against active workingDays and
    [checkInOpenTime, checkInCloseTime] configured in Firestore `attendance_settings/default`.
    """
    working_days = org.get("workingDays")
    if working_days is not None and now_wib.isoweekday() not in working_days:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Presensi Masuk belum dibuka atau sudah ditutup",
        )

    open_str = org.get("checkInOpenTime")
    close_str = org.get("checkInCloseTime")
    if open_str and close_str:
        open_dt = _at_hhmm(now_wib, open_str)
        close_dt = _at_hhmm(now_wib, close_str)
        # Allow through the end of the closing minute (e.g. 09:00:59 for 09:00)
        close_dt_inclusive = close_dt.replace(second=59, microsecond=999999)
        if now_wib < open_dt or now_wib > close_dt_inclusive:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Presensi Masuk belum dibuka atau sudah ditutup",
            )


def _validate_checkout_schedule_window(now_wib: datetime, org: dict) -> None:
    """
    Validates server time (Asia/Jakarta WIB) against active workingDays and
    [checkOutOpenTime, checkOutCloseTime] configured in Firestore `attendance_settings/default`.
    """
    working_days = org.get("workingDays")
    if working_days is not None and now_wib.isoweekday() not in working_days:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Presensi Pulang belum dibuka",
        )

    open_str = org.get("checkOutOpenTime")
    close_str = org.get("checkOutCloseTime")
    if open_str and close_str:
        open_dt = _at_hhmm(now_wib, open_str)
        close_dt = _at_hhmm(now_wib, close_str).replace(second=59, microsecond=999999)
        if now_wib < open_dt or now_wib > close_dt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Presensi Pulang belum dibuka",
            )

def _format_attendance_doc(doc_id: str, data: dict) -> AttendanceRecordResponse:
    """Safely converts Firestore document fields into AttendanceRecordResponse."""
    def _convert_dt(dt):
        if dt is None:
            return None
        if hasattr(dt, "to_datetime"):
            return dt.to_datetime()
        if isinstance(dt, datetime):
            return dt
        return None

    # Parse locations
    in_loc = None
    if "checkInLocation" in data and isinstance(data["checkInLocation"], dict):
        in_loc = GeoLocation(**data["checkInLocation"])

    out_loc = None
    if "checkOutLocation" in data and isinstance(data["checkOutLocation"], dict):
        out_loc = GeoLocation(**data["checkOutLocation"])

    return AttendanceRecordResponse(
        attendanceId=doc_id,
        uid=data.get("uid", ""),
        employeeId=data.get("employeeId", ""),
        employeeName=data.get("employeeName"),
        department=data.get("department"),
        workDate=data.get("workDate", ""),
        scheduleId=data.get("scheduleId", "standard_schedule"),
        checkInAt=_convert_dt(data.get("checkInAt")),
        checkOutAt=_convert_dt(data.get("checkOutAt")),
        checkInStatus=data.get("checkInStatus"),
        checkOutStatus=data.get("checkOutStatus"),
        lateMinutes=data.get("lateMinutes", 0),
        checkInLocation=in_loc,
        checkOutLocation=out_loc,
        faceVerified=data.get("faceVerified", False),
        faceSimilarity=data.get("faceSimilarity"),
        livenessVerified=data.get("livenessVerified", False),
        notes=data.get("notes"),
        createdAt=_convert_dt(data.get("createdAt")),
        updatedAt=_convert_dt(data.get("updatedAt"))
    )

def _enforce_liveness(
    org: dict,
    session_id: Optional[str],
    uid: str,
    live_embedding,
    db=None,
    request: Optional[Request] = None,
    attendance_label: str = "Presensi",
) -> bool:
    """
    Returns True when a liveness challenge was verified and spent. Raises 403 when liveness
    is required (admin setting) but no valid, unspent, same-person session was supplied.
    When liveness is disabled by the admin, returns False (recorded as livenessVerified=False).
    """
    if not org.get("livenessEnabled", True):
        return False
    if not liveness_service.consume_passed_session(session_id, uid, live_embedding):
        record_audit_event(
            db,
            action=ACTION_SPOOF_ATTEMPT,
            actor_uid=uid,
            target_id=session_id or uid,
            status="failed",
            details=(
                f"{attendance_label} ditolak oleh proteksi anti-spoofing: "
                "sesi liveness tidak valid, kedaluwarsa, pernah dipakai, atau identitas wajah berbeda."
            ),
            request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Verifikasi liveness (anti-spoofing) wajib diselesaikan sebelum presensi. "
                   "Ulangi tantangan liveness lalu coba kembali."
        )
    return True

@router.post("/check-in", response_model=AttendanceRecordResponse)
async def check_in(
    payload: AttendanceCheckInRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(attendance_rate_limit)
):
    """
    Biometric Check-In Endpoint:
    1. Validates GPS geofence against the organization settings configured by the admin.
    2. Validates image quality (blur, illumination, single face).
    3. Performs 1:1 biometric facial verification against employee's encrypted template.
    4. Requires a passed, single-use liveness challenge performed by the same person (if enabled).
    5. Calculates late arrival status and minutes against the configured schedule.
    6. Writes attendance record to Cloud Firestore.
    """
    now_wib = _get_current_wib()
    today_str = now_wib.strftime("%Y-%m-%d")
    org = get_effective_org_settings()
    db = get_firestore_client()

    # 1. Validate GPS Geofence (when enabled in Admin Settings)
    gps_enabled = bool(org.get("gpsEnabled", True))
    distance: Optional[float] = None
    if gps_enabled:
        if payload.latitude is None or payload.longitude is None:
            record_audit_event(
                db,
                action=ACTION_ATTENDANCE_REJECTED,
                actor_uid=current_user.uid,
                status="failed",
                details="Presensi Masuk ditolak: Koordinat lokasi GPS tidak dikirimkan.",
                request=request,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Koordinat lokasi GPS wajib diaktifkan untuk melakukan presensi.",
            )
        is_valid_geo, distance, geo_msg = validate_office_geofence(
            client_lat=payload.latitude,
            client_lon=payload.longitude,
            client_accuracy=payload.accuracy,
            office_lat=org["latitude"],
            office_lon=org["longitude"],
            radius_meters=org["radiusMeters"],
            max_accuracy_meters=org["maxGpsAccuracyMeters"],
        )
        if not is_valid_geo:
            record_audit_event(
                db,
                action=ACTION_ATTENDANCE_REJECTED,
                actor_uid=current_user.uid,
                status="failed",
                details=f"Presensi Masuk ditolak (Geofence): {geo_msg}",
                request=request,
                metadata={"distanceMeters": distance, "radiusMeters": org["radiusMeters"]},
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=geo_msg)
    elif payload.latitude is not None and payload.longitude is not None:
        distance = round(
            calculate_haversine_distance(
                payload.latitude, payload.longitude, org["latitude"], org["longitude"]
            ),
            2,
        )

    # 1b. Authoritative Server-Side Working Day & Check-In Window Validation (Asia/Jakarta WIB)
    try:
        _validate_checkin_schedule_window(now_wib, org)
    except HTTPException as sched_exc:
        record_audit_event(
            db,
            action=ACTION_ATTENDANCE_REJECTED,
            actor_uid=current_user.uid,
            status="failed",
            details=(
                f"Presensi Masuk ditolak (Di luar jam/hari kerja): mencoba pukul {now_wib.strftime('%H:%M')} WIB "
                f"(jendela Check-In: {org.get('checkInOpenTime')}–{org.get('checkInCloseTime')} WIB)."
            ),
            request=request,
            metadata={"distanceMeters": distance},
        )
        raise sched_exc

    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service unavailable.")

    # 2. Check duplicate check-in for today
    doc_id = f"att_{current_user.uid}_{today_str}"
    att_ref = db.collection("attendance").document(doc_id)
    existing_doc = att_ref.get()

    if existing_doc.exists:
        existing_data = existing_doc.to_dict()
        if existing_data.get("checkInAt"):
            in_time = existing_data.get("checkInAt")
            time_display = in_time.strftime("%H:%M WIB") if hasattr(in_time, "strftime") else "hari ini"
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Anda sudah melakukan presensi masuk hari ini pada pukul {time_display}."
            )

    # 3. Retrieve Employee Profile & Verify Face Registration Status
    user_doc = db.collection("users").document(current_user.uid).get()
    if not user_doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profil pegawai tidak ditemukan.")

    user_data = user_doc.to_dict() or {}
    emp_name = user_data.get("name") or current_user.name or "Karyawan"
    emp_niy = user_data.get("niy") or user_data.get("employeeId") or "-"
    emp_dept = user_data.get("department") or "-"

    face_info = resolve_face_verification_fields(user_data)
    if face_info["status"] != FaceVerificationStatus.APPROVED or not face_info["face_registered"]:
        if face_info["status"] == FaceVerificationStatus.PENDING_APPROVAL:
            block_detail = (
                "Status Biometrik: Menunggu Persetujuan Admin. "
                "Hubungi pihak Tata Usaha/Admin untuk verifikasi wajah sebelum melakukan presensi."
            )
        elif face_info["status"] == FaceVerificationStatus.REJECTED:
            rej_reason = face_info["rejection_reason"] or "Tidak memenuhi standar verifikasi"
            block_detail = (
                f"Verifikasi Wajah Ditolak: {rej_reason}. "
                "Silakan lakukan pendaftaran wajah ulang terlebih dahulu."
            )
        else:
            block_detail = (
                "Wajah Anda belum didaftarkan di sistem. Harap lakukan pendaftaran biometrik terlebih dahulu."
            )
        record_audit_event(
            db,
            action=ACTION_ATTENDANCE_REJECTED,
            actor_uid=current_user.uid,
            employee_name=emp_name,
            niy=emp_niy,
            department=emp_dept,
            status="failed",
            details=f"Presensi Masuk ditolak ({face_info['status'].value}): {block_detail}",
            request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN
            if face_info["status"] in (FaceVerificationStatus.PENDING_APPROVAL, FaceVerificationStatus.REJECTED)
            else status.HTTP_400_BAD_REQUEST,
            detail=block_detail,
        )

    # 4. Biometric Face Verification
    face_profile_doc = db.collection("face_profiles").document(current_user.uid).get()
    if not face_profile_doc.exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Template biometrik wajah tidak ditemukan. Silakan hubungi admin atau daftarkan ulang wajah Anda."
        )

    profile_data = face_profile_doc.to_dict()
    enc_embedding = profile_data.get("encryptedEmbedding")
    if not enc_embedding:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Data biometrik rusak.")

    try:
        enrolled_emb = biometric_crypto.decrypt_embedding(enc_embedding)
    except Exception as e:
        logger.error(f"Biometric decryption error for {current_user.uid}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Gagal mendekripsi template biometrik.")

    # Process submitted frame
    try:
        img = face_engine.decode_image(payload.image)
        quality = face_engine.check_image_quality(img)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Gagal memproses gambar webcam: {str(e)}")

    if not quality.isQualityPassed:
        record_audit_event(
            db,
            action=ACTION_ATTENDANCE_REJECTED,
            actor_uid=current_user.uid,
            employee_name=emp_name,
            niy=emp_niy,
            department=emp_dept,
            status="failed",
            details=f"Presensi Masuk ditolak (Kualitas Foto): {quality.message}",
            request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Kualitas foto presensi tidak memenuhi standar: {quality.message}"
        )

    # Extract live embedding and verify
    live_emb = face_engine.extract_embedding(img, enforce_quality=False)
    face_threshold = org["faceThreshold"]
    is_match, similarity, confidence = face_engine.verify_match(
        live_embedding=live_emb,
        enrolled_embedding=enrolled_emb,
        threshold=face_threshold
    )

    if not is_match:
        logger.warning(
            f"Biometric mismatch for {current_user.uid}: sim={similarity:.4f} < threshold={face_threshold}"
        )
        record_audit_event(
            db,
            action=ACTION_ATTENDANCE_REJECTED,
            actor_uid=current_user.uid,
            employee_name=emp_name,
            niy=emp_niy,
            department=emp_dept,
            status="failed",
            details=(
                f"Presensi Masuk ditolak (Wajah tidak cocok): cosine similarity SFace {similarity:.3f} "
                f"< batas minimal {face_threshold}."
            ),
            request=request,
            metadata={"faceSimilarity": round(float(similarity), 4), "threshold": face_threshold, "distanceMeters": distance},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Verifikasi biometrik wajah gagal. Wajah tidak cocok dengan template terdaftar "
                   f"(skor kemiripan: {similarity:.3f}, batas minimal: {face_threshold})."
        )

    # 4b. Anti-spoofing: a passed liveness challenge, performed by this same face, is mandatory
    liveness_ok = _enforce_liveness(
        org,
        payload.livenessSessionId,
        current_user.uid,
        live_emb,
        db=db,
        request=request,
        attendance_label="Presensi Masuk",
    )

    # 5. Calculate Schedule & Late Arrival (from Official Start Time + Grace Period)
    official_start_str = org.get("officialStartTime") or org.get("startTime") or "08:00"
    schedule_start = _at_hhmm(now_wib, official_start_str)
    grace_cutoff = schedule_start + timedelta(minutes=int(org["gracePeriodMinutes"]))

    if now_wib > grace_cutoff:
        check_in_status = CheckInStatus.LATE
        late_minutes = max(1, int((now_wib - schedule_start).total_seconds() // 60))
    else:
        check_in_status = CheckInStatus.ON_TIME
        late_minutes = 0

    # 6. Save Attendance Document to Firestore
    location_payload = (
        {
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "accuracy": payload.accuracy,
            "distanceMeters": distance,
        }
        if payload.latitude is not None and payload.longitude is not None
        else None
    )

    record_data = {
        "attendanceId": doc_id,
        "uid": current_user.uid,
        "employeeId": user_data.get("employeeId") or emp_niy,
        "employeeName": user_data.get("name", ""),
        "department": user_data.get("department", ""),
        "workDate": today_str,
        "scheduleId": "schedule_paud_lab_um",
        "checkInAt": now_wib,
        "checkOutAt": None,
        "checkInStatus": check_in_status.value,
        "checkOutStatus": None,
        "lateMinutes": late_minutes,
        "checkInLocation": location_payload,
        "checkOutLocation": None,
        "faceVerified": True,
        "faceSimilarity": similarity,
        "livenessVerified": liveness_ok,
        "notes": payload.notes,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP
    }

    # create() fails atomically if a concurrent request already wrote today's record
    try:
        att_ref.create(record_data)
    except AlreadyExists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Anda sudah melakukan presensi masuk hari ini."
        )

    # 7. Audit log (ATTENDANCE_CHECKIN with GPS distance + SFace cosine similarity)
    dist_display = f"{distance:.1f}m" if distance is not None else "GPS Nonaktif"
    status_id_label = f"Terlambat {late_minutes}m" if check_in_status == CheckInStatus.LATE else "Tepat Waktu"
    record_audit_event(
        db,
        action=ACTION_ATTENDANCE_CHECKIN,
        actor_uid=current_user.uid,
        target_id=doc_id,
        employee_name=emp_name,
        niy=emp_niy,
        department=emp_dept,
        status="success",
        details=(
            f"Presensi Masuk pukul {now_wib.strftime('%H:%M:%S')} WIB ({status_id_label}) • "
            f"Jarak GPS: {dist_display} • Cosine Similarity SFace: {similarity:.3f}"
        ),
        request=request,
        metadata={
            "distanceMeters": distance,
            "faceSimilarity": round(float(similarity), 4),
            "lateMinutes": late_minutes,
            "checkInStatus": check_in_status.value,
        },
    )

    # 8. Schedule asynchronous Email & WhatsApp notifications (non-blocking)
    try:
        checkin_status_label = (
            f"Terlambat {late_minutes} Menit"
            if check_in_status == CheckInStatus.LATE
            else "Tepat Waktu"
        )
        email_enabled = bool(org.get("emailNotificationEnabled", True))
        target_email = (user_data.get("email") or current_user.email) if email_enabled else None
        background_tasks.add_task(
            dispatch_attendance_notifications,
            employee_name=emp_name,
            niy=emp_niy,
            email=target_email,
            phone=user_data.get("phone") or user_data.get("phoneNumber"),
            attendance_type="Check-In",
            work_date=today_str,
            time_wib=now_wib.strftime("%H:%M"),
            status_label=checkin_status_label,
            notes=payload.notes,
        )
    except Exception as notif_err:
        logger.warning(f"Failed to enqueue check-in notifications for {current_user.uid}: {notif_err}")

    logger.info(f"Check-in success for {current_user.uid} ({check_in_status.value}, late: {late_minutes}m)")
    return _format_attendance_doc(doc_id, record_data)

@router.post("/check-out", response_model=AttendanceRecordResponse)
async def check_out(
    payload: AttendanceCheckOutRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(attendance_rate_limit)
):
    """
    Biometric Check-Out Endpoint:
    1. Validates GPS geofence (admin-configured).
    2. Verifies check-in has been performed today.
    3. Performs biometric facial verification and (if enabled) a single-use liveness challenge.
    4. Records departure timestamp and status (normal / early departure).
    """
    now_wib = _get_current_wib()
    today_str = now_wib.strftime("%Y-%m-%d")
    org = get_effective_org_settings()
    db = get_firestore_client()

    # 1. Validate GPS Geofence (when enabled in Admin Settings)
    gps_enabled = bool(org.get("gpsEnabled", True))
    distance: Optional[float] = None
    if gps_enabled:
        if payload.latitude is None or payload.longitude is None:
            record_audit_event(
                db,
                action=ACTION_ATTENDANCE_REJECTED,
                actor_uid=current_user.uid,
                status="failed",
                details="Presensi Pulang ditolak: Koordinat lokasi GPS tidak dikirimkan.",
                request=request,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Koordinat lokasi GPS wajib diaktifkan untuk melakukan presensi.",
            )
        is_valid_geo, distance, geo_msg = validate_office_geofence(
            client_lat=payload.latitude,
            client_lon=payload.longitude,
            client_accuracy=payload.accuracy,
            office_lat=org["latitude"],
            office_lon=org["longitude"],
            radius_meters=org["radiusMeters"],
            max_accuracy_meters=org["maxGpsAccuracyMeters"],
        )
        if not is_valid_geo:
            record_audit_event(
                db,
                action=ACTION_ATTENDANCE_REJECTED,
                actor_uid=current_user.uid,
                status="failed",
                details=f"Presensi Pulang ditolak (Geofence): {geo_msg}",
                request=request,
                metadata={"distanceMeters": distance, "radiusMeters": org["radiusMeters"]},
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=geo_msg)
    elif payload.latitude is not None and payload.longitude is not None:
        distance = round(
            calculate_haversine_distance(
                payload.latitude, payload.longitude, org["latitude"], org["longitude"]
            ),
            2,
        )

    # 1b. Authoritative Server-Side Working Day & Check-Out Window Validation (Asia/Jakarta WIB)
    try:
        _validate_checkout_schedule_window(now_wib, org)
    except HTTPException as sched_exc:
        record_audit_event(
            db,
            action=ACTION_ATTENDANCE_REJECTED,
            actor_uid=current_user.uid,
            status="failed",
            details=(
                f"Presensi Pulang ditolak (Di luar jam/hari kerja): mencoba pukul {now_wib.strftime('%H:%M')} WIB "
                f"(jendela Check-Out: {org.get('checkOutOpenTime')}–{org.get('checkOutCloseTime')} WIB)."
            ),
            request=request,
            metadata={"distanceMeters": distance},
        )
        raise sched_exc

    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable.")

    # 2. Check today's check-in record
    doc_id = f"att_{current_user.uid}_{today_str}"
    att_ref = db.collection("attendance").document(doc_id)
    att_doc = att_ref.get()

    if not att_doc.exists or not att_doc.to_dict().get("checkInAt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Anda belum melakukan presensi masuk hari ini. Presensi pulang hanya dapat dilakukan setelah presensi masuk."
        )

    att_data = att_doc.to_dict()
    if att_data.get("checkOutAt"):
        out_time = att_data.get("checkOutAt")
        time_display = out_time.strftime("%H:%M WIB") if hasattr(out_time, "strftime") else "hari ini"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Anda sudah melakukan presensi pulang hari ini pada pukul {time_display}."
        )

    user_doc = db.collection("users").document(current_user.uid).get()
    user_data = (user_doc.to_dict() or {}) if user_doc.exists else {}
    emp_name = user_data.get("name") or att_data.get("employeeName") or current_user.name or "Karyawan"
    emp_niy = user_data.get("niy") or user_data.get("employeeId") or att_data.get("employeeId") or "-"
    emp_dept = user_data.get("department") or att_data.get("department") or "-"

    co_face_info = resolve_face_verification_fields(user_data)
    if co_face_info["status"] != FaceVerificationStatus.APPROVED or not co_face_info["face_registered"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Status Biometrik wajah belum disetujui Administrator sehingga tidak dapat digunakan untuk presensi."
            ),
        )

    # 3. Biometric Face Verification
    face_profile_doc = db.collection("face_profiles").document(current_user.uid).get()
    if not face_profile_doc.exists:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Template biometrik wajah tidak ditemukan.")

    profile_data = face_profile_doc.to_dict()
    try:
        enrolled_emb = biometric_crypto.decrypt_embedding(profile_data["encryptedEmbedding"])
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Gagal mendekripsi template biometrik.")

    try:
        img = face_engine.decode_image(payload.image)
        quality = face_engine.check_image_quality(img)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Gagal memproses gambar webcam: {str(e)}")

    if not quality.isQualityPassed:
        record_audit_event(
            db,
            action=ACTION_ATTENDANCE_REJECTED,
            actor_uid=current_user.uid,
            employee_name=emp_name,
            niy=emp_niy,
            department=emp_dept,
            status="failed",
            details=f"Presensi Pulang ditolak (Kualitas Foto): {quality.message}",
            request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Kualitas foto presensi tidak memenuhi standar: {quality.message}"
        )

    live_emb = face_engine.extract_embedding(img, enforce_quality=False)
    face_threshold = org["faceThreshold"]
    is_match, similarity, confidence = face_engine.verify_match(
        live_embedding=live_emb,
        enrolled_embedding=enrolled_emb,
        threshold=face_threshold
    )

    if not is_match:
        record_audit_event(
            db,
            action=ACTION_ATTENDANCE_REJECTED,
            actor_uid=current_user.uid,
            employee_name=emp_name,
            niy=emp_niy,
            department=emp_dept,
            status="failed",
            details=(
                f"Presensi Pulang ditolak (Wajah tidak cocok): cosine similarity SFace {similarity:.3f} "
                f"< batas minimal {face_threshold}."
            ),
            request=request,
            metadata={"faceSimilarity": round(float(similarity), 4), "threshold": face_threshold, "distanceMeters": distance},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Verifikasi wajah pulang gagal (skor: {similarity:.3f}, minimal: {face_threshold})."
        )

    _enforce_liveness(
        org,
        payload.livenessSessionId,
        current_user.uid,
        live_emb,
        db=db,
        request=request,
        attendance_label="Presensi Pulang",
    )

    # 4. Check Departure Schedule (admin-configured official end time)
    official_end_str = org.get("officialEndTime") or org.get("endTime") or "17:00"
    schedule_end = _at_hhmm(now_wib, official_end_str)

    if now_wib < schedule_end:
        checkout_status = CheckOutStatus.EARLY_DEPARTURE
    else:
        checkout_status = CheckOutStatus.NORMAL

    # 5. Update Record in Firestore
    location_payload = (
        {
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "accuracy": payload.accuracy,
            "distanceMeters": distance,
        }
        if payload.latitude is not None and payload.longitude is not None
        else None
    )

    update_fields = {
        "checkOutAt": now_wib,
        "checkOutStatus": checkout_status.value,
        "checkOutLocation": location_payload,
        "updatedAt": firestore.SERVER_TIMESTAMP
    }
    if payload.notes:
        existing_notes = att_data.get("notes") or ""
        update_fields["notes"] = f"{existing_notes} [Pulang: {payload.notes}]".strip()

    att_ref.update(update_fields)

    # 6. Audit log (ATTENDANCE_CHECKOUT with GPS distance + SFace cosine similarity)
    dist_display = f"{distance:.1f}m" if distance is not None else "GPS Nonaktif"
    out_status_label = "Pulang Cepat" if checkout_status == CheckOutStatus.EARLY_DEPARTURE else "Normal"
    record_audit_event(
        db,
        action=ACTION_ATTENDANCE_CHECKOUT,
        actor_uid=current_user.uid,
        target_id=doc_id,
        employee_name=emp_name,
        niy=emp_niy,
        department=emp_dept,
        status="success",
        details=(
            f"Presensi Pulang pukul {now_wib.strftime('%H:%M:%S')} WIB ({out_status_label}) • "
            f"Jarak GPS: {dist_display} • Cosine Similarity SFace: {similarity:.3f}"
        ),
        request=request,
        metadata={
            "distanceMeters": distance,
            "faceSimilarity": round(float(similarity), 4),
            "checkOutStatus": checkout_status.value,
        },
    )

    # 7. Schedule asynchronous Email & WhatsApp notifications (non-blocking)
    try:
        checkout_status_label = (
            "Pulang Cepat"
            if checkout_status == CheckOutStatus.EARLY_DEPARTURE
            else "Tepat Waktu"
        )
        email_enabled = bool(org.get("emailNotificationEnabled", True))
        target_email = (user_data.get("email") or current_user.email) if email_enabled else None
        background_tasks.add_task(
            dispatch_attendance_notifications,
            employee_name=emp_name,
            niy=emp_niy,
            email=target_email,
            phone=user_data.get("phone") or user_data.get("phoneNumber"),
            attendance_type="Check-Out",
            work_date=today_str,
            time_wib=now_wib.strftime("%H:%M"),
            status_label=checkout_status_label,
            notes=payload.notes or att_data.get("notes"),
        )
    except Exception as notif_err:
        logger.warning(f"Failed to enqueue check-out notifications for {current_user.uid}: {notif_err}")

    # Return merged updated record
    merged_data = {**att_data, **update_fields}
    return _format_attendance_doc(doc_id, merged_data)

@router.get("/today", response_model=Optional[AttendanceRecordResponse])
async def get_today_attendance(current_user: TokenData = Depends(get_current_user)):
    """
    Returns today's attendance record for the signed-in employee, or None if not yet checked in.
    """
    db = get_firestore_client()
    if db is None:
        return None

    today_str = _get_current_wib().strftime("%Y-%m-%d")
    doc_id = f"att_{current_user.uid}_{today_str}"
    doc = db.collection("attendance").document(doc_id).get()

    if not doc.exists:
        return None

    return _format_attendance_doc(doc_id, doc.to_dict() or {})

@router.get("/me", response_model=List[AttendanceRecordResponse])
async def get_my_attendance_history(
    limit: int = Query(30, ge=1, le=100),
    startDate: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
    endDate: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Returns the attendance history strictly for the signed-in employee.
    Filters by user ID and sorts descending by workDate.
    """
    db = get_firestore_client()
    if db is None:
        return []

    docs = db.collection("attendance").where("uid", "==", current_user.uid).stream()
    records = []
    for doc in docs:
        rec = _format_attendance_doc(doc.id, doc.to_dict() or {})
        if startDate and rec.workDate < startDate:
            continue
        if endDate and rec.workDate > endDate:
            continue
        records.append(rec)

    records.sort(key=lambda r: r.workDate, reverse=True)
    return records[:limit]

@router.get("/summary/monthly", response_model=AttendanceSummaryResponse)
async def get_monthly_attendance_summary(
    month: Optional[str] = Query(None, description="Format: YYYY-MM (defaults to current month)"),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Returns personal attendance summary for the specified month (attended, on-time, late, total minutes).
    """
    target_month = month or _get_current_wib().strftime("%Y-%m")
    db = get_firestore_client()
    if db is None:
        return AttendanceSummaryResponse(month=target_month)

    docs = db.collection("attendance").where("uid", "==", current_user.uid).stream()

    attended_days = 0
    on_time = 0
    late = 0
    total_late_min = 0

    for doc in docs:
        data = doc.to_dict()
        work_date = data.get("workDate", "")
        if not work_date.startswith(target_month):
            continue

        if data.get("checkInAt"):
            attended_days += 1
            if data.get("checkInStatus") == CheckInStatus.ON_TIME.value:
                on_time += 1
            elif data.get("checkInStatus") == CheckInStatus.LATE.value:
                late += 1
                total_late_min += data.get("lateMinutes", 0)

    return AttendanceSummaryResponse(
        month=target_month,
        totalDays=attended_days,
        attendedDays=attended_days,
        onTimeCount=on_time,
        lateCount=late,
        totalLateMinutes=total_late_min
    )

@router.get("/organization", response_model=List[AttendanceRecordResponse])
async def list_organization_attendance(
    workDate: Optional[str] = Query(None, description="Format: YYYY-MM-DD (defaults to today)"),
    department: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    admin_user: TokenData = Depends(require_admin)
):
    """
    Admin-only: Inspects and audits attendance records across the organization.
    Supports filtering by workDate, department, status, and employee search.
    """
    db = get_firestore_client()
    if db is None:
        return []

    target_date = workDate or _get_current_wib().strftime("%Y-%m-%d")
    query = db.collection("attendance").where("workDate", "==", target_date)

    if department:
        query = query.where("department", "==", department)
    if status:
        query = query.where("checkInStatus", "==", status)

    docs = query.limit(200).stream()
    records = []
    q_lower = q.lower().strip() if q else None

    for doc in docs:
        rec = _format_attendance_doc(doc.id, doc.to_dict() or {})
        if q_lower:
            match_name = q_lower in (rec.employeeName or "").lower()
            match_id = q_lower in rec.employeeId.lower()
            if not match_name and not match_id:
                continue
        records.append(rec)

    return records

@router.get("/stats/today", response_model=TodayAttendanceStatsResponse)
async def get_today_attendance_stats(admin_user: TokenData = Depends(require_admin)):
    """
    Admin-only: Returns high-level metrics for today's attendance dashboard.
    """
    today_str = _get_current_wib().strftime("%Y-%m-%d")
    db = get_firestore_client()
    if db is None:
        return TodayAttendanceStatsResponse(workDate=today_str)

    # 1. Total active employees
    total_employees = 0
    try:
        users_stream = db.collection("users").where("isActive", "==", True).stream()
        for _ in users_stream:
            total_employees += 1
    except Exception as e:
        logger.warning(f"Failed to count users: {e}")

    # 2. Today's attendance records
    attended = 0
    on_time = 0
    late = 0

    try:
        att_stream = db.collection("attendance").where("workDate", "==", today_str).stream()
        for doc in att_stream:
            data = doc.to_dict()
            if data.get("checkInAt"):
                attended += 1
                if data.get("checkInStatus") == CheckInStatus.ON_TIME.value:
                    on_time += 1
                elif data.get("checkInStatus") == CheckInStatus.LATE.value:
                    late += 1
    except Exception as e:
        logger.warning(f"Failed to stream today's attendance: {e}")

    not_yet = max(0, total_employees - attended)

    return TodayAttendanceStatsResponse(
        workDate=today_str,
        totalEmployees=total_employees,
        totalAttended=attended,
        onTimeCount=on_time,
        lateCount=late,
        notYetAttended=not_yet
    )
