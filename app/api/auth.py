import time
import logging
from collections import defaultdict, deque
from fastapi import APIRouter, Depends, HTTPException, Request, status
from firebase_admin import auth as fb_auth, firestore
from app.models.auth import (
    TokenData,
    VerifyTokenRequest,
    LoginAuditRequest,
    TokenValidationResponse,
    SelfRegisterRequest,
    SelfRegisterResponse,
)
from app.models.employee import resolve_face_verification_fields
from app.models.user import FaceVerificationStatus, UserRole
from app.security.auth import get_current_user
from app.core.firebase import (
    FirebaseServiceUnavailableError,
    execute_firebase_with_retry_async,
    initialize_firebase,
    is_firebase_transient_error,
    verify_firebase_id_token_async,
    get_firestore_client,
    is_firebase_ready,
)
from app.core.timezone import now_wib
from app.core.config import settings
from app.services.audit_logger import (
    ACTION_EMPLOYEE_REGISTERED,
    ACTION_LOGIN_ATTEMPT,
    extract_client_metadata,
    record_audit_event,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# Lightweight per-IP rate limiter for unauthenticated login audit events (20 events / 60s)
_LOGIN_AUDIT_BUCKETS: dict[str, deque] = defaultdict(deque)
_LOGIN_AUDIT_MAX_PER_MIN = 20


def _allow_login_audit_ip(ip_addr: str) -> bool:
    now = time.monotonic()
    dq = _LOGIN_AUDIT_BUCKETS[ip_addr]
    while dq and now - dq[0] > 60.0:
        dq.popleft()
    if len(dq) >= _LOGIN_AUDIT_MAX_PER_MIN:
        return False
    dq.append(now)
    return True


@router.post("/register", response_model=SelfRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_employee_self(payload: SelfRegisterRequest, request: Request):
    """
    Public Self-Registration Endpoint (no Admin token required):
    1. Ensures Firebase Admin SDK + Cloud Firestore connectivity (with automatic retry).
    2. Checks duplicate NIY/NIP and Email in Cloud Firestore `users`.
    3. Registers or links the employee in Firebase Authentication with `role='employee'`.
    4. Saves complete profile in Firestore `users/{uid}` and logs immutable audit event.
    """
    db = get_firestore_client()
    if db is None or not is_firebase_ready():
        initialize_firebase()
        db = get_firestore_client()

    if db is None or not is_firebase_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Layanan Firebase Admin SDK / Cloud Firestore belum siap pada server backend. "
                "Pastikan file kredensial serviceAccountKey.json terpasang dan koneksi internet server aktif."
            ),
        )

    niy_val = payload.niy.strip()
    email_val = str(payload.email).strip().lower()

    # Optional: verify Firebase ID token when account was created via Firebase Client SDK (`createUserWithEmailAndPassword`)
    client_uid = None
    if payload.idToken:
        try:
            decoded_client = await verify_firebase_id_token_async(payload.idToken)
            client_uid = decoded_client.get("uid")
            token_email = str(decoded_client.get("email") or "").strip().lower()
            if token_email and token_email != email_val:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email pada token autentikasi tidak sesuai dengan email formulir pendaftaran.",
                )
        except HTTPException:
            raise
        except FirebaseServiceUnavailableError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token autentikasi pendaftaran tidak valid: {str(e)}",
            )

    def _rollback_uncommitted_client_auth():
        if client_uid:
            try:
                existing_profile = db.collection("users").document(client_uid).get()
                if not existing_profile.exists:
                    fb_auth.delete_user(client_uid)
            except Exception:
                pass

    # 1. Validate unique NIY / employeeId & Email in Firestore (with transient retry)
    def _check_duplicates():
        dup_n = [
            d for d in db.collection("users").where("niy", "==", niy_val).limit(2).stream()
            if d.id != client_uid
        ]
        if not dup_n:
            dup_n = [
                d for d in db.collection("users").where("employeeId", "==", niy_val).limit(2).stream()
                if d.id != client_uid
            ]
        if dup_n:
            return "niy"

        dup_e = [
            d for d in db.collection("users").where("email", "==", email_val).limit(2).stream()
            if d.id != client_uid
        ]
        if not dup_e and email_val != str(payload.email).strip():
            dup_e = [
                d
                for d in db.collection("users").where("email", "==", str(payload.email).strip()).limit(2).stream()
                if d.id != client_uid
            ]
        if dup_e:
            return "email"
        return None

    try:
        conflict_kind = await execute_firebase_with_retry_async(
            _check_duplicates,
            operation_name="validasi duplikasi NIY/Email di Cloud Firestore",
        )
    except FirebaseServiceUnavailableError as e:
        _rollback_uncommitted_client_auth()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    if conflict_kind == "niy":
        _rollback_uncommitted_client_auth()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"NIY / NIP '{niy_val}' sudah terdaftar di sistem.",
        )
    if conflict_kind == "email":
        _rollback_uncommitted_client_auth()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Alamat email '{email_val}' sudah terdaftar. Silakan masuk menggunakan akun Anda.",
        )

    is_active = bool(settings.SELF_REGISTER_AUTO_ACTIVE)
    # Security invariant: public registration can ONLY issue the 'employee' role
    enforced_role = UserRole.EMPLOYEE.value

    # 2. Create or finalize user in Firebase Authentication (with transient retry)
    def _provision_auth_user() -> str:
        if client_uid:
            uid_out = client_uid
            fb_auth.update_user(
                uid_out,
                display_name=payload.name,
            )
        else:
            user_record = fb_auth.create_user(
                email=email_val,
                password=payload.password,
                display_name=payload.name,
                disabled=not is_active,
            )
            uid_out = user_record.uid
        fb_auth.set_custom_user_claims(uid_out, {"role": enforced_role})
        return uid_out

    try:
        target_uid = await execute_firebase_with_retry_async(
            _provision_auth_user,
            operation_name="pendaftaran akun di Firebase Authentication",
        )
    except fb_auth.EmailAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Alamat email '{email_val}' sudah terdaftar pada sistem autentikasi.",
        )
    except FirebaseServiceUnavailableError as e:
        _rollback_uncommitted_client_auth()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Self-registration Firebase Auth error for {email_val}: {e}")
        if is_firebase_transient_error(e):
            _rollback_uncommitted_client_auth()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Layanan Firebase Authentication mengalami timeout koneksi sementara. Silakan coba kembali.",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gagal mendaftarkan akun autentikasi: {str(e)}",
        )

    # 3. Save complete employee profile to Firestore `users/{uid}` (joinDate aligned to server WIB)
    current_wib = now_wib()
    user_doc_data = {
        "uid": target_uid,
        "niy": niy_val,
        "employeeId": niy_val,
        "name": payload.name,
        "email": email_val,
        "phone": payload.phone,
        "phoneNumber": payload.phone,
        "department": payload.department,
        "position": payload.position,
        "employmentStatus": "Tetap",
        "joinDate": current_wib.strftime("%Y-%m-%d"),
        "role": enforced_role,
        "is_active": is_active,
        "isActive": is_active,
        "face_registered": False,
        "faceRegistered": False,
        "face_verification_status": FaceVerificationStatus.NOT_ENROLLED.value,
        "faceVerificationStatus": FaceVerificationStatus.NOT_ENROLLED.value,
        "face_rejection_reason": None,
        "faceRejectionReason": None,
        "face_submitted_at": None,
        "faceSubmittedAt": None,
        "created_at": firestore.SERVER_TIMESTAMP,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    try:
        await execute_firebase_with_retry_async(
            lambda: db.collection("users").document(target_uid).set(user_doc_data),
            operation_name="penyimpanan profil karyawan ke Cloud Firestore",
        )
        record_audit_event(
            db,
            action=ACTION_EMPLOYEE_REGISTERED,
            actor_uid=target_uid,
            target_id=target_uid,
            employee_name=payload.name,
            niy=niy_val,
            department=payload.department,
            status="success",
            details=f"Registrasi mandiri karyawan {payload.name} ({niy_val}) pada unit {payload.department}.",
            request=request,
        )
    except FirebaseServiceUnavailableError as e:
        logger.error(f"Firestore profile save timed out for {target_uid}; rolling back Auth user: {e}")
        try:
            fb_auth.delete_user(target_uid)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Firestore profile save failed for {target_uid}; rolling back Auth user: {e}")
        try:
            fb_auth.delete_user(target_uid)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal menyimpan profil karyawan ke database.",
        )

    status_msg = (
        "Pendaftaran akun karyawan berhasil! Akun Anda sudah aktif, silakan masuk ke portal."
        if is_active
        else "Pendaftaran berhasil! Akun Anda sedang menunggu verifikasi dan aktivasi oleh Administrator."
    )

    return SelfRegisterResponse(
        uid=target_uid,
        niy=niy_val,
        name=payload.name,
        email=email_val,
        phone=payload.phone,
        department=payload.department,
        position=payload.position,
        role=UserRole.EMPLOYEE,
        is_active=is_active,
        isActive=is_active,
        face_registered=False,
        faceRegistered=False,
        face_verification_status=FaceVerificationStatus.NOT_ENROLLED,
        faceVerificationStatus=FaceVerificationStatus.NOT_ENROLLED,
        message=status_msg,
    )


@router.post("/verify-token", response_model=TokenValidationResponse)
async def verify_token_endpoint(payload: VerifyTokenRequest, request: Request):
    """
    Validates a Firebase ID Token provided by the client.
    Returns user details and verified server-side role, and logs explicit LOGIN_ATTEMPT events.
    """
    try:
        decoded = await verify_firebase_id_token_async(payload.idToken)
        uid = decoded.get("uid")
        email = decoded.get("email")
        name = decoded.get("name")
        role = UserRole.EMPLOYEE
        face_registered = False
        face_status = FaceVerificationStatus.NOT_ENROLLED
        face_rejection_reason = None
        employee_id = None
        dept = "-"

        db = get_firestore_client()
        if db is not None:
            user_doc = db.collection("users").document(uid).get()
            if user_doc.exists:
                user_data = user_doc.to_dict() or {}
                name = user_data.get("name") or name
                employee_id = user_data.get("niy") or user_data.get("employeeId")
                dept = user_data.get("department") or "-"
                active_flag = user_data.get("isActive", user_data.get("is_active", True))
                if not active_flag:
                    record_audit_event(
                        db,
                        action=ACTION_LOGIN_ATTEMPT,
                        actor_uid=uid,
                        target_id=uid,
                        employee_name=name,
                        niy=employee_id,
                        department=dept,
                        status="failed",
                        details=f"Percobaan login ditolak: Akun {email or uid} berstatus nonaktif.",
                        request=request,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Akun karyawan belum aktif atau telah dinonaktifkan oleh Administrator.",
                    )
                if user_data.get("role") == "admin":
                    role = UserRole.ADMIN
                face_info = resolve_face_verification_fields(user_data)
                face_status = face_info["status"]
                face_registered = face_info["face_registered"]
                face_rejection_reason = face_info["rejection_reason"]
            elif decoded.get("role") == "admin":
                role = UserRole.ADMIN
        elif decoded.get("role") == "admin":
            role = UserRole.ADMIN

        if payload.isLoginAttempt and db is not None:
            record_audit_event(
                db,
                action=ACTION_LOGIN_ATTEMPT,
                actor_uid=uid,
                target_id=uid,
                employee_name=name or email or "Pengguna",
                niy=employee_id or ("ADM-PAUD-001" if role == UserRole.ADMIN else "-"),
                department=dept,
                status="success",
                details=f"Login berhasil ke portal CheckHadir.Id ({email or uid}, role: {role.value}).",
                request=request,
            )

        return TokenValidationResponse(
            valid=True,
            uid=uid,
            email=email,
            name=name,
            role=role,
            employeeId=employee_id,
            faceRegistered=face_registered,
            face_registered=face_registered,
            faceVerificationStatus=face_status,
            face_verification_status=face_status,
            faceRejectionReason=face_rejection_reason,
            face_rejection_reason=face_rejection_reason,
        )
    except HTTPException:
        raise
    except FirebaseServiceUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected token verification failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to verify authentication token.",
        )


@router.post("/login-audit", status_code=status.HTTP_202_ACCEPTED)
async def record_login_attempt_audit(payload: LoginAuditRequest, request: Request):
    """
    Records client-side Firebase authentication outcomes (such as unverified email blocks,
    wrong password, or locked account attempts) into the immutable `audit_logs` ledger.
    Rate-limited per IP to prevent abuse.
    """
    client_meta = extract_client_metadata(request)
    if not _allow_login_audit_ip(client_meta["ipAddress"]):
        return {"recorded": False}

    db = get_firestore_client()
    if db is None:
        return {"recorded": False}

    cleaned_email = payload.email.strip().lower()
    actor_uid = "anonymous"
    emp_name = cleaned_email
    emp_niy = "-"
    emp_dept = "-"

    if payload.idToken:
        try:
            decoded = await verify_firebase_id_token_async(payload.idToken)
            actor_uid = decoded.get("uid") or actor_uid
            emp_name = decoded.get("name") or emp_name
        except Exception:
            pass

    try:
        if actor_uid != "anonymous":
            snap = db.collection("users").document(actor_uid).get()
            if snap.exists:
                u = snap.to_dict() or {}
                emp_name = u.get("name") or emp_name
                emp_niy = u.get("niy") or u.get("employeeId") or emp_niy
                emp_dept = u.get("department") or emp_dept
        else:
            matches = list(db.collection("users").where("email", "==", cleaned_email).limit(1).stream())
            if matches:
                actor_uid = matches[0].id
                u = matches[0].to_dict() or {}
                emp_name = u.get("name") or emp_name
                emp_niy = u.get("niy") or u.get("employeeId") or emp_niy
                emp_dept = u.get("department") or emp_dept
    except Exception:
        pass

    reason_text = (payload.reason or "").strip()
    if not reason_text:
        reason_text = (
            f"Login berhasil ({cleaned_email})."
            if payload.status == "success"
            else f"Percobaan login gagal untuk akun {cleaned_email}."
        )
    elif cleaned_email not in reason_text.lower():
        reason_text = f"{reason_text} ({cleaned_email})"

    record_audit_event(
        db,
        action=ACTION_LOGIN_ATTEMPT,
        actor_uid=actor_uid,
        target_id=actor_uid if actor_uid != "anonymous" else cleaned_email,
        employee_name=emp_name,
        niy=emp_niy,
        department=emp_dept,
        status=payload.status,
        details=reason_text,
        request=request,
    )
    return {"recorded": True}


@router.get("/me", response_model=TokenData)
async def get_me(current_user: TokenData = Depends(get_current_user)):
    """
    Returns the currently authenticated user's verified token information.
    """
    return current_user
