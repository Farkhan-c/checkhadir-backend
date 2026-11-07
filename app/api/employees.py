"""
Employees API Router (Phase 2 — Employee Management)
Provides real Cloud Firestore + Firebase Authentication Full CRUD (Create, Read, Update, Delete),
search, department filtering, pagination, account status toggling, and biometric face reset workflows
for PAUD LAB UM KOTA BLITAR.
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from firebase_admin import auth as fb_auth, firestore

from app.models.auth import TokenData
from app.models.employee import (
    DepartmentModel,
    EmployeeCreate,
    EmployeeResponse,
    EmployeeUpdate,
    FaceVerificationStatus,
    UserRole,
    resolve_face_verification_fields,
)
from app.security.auth import get_current_user, require_admin
from app.core.firebase import (
    FirebaseServiceUnavailableError,
    execute_firebase_with_retry_async,
    get_firestore_client,
    is_firebase_ready,
)
from app.core.timezone import now_wib, to_wib
from app.services.audit_logger import (
    ACTION_EMPLOYEE_CREATED,
    ACTION_EMPLOYEE_DELETED,
    ACTION_EMPLOYEE_UPDATED,
    ACTION_FACE_RESET,
    record_audit_event,
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/employees", tags=["Employees"])

# Default Initial Departments for PAUD LAB UM KOTA BLITAR
DEFAULT_DEPARTMENTS = [
    {
        "departmentId": "dept_guru",
        "name": "Tenaga Pendidik (Guru)",
        "description": "Guru Kelas, Guru Pendamping, dan Pengajar PAUD/TK",
    },
    {
        "departmentId": "dept_tu",
        "name": "Tata Usaha & Administrasi",
        "description": "Administrasi, Keuangan, dan Pengelolaan Dokumen Sekolah",
    },
    {
        "departmentId": "dept_manajemen",
        "name": "Pimpinan & Manajemen",
        "description": "Kepala Sekolah, Wakil Kepala, dan Koordinator Kurikulum",
    },
    {
        "departmentId": "dept_sarpras",
        "name": "Sarana & Prasarana",
        "description": "Pengelolaan Fasilitas dan Kebersihan Lingkungan Sekolah",
    },
]


def _require_firestore():
    """Returns live Cloud Firestore client or raises 503 (never returns mock data)."""
    db = get_firestore_client()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Layanan database Cloud Firestore tidak tersedia.",
        )
    return db


def _parse_dt(val) -> datetime:
    if hasattr(val, "to_datetime"):
        return to_wib(val.to_datetime())
    if isinstance(val, datetime):
        return to_wib(val)
    if isinstance(val, str):
        try:
            return to_wib(datetime.fromisoformat(val))
        except Exception:
            pass
    return now_wib()


def _parse_optional_dt(val) -> Optional[datetime]:
    if val is None:
        return None
    if hasattr(val, "to_datetime"):
        return to_wib(val.to_datetime())
    if isinstance(val, datetime):
        return to_wib(val)
    if isinstance(val, str) and val.strip():
        try:
            return to_wib(datetime.fromisoformat(val.strip()))
        except Exception:
            return None
    return None


def _format_user_doc(uid: str, data: dict) -> EmployeeResponse:
    """Converts a Firestore `users/{uid}` document into EmployeeResponse."""
    resolved_id = data.get("niy") or data.get("employeeId") or f"NIY-{uid[:7].upper()}"
    is_active = data.get("isActive", data.get("is_active", True))
    face_info = resolve_face_verification_fields(data)
    face_status = face_info["status"]
    face_registered = face_info["face_registered"]
    rejection_reason = face_info["rejection_reason"]
    submitted_dt = _parse_optional_dt(face_info["submitted_raw"])
    created_raw = data.get("createdAt") or data.get("created_at")
    return EmployeeResponse(
        uid=uid,
        employeeId=data.get("employeeId") or resolved_id,
        niy=resolved_id,
        name=data.get("name", "Tanpa Nama"),
        email=data.get("email", ""),
        phoneNumber=data.get("phoneNumber") or data.get("phone"),
        department=data.get("department", "Umum"),
        departmentId=data.get("departmentId"),
        position=data.get("position", "Staf"),
        joinDate=data.get("joinDate"),
        employmentStatus=data.get("employmentStatus", "Tetap"),
        role=data.get("role", UserRole.EMPLOYEE),
        isActive=bool(is_active),
        faceRegistered=bool(face_registered),
        face_registered=bool(face_registered),
        faceVerificationStatus=face_status,
        face_verification_status=face_status,
        faceRejectionReason=rejection_reason,
        face_rejection_reason=rejection_reason,
        faceSubmittedAt=submitted_dt,
        face_submitted_at=submitted_dt,
        createdAt=_parse_dt(created_raw),
        updatedAt=_parse_dt(data.get("updatedAt") or created_raw),
    )


@router.get("/me", response_model=EmployeeResponse)
async def get_my_profile(current_user: TokenData = Depends(get_current_user)):
    """Returns the signed-in user's own profile from Cloud Firestore."""
    db = _require_firestore()

    doc = db.collection("users").document(current_user.uid).get()
    if not doc.exists:
        # Auto-provision initial Firestore profile if authenticated via Firebase Auth
        generated_niy = f"NIY-{current_user.uid[:7].upper()}"
        new_data = {
            "uid": current_user.uid,
            "employeeId": generated_niy,
            "niy": generated_niy,
            "name": current_user.name
            or ("Administrator" if current_user.role == UserRole.ADMIN else "Karyawan PAUD Lab UM"),
            "email": current_user.email or f"{current_user.uid}@paudlabum.sch.id",
            "department": "Pimpinan & Manajemen"
            if current_user.role == UserRole.ADMIN
            else "Tenaga Pendidik (Guru)",
            "position": "Kepala Sekolah"
            if current_user.role == UserRole.ADMIN
            else "Guru Pendamping",
            "role": current_user.role.value,
            "isActive": True,
            "is_active": True,
            "faceRegistered": False,
            "face_registered": False,
            "faceVerificationStatus": FaceVerificationStatus.NOT_ENROLLED.value,
            "face_verification_status": FaceVerificationStatus.NOT_ENROLLED.value,
            "faceRejectionReason": None,
            "face_rejection_reason": None,
            "faceSubmittedAt": None,
            "face_submitted_at": None,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
        db.collection("users").document(current_user.uid).set(new_data)
        doc = db.collection("users").document(current_user.uid).get()

    return _format_user_doc(current_user.uid, doc.to_dict() or {})


@router.get("/departments/list", response_model=List[DepartmentModel])
async def list_departments(current_user: TokenData = Depends(get_current_user)):
    """Returns organization departments with real employee counts from Cloud Firestore."""
    db = _require_firestore()
    departments = [DepartmentModel(**d, employeeCount=0) for d in DEFAULT_DEPARTMENTS]

    users_docs = db.collection("users").stream()
    counts = {}
    for doc in users_docs:
        dept_name = (doc.to_dict() or {}).get("department")
        if dept_name:
            counts[dept_name] = counts.get(dept_name, 0) + 1

    for dept in departments:
        dept.employeeCount = counts.get(dept.name, 0)

    return departments


@router.get("", response_model=List[EmployeeResponse])
async def list_employees(
    response: Response,
    q: Optional[str] = Query(None, description="Cari berdasarkan nama, NIY, atau email"),
    department: Optional[str] = Query(None, description="Filter berdasarkan nama departemen"),
    isActive: Optional[bool] = Query(None, description="Filter status aktif akun"),
    role: Optional[UserRole] = Query(None, description="Filter berdasarkan role"),
    skip: int = Query(0, ge=0, description="Offset pagination"),
    limit: int = Query(50, ge=1, le=200, description="Batas jumlah baris per halaman"),
    admin_user: TokenData = Depends(require_admin),
):
    """
    Admin-only: Lists employees from Cloud Firestore with search (name/NIY/email),
    department/status/role filtering, and pagination (`skip`, `limit`, `X-Total-Count` header).
    """
    db = _require_firestore()
    docs = db.collection("users").stream()

    results: List[EmployeeResponse] = []
    q_lower = q.strip().lower() if q else None

    for doc in docs:
        data = doc.to_dict() or {}
        emp = _format_user_doc(doc.id, data)

        if department and emp.department != department:
            continue
        if isActive is not None and emp.isActive != isActive:
            continue
        if role and emp.role != role:
            continue
        if q_lower:
            matched = (
                q_lower in emp.name.lower()
                or q_lower in emp.employeeId.lower()
                or q_lower in emp.niy.lower()
                or q_lower in emp.email.lower()
                or (emp.phoneNumber and q_lower in emp.phoneNumber.lower())
            )
            if not matched:
                continue

        results.append(emp)

    # Deterministic alphabetical sort before slicing pagination window
    results.sort(key=lambda item: (item.name.lower(), item.niy.lower()))
    response.headers["X-Total-Count"] = str(len(results))
    return results[skip : skip + limit]


@router.get("/{uid}", response_model=EmployeeResponse)
async def get_employee_detail(
    uid: str,
    admin_user: TokenData = Depends(require_admin),
):
    """Admin-only: Retrieves an employee's complete profile from Cloud Firestore."""
    db = _require_firestore()
    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Karyawan dengan UID '{uid}' tidak ditemukan.",
        )

    return _format_user_doc(uid, doc.to_dict() or {})


@router.post("", response_model=EmployeeResponse, status_code=status.HTTP_201_CREATED)
async def create_employee(
    payload: EmployeeCreate,
    request: Request,
    admin_user: TokenData = Depends(require_admin),
):
    """
    Admin-only: Creates a new employee in Firebase Authentication via Admin SDK
    and stores the corresponding profile document in Firestore `users/{uid}`.
    """
    db = _require_firestore()
    niy_value = (payload.niy or payload.employeeId or "").strip()
    emp_id_value = (payload.employeeId or niy_value).strip()
    email_value = str(payload.email).strip().lower()

    # 1. Check duplicate NIY / employeeId or email in Firestore
    existing_emp = list(
        db.collection("users").where("employeeId", "==", emp_id_value).limit(1).stream()
    )
    if not existing_emp:
        existing_emp = list(
            db.collection("users").where("niy", "==", niy_value).limit(1).stream()
        )
    if existing_emp:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"NIY / ID Karyawan '{niy_value}' sudah terdaftar.",
        )

    existing_email = list(
        db.collection("users").where("email", "==", email_value).limit(1).stream()
    )
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Alamat email '{email_value}' sudah digunakan.",
        )

    # 2. Provision Firebase Authentication user via Admin SDK
    target_uid = payload.uid
    if not target_uid:
        if not is_firebase_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Firebase Admin SDK belum siap untuk membuat akun autentikasi.",
            )
        try:
            user_record = fb_auth.create_user(
                email=email_value,
                password=payload.password,
                display_name=payload.name,
                email_verified=True,
                disabled=not payload.isActive,
            )
            target_uid = user_record.uid
            fb_auth.set_custom_user_claims(target_uid, {"role": payload.role.value})
            logger.info(f"Created Firebase Auth user: {target_uid} for {email_value}")
        except fb_auth.EmailAlreadyExistsError:
            existing_fb_user = fb_auth.get_user_by_email(email_value)
            if db.collection("users").document(existing_fb_user.uid).get().exists:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Akun autentikasi untuk email '{email_value}' sudah terdaftar.",
                )
            target_uid = existing_fb_user.uid
            fb_auth.update_user(
                target_uid,
                display_name=payload.name,
                email_verified=True,
                disabled=not payload.isActive,
            )
            fb_auth.set_custom_user_claims(target_uid, {"role": payload.role.value})
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create Firebase Auth user: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Gagal membuat akun Firebase Authentication: {str(e)}",
            )

    # 3. Persist employee document in Firestore `users/{uid}`
    current_wib = now_wib()
    cleaned_phone = payload.phoneNumber.strip() if payload.phoneNumber else None
    user_data = {
        "uid": target_uid,
        "employeeId": emp_id_value,
        "niy": niy_value,
        "name": payload.name.strip(),
        "email": email_value,
        "phone": cleaned_phone,
        "phoneNumber": cleaned_phone,
        "department": payload.department.strip(),
        "departmentId": payload.departmentId,
        "position": payload.position.strip(),
        "joinDate": payload.joinDate or current_wib.strftime("%Y-%m-%d"),
        "employmentStatus": payload.employmentStatus.value,
        "role": payload.role.value,
        "is_active": payload.isActive,
        "isActive": payload.isActive,
        "email_verified": True,
        "emailVerified": True,
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

    db.collection("users").document(target_uid).set(user_data)
    record_audit_event(
        db,
        action=ACTION_EMPLOYEE_CREATED,
        actor_uid=admin_user.uid,
        target_id=target_uid,
        employee_name=payload.name.strip(),
        niy=niy_value,
        department=payload.department.strip(),
        status="success",
        details=f"Admin menambahkan karyawan baru: {payload.name.strip()} ({niy_value}) di unit {payload.department.strip()}.",
        request=request,
        metadata={
            "name": payload.name.strip(),
            "niy": niy_value,
            "department": payload.department.strip(),
        },
    )

    return _format_user_doc(target_uid, {**user_data, "createdAt": current_wib, "updatedAt": current_wib})


@router.patch("/{uid}", response_model=EmployeeResponse)
async def update_employee(
    uid: str,
    payload: EmployeeUpdate,
    request: Request,
    admin_user: TokenData = Depends(require_admin),
):
    """
    Admin-only: Updates employee data (Name, NIY, Department, Position, WhatsApp, Role, Status)
    in Cloud Firestore and synchronizes Firebase Authentication.
    """
    db = _require_firestore()
    doc_ref = db.collection("users").document(uid)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Karyawan dengan UID '{uid}' tidak ditemukan.",
        )

    current_data = doc.to_dict() or {}
    update_fields = {}

    raw_new_niy = payload.niy if payload.niy is not None else payload.employeeId
    new_niy = raw_new_niy.strip() if raw_new_niy is not None else None
    if new_niy and new_niy != current_data.get("niy") and new_niy != current_data.get("employeeId"):
        dup = [
            d for d in db.collection("users").where("niy", "==", new_niy).limit(2).stream()
            if d.id != uid
        ]
        if not dup:
            dup = [
                d for d in db.collection("users").where("employeeId", "==", new_niy).limit(2).stream()
                if d.id != uid
            ]
        if dup:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"NIY / ID Karyawan '{new_niy}' sudah digunakan pegawai lain.",
            )
        update_fields["niy"] = new_niy
        update_fields["employeeId"] = (payload.employeeId or new_niy).strip()

    if payload.name is not None:
        update_fields["name"] = payload.name.strip()
    if payload.phoneNumber is not None:
        cleaned_phone = payload.phoneNumber.strip()
        update_fields["phoneNumber"] = cleaned_phone
        update_fields["phone"] = cleaned_phone
    if payload.department is not None:
        update_fields["department"] = payload.department.strip()
    if payload.departmentId is not None:
        update_fields["departmentId"] = payload.departmentId
    if payload.position is not None:
        update_fields["position"] = payload.position.strip()
    if payload.joinDate is not None:
        update_fields["joinDate"] = payload.joinDate
    if payload.employmentStatus is not None:
        update_fields["employmentStatus"] = payload.employmentStatus.value
    if payload.role is not None:
        update_fields["role"] = payload.role.value
        if is_firebase_ready():
            try:
                fb_auth.set_custom_user_claims(uid, {"role": payload.role.value})
            except Exception as e:
                logger.warning(f"Could not update custom claim for {uid}: {e}")
    if payload.isActive is not None:
        update_fields["isActive"] = payload.isActive
        update_fields["is_active"] = payload.isActive

    if is_firebase_ready() and (payload.name is not None or payload.isActive is not None):
        auth_kwargs = {}
        if payload.name is not None:
            auth_kwargs["display_name"] = payload.name.strip()
        if payload.isActive is not None:
            auth_kwargs["disabled"] = not payload.isActive
        try:
            fb_auth.update_user(uid, **auth_kwargs)
        except Exception as e:
            logger.warning(f"Could not sync Firebase Auth user {uid}: {e}")

    changed_keys = sorted([k for k in update_fields.keys()])
    update_fields["updatedAt"] = firestore.SERVER_TIMESTAMP
    doc_ref.update(update_fields)

    updated_data = doc_ref.get().to_dict() or {}
    emp_name = updated_data.get("name") or current_data.get("name") or uid
    emp_niy = updated_data.get("niy") or updated_data.get("employeeId") or "-"

    record_audit_event(
        db,
        action=ACTION_EMPLOYEE_UPDATED,
        actor_uid=admin_user.uid,
        target_id=uid,
        employee_name=emp_name,
        niy=emp_niy,
        department=updated_data.get("department"),
        status="success",
        details=f"Data karyawan {emp_name} ({emp_niy}) diperbarui oleh Admin ({', '.join(changed_keys) or 'tanpa perubahan'}).",
        request=request,
    )

    return _format_user_doc(uid, updated_data)


@router.delete("/{uid}")
async def delete_employee(
    uid: str,
    request: Request,
    admin_user: TokenData = Depends(require_admin),
):
    """
    Admin-only: Permanently deletes an employee account:
    1. Purges biometric vector in Firestore `face_profiles/{uid}`.
    2. Deletes profile document in Firestore `users/{uid}`.
    3. Deletes Firebase Authentication account via `auth.delete_user(uid)`.
    """
    if uid == admin_user.uid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Anda tidak dapat menghapus akun Administrator yang sedang Anda gunakan sendiri.",
        )

    db = _require_firestore()
    user_ref = db.collection("users").document(uid)
    user_snap = user_ref.get()

    emp_data = user_snap.to_dict() or {} if user_snap.exists else {}
    emp_name = emp_data.get("name") or uid
    emp_niy = emp_data.get("niy") or emp_data.get("employeeId") or "-"
    emp_email = emp_data.get("email") or "-"
    emp_dept = emp_data.get("department") or "-"

    # Verify existence in either Firestore or Firebase Auth before deleting
    auth_exists = False
    if is_firebase_ready():
        try:
            fb_user = fb_auth.get_user(uid)
            auth_exists = True
            if emp_name == uid and fb_user.display_name:
                emp_name = fb_user.display_name
            if emp_email == "-" and fb_user.email:
                emp_email = fb_user.email
        except fb_auth.UserNotFoundError:
            auth_exists = False
        except Exception as e:
            logger.warning(f"Firebase Auth lookup warning prior to delete for {uid}: {e}")
            auth_exists = True  # still attempt delete_user below

    if not user_snap.exists and not auth_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Karyawan dengan UID '{uid}' tidak ditemukan.",
        )

    try:
        # 1. Delete biometric template in `face_profiles/{uid}`
        await execute_firebase_with_retry_async(
            lambda: db.collection("face_profiles").document(uid).delete(),
            operation_name="penghapusan profil biometrik wajah",
        )

        # 2. Delete user document in `users/{uid}`
        if user_snap.exists:
            await execute_firebase_with_retry_async(
                lambda: user_ref.delete(),
                operation_name="penghapusan dokumen profil karyawan",
            )

        # 3. Delete Firebase Authentication account
        if is_firebase_ready():
            try:
                await execute_firebase_with_retry_async(
                    lambda: fb_auth.delete_user(uid),
                    operation_name="penghapusan akun Firebase Authentication",
                )
            except fb_auth.UserNotFoundError:
                logger.info(f"Firebase Auth user {uid} already absent during permanent delete.")
    except FirebaseServiceUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to permanently delete employee {uid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal menghapus akun karyawan secara permanen: {str(e)}",
        )

    record_audit_event(
        db,
        action=ACTION_EMPLOYEE_DELETED,
        actor_uid=admin_user.uid,
        target_id=uid,
        employee_name=emp_name,
        niy=emp_niy,
        department=emp_dept,
        status="success",
        details=f"Akun karyawan {emp_name} ({emp_niy} • {emp_email}) beserta biometrik wajah dihapus permanen oleh Admin.",
        request=request,
    )

    return {
        "success": True,
        "uid": uid,
        "message": f"Karyawan {emp_name} ({emp_niy}) beserta profil biometrik wajah dan akun login berhasil dihapus permanen.",
    }


@router.post("/{uid}/deactivate", response_model=EmployeeResponse)
async def deactivate_employee(
    uid: str,
    request: Request,
    admin_user: TokenData = Depends(require_admin),
):
    """Admin-only: Deactivates an employee account in Firestore and Firebase Auth."""
    db = _require_firestore()
    doc_ref = db.collection("users").document(uid)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Karyawan tidak ditemukan.")

    u_data = doc.to_dict() or {}
    doc_ref.update({"isActive": False, "is_active": False, "updatedAt": firestore.SERVER_TIMESTAMP})
    if is_firebase_ready():
        try:
            fb_auth.update_user(uid, disabled=True)
        except Exception as e:
            logger.warning(f"Failed to disable Firebase Auth user: {e}")

    emp_name = u_data.get("name") or uid
    emp_niy = u_data.get("niy") or u_data.get("employeeId") or "-"
    record_audit_event(
        db,
        action=ACTION_EMPLOYEE_UPDATED,
        actor_uid=admin_user.uid,
        target_id=uid,
        employee_name=emp_name,
        niy=emp_niy,
        department=u_data.get("department"),
        status="success",
        details=f"Akun karyawan {emp_name} ({emp_niy}) dinonaktifkan oleh Admin.",
        request=request,
    )
    return _format_user_doc(uid, doc_ref.get().to_dict() or {})


@router.post("/{uid}/activate", response_model=EmployeeResponse)
async def activate_employee(
    uid: str,
    request: Request,
    admin_user: TokenData = Depends(require_admin),
):
    """Admin-only: Reactivates an employee account in Firestore and Firebase Auth."""
    db = _require_firestore()
    doc_ref = db.collection("users").document(uid)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Karyawan tidak ditemukan.")

    u_data = doc.to_dict() or {}
    doc_ref.update({"isActive": True, "is_active": True, "updatedAt": firestore.SERVER_TIMESTAMP})
    if is_firebase_ready():
        try:
            fb_auth.update_user(uid, disabled=False)
        except Exception as e:
            logger.warning(f"Failed to enable Firebase Auth user: {e}")

    emp_name = u_data.get("name") or uid
    emp_niy = u_data.get("niy") or u_data.get("employeeId") or "-"
    record_audit_event(
        db,
        action=ACTION_EMPLOYEE_UPDATED,
        actor_uid=admin_user.uid,
        target_id=uid,
        employee_name=emp_name,
        niy=emp_niy,
        department=u_data.get("department"),
        status="success",
        details=f"Akun karyawan {emp_name} ({emp_niy}) diaktifkan kembali oleh Admin.",
        request=request,
    )
    return _format_user_doc(uid, doc_ref.get().to_dict() or {})


@router.post("/{uid}/reset-face", response_model=EmployeeResponse)
async def reset_employee_face(
    uid: str,
    request: Request,
    admin_user: TokenData = Depends(require_admin),
):
    """Admin-only: Purges biometric template in `face_profiles/{uid}` and resets `face_registered: false`."""
    db = _require_firestore()
    doc_ref = db.collection("users").document(uid)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Karyawan tidak ditemukan.")

    u_data = doc.to_dict() or {}
    emp_name = u_data.get("name") or uid
    emp_niy = u_data.get("niy") or u_data.get("employeeId") or "-"

    doc_ref.update(
        {
            "faceRegistered": False,
            "face_registered": False,
            "faceVerificationStatus": FaceVerificationStatus.NOT_ENROLLED.value,
            "face_verification_status": FaceVerificationStatus.NOT_ENROLLED.value,
            "faceRejectionReason": None,
            "face_rejection_reason": None,
            "faceSubmittedAt": None,
            "face_submitted_at": None,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
    )

    db.collection("face_profiles").document(uid).delete()
    logger.info(f"Purged biometric face profile for {uid} ({emp_name})")

    record_audit_event(
        db,
        action=ACTION_FACE_RESET,
        actor_uid=admin_user.uid,
        target_id=uid,
        employee_name=emp_name,
        niy=emp_niy,
        department=u_data.get("department"),
        status="success",
        details=f"Biometrik wajah karyawan {emp_name} ({emp_niy}) telah di-reset oleh Admin agar dapat didaftarkan ulang.",
        request=request,
    )
    return _format_user_doc(uid, doc_ref.get().to_dict() or {})
