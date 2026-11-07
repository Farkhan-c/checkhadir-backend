"""
Unit Tests for Registration 503 Resilience, Admin Employee Full CRUD & Biometric Reset,
and Real-Time Activity Monitoring / Audit Trail (`audit_logs`).
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException

from app.core import firebase as fb_core
from app.models.auth import LoginAuditRequest, SelfRegisterRequest, TokenData, UserRole
from app.models.employee import EmployeeUpdate
from app.services.audit_logger import (
    ACTION_ATTENDANCE_CHECKIN,
    ACTION_ATTENDANCE_CHECKOUT,
    ACTION_ATTENDANCE_REJECTED,
    ACTION_EMPLOYEE_DELETED,
    ACTION_EMPLOYEE_UPDATED,
    ACTION_FACE_APPROVAL_ACCEPTED,
    ACTION_FACE_APPROVAL_REJECTED,
    ACTION_FACE_ENROLLMENT,
    ACTION_FACE_RESET,
    ACTION_LEAVE_APPLICATION,
    ACTION_LOGIN_ATTEMPT,
    ACTION_SPOOF_ATTEMPT,
    extract_client_metadata,
    normalize_audit_action,
)


def _drive_coro(coro):
    """Drives a pure coroutine synchronously without spawning an OS socketpair event loop."""
    try:
        while True:
            coro.send(None)
    except StopIteration as stop:
        return stop.value


def _make_fake_request(ip: str = "10.77.209.55", ua: str = "Mozilla/5.0 (Linux; Android 14) Chrome/131.0 Mobile"):
    headers = {
        "x-forwarded-for": f"{ip}, 127.0.0.1",
        "user-agent": ua,
    }
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host="127.0.0.1"),
    )


def test_fastapi_app_registers_all_crud_and_audit_routes():
    """Ensures `app.main:app` imports cleanly with all required CRUD, auth, and audit endpoints."""
    from app.main import app

    schema_paths = app.openapi()["paths"]

    assert "/api/auth/register" in schema_paths
    assert "/api/auth/login-audit" in schema_paths
    assert "/api/employees/{uid}" in schema_paths
    assert "/api/employees/{uid}/reset-face" in schema_paths
    assert "/api/admin/face-approvals" in schema_paths
    assert "/api/admin/face-approvals/{uid}/approve" in schema_paths
    assert "/api/admin/face-approvals/{uid}/reject" in schema_paths
    assert "/api/audit" in schema_paths

    emp_uid_methods = {m.upper() for m in schema_paths["/api/employees/{uid}"].keys()}
    assert {"GET", "PATCH", "DELETE"}.issubset(emp_uid_methods)


def test_initialize_firebase_reuses_existing_default_app_without_503():
    """Prevents reload-induced ValueError from leaving `_firebase_app = None` (HTTP 503 root cause)."""
    fake_existing_app = MagicMock(name="ExistingFirebaseApp")
    fake_db = MagicMock(name="FirestoreClient")

    with patch.object(fb_core, "_firebase_app", None), patch.object(
        fb_core, "_firestore_db", None
    ), patch.object(fb_core, "_is_connected_to_live", False), patch.object(
        fb_core.firebase_admin, "_apps", {"[DEFAULT]": fake_existing_app}
    ), patch.object(
        fb_core.firebase_admin, "get_app", return_value=fake_existing_app
    ), patch.object(
        fb_core.firestore, "client", return_value=fake_db
    ):
        app_ref = fb_core.initialize_firebase()
        assert app_ref is fake_existing_app
        assert fb_core.is_firebase_ready() is True
        assert fb_core.get_firestore_client() is fake_db


def test_initialize_firebase_from_service_account_json_env_when_file_missing(monkeypatch):
    """
    Verifies cloud deployment mode: when `serviceAccountKey.json` is not present on disk,
    `initialize_firebase()` reads credentials from `FIREBASE_SERVICE_ACCOUNT_JSON`.
    """
    import json

    fake_sa = {
        "type": "service_account",
        "project_id": "employee-attendance-syst-39d40",
        "private_key": "-----BEGIN PRIVATE KEY-----\\nFAKE_KEY_LINE\\n-----END PRIVATE KEY-----\\n",
        "client_email": "firebase-adminsdk@employee-attendance-syst-39d40.iam.gserviceaccount.com",
    }
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_JSON", json.dumps(fake_sa))

    fake_cert = MagicMock(name="FakeCert")
    fake_app = MagicMock(name="FakeCloudApp")

    with patch.object(fb_core, "_firebase_app", None), patch.object(
        fb_core, "_firestore_db", None
    ), patch.object(fb_core, "_is_connected_to_live", False), patch.object(
        fb_core.firebase_admin, "_apps", {}
    ), patch.object(
        fb_core, "detect_service_account_path", return_value=None
    ), patch.object(
        fb_core.credentials, "Certificate", return_value=fake_cert
    ) as mock_cert, patch.object(
        fb_core.firebase_admin, "initialize_app", return_value=fake_app
    ) as mock_init_app:
        app_out = fb_core.initialize_firebase()
        assert app_out is fake_app
        assert fb_core.is_firebase_ready() is True
        passed_dict = mock_cert.call_args[0][0]
        assert passed_dict["project_id"] == "employee-attendance-syst-39d40"
        assert "\nFAKE_KEY_LINE\n" in passed_dict["private_key"]
        mock_init_app.assert_called_once()


def test_self_register_retries_transient_503_and_formats_unavailable_error():
    """
    Verifies `POST /api/auth/register`:
    1. Automatically retries transient Firebase `503 Service Unavailable` once and succeeds.
    2. Returns a clear Indonesian HTTP 503 message when Firebase remains unreachable.
    """
    from app.api import auth as auth_api

    payload = SelfRegisterRequest(
        niy="NIY-2026-777",
        name="Siti Aminah",
        email="siti.aminah@paudlabum.sch.id",
        phone="081234567891",
        password="Password123!",
        department="Tenaga Pendidik (Guru)",
        position="Guru Kelompok B",
    )

    # Case A: First Firestore query throws transient 503, retry succeeds
    attempts = {"where_calls": 0}
    user_doc_ref = MagicMock()
    audit_col = MagicMock()
    users_col = MagicMock()

    def flaky_where(*args, **kwargs):
        attempts["where_calls"] += 1
        if attempts["where_calls"] == 1:
            raise RuntimeError("503 Service Unavailable: Socket closed")
        q = MagicMock()
        q.limit.return_value.stream.return_value = []
        return q

    users_col.where.side_effect = flaky_where
    users_col.document.return_value = user_doc_ref

    db = MagicMock()
    db.collection.side_effect = lambda name: users_col if name == "users" else audit_col

    async def no_sleep(_seconds: float):
        return None

    with patch.object(auth_api, "is_firebase_ready", return_value=True), patch.object(
        auth_api, "get_firestore_client", return_value=db
    ), patch.object(
        auth_api.fb_auth, "create_user", return_value=SimpleNamespace(uid="uid_siti_777")
    ), patch.object(
        auth_api.fb_auth, "set_custom_user_claims"
    ), patch.object(
        fb_core, "_async_sleep", side_effect=no_sleep
    ):
        resp = _drive_coro(auth_api.register_employee_self(payload, _make_fake_request()))

    assert resp.uid == "uid_siti_777"
    assert resp.role == UserRole.EMPLOYEE
    assert attempts["where_calls"] >= 2  # Retried after initial 503
    user_doc_ref.set.assert_called_once()

    # Case B: Persistent network outage -> HTTP 503 with informative Indonesian message
    broken_users_col = MagicMock()
    broken_users_col.where.side_effect = RuntimeError("DeadlineExceeded: Timeout contacting Firestore")
    broken_db = MagicMock()
    broken_db.collection.return_value = broken_users_col

    with patch.object(auth_api, "is_firebase_ready", return_value=True), patch.object(
        auth_api, "get_firestore_client", return_value=broken_db
    ), patch.object(fb_core, "_async_sleep", side_effect=no_sleep):
        with pytest.raises(HTTPException) as exc_info:
            _drive_coro(auth_api.register_employee_self(payload, _make_fake_request()))
        assert exc_info.value.status_code == 503
        assert "Firebase" in exc_info.value.detail or "timeout" in exc_info.value.detail.lower()


def test_admin_delete_reset_face_and_edit_employee():
    """
    Verifies Full Employee CRUD & Biometric Reset in `app/api/employees.py`:
    - `PATCH /api/employees/{uid}` edits Nama, NIY, Departemen, Posisi, Nomor WhatsApp.
    - `POST /api/employees/{uid}/reset-face` deletes `face_profiles/{uid}` & sets `face_registered: False`.
    - `DELETE /api/employees/{uid}` blocks self-delete (400), deletes `face_profiles/{uid}`, `users/{uid}`,
      and `fb_auth.delete_user(uid)`, and records `EMPLOYEE_DELETED`.
    """
    from app.api import employees as emp_api

    admin_token = TokenData(uid="admin_uid_001", email="admin@checkhadir.id", role=UserRole.ADMIN)
    req = _make_fake_request()

    stored_user = {
        "uid": "emp_uid_101",
        "niy": "NIY-101",
        "employeeId": "NIY-101",
        "name": "Rina Kartika",
        "email": "rina@paudlabum.sch.id",
        "phone": "+628123000111",
        "phoneNumber": "+628123000111",
        "department": "Tenaga Pendidik (Guru)",
        "position": "Guru Kelas A",
        "role": "employee",
        "isActive": True,
        "is_active": True,
        "faceRegistered": True,
        "face_registered": True,
    }

    user_doc_ref = MagicMock()
    user_doc_ref.get.side_effect = lambda: SimpleNamespace(exists=True, to_dict=lambda: dict(stored_user))

    def apply_update(fields: dict):
        stored_user.update(fields)

    user_doc_ref.update.side_effect = apply_update

    users_col = MagicMock()
    users_col.document.return_value = user_doc_ref
    empty_query = MagicMock()
    empty_query.limit.return_value.stream.return_value = []
    users_col.where.return_value = empty_query

    face_doc_ref = MagicMock()
    faces_col = MagicMock()
    faces_col.document.return_value = face_doc_ref

    audit_events = []
    audit_col = MagicMock()
    audit_col.add.side_effect = lambda data: audit_events.append(data)

    def route_col(name: str):
        if name == "users":
            return users_col
        if name == "face_profiles":
            return faces_col
        if name == "audit_logs":
            return audit_col
        return MagicMock()

    db = MagicMock()
    db.collection.side_effect = route_col

    with patch.object(emp_api, "_require_firestore", return_value=db), patch.object(
        emp_api, "is_firebase_ready", return_value=True
    ), patch.object(emp_api.fb_auth, "update_user") as mock_update_auth, patch.object(
        emp_api.fb_auth, "get_user", return_value=SimpleNamespace(uid="emp_uid_101", display_name="Rina", email="rina@paudlabum.sch.id")
    ), patch.object(
        emp_api.fb_auth, "delete_user"
    ) as mock_delete_auth:
        # 1. Edit employee profile (Nama, NIY, Departemen, Posisi, Nomor WhatsApp)
        edit_payload = EmployeeUpdate(
            niy="NIY-101-REV",
            name="Rina Kartika, S.Pd.",
            department="Manajemen & Kepala Sekolah",
            position="Koordinator Kurikulum",
            phoneNumber="+6285799998888",
        )
        edited = _drive_coro(
            emp_api.update_employee("emp_uid_101", edit_payload, req, admin_user=admin_token)
        )
        assert edited.name == "Rina Kartika, S.Pd."
        assert edited.niy == "NIY-101-REV"
        assert edited.department == "Manajemen & Kepala Sekolah"
        assert edited.position == "Koordinator Kurikulum"
        assert edited.phoneNumber == "+6285799998888"
        mock_update_auth.assert_called_once()
        assert audit_events[-1]["action"] == ACTION_EMPLOYEE_UPDATED

        # 2. Reset biometric face profile
        reset_res = _drive_coro(
            emp_api.reset_employee_face("emp_uid_101", req, admin_user=admin_token)
        )
        assert reset_res.faceRegistered is False
        assert stored_user["face_registered"] is False
        face_doc_ref.delete.assert_called_once()
        assert audit_events[-1]["action"] == ACTION_FACE_RESET

        # 3. Admin self-delete must be blocked with HTTP 400
        with pytest.raises(HTTPException) as self_del_exc:
            _drive_coro(emp_api.delete_employee("admin_uid_001", req, admin_user=admin_token))
        assert self_del_exc.value.status_code == 400

        # 4. Permanent delete employee
        face_doc_ref.delete.reset_mock()
        del_res = _drive_coro(emp_api.delete_employee("emp_uid_101", req, admin_user=admin_token))
        assert del_res["success"] is True
        face_doc_ref.delete.assert_called_once()
        user_doc_ref.delete.assert_called_once()
        mock_delete_auth.assert_called_once_with("emp_uid_101")
        assert audit_events[-1]["action"] == ACTION_EMPLOYEE_DELETED


def test_login_audit_and_audit_logs_filtering():
    """
    Verifies:
    - `POST /api/auth/login-audit` logs failed login attempts (e.g., unverified email) with IP & device metadata.
    - `GET /api/audit` normalizes all 6 required activity categories, formats WIB timestamps,
      and filters by employee name/NIY (`q`) and WIB date (`startDate`/`endDate`).
    """
    from app.api import audit as audit_api
    from app.api import auth as auth_api

    req = _make_fake_request(
        ip="114.122.10.25",
        ua="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
    )
    meta = extract_client_metadata(req)
    assert meta["ipAddress"] == "114.122.10.25"
    assert "Chrome" in meta["deviceInfo"] and "Windows" in meta["deviceInfo"]

    recorded_logs = []
    audit_col = MagicMock()
    audit_col.add.side_effect = lambda doc: recorded_logs.append(doc)

    emp_doc = SimpleNamespace(
        id="emp_budi",
        exists=True,
        to_dict=lambda: {
            "name": "Budi Santoso",
            "niy": "NIY-2026-012",
            "department": "Tenaga Pendidik (Guru)",
        },
    )
    users_col = MagicMock()
    users_col.where.return_value.limit.return_value.stream.return_value = [emp_doc]
    users_col.stream.return_value = [emp_doc]

    db = MagicMock()
    db.collection.side_effect = lambda name: users_col if name == "users" else audit_col

    with patch.object(auth_api, "get_firestore_client", return_value=db):
        res = _drive_coro(
            auth_api.record_login_attempt_audit(
                LoginAuditRequest(
                    email="budi@paudlabum.sch.id",
                    status="failed",
                    reason="Login ditolak karena email belum diverifikasi",
                ),
                req,
            )
        )
    assert res["recorded"] is True
    assert len(recorded_logs) == 1
    assert recorded_logs[0]["action"] == ACTION_LOGIN_ATTEMPT
    assert recorded_logs[0]["status"] == "failed"
    assert recorded_logs[0]["niy"] == "NIY-2026-012"
    assert recorded_logs[0]["ipAddress"] == "114.122.10.25"

    # Verify all 6 required activity categories normalize properly & filter in `list_audit_logs`
    sample_docs = [
        ("log_1", {
            "actorUid": "emp_budi",
            "employeeName": "Budi Santoso",
            "niy": "NIY-2026-012",
            "action": ACTION_LOGIN_ATTEMPT,
            "status": "failed",
            "details": "Email belum diverifikasi",
            "ipAddress": "114.122.10.25",
            "deviceInfo": "Chrome • Windows",
            "timestamp": datetime(2026, 11, 8, 0, 30, 0, tzinfo=timezone.utc),  # 07:30 WIB on 2026-11-08
            "workDateWib": "2026-11-08",
        }),
        ("log_2", {
            "actorUid": "emp_budi",
            "employeeName": "Budi Santoso",
            "niy": "NIY-2026-012",
            "action": ACTION_FACE_ENROLLMENT,
            "status": "success",
            "details": "Pendaftaran profil biometrik wajah berhasil.",
            "ipAddress": "114.122.10.25",
            "deviceInfo": "Chrome • Android",
            "timestamp": datetime(2026, 11, 8, 0, 35, 0, tzinfo=timezone.utc),
            "workDateWib": "2026-11-08",
        }),
        ("log_3", {
            "actorUid": "emp_budi",
            "employeeName": "Budi Santoso",
            "niy": "NIY-2026-012",
            "action": ACTION_ATTENDANCE_CHECKIN,
            "status": "success",
            "details": "Check-In berhasil (Hadir). Jarak GPS: 14.2m • Skor SFace: 0.812.",
            "metadata": {"distanceMeters": 14.2, "similarity": 0.812},
            "ipAddress": "114.122.10.25",
            "deviceInfo": "Chrome • Android",
            "timestamp": datetime(2026, 11, 8, 0, 40, 0, tzinfo=timezone.utc),
            "workDateWib": "2026-11-08",
        }),
        ("log_4", {
            "actorUid": "emp_siti",
            "employeeName": "Siti Aminah",
            "niy": "NIY-2026-099",
            "action": ACTION_ATTENDANCE_REJECTED,
            "status": "failed",
            "details": "Presensi Check-In ditolak: berada di luar radius kantor (245.0m > 100m).",
            "ipAddress": "10.77.209.88",
            "deviceInfo": "Safari • iOS",
            "timestamp": datetime(2026, 11, 7, 1, 0, 0, tzinfo=timezone.utc),
            "workDateWib": "2026-11-07",
        }),
        ("log_5", {
            "actorUid": "emp_siti",
            "employeeName": "Siti Aminah",
            "niy": "NIY-2026-099",
            "action": ACTION_SPOOF_ATTEMPT,
            "status": "failed",
            "details": "Indikasi spoofing layar/foto terdeteksi saat verifikasi liveness.",
            "ipAddress": "10.77.209.88",
            "deviceInfo": "Safari • iOS",
            "timestamp": datetime(2026, 11, 8, 1, 10, 0, tzinfo=timezone.utc),
            "workDateWib": "2026-11-08",
        }),
        ("log_6", {
            "actorUid": "emp_budi",
            "employeeName": "Budi Santoso",
            "niy": "NIY-2026-012",
            "action": ACTION_LEAVE_APPLICATION,
            "status": "success",
            "details": "Pengajuan izin/cuti baru diajukan.",
            "ipAddress": "114.122.10.25",
            "deviceInfo": "Chrome • Windows",
            "timestamp": datetime(2026, 11, 8, 2, 0, 0, tzinfo=timezone.utc),
            "workDateWib": "2026-11-08",
        }),
    ]

    assert normalize_audit_action("attendance_checkin") == ACTION_ATTENDANCE_CHECKIN
    assert normalize_audit_action("attendance_checkout") == ACTION_ATTENDANCE_CHECKOUT

    mock_audit_query = MagicMock()
    mock_audit_query.order_by.return_value.limit.return_value.stream.return_value = [
        SimpleNamespace(id=doc_id, to_dict=lambda d=d: d) for doc_id, d in sample_docs
    ]
    query_db = MagicMock()
    query_db.collection.side_effect = lambda name: mock_audit_query if name == "audit_logs" else users_col

    admin_token = TokenData(uid="admin_uid_001", email="admin@checkhadir.id", role=UserRole.ADMIN)

    with patch.object(audit_api, "get_firestore_client", return_value=query_db):
        # Filter by WIB date 2026-11-08 -> excludes log_4 (2026-11-07)
        today_logs = _drive_coro(
            audit_api.list_audit_logs(
                action=None,
                status=None,
                q=None,
                date=None,
                startDate="2026-11-08",
                endDate="2026-11-08",
                limit=100,
                admin_user=admin_token,
            )
        )
        assert len(today_logs) == 5
        assert all(item.workDateWib == "2026-11-08" for item in today_logs)
        assert all(item.timestampWib and "WIB" in item.timestampWib for item in today_logs)

        # Filter by employee NIY search (`q="NIY-2026-099"`)
        siti_logs = _drive_coro(
            audit_api.list_audit_logs(
                action=None,
                status=None,
                q="NIY-2026-099",
                date=None,
                startDate=None,
                endDate=None,
                limit=100,
                admin_user=admin_token,
            )
        )
        assert len(siti_logs) == 2
        assert {item.action for item in siti_logs} == {ACTION_ATTENDANCE_REJECTED, ACTION_SPOOF_ATTEMPT}


def test_face_verification_approval_and_rejection_workflow():
    """
    Verifies the complete Face Verification & Admin Approval Workflow:
    1. Employee self-enrollment (`POST /api/face/register`) sets `PENDING_APPROVAL`,
       `face_registered = False`, `face_submitted_at` (WIB), and stores `previewThumbnail` in `face_profiles/{uid}`.
    2. `GET /api/admin/face-approvals` lists pending employees with `previewThumbnail` & profile info.
    3. `POST /api/admin/face-approvals/{uid}/reject` sets `REJECTED`, `face_registered = False`,
       stores `face_rejection_reason`, deletes `face_profiles/{uid}`, and logs `FACE_APPROVAL_REJECTED`.
    4. Rejected employee can cleanly re-enroll (`PENDING_APPROVAL`), and
       `POST /api/admin/face-approvals/{uid}/approve` sets `APPROVED`, `face_registered = True`,
       and logs `FACE_APPROVAL_ACCEPTED` with the approving Admin's name.
    """
    import numpy as np
    from app.api import face as face_api
    from app.models.face import FaceApprovalRejectRequest, FaceRegisterRequest
    from app.models.user import FaceVerificationStatus

    req = _make_fake_request()
    emp_token = TokenData(uid="emp_dewi_01", email="dewi@paudlabum.sch.id", name="Dewi Lestari", role=UserRole.EMPLOYEE)
    admin_token = TokenData(
        uid="admin_uid_001",
        email="admin@checkhadir.id",
        name="Kepala Sekolah Admin",
        role=UserRole.ADMIN,
    )

    users_store = {
        "emp_dewi_01": {
            "uid": "emp_dewi_01",
            "niy": "NIY-2026-205",
            "employeeId": "NIY-2026-205",
            "name": "Dewi Lestari",
            "email": "dewi@paudlabum.sch.id",
            "department": "Tenaga Pendidik (Guru)",
            "position": "Guru Kelompok A",
            "role": "employee",
            "isActive": True,
            "faceRegistered": False,
            "face_registered": False,
            "faceVerificationStatus": "NOT_ENROLLED",
            "face_verification_status": "NOT_ENROLLED",
        },
        "admin_uid_001": {
            "uid": "admin_uid_001",
            "niy": "ADM-001",
            "name": "Kepala Sekolah Admin",
            "role": "admin",
        },
    }
    face_profiles_store: dict[str, dict] = {}
    audit_events: list[dict] = []

    def make_doc_ref(store: dict, doc_id: str):
        ref = MagicMock()
        ref.get.side_effect = lambda: SimpleNamespace(
            id=doc_id,
            exists=doc_id in store,
            to_dict=lambda: dict(store[doc_id]) if doc_id in store else {},
        )
        ref.set.side_effect = lambda data: store.__setitem__(doc_id, dict(data))
        ref.update.side_effect = lambda fields: store.setdefault(doc_id, {}).update(fields)
        ref.delete.side_effect = lambda: store.pop(doc_id, None)
        return ref

    users_col = MagicMock()
    users_col.document.side_effect = lambda doc_id: make_doc_ref(users_store, doc_id)

    def users_where(field_name: str, op: str, val: str):
        q = MagicMock()
        q.stream.side_effect = lambda: [
            SimpleNamespace(id=k, to_dict=lambda v=v: dict(v))
            for k, v in users_store.items()
            if v.get(field_name) == val
        ]
        return q

    users_col.where.side_effect = users_where

    faces_col = MagicMock()
    faces_col.document.side_effect = lambda doc_id: make_doc_ref(face_profiles_store, doc_id)
    faces_col.stream.side_effect = lambda: [
        SimpleNamespace(id=k, to_dict=lambda v=v: dict(v))
        for k, v in face_profiles_store.items()
    ]

    audit_col = MagicMock()
    audit_col.add.side_effect = lambda doc: audit_events.append(doc)

    def collection_router(name: str):
        if name == "users":
            return users_col
        if name == "face_profiles":
            return faces_col
        if name == "audit_logs":
            return audit_col
        return MagicMock()

    db = MagicMock()
    db.collection.side_effect = collection_router

    dummy_emb = (np.ones(128, dtype=np.float32) / np.sqrt(128))
    fake_samples = [
        "data:image/jpeg;base64,SAMPLE_1",
        "data:image/jpeg;base64,SAMPLE_2",
        "data:image/jpeg;base64,SAMPLE_3",
    ]

    with patch.object(face_api, "get_firestore_client", return_value=db), patch.object(
        face_api.face_engine, "process_enrollment_samples", return_value=(dummy_emb, [])
    ), patch.object(
        face_api.face_engine,
        "create_preview_thumbnail",
        return_value="data:image/jpeg;base64,THUMBNAIL_PREVIEW_B64",
    ), patch.object(
        face_api, "get_effective_org_settings", return_value={"faceThreshold": 0.45}
    ):
        # 1. Employee enrolls 3 face samples -> PENDING_APPROVAL & face_registered=False
        reg_resp = _drive_coro(
            face_api.register_face_biometrics(
                FaceRegisterRequest(samples=fake_samples, consentAgreed=True),
                req,
                target_uid=None,
                current_user=emp_token,
            )
        )
        assert reg_resp.success is True
        assert reg_resp.faceVerificationStatus == FaceVerificationStatus.PENDING_APPROVAL
        assert reg_resp.faceRegistered is False
        assert "menunggu verifikasi dan persetujuan dari Administrator" in reg_resp.message
        assert users_store["emp_dewi_01"]["face_verification_status"] == "PENDING_APPROVAL"
        assert users_store["emp_dewi_01"]["face_registered"] is False
        assert users_store["emp_dewi_01"]["face_submitted_at"] is not None
        assert (
            face_profiles_store["emp_dewi_01"]["previewThumbnail"]
            == "data:image/jpeg;base64,THUMBNAIL_PREVIEW_B64"
        )

        # 2. Admin lists pending face approvals -> includes Dewi + thumbnail
        pending_list = _drive_coro(face_api.list_pending_face_approvals(admin_user=admin_token))
        assert len(pending_list) == 1
        assert pending_list[0].uid == "emp_dewi_01"
        assert pending_list[0].name == "Dewi Lestari"
        assert pending_list[0].niy == "NIY-2026-205"
        assert pending_list[0].previewThumbnail == "data:image/jpeg;base64,THUMBNAIL_PREVIEW_B64"

        # 3. Admin rejects face with reason -> REJECTED, face_profiles purged, FACE_APPROVAL_REJECTED logged
        rej_resp = _drive_coro(
            face_api.reject_employee_face(
                "emp_dewi_01",
                FaceApprovalRejectRequest(reason="Foto buram dan wajah terhalang masker"),
                req,
                admin_user=admin_token,
            )
        )
        assert rej_resp.faceVerificationStatus == FaceVerificationStatus.REJECTED
        assert rej_resp.faceRegistered is False
        assert rej_resp.faceRejectionReason == "Foto buram dan wajah terhalang masker"
        assert "emp_dewi_01" not in face_profiles_store  # Temporary template deleted!
        assert audit_events[-1]["action"] == ACTION_FACE_APPROVAL_REJECTED
        assert "Foto buram" in audit_events[-1]["details"]

        # 4. Rejected employee re-enrolls cleanly -> PENDING_APPROVAL again, then Admin approves!
        re_reg = _drive_coro(
            face_api.register_face_biometrics(
                FaceRegisterRequest(samples=fake_samples, consentAgreed=True),
                req,
                target_uid=None,
                current_user=emp_token,
            )
        )
        assert re_reg.faceVerificationStatus == FaceVerificationStatus.PENDING_APPROVAL
        assert users_store["emp_dewi_01"]["face_rejection_reason"] is None

        app_resp = _drive_coro(
            face_api.approve_employee_face("emp_dewi_01", req, admin_user=admin_token)
        )
        assert app_resp.faceVerificationStatus == FaceVerificationStatus.APPROVED
        assert app_resp.faceRegistered is True
        assert users_store["emp_dewi_01"]["face_verification_status"] == "APPROVED"
        assert users_store["emp_dewi_01"]["face_registered"] is True
        assert audit_events[-1]["action"] == ACTION_FACE_APPROVAL_ACCEPTED
        assert "Kepala Sekolah Admin" in audit_events[-1]["details"]

