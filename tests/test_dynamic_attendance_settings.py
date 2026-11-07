import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from firebase_admin import firestore

from app.main import app
from app.core.firebase import initialize_firebase, get_firestore_client
from app.core.org_settings import invalidate_org_settings_cache
from app.face_recognition.engine import FaceQualityReport
from app.security.encryption import biometric_crypto

initialize_firebase()
client = TestClient(app)
WIB = timezone(timedelta(hours=7))


@pytest.fixture
def restore_attendance_settings():
    """Snapshots `attendance_settings/default` and `settings/organization` before test and restores afterward."""
    db = get_firestore_client()
    att_existed = False
    att_backup = None
    org_existed = False
    org_backup = None

    if db is not None:
        att_snap = db.collection("attendance_settings").document("default").get()
        if att_snap.exists:
            att_existed = True
            att_backup = att_snap.to_dict()
        org_snap = db.collection("settings").document("organization").get()
        if org_snap.exists:
            org_existed = True
            org_backup = org_snap.to_dict()

    yield

    if db is not None:
        if att_existed and att_backup is not None:
            db.collection("attendance_settings").document("default").set(att_backup)
        else:
            db.collection("attendance_settings").document("default").delete()

        if org_existed and org_backup is not None:
            db.collection("settings").document("organization").set(org_backup)
        elif not org_existed:
            db.collection("settings").document("organization").delete()

    invalidate_org_settings_cache()


@pytest.fixture
def temp_biometric_employee():
    """Creates a temporary enrolled employee in Firestore and removes all traces on teardown."""
    db = get_firestore_client()
    uid = f"dyn_emp_{uuid.uuid4().hex[:8]}"
    fake_vec = np.ones(128, dtype=np.float32) / np.sqrt(128.0)
    encrypted_vec = biometric_crypto.encrypt_embedding(fake_vec)

    if db is not None:
        db.collection("users").document(uid).set(
            {
                "uid": uid,
                "niy": "NIY-DYN-01",
                "employeeId": "NIY-DYN-01",
                "name": "Guru Jadwal Dinamis",
                "email": "jadwal.dinamis@paudlabum.sch.id",
                "phone": "6281234567899",
                "department": "Tenaga Pendidik (Guru)",
                "position": "Guru Kelas",
                "role": "employee",
                "is_active": True,
                "isActive": True,
                "face_registered": True,
                "faceRegistered": True,
                "created_at": firestore.SERVER_TIMESTAMP,
            }
        )
        db.collection("face_profiles").document(uid).set(
            {
                "uid": uid,
                "encryptedEmbedding": encrypted_vec,
                "keyVersion": biometric_crypto.version,
                "algorithm": "SFace-v1",
                "vectorDimension": 128,
                "sampleCount": 3,
            }
        )

    yield uid, fake_vec

    if db is not None:
        for col in ("users", "face_profiles"):
            try:
                db.collection(col).document(uid).delete()
            except Exception:
                pass
        try:
            for att_doc in db.collection("attendance").where("uid", "==", uid).stream():
                att_doc.reference.delete()
        except Exception:
            pass


def test_get_and_patch_dynamic_attendance_settings(restore_attendance_settings):
    """
    Verify:
    - GET /api/settings/attendance is publicly accessible (200 OK).
    - PATCH /api/settings/attendance enforces Admin RBAC (401 unauth, 403 employee, 200 admin).
    - Persists directly to Firestore `attendance_settings/default`.
    """
    # 1. Public GET
    get_res = client.get("/api/settings/attendance")
    assert get_res.status_code == 200, get_res.text
    initial = get_res.json()
    assert "workingDays" in initial
    assert "checkInOpenTime" in initial
    assert "checkOutOpenTime" in initial
    assert "gpsEnabled" in initial
    assert "livenessEnabled" in initial
    assert "emailNotificationEnabled" in initial

    # 2. Unauthenticated PATCH -> 401
    assert client.patch("/api/settings/attendance", json={"gracePeriodMinutes": 10}).status_code == 401

    # 3. Employee PATCH -> 403 Forbidden
    emp_headers = {"Authorization": "Bearer dev-token-emp001-employee"}
    assert client.patch("/api/settings/attendance", json={"gracePeriodMinutes": 10}, headers=emp_headers).status_code == 403

    # 4. Admin PATCH -> 200 OK & persisted to Firestore `attendance_settings/default`
    admin_headers = {"Authorization": "Bearer dev-token-admin001-admin"}
    patch_payload = {
        "workingDays": [1, 2, 3, 4, 5, 6],
        "checkInOpenTime": "06:15",
        "checkInCloseTime": "08:45",
        "officialStartTime": "07:30",
        "gracePeriodMinutes": 10,
        "checkOutOpenTime": "15:30",
        "checkOutCloseTime": "18:30",
        "officialEndTime": "16:00",
        "radiusMeters": 120.0,
        "gpsEnabled": True,
        "livenessEnabled": True,
        "emailNotificationEnabled": False,
    }
    patch_res = client.patch("/api/settings/attendance", json=patch_payload, headers=admin_headers)
    assert patch_res.status_code == 200, patch_res.text
    updated = patch_res.json()
    assert updated["workingDays"] == [1, 2, 3, 4, 5, 6]
    assert updated["checkInOpenTime"] == "06:15"
    assert updated["officialStartTime"] == "07:30"
    assert updated["gracePeriodMinutes"] == 10
    assert updated["emailNotificationEnabled"] is False

    db = get_firestore_client()
    snap = db.collection("attendance_settings").document("default").get()
    assert snap.exists
    stored = snap.to_dict() or {}
    assert stored["checkInOpenTime"] == "06:15"
    assert stored["checkInCloseTime"] == "08:45"
    assert stored["officialStartTime"] == "07:30"
    assert stored["radiusMeters"] == 120.0
    assert stored["emailNotificationEnabled"] is False

    # 5. Invalid window (open > close) -> 400 Bad Request
    bad_window = client.patch(
        "/api/settings/attendance",
        json={"checkInOpenTime": "10:00", "checkInCloseTime": "08:00"},
        headers=admin_headers,
    )
    assert bad_window.status_code == 400


def test_server_side_schedule_windows_late_calculation_and_toggles(temp_biometric_employee):
    """
    Verify server-side Asia/Jakarta WIB validation on POST /api/attendance/check-in & check-out:
    - Outside check-in window or non-working day -> HTTP 400: 'Presensi Masuk belum dibuka atau sudah ditutup'
    - Outside check-out window -> HTTP 400: 'Presensi Pulang belum dibuka'
    - Automatic late calculation from officialStartTime + gracePeriodMinutes
    - gpsEnabled=False allows GPS-less attendance; emailNotificationEnabled=False suppresses email target
    """
    uid, fake_vec = temp_biometric_employee
    emp_headers = {"Authorization": f"Bearer dev-token-{uid}-employee"}

    quality_ok = FaceQualityReport(
        isQualityPassed=True,
        blurScore=190.0,
        isBlurry=False,
        brightnessScore=128.0,
        isTooDark=False,
        isTooBright=False,
        faceCount=1,
        message="Kualitas optimal",
    )

    base_policy = {
        "latitude": -8.095469,
        "longitude": 112.170750,
        "radiusMeters": 100.0,
        "maxGpsAccuracyMeters": 50.0,
        "workingDays": [1, 2, 3, 4, 5],  # Senin - Jumat
        "checkInOpenTime": "06:30",
        "checkInCloseTime": "09:00",
        "officialStartTime": "08:00",
        "startTime": "08:00",
        "gracePeriodMinutes": 15,
        "checkOutOpenTime": "16:00",
        "checkOutCloseTime": "19:00",
        "officialEndTime": "17:00",
        "endTime": "17:00",
        "faceThreshold": 0.637,
        "gpsEnabled": True,
        "livenessEnabled": False,
        "emailNotificationEnabled": True,
    }

    sample_payload = {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        "latitude": -8.095469,
        "longitude": 112.170750,
        "accuracy": 10.0,
    }

    # 1. Before Check-In Open (Monday 05:45 WIB) -> 400
    with patch("app.api.attendance._get_current_wib", return_value=datetime(2026, 11, 9, 5, 45, tzinfo=WIB)), patch(
        "app.api.attendance.get_effective_org_settings", return_value=base_policy
    ):
        r_early = client.post("/api/attendance/check-in", headers=emp_headers, json=sample_payload)
        assert r_early.status_code == 400
        assert r_early.json()["detail"] == "Presensi Masuk belum dibuka atau sudah ditutup"

    # 2. After Check-In Close (Monday 09:15 WIB) -> 400
    with patch("app.api.attendance._get_current_wib", return_value=datetime(2026, 11, 9, 9, 15, tzinfo=WIB)), patch(
        "app.api.attendance.get_effective_org_settings", return_value=base_policy
    ):
        r_closed = client.post("/api/attendance/check-in", headers=emp_headers, json=sample_payload)
        assert r_closed.status_code == 400
        assert r_closed.json()["detail"] == "Presensi Masuk belum dibuka atau sudah ditutup"

    # 3. Non-working day (Sunday 2026-11-08 07:45 WIB) -> 400
    with patch("app.api.attendance._get_current_wib", return_value=datetime(2026, 11, 8, 7, 45, tzinfo=WIB)), patch(
        "app.api.attendance.get_effective_org_settings", return_value=base_policy
    ):
        r_sunday = client.post("/api/attendance/check-in", headers=emp_headers, json=sample_payload)
        assert r_sunday.status_code == 400
        assert r_sunday.json()["detail"] == "Presensi Masuk belum dibuka atau sudah ditutup"

    # 4. Within grace period (Monday 2026-11-09 08:12 WIB <= 08:15 WIB) -> on_time, lateMinutes == 0
    with patch("app.api.attendance._get_current_wib", return_value=datetime(2026, 11, 9, 8, 12, tzinfo=WIB)), patch(
        "app.api.attendance.get_effective_org_settings", return_value=base_policy
    ), patch("app.api.attendance.face_engine.decode_image", return_value=np.zeros((64, 64, 3), dtype=np.uint8)), patch(
        "app.api.attendance.face_engine.check_image_quality", return_value=quality_ok
    ), patch("app.api.attendance.face_engine.extract_embedding", return_value=fake_vec), patch(
        "app.api.attendance.dispatch_attendance_notifications"
    ):
        r_ontime = client.post("/api/attendance/check-in", headers=emp_headers, json=sample_payload)
        assert r_ontime.status_code == 200, r_ontime.text
        data_ontime = r_ontime.json()
        assert data_ontime["checkInStatus"] == "on_time"
        assert data_ontime["lateMinutes"] == 0

    # 5. Check-out BEFORE checkOutOpenTime (Monday 14:30 WIB < 16:00 WIB) -> 400: 'Presensi Pulang belum dibuka'
    with patch("app.api.attendance._get_current_wib", return_value=datetime(2026, 11, 9, 14, 30, tzinfo=WIB)), patch(
        "app.api.attendance.get_effective_org_settings", return_value=base_policy
    ):
        r_out_early = client.post("/api/attendance/check-out", headers=emp_headers, json=sample_payload)
        assert r_out_early.status_code == 400
        assert r_out_early.json()["detail"] == "Presensi Pulang belum dibuka"

    # 6. Late Check-In on Tuesday (2026-11-10 08:25 WIB > 08:15 WIB) with gpsEnabled=False & emailNotificationEnabled=False
    tuesday_policy = {
        **base_policy,
        "gpsEnabled": False,
        "emailNotificationEnabled": False,
    }
    dispatched = []
    with patch("app.api.attendance._get_current_wib", return_value=datetime(2026, 11, 10, 8, 25, tzinfo=WIB)), patch(
        "app.api.attendance.get_effective_org_settings", return_value=tuesday_policy
    ), patch("app.api.attendance.face_engine.decode_image", return_value=np.zeros((64, 64, 3), dtype=np.uint8)), patch(
        "app.api.attendance.face_engine.check_image_quality", return_value=quality_ok
    ), patch("app.api.attendance.face_engine.extract_embedding", return_value=fake_vec), patch(
        "app.api.attendance.dispatch_attendance_notifications",
        side_effect=lambda **kwargs: dispatched.append(kwargs),
    ):
        # Notice: no latitude/longitude sent because gpsEnabled=False
        r_late = client.post(
            "/api/attendance/check-in",
            headers=emp_headers,
            json={"image": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="},
        )
        assert r_late.status_code == 200, r_late.text
        data_late = r_late.json()
        assert data_late["checkInStatus"] == "late"
        assert data_late["lateMinutes"] == 25  # 25 minutes past officialStartTime (08:00)

    assert len(dispatched) == 1
    assert dispatched[0]["email"] is None  # Suppressed by emailNotificationEnabled=False
    assert dispatched[0]["status_label"] == "Terlambat 25 Menit"
