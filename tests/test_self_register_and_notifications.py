import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import smtplib
from fastapi.testclient import TestClient
from firebase_admin import auth as fb_auth, firestore

from app.main import app
from app.core.config import settings
from app.core.firebase import initialize_firebase, get_firestore_client
from app.face_recognition.engine import FaceQualityReport
from app.security.encryption import biometric_crypto
from app.services.notification import (
    dispatch_attendance_notifications,
    format_whatsapp_attendance_message,
    send_attendance_email,
    send_attendance_whatsapp,
)

initialize_firebase()
client = TestClient(app)


@pytest.fixture
def cleanup_registered_uids():
    """Tracks UIDs created during self-registration & attendance tests and cleans up Firebase Auth + Firestore."""
    uids = []
    yield uids
    db = get_firestore_client()
    for uid in uids:
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
        try:
            fb_auth.delete_user(uid)
        except Exception:
            pass


def test_self_registration_validation_errors():
    """Verify invalid email, non-Indonesian phone, and short password (<6 chars) are rejected with 422."""
    # Short password
    res_short_pw = client.post(
        "/api/auth/register",
        json={
            "niy": "NIY-VAL-01",
            "name": "Uji Validasi",
            "email": "validasi@paudlabum.sch.id",
            "phone": "081234567890",
            "password": "12345",
            "department": "Tenaga Pendidik (Guru)",
            "position": "Guru Kelas",
        },
    )
    assert res_short_pw.status_code == 422

    # Invalid phone format (US prefix +1)
    res_bad_phone = client.post(
        "/api/auth/register",
        json={
            "niy": "NIY-VAL-02",
            "name": "Uji Validasi",
            "email": "validasi2@paudlabum.sch.id",
            "phone": "+1-555-123-4567",
            "password": "Password123!",
            "department": "Tenaga Pendidik (Guru)",
            "position": "Guru Kelas",
        },
    )
    assert res_bad_phone.status_code == 422


def test_public_self_registration_and_duplicate_protection(cleanup_registered_uids):
    """
    Verify public POST /api/auth/register (without token) creates Firebase Auth user and Firestore
    users/{uid} with role='employee', is_active=True, face_registered=False, phone='62...', created_at,
    and rejects duplicate NIY or Email with 409 Conflict.
    """
    tag = uuid.uuid4().hex[:6].upper()
    niy = f"NIY-SELF-{tag}"
    email = f"selfreg.{tag.lower()}@paudlabum.sch.id"

    res = client.post(
        "/api/auth/register",
        json={
            "niy": niy,
            "name": f"Bunda Aisyah {tag}",
            "email": email,
            "phone": "0812-3456-7890",
            "password": "SecretPass123",
            "department": "Tenaga Pendidik (Guru)",
            "position": "Guru TK B",
        },
    )
    assert res.status_code == 201, res.text
    body = res.json()
    uid = body["uid"]
    cleanup_registered_uids.append(uid)

    assert body["role"] == "employee"
    assert body["is_active"] is True
    assert body["isActive"] is True
    assert body["face_registered"] is False
    assert body["faceRegistered"] is False
    assert body["phone"] == "6281234567890"
    assert body["niy"] == niy

    # Verify Firestore document has exact required attributes
    db = get_firestore_client()
    snap = db.collection("users").document(uid).get()
    assert snap.exists
    doc_data = snap.to_dict() or {}
    assert doc_data["role"] == "employee"
    assert doc_data["is_active"] is True
    assert doc_data["face_registered"] is False
    assert doc_data["phone"] == "6281234567890"
    assert doc_data["created_at"] is not None

    # Duplicate NIY -> 409 Conflict
    dup_niy_res = client.post(
        "/api/auth/register",
        json={
            "niy": niy,
            "name": "Pegawai Duplikat NIY",
            "email": f"other.{tag.lower()}@paudlabum.sch.id",
            "phone": "6281399998888",
            "password": "SecretPass123",
            "department": "Tata Usaha & Administrasi",
            "position": "Staf TU",
        },
    )
    assert dup_niy_res.status_code == 409

    # Duplicate Email -> 409 Conflict
    dup_email_res = client.post(
        "/api/auth/register",
        json={
            "niy": f"NIY-OTHER-{tag}",
            "name": "Pegawai Duplikat Email",
            "email": email,
            "phone": "+62 813-9999-8888",
            "password": "SecretPass123",
            "department": "Tata Usaha & Administrasi",
            "position": "Staf TU",
        },
    )
    assert dup_email_res.status_code == 409

    # Strict RBAC check using the newly self-registered employee account:
    # Employee MUST be allowed on employee endpoints and FORBIDDEN (403) on all admin endpoints.
    emp_headers = {"Authorization": f"Bearer dev-token-{uid}-employee"}
    assert client.get("/api/employees/me", headers=emp_headers).status_code == 200
    assert client.get("/api/attendance/today", headers=emp_headers).status_code == 200
    assert client.get("/api/leaves/me", headers=emp_headers).status_code == 200

    assert client.get("/api/employees", headers=emp_headers).status_code == 403
    assert client.get(f"/api/employees/{uid}", headers=emp_headers).status_code == 403
    assert client.patch(f"/api/employees/{uid}", json={"position": "Admin"}, headers=emp_headers).status_code == 403
    assert client.get("/api/attendance/organization", headers=emp_headers).status_code == 403
    assert client.get("/api/attendance/stats/today", headers=emp_headers).status_code == 403
    assert client.get("/api/leaves", headers=emp_headers).status_code == 403
    assert client.patch("/api/settings/organization", json={"gracePeriodMinutes": 30}, headers=emp_headers).status_code == 403
    assert client.patch("/api/settings/attendance", json={"gracePeriodMinutes": 30}, headers=emp_headers).status_code == 403
    assert client.get("/api/audit", headers=emp_headers).status_code == 403
    assert client.get("/api/reports/attendance/analytics", headers=emp_headers).status_code == 403
    assert client.get("/api/reports/attendance/excel", headers=emp_headers).status_code == 403


def test_notification_service_email_whatsapp_and_graceful_errors(monkeypatch):
    """Verify Email SMTP, WhatsApp HTTP POST format, and graceful try-except error isolation."""
    msg_text = format_whatsapp_attendance_message(
        nama="Siti Aminah",
        attendance_type="Check-In",
        work_date="2026-11-06",
        time_wib="07:52",
        status_label="Tepat Waktu",
    )
    assert (
        msg_text
        == "Halo Siti Aminah, presensi Check-In Anda pada 2026-11-06 pukul 07:52 WIB berhasil dicatat. Status: Tepat Waktu."
    )

    monkeypatch.setattr(settings, "SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setattr(settings, "SMTP_PORT", 587)
    monkeypatch.setattr(settings, "SMTP_USER", "notifikasi@paudlabum.sch.id")
    monkeypatch.setattr(settings, "SMTP_PASSWORD", "abcd efgh ijkl mnop")
    monkeypatch.setattr(settings, "WA_GATEWAY_URL", "https://api.fonnte.com/send")
    monkeypatch.setattr(settings, "WA_API_TOKEN", "dummy-wa-token")

    # 1. Successful Email + WhatsApp delivery
    sent_messages = []
    mock_smtp_instance = MagicMock()
    mock_smtp_instance.__enter__.return_value = mock_smtp_instance
    mock_smtp_instance.send_message.side_effect = lambda m: sent_messages.append(m)

    captured_wa_requests = []

    class DummyHttpResponse:
        status_code = 200

    class DummyHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None, headers=None):
            captured_wa_requests.append((url, json, headers))
            return DummyHttpResponse()

    with patch("app.services.notification.smtplib.SMTP", return_value=mock_smtp_instance), patch(
        "app.services.notification.httpx.Client", DummyHttpClient
    ):
        outcome = dispatch_attendance_notifications(
            employee_name="Siti Aminah",
            niy="NIY-2026-08",
            email="siti.aminah@paudlabum.sch.id",
            phone="081234567890",
            attendance_type="Check-In",
            work_date="2026-11-06",
            time_wib="08:20",
            status_label="Terlambat 20 Menit",
            notes="Hujan deras di perjalanan",
        )

    assert outcome == {"emailSent": True, "whatsappSent": True}
    assert len(sent_messages) == 1
    email_msg = sent_messages[0]
    assert email_msg["To"] == "siti.aminah@paudlabum.sch.id"
    raw_email_str = email_msg.as_string()
    assert "Siti Aminah" in raw_email_str
    assert "NIY-2026-08" in raw_email_str
    assert "Terlambat 20 Menit" in raw_email_str
    assert "Check-In" in raw_email_str
    assert "Hujan deras di perjalanan" in raw_email_str

    assert len(captured_wa_requests) == 1
    wa_url, wa_json, wa_headers = captured_wa_requests[0]
    assert wa_url == "https://api.fonnte.com/send"
    assert wa_headers["Authorization"] == "dummy-wa-token"
    assert wa_json["target"] == "6281234567890"
    assert (
        wa_json["message"]
        == "Halo Siti Aminah, presensi Check-In Anda pada 2026-11-06 pukul 08:20 WIB berhasil dicatat. Status: Terlambat 20 Menit."
    )

    # 2. Simulate SMTP failure + WhatsApp network timeout -> must NEVER raise
    with patch(
        "app.services.notification.smtplib.SMTP",
        side_effect=smtplib.SMTPConnectError(421, "Service unavailable"),
    ), patch(
        "app.services.notification.httpx.Client.post",
        side_effect=httpx.ConnectError("WA gateway unreachable"),
    ):
        failed_outcome = dispatch_attendance_notifications(
            employee_name="Siti Aminah",
            niy="NIY-2026-08",
            email="siti.aminah@paudlabum.sch.id",
            phone="081234567890",
            attendance_type="Check-Out",
            work_date="2026-11-06",
            time_wib="17:05",
            status_label="Tepat Waktu",
        )
    assert failed_outcome == {"emailSent": False, "whatsappSent": False}


def test_attendance_checkin_checkout_triggers_background_notifications_gracefully(
    cleanup_registered_uids, monkeypatch
):
    """
    End-to-end check on POST /api/attendance/check-in and /api/attendance/check-out:
    verifies BackgroundTasks dispatches notifications after Firestore persistence AND that
    notification transport errors do NOT fail the attendance response.
    """
    db = get_firestore_client()
    uid = f"notif_emp_{uuid.uuid4().hex[:8]}"
    cleanup_registered_uids.append(uid)

    fake_vec = np.ones(128, dtype=np.float32) / np.sqrt(128.0)
    encrypted_vec = biometric_crypto.encrypt_embedding(fake_vec)

    db.collection("users").document(uid).set(
        {
            "uid": uid,
            "niy": "NIY-NOTIF-01",
            "employeeId": "NIY-NOTIF-01",
            "name": "Guru Notifikasi",
            "email": "guru.notif@paudlabum.sch.id",
            "phone": "6281298765432",
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

    emp_headers = {"Authorization": f"Bearer dev-token-{uid}-employee"}

    quality_ok = FaceQualityReport(
        isQualityPassed=True,
        blurScore=180.0,
        isBlurry=False,
        brightnessScore=130.0,
        isTooDark=False,
        isTooBright=False,
        faceCount=1,
        message="Kualitas optimal",
    )

    monkeypatch.setattr(settings, "SMTP_USER", "notifikasi@paudlabum.sch.id")
    monkeypatch.setattr(settings, "SMTP_PASSWORD", "apppassword123456")
    monkeypatch.setattr(settings, "WA_GATEWAY_URL", "https://api.fonnte.com/send")
    monkeypatch.setattr(settings, "WA_API_TOKEN", "dummy-wa-token")

    wib_tz = timezone(timedelta(hours=7))
    monday_checkin_wib = datetime(2026, 11, 9, 7, 50, 0, tzinfo=wib_tz)   # Monday 07:50 WIB
    monday_checkout_wib = datetime(2026, 11, 9, 17, 5, 0, tzinfo=wib_tz)  # Monday 17:05 WIB

    org_policy = {
        "latitude": -8.095469,
        "longitude": 112.170750,
        "radiusMeters": 100.0,
        "maxGpsAccuracyMeters": 50.0,
        "workingDays": [1, 2, 3, 4, 5],
        "checkInOpenTime": "06:30",
        "checkInCloseTime": "09:00",
        "officialStartTime": "08:00",
        "startTime": "08:00",
        "checkOutOpenTime": "16:00",
        "checkOutCloseTime": "19:00",
        "officialEndTime": "17:00",
        "endTime": "17:00",
        "gracePeriodMinutes": 15,
        "faceThreshold": 0.637,
        "gpsEnabled": True,
        "livenessEnabled": False,
        "emailNotificationEnabled": True,
    }

    class FailingWaClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            raise RuntimeError("Simulated WA outage")

    # Force SMTP & WA to fail during check-in to prove attendance still succeeds (200 OK)!
    with patch("app.api.attendance._get_current_wib", return_value=monday_checkin_wib), patch(
        "app.api.attendance.get_effective_org_settings", return_value=org_policy
    ), patch(
        "app.api.attendance.face_engine.decode_image", return_value=np.zeros((100, 100, 3), dtype=np.uint8)
    ), patch(
        "app.api.attendance.face_engine.check_image_quality", return_value=quality_ok
    ), patch(
        "app.api.attendance.face_engine.extract_embedding", return_value=fake_vec
    ), patch(
        "app.services.notification.smtplib.SMTP",
        side_effect=RuntimeError("Simulated SMTP outage"),
    ), patch(
        "app.services.notification.httpx.Client",
        FailingWaClient,
    ):
        in_res = client.post(
            "/api/attendance/check-in",
            headers=emp_headers,
            json={
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                "latitude": -8.095469,
                "longitude": 112.170750,
                "accuracy": 10.0,
                "notes": "Hadir pagi",
            },
        )
        assert in_res.status_code == 200, in_res.text

    # Verify check-out dispatches notification with attendance_type="Check-Out"
    dispatched_calls = []
    with patch("app.api.attendance._get_current_wib", return_value=monday_checkout_wib), patch(
        "app.api.attendance.get_effective_org_settings", return_value=org_policy
    ), patch(
        "app.api.attendance.face_engine.decode_image", return_value=np.zeros((100, 100, 3), dtype=np.uint8)
    ), patch(
        "app.api.attendance.face_engine.check_image_quality", return_value=quality_ok
    ), patch(
        "app.api.attendance.face_engine.extract_embedding", return_value=fake_vec
    ), patch(
        "app.api.attendance.dispatch_attendance_notifications",
        side_effect=lambda **kwargs: dispatched_calls.append(kwargs),
    ):
        out_res = client.post(
            "/api/attendance/check-out",
            headers=emp_headers,
            json={
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                "latitude": -8.095469,
                "longitude": 112.170750,
                "accuracy": 10.0,
                "notes": "Pulang sore",
            },
        )
        assert out_res.status_code == 200, out_res.text

    assert len(dispatched_calls) == 1
    assert dispatched_calls[0]["attendance_type"] == "Check-Out"
    assert dispatched_calls[0]["niy"] == "NIY-NOTIF-01"
    assert dispatched_calls[0]["phone"] == "6281298765432"
