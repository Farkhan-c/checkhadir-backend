"""
Unit & Integration Tests for Production Auth Hardening, Auto-Liveness Stability Hold,
and Clean-Slate Production Initialization Utility (`scripts/init_production_db.py`).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from app.models.auth import SelfRegisterRequest
from app.face_recognition.liveness import LivenessService
from scripts.init_production_db import (
    PURGE_COLLECTIONS,
    build_default_attendance_settings,
    ensure_primary_admin,
    purge_collection,
    purge_non_admin_users,
    seed_default_attendance_settings,
)


def test_self_register_forbids_role_escalation_payload():
    """Public registration schema must strictly forbid supplying a `role` field."""
    with pytest.raises(ValidationError):
        SelfRegisterRequest(
            niy="NIY-2026-099",
            name="Budi Santoso",
            email="budi@paudlabum.sch.id",
            phone="081234567890",
            password="password123",
            department="Tenaga Pendidik (Guru)",
            position="Guru Kelas",
            role="admin",  # type: ignore[call-arg]
        )


def test_liveness_stability_hold_session_and_static_replay_rejection():
    """`stability_hold` challenge issues expected instruction and rejects identical frame replays."""
    svc = LivenessService()
    challenge = svc.create_challenge(uid="emp_001", preferred_challenge="stability_hold")
    assert challenge["challengeType"] == "stability_hold"
    assert "Wajah Terdeteksi" in challenge["instruction"]

    # Mock face detection, passive anti-spoofing, and embedding extraction to reach geometry/replay check
    fake_face = [40, 30, 80, 90, 60, 55, 100, 55, 80, 75, 65, 95, 95, 95, 0.99]
    import numpy as np

    dummy_img = np.full((160, 160, 3), 128, dtype=np.uint8)
    dummy_emb = np.ones(128, dtype=np.float32) / np.sqrt(128)

    with patch("app.face_recognition.liveness.face_engine.decode_image", return_value=dummy_img), patch(
        "app.face_recognition.liveness.face_engine.detect_faces", return_value=[np.array(fake_face, dtype=np.float32)]
    ), patch.object(svc, "check_passive_spoofing", return_value=(True, "ok", 0.9)), patch(
        "app.face_recognition.liveness.face_engine.extract_embedding", return_value=dummy_emb
    ):
        res = svc.verify_liveness(
            session_id=challenge["sessionId"],
            baseline_input="data:image/jpeg;base64,IDENTICAL_FRAME",
            action_input="data:image/jpeg;base64,IDENTICAL_FRAME",
            expected_uid="emp_001",
        )
        assert res["passed"] is False
        assert "statis" in res["message"].lower() or "spoofing" in res["message"].lower()


def test_init_production_db_provisions_admin_purges_dummies_and_seeds_settings():
    """Verifies clean-slate script logic using in-memory mocks (zero live DB destruction)."""
    assert "attendances" in PURGE_COLLECTIONS
    assert "attendance" in PURGE_COLLECTIONS
    assert "audit_logs" in PURGE_COLLECTIONS
    assert "face_profiles" in PURGE_COLLECTIONS

    # Mock Firebase Auth module
    class FakeUserNotFoundError(Exception):
        pass

    fake_auth = MagicMock()
    fake_auth.UserNotFoundError = FakeUserNotFoundError
    fake_auth.get_user_by_email.side_effect = FakeUserNotFoundError()
    fake_auth.create_user.return_value = SimpleNamespace(uid="admin_primary_uid")

    dummy_auth_users = [
        SimpleNamespace(uid="admin_primary_uid", email="admin@paudlabum.sch.id"),
        SimpleNamespace(uid="dummy_emp_1", email="dummy1@test.local"),
        SimpleNamespace(uid="dummy_emp_2", email="dummy2@test.local"),
    ]
    fake_auth.list_users.return_value = SimpleNamespace(iterate_all=lambda: iter(dummy_auth_users))
    fake_auth.delete_users.return_value = SimpleNamespace(success_count=2, failure_count=0)

    # Mock Firestore DB
    db = MagicMock()
    batch = MagicMock()
    db.batch.return_value = batch

    # Admin user doc ref
    admin_doc_ref = MagicMock()
    admin_doc_ref.get.return_value = SimpleNamespace(exists=False)

    # Users collection stream contains admin + 2 dummy employees
    admin_fs_doc = SimpleNamespace(
        id="admin_primary_uid",
        reference=MagicMock(),
        to_dict=lambda: {"email": "admin@paudlabum.sch.id", "role": "admin"},
    )
    dummy_fs_1 = SimpleNamespace(
        id="dummy_emp_1",
        reference=MagicMock(),
        to_dict=lambda: {"email": "dummy1@test.local", "role": "employee"},
    )
    dummy_fs_2 = SimpleNamespace(
        id="dummy_emp_2",
        reference=MagicMock(),
        to_dict=lambda: {"email": "dummy2@test.local", "role": "employee"},
    )

    att_doc_1 = SimpleNamespace(id="att_1", reference=MagicMock())
    settings_default_ref = MagicMock()
    org_settings_ref = MagicMock()

    def collection_router(name: str):
        col = MagicMock()
        if name == "users":
            col.document.return_value = admin_doc_ref
            col.stream.return_value = [admin_fs_doc, dummy_fs_1, dummy_fs_2]
        elif name == "attendances":
            col.stream.return_value = [att_doc_1]
        elif name == "attendance_settings":
            col.document.return_value = settings_default_ref
        elif name == "settings":
            col.document.return_value = org_settings_ref
        else:
            col.stream.return_value = []
        return col

    db.collection.side_effect = collection_router

    # 1. Ensure primary admin
    uid = ensure_primary_admin(
        db=db,
        admin_email="Admin@PaudLabUM.sch.id",
        admin_password="StrongPassword123!",
        admin_name="Admin Utama",
        admin_niy="ADM-001",
        auth_module=fake_auth,
    )
    assert uid == "admin_primary_uid"
    fake_auth.set_custom_user_claims.assert_called_once_with("admin_primary_uid", {"role": "admin"})
    saved_admin = admin_doc_ref.set.call_args[0][0]
    assert saved_admin["role"] == "admin"
    assert saved_admin["is_active"] is True

    # 2. Purge collection
    deleted_att = purge_collection(db, "attendances", dry_run=False)
    assert deleted_att == 1
    batch.delete.assert_any_call(att_doc_1.reference)

    # 3. Purge non-admin users (must preserve admin_primary_uid!)
    del_fs, del_auth = purge_non_admin_users(
        db=db,
        admin_uid="admin_primary_uid",
        admin_email="admin@paudlabum.sch.id",
        dry_run=False,
        auth_module=fake_auth,
    )
    assert del_fs == 2
    assert del_auth == 2
    fake_auth.delete_users.assert_called_once_with(["dummy_emp_1", "dummy_emp_2"])

    # 4. Seed default schedule settings
    seeded = seed_default_attendance_settings(db=db, admin_uid="admin_primary_uid", dry_run=False)
    assert seeded["workingDays"] == [1, 2, 3, 4, 5]
    assert seeded["gracePeriodMinutes"] == 15
    assert seeded["radiusMeters"] == 100.0
    settings_default_ref.set.assert_called_once()

    defaults = build_default_attendance_settings("admin_primary_uid")
    assert defaults["workingDays"] == [1, 2, 3, 4, 5]
    assert defaults["gracePeriodMinutes"] == 15
    assert defaults["radiusMeters"] == 100.0


def test_wib_timezone_offset_is_utc_plus_7():
    """Ensures server WIB helper always produces UTC+07:00 timezone-aware timestamps."""
    from datetime import timedelta
    from app.core.timezone import now_wib

    current = now_wib()
    assert current.tzinfo is not None
    assert current.utcoffset() == timedelta(hours=7)


def test_verify_firebase_id_token_clock_skew_and_single_retry():
    """
    Verifies:
    1. `clock_skew_seconds=10` is passed to `firebase_admin.auth.verify_id_token`.
    2. When `InvalidIdTokenError('Token used too early...')` occurs, `verify_firebase_id_token_async`
       waits 1.5s and retries once before succeeding or raising ValueError.
    """
    from firebase_admin import auth as fb_auth
    from app.core import firebase as fb_core

    calls = []
    slept_durations = []

    def flaky_verify(token, check_revoked=True, clock_skew_seconds=0):
        calls.append((token, check_revoked, clock_skew_seconds))
        if len(calls) == 1:
            raise fb_auth.InvalidIdTokenError(
                "Token used too early, 1794071705 < 1794071707. Check that your computer's clock is set correctly."
            )
        return {"uid": "emp_skew_ok", "email": "emp@paudlabum.sch.id", "role": "employee"}

    async def fake_async_sleep(seconds: float) -> None:
        slept_durations.append(seconds)

    with patch.object(fb_core, "is_firebase_ready", return_value=True), patch.object(
        fb_core.auth, "verify_id_token", side_effect=flaky_verify
    ), patch.object(fb_core, "_async_sleep", side_effect=fake_async_sleep):
        coro = fb_core.verify_firebase_id_token_async("real-jwt-token")
        try:
            coro.send(None)
            pytest.fail("Expected StopIteration with return value")
        except StopIteration as stop:
            decoded = stop.value

    assert decoded["uid"] == "emp_skew_ok"
    assert len(calls) == 2
    assert calls[0][2] == 10  # Initial call uses clock_skew_seconds=10
    assert calls[1][2] >= 15  # Retry widens skew tolerance
    assert slept_durations == [1.5]
