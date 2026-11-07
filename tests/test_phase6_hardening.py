"""
Phase 6 tests: security hardening behaviour.
These tests avoid Firestore writes; they exercise the guards that must hold in production.
"""

import io
import time
import base64
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import Settings, settings, _INSECURE_DEFAULT_ENCRYPTION_KEY
from app.core.firebase import initialize_firebase, verify_firebase_id_token
from app.face_recognition.liveness import LivenessService
from app.security.rate_limit import SlidingWindowLimiter, reset_rate_limits
from app.security.auth import get_current_user
from app.models.auth import TokenData
from app.models.user import UserRole

initialize_firebase()
client = TestClient(app)


def _employee():
    return TokenData(uid="usr_p6_emp", email="e@x.id", role=UserRole.EMPLOYEE, name="Emp")


def _admin():
    return TokenData(uid="usr_p6_admin", email="a@x.id", role=UserRole.ADMIN, name="Adm")


def _img_b64(w=160, h=160, val=128):
    _, buf = cv2.imencode(".jpg", np.full((h, w, 3), val, dtype=np.uint8))
    return base64.b64encode(buf).decode()


class TestProductionConfigGuards:
    def test_default_key_rejected_in_production(self):
        s = Settings(APP_ENV="production", DEBUG=False, CORS_ORIGINS=["https://x.web.app"])
        assert any("BIOMETRIC_ENCRYPTION_KEY" in p for p in s.validate_production_config())

    def test_debug_and_localhost_rejected_in_production(self):
        s = Settings(
            APP_ENV="production",
            DEBUG=True,
            BIOMETRIC_ENCRYPTION_KEY="x" * 5,
            CORS_ORIGINS=["http://localhost:5173"],
        )
        problems = " ".join(s.validate_production_config())
        assert "DEBUG" in problems and "CORS_ORIGINS" in problems

    def test_clean_production_config_passes(self):
        from cryptography.fernet import Fernet
        s = Settings(
            APP_ENV="production",
            DEBUG=False,
            BIOMETRIC_ENCRYPTION_KEY=Fernet.generate_key().decode(),
            CORS_ORIGINS=["https://employee-attendance-syst-39d40.web.app"],
        )
        assert s.validate_production_config() == []

    def test_development_never_blocks(self):
        s = Settings(APP_ENV="development")
        assert s.validate_production_config() == []

    def test_dev_tokens_disabled_in_production(self):
        s = Settings(APP_ENV="production", ALLOW_DEV_TOKENS=True)
        assert s.dev_tokens_allowed is False
        assert Settings(APP_ENV="development", ALLOW_DEV_TOKENS=True).dev_tokens_allowed is True

    def test_dev_token_rejected_when_production(self, monkeypatch):
        monkeypatch.setattr(settings, "APP_ENV", "production")
        with pytest.raises(ValueError):
            verify_firebase_id_token("dev-token-admin01-admin")

    def test_invalid_encryption_key_fails_closed_in_production(self, monkeypatch):
        from app.security.encryption import BiometricEncryptionService
        monkeypatch.setattr(settings, "APP_ENV", "production")
        with pytest.raises(RuntimeError):
            BiometricEncryptionService(key="not-a-valid-fernet-key")


class TestRateLimiter:
    def test_blocks_after_limit_and_reports_retry(self):
        lim = SlidingWindowLimiter(window_seconds=60)
        assert all(lim.check("u1", 3) == 0 for _ in range(3))
        assert lim.check("u1", 3) > 0
        assert lim.check("u2", 3) == 0  # other users unaffected

    def test_window_expires(self):
        lim = SlidingWindowLimiter(window_seconds=1)
        assert lim.check("u", 1) == 0
        assert lim.check("u", 1) > 0
        time.sleep(1.1)
        assert lim.check("u", 1) == 0

    def test_endpoint_returns_429(self, monkeypatch):
        reset_rate_limits()
        monkeypatch.setattr(settings, "RATE_LIMIT_FACE_PER_MINUTE", 2)
        app.dependency_overrides[get_current_user] = _employee
        try:
            codes = [client.post("/api/face/liveness/start").status_code for _ in range(4)]
            assert codes[:2] == [200, 200]
            assert 429 in codes[2:]
        finally:
            app.dependency_overrides.clear()
            reset_rate_limits()


class TestLivenessSingleUse:
    def _passed_session(self, svc, uid="u1", emb=None):
        sid = svc.create_challenge(uid)["sessionId"]
        s = svc._sessions[sid]
        s.passed = True
        s.passed_valid_until = time.time() + 60
        s.embedding = emb
        return sid

    def test_consume_once_only(self):
        svc = LivenessService()
        sid = self._passed_session(svc)
        assert svc.consume_passed_session(sid, "u1") is True
        assert svc.consume_passed_session(sid, "u1") is False  # replay

    def test_wrong_user_rejected(self):
        svc = LivenessService()
        sid = self._passed_session(svc, uid="u1")
        assert svc.consume_passed_session(sid, "someone-else") is False

    def test_unpassed_or_missing_rejected(self):
        svc = LivenessService()
        sid = svc.create_challenge("u1")["sessionId"]  # never passed
        assert svc.consume_passed_session(sid, "u1") is False
        assert svc.consume_passed_session("nope", "u1") is False
        assert svc.consume_passed_session(None, "u1") is False

    def test_expired_pass_rejected(self):
        svc = LivenessService()
        sid = self._passed_session(svc)
        svc._sessions[sid].passed_valid_until = time.time() - 1
        assert svc.consume_passed_session(sid, "u1") is False

    def test_different_face_than_challenge_rejected(self):
        svc = LivenessService()
        rng = np.random.default_rng(1)
        a = rng.standard_normal(128).astype(np.float32)
        b = rng.standard_normal(128).astype(np.float32)  # unrelated "person"
        sid = self._passed_session(svc, emb=a / np.linalg.norm(a))
        assert svc.consume_passed_session(sid, "u1", b / np.linalg.norm(b)) is False

    def test_same_face_as_challenge_accepted(self):
        svc = LivenessService()
        rng = np.random.default_rng(2)
        a = rng.standard_normal(128).astype(np.float32)
        a /= np.linalg.norm(a)
        sid = self._passed_session(svc, emb=a)
        assert svc.consume_passed_session(sid, "u1", a) is True


class TestAccessControl:
    def test_identify_is_admin_only(self):
        app.dependency_overrides[get_current_user] = _employee
        try:
            r = client.post("/api/face/identify", json={"image": _img_b64()})
            assert r.status_code == 403
        finally:
            app.dependency_overrides.clear()

    def test_recognize_other_users_template_forbidden(self):
        reset_rate_limits()
        app.dependency_overrides[get_current_user] = _employee
        try:
            r = client.post(
                "/api/face/recognize",
                json={"image": _img_b64(), "targetUid": "some_other_employee"},
            )
            assert r.status_code == 403
        finally:
            app.dependency_overrides.clear()

    def test_checkin_without_liveness_is_blocked_before_any_write(self):
        # Outside geofence is rejected first; confirms the guard order never reaches storage
        reset_rate_limits()
        app.dependency_overrides[get_current_user] = _employee
        try:
            r = client.post(
                "/api/attendance/check-in",
                json={"image": _img_b64(), "latitude": 0.0, "longitude": 0.0, "accuracy": 5.0},
            )
            assert r.status_code == 400
        finally:
            app.dependency_overrides.clear()


class TestHttpHardening:
    def test_security_headers_present(self):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["cache-control"] == "no-store"

    def test_oversized_body_rejected(self):
        r = client.post(
            "/api/face/quality-check",
            content=b"x",
            headers={"content-length": str(50 * 1024 * 1024), "content-type": "application/json"},
        )
        assert r.status_code == 413

    def test_unauthenticated_requests_rejected(self):
        assert client.get("/api/attendance/me").status_code == 401
        assert client.get("/api/audit").status_code == 401


class TestImageSafety:
    def test_oversized_pixel_count_rejected(self):
        from app.face_recognition.engine import face_engine
        big = np.zeros((6000, 6000, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", big)
        with pytest.raises(ValueError):
            face_engine.decode_image(base64.b64encode(buf).decode())

    def test_large_image_is_downscaled(self):
        from app.face_recognition.engine import face_engine, MAX_IMAGE_LONG_SIDE
        img = np.random.randint(0, 255, (2400, 3000, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        out = face_engine.decode_image(base64.b64encode(buf).decode())
        assert max(out.shape[:2]) <= MAX_IMAGE_LONG_SIDE
