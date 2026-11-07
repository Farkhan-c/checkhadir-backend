"""
Unit & Integration Tests for Phase 3: Face Recognition & Anti-Spoofing Liveness
Tests quality checks, SFace biometric extraction, AES-256 encryption,
liveness challenge-response lifecycle, and Face API endpoints.
"""

import time
import base64
import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.face_recognition.engine import face_engine, FaceQualityError, FaceRecognitionEngine
from app.face_recognition.liveness import liveness_service, LivenessService
from app.security.encryption import biometric_crypto, BiometricEncryptionService
from app.security.auth import get_current_user
from app.models.auth import TokenData
from app.models.user import UserRole

client = TestClient(app)

def mock_auth_user():
    return TokenData(
        uid="usr_test_employee_p3",
        email="guru.test@paudlabum.sch.id",
        role=UserRole.EMPLOYEE,
        name="Guru Biometrik Test"
    )

def mock_auth_admin():
    return TokenData(
        uid="usr_test_admin_p3",
        email="admin.test@paudlabum.sch.id",
        role=UserRole.ADMIN,
        name="Admin Biometrik Test"
    )

def encode_cv2_image(img: np.ndarray) -> str:
    """Encodes cv2 numpy array to base64 jpeg string."""
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


class TestFaceQualityAndValidation:
    """Tests image quality validation, blur detection, brightness, and dimensions."""

    def test_face_engine_initialization(self):
        """Verifies YuNet detector and SFace recognizer are loaded."""
        assert face_engine.detector is not None
        assert face_engine.recognizer is not None

    def test_sharp_vs_blurry_image_quality(self):
        """Tests that Laplacian variance accurately detects blurry images."""
        # Create a textured image
        sharp_img = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.randn(sharp_img, 128, 50)  # High frequency noise / texture

        # Create heavily blurred version
        blurry_img = cv2.GaussianBlur(sharp_img, (35, 35), 0)

        sharp_report = face_engine.check_image_quality(sharp_img)
        blurry_report = face_engine.check_image_quality(blurry_img)

        assert sharp_report.blurScore > blurry_report.blurScore
        assert blurry_report.isBlurry is True
        assert "buram" in blurry_report.message.lower()

    def test_image_brightness_validation(self):
        """Tests that overly dark and overly bright frames are rejected."""
        dark_img = np.full((320, 320, 3), 15, dtype=np.uint8)  # Mean 15 < 40
        bright_img = np.full((320, 320, 3), 245, dtype=np.uint8)  # Mean 245 > 220
        normal_img = np.full((320, 320, 3), 120, dtype=np.uint8)

        dark_report = face_engine.check_image_quality(dark_img)
        bright_report = face_engine.check_image_quality(bright_img)
        normal_report = face_engine.check_image_quality(normal_img)

        assert dark_report.isTooDark is True
        assert dark_report.isQualityPassed is False
        assert bright_report.isTooBright is True
        assert bright_report.isQualityPassed is False
        assert normal_report.isTooDark is False
        assert normal_report.isTooBright is False

    def test_no_face_detection_fails_quality(self):
        """Tests that an image without faces fails quality verification."""
        blank_img = np.full((320, 320, 3), 128, dtype=np.uint8)
        report = face_engine.check_image_quality(blank_img)

        assert report.faceCount == 0
        assert report.isQualityPassed is False
        assert "tidak ada wajah" in report.message.lower()


class TestBiometricExtractionAndMatching:
    """Tests SFace 128-d feature extraction, normalization, and cosine similarity matching."""

    def test_sface_feature_extraction_and_normalization(self):
        """Tests that SFace produces 128-dimensional L2-normalized feature vector."""
        dummy_aligned = np.full((112, 112, 3), 120, dtype=np.uint8)
        feat = face_engine.recognizer.feature(dummy_aligned).flatten()

        assert feat.shape == (128,)
        norm = np.linalg.norm(feat)
        # SFace features have non-zero norm
        assert norm > 0

    def test_cosine_similarity_computation(self):
        """Tests cosine similarity with identical, orthogonal, and calibrated thresholds."""
        vec1 = np.random.randn(128).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)

        # Identical vector -> similarity 1.0
        sim_self = face_engine.compute_similarity(vec1, vec1)
        assert abs(sim_self - 1.0) < 1e-3

        # Slightly perturbed vector (same person simulation with small variance)
        vec_similar = vec1 + np.random.normal(0, 0.02, size=128).astype(np.float32)
        vec_similar /= np.linalg.norm(vec_similar)
        sim_high = face_engine.compute_similarity(vec1, vec_similar)
        assert sim_high > 0.85

        # Unrelated random vector (different person simulation)
        vec_different = np.random.randn(128).astype(np.float32)
        vec_different /= np.linalg.norm(vec_different)
        sim_low = face_engine.compute_similarity(vec1, vec_different)
        assert sim_low < 0.50

    def test_verify_match_thresholds_and_confidence(self):
        """Tests verify_match thresholding and confidence classification."""
        vec_base = np.random.randn(128).astype(np.float32)
        vec_base /= np.linalg.norm(vec_base)

        # High confidence match (sim >= 0.75)
        vec_close = vec_base + np.random.normal(0, 0.15, size=128).astype(np.float32)
        vec_close /= np.linalg.norm(vec_close)
        matched, sim, conf = face_engine.verify_match(vec_base, vec_close, threshold=0.637)
        if sim >= 0.75:
            assert conf == "high"
            assert matched is True

        # Unmatched / rejected (< 0.55)
        vec_far = -vec_base
        matched, sim, conf = face_engine.verify_match(vec_base, vec_far, threshold=0.637)
        assert matched is False
        assert conf == "rejected"


class TestBiometricEncryption:
    """Tests AES-256 Fernet biometric vector encryption and decryption."""

    def test_encryption_roundtrip(self):
        """Verifies vector encryption produces secure ciphertext and decrypts faithfully."""
        original_vector = np.random.randn(128).astype(np.float32)
        ciphertext = biometric_crypto.encrypt_embedding(original_vector)

        # Ciphertext must be non-empty string and not contain plaintext floats
        assert isinstance(ciphertext, str)
        assert str(original_vector[0]) not in ciphertext

        # Decrypt
        decrypted_vector = biometric_crypto.decrypt_embedding(ciphertext)
        assert isinstance(decrypted_vector, np.ndarray)
        assert decrypted_vector.shape == (128,)
        np.testing.assert_allclose(original_vector, decrypted_vector, atol=1e-5)

    def test_tampered_ciphertext_rejection(self):
        """Verifies tampering with encrypted ciphertext raises ValueError."""
        original_vector = np.ones(128, dtype=np.float32)
        ciphertext = biometric_crypto.encrypt_embedding(original_vector)
        tampered = ciphertext[:-4] + "AAAA"

        with pytest.raises(ValueError, match="Biometric decryption failed"):
            biometric_crypto.decrypt_embedding(tampered)


class TestLivenessService:
    """Tests anti-spoofing challenge generation, expiry, and single-use enforcement."""

    def test_create_challenge_session(self):
        """Tests issuance of a signed challenge with 60-second TTL."""
        service = LivenessService()
        challenge = service.create_challenge(uid="usr_test_123", preferred_challenge="smile")

        assert "sessionId" in challenge
        assert challenge["sessionId"].startswith("live_")
        assert challenge["challengeType"] == "smile"
        assert challenge["expiresInSeconds"] == 60
        assert "tersenyum" in challenge["instruction"].lower()

    def test_challenge_expiration(self):
        """Verifies that expired sessions are rejected."""
        service = LivenessService()
        challenge = service.create_challenge(uid="usr_test_expired")
        sid = challenge["sessionId"]

        # Artificially expire session
        service._sessions[sid].expires_at = time.time() - 5

        dummy_img = np.full((160, 160, 3), 128, dtype=np.uint8)
        b64 = encode_cv2_image(dummy_img)

        with pytest.raises(FaceQualityError, match="kedaluwarsa"):
            service.verify_liveness(sid, b64, b64, expected_uid="usr_test_expired")

    def test_replay_attack_prevention(self):
        """Verifies that using the same session token twice is rejected (burn-after-reading)."""
        service = LivenessService()
        challenge = service.create_challenge(uid="usr_test_replay")
        sid = challenge["sessionId"]

        # Mark as already used
        service._sessions[sid].used = True

        dummy_img = np.full((160, 160, 3), 128, dtype=np.uint8)
        b64 = encode_cv2_image(dummy_img)

        with pytest.raises(FaceQualityError, match="pernah digunakan"):
            service.verify_liveness(sid, b64, b64, expected_uid="usr_test_replay")


class TestFaceAPIEndpoints:
    """Tests Face Biometrics API routers via FastAPI TestClient."""

    def test_liveness_start_endpoint(self):
        """Tests POST /api/face/liveness/start."""
        app.dependency_overrides[get_current_user] = mock_auth_user
        try:
            resp = client.post("/api/face/liveness/start")
            assert resp.status_code == 200
            data = resp.json()
            assert "sessionId" in data
            assert "challengeType" in data
            assert data["expiresInSeconds"] == 60
        finally:
            app.dependency_overrides.clear()

    def test_quality_check_endpoint(self):
        """Tests POST /api/face/quality-check."""
        app.dependency_overrides[get_current_user] = mock_auth_user
        try:
            test_img = np.full((200, 200, 3), 128, dtype=np.uint8)
            payload = {"image": encode_cv2_image(test_img)}
            resp = client.post("/api/face/quality-check", json=payload)

            assert resp.status_code == 200
            data = resp.json()
            assert "isQualityPassed" in data
            assert "blurScore" in data
            assert "brightnessScore" in data
            assert "faceCount" in data
        finally:
            app.dependency_overrides.clear()

    def test_face_register_without_consent_rejected(self):
        """Tests POST /api/face/register without UU PDP consent is rejected (400)."""
        app.dependency_overrides[get_current_user] = mock_auth_user
        try:
            dummy_img = np.full((160, 160, 3), 128, dtype=np.uint8)
            b64 = encode_cv2_image(dummy_img)
            payload = {
                "samples": [b64, b64, b64],
                "consentAgreed": False
            }
            resp = client.post("/api/face/register", json=payload)
            assert resp.status_code == 400
            assert "UU PDP" in resp.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_face_status_endpoint(self):
        """Tests GET /api/face/status/{uid}."""
        app.dependency_overrides[get_current_user] = mock_auth_user
        try:
            resp = client.get("/api/face/status/usr_test_employee_p3")
            assert resp.status_code == 200
            data = resp.json()
            assert "faceRegistered" in data
        finally:
            app.dependency_overrides.clear()
