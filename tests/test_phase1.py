import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.security.encryption import BiometricEncryptionService

client = TestClient(app)

def test_health_endpoint():
    """Verify health endpoint responds with 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "firebaseReady" in data

def test_biometric_encryption_roundtrip():
    """Verify application-level biometric vector encryption and decryption."""
    crypto = BiometricEncryptionService()
    
    # Generate a realistic 128-dimensional ArcFace/SFace float32 vector
    rng = np.random.default_rng(42)
    original_vector = rng.standard_normal(128).astype(np.float32)
    # L2 normalize
    original_vector /= np.linalg.norm(original_vector)
    
    # Encrypt
    encrypted_str = crypto.encrypt_embedding(original_vector)
    assert isinstance(encrypted_str, str)
    assert len(encrypted_str) > 0
    # The raw numbers should not be plain visible in ciphertext
    assert "128" not in encrypted_str
    
    # Decrypt
    decrypted_vector = crypto.decrypt_embedding(encrypted_str)
    assert isinstance(decrypted_vector, np.ndarray)
    assert decrypted_vector.shape == (128,)
    # Verify values match with high float precision
    assert np.allclose(original_vector, decrypted_vector, atol=1e-5)

def test_biometric_tampering_rejection():
    """Verify that tampering with an encrypted biometric payload raises ValueError."""
    crypto = BiometricEncryptionService()
    tampered_str = "gAAAAABinvalidTamperedPayloadStringHere="
    with pytest.raises(ValueError):
        crypto.decrypt_embedding(tampered_str)

def test_unauthenticated_protected_endpoint():
    """Verify protected endpoints reject requests without Authorization header."""
    response = client.get("/api/employees/me")
    assert response.status_code == 401

def test_dev_token_verification():
    """Verify the development token mechanism works when Firebase SDK runs in dev mode."""
    # Test valid dev token for employee
    response = client.post("/api/auth/verify-token", json={"idToken": "dev-token-emp01-employee"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["uid"] == "emp01"
    assert data["role"] == "employee"

    # Test valid dev token for admin
    response_admin = client.post("/api/auth/verify-token", json={"idToken": "dev-token-admin01-admin"})
    assert response_admin.status_code == 200
    data_admin = response_admin.json()
    assert data_admin["valid"] is True
    assert data_admin["role"] == "admin"

def test_role_authorization_admin_only():
    """Verify admin endpoints reject non-admin users."""
    # Employee token trying to call admin list employees endpoint
    headers_employee = {"Authorization": "Bearer dev-token-emp01-employee"}
    response = client.get("/api/employees", headers=headers_employee)
    assert response.status_code == 403

    # Admin token calling admin list employees endpoint
    headers_admin = {"Authorization": "Bearer dev-token-admin01-admin"}
    response = client.get("/api/employees", headers=headers_admin)
    assert response.status_code == 200
