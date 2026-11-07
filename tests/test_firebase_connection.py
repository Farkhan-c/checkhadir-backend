import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.core.firebase import (
    detect_service_account_path,
    initialize_firebase,
    is_firebase_ready,
    test_firestore_connection as run_firestore_connection_test
)

client = TestClient(app)

def test_service_account_detection():
    """Verify serviceAccountKey.json is automatically discovered in backend folder."""
    path = detect_service_account_path()
    assert path is not None, "serviceAccountKey.json was not found by automatic detection."
    assert os.path.exists(path), f"Detected path does not exist: {path}"
    assert path.endswith(".json")

def test_firebase_admin_initialization():
    """Verify Firebase Admin SDK initializes successfully on project employee-attendance-syst-39d40."""
    firebase_app = initialize_firebase()
    assert firebase_app is not None, "Firebase app failed to initialize."
    assert is_firebase_ready() is True
    assert firebase_app.project_id == "employee-attendance-syst-39d40"

def test_live_firestore_connectivity():
    """Verify real connectivity to Cloud Firestore without private key leakage."""
    diag = run_firestore_connection_test()
    assert diag["connected"] is True, f"Firestore connection failed: {diag.get('error')}"
    assert diag["projectId"] == "employee-attendance-syst-39d40"
    assert "latencyMs" in diag
    assert diag["latencyMs"] >= 0
    # Security check: Ensure private key is NOT present in diagnostic payload
    assert "private_key" not in str(diag)

def test_health_check_live_firebase():
    """Verify /health endpoint reports firestoreConnected: True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "CheckHadir.Id"
    assert data["organization"] == "PAUD LAB UM KOTA BLITAR"
    assert data["firebaseReady"] is True
    assert data["firestoreConnected"] is True
    assert data["projectId"] == "employee-attendance-syst-39d40"
