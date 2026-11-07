"""
Unit & Integration Tests for Phase 4: Attendance & GPS Geofencing Engine
Tests Haversine distance, geofence radius boundary, schedule calculation,
check-in, check-out, and attendance history endpoints.
"""

from datetime import datetime, timezone, timedelta
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.core.geofence import calculate_haversine_distance, validate_office_geofence
from app.models.attendance import CheckInStatus, CheckOutStatus
from app.models.auth import TokenData
from app.models.user import UserRole
from app.security.auth import get_current_user

client = TestClient(app)

WIB = timezone(timedelta(hours=7))

def mock_auth_employee():
    return TokenData(
        uid="usr_guru_test_p4",
        email="guru.p4@paudlabum.sch.id",
        role=UserRole.EMPLOYEE,
        name="Ibu Guru Test P4"
    )

def mock_auth_admin():
    return TokenData(
        uid="usr_admin_test_p4",
        email="admin.p4@paudlabum.sch.id",
        role=UserRole.ADMIN,
        name="Admin Sekolah P4"
    )


class TestGeofenceAndDistanceCalculations:
    """Tests Haversine great-circle distance and boundary enforcement."""

    def test_haversine_exact_point(self):
        """Distance between identical coordinates must be 0 meters."""
        lat, lon = settings.DEFAULT_OFFICE_LATITUDE, settings.DEFAULT_OFFICE_LONGITUDE
        dist = calculate_haversine_distance(lat, lon, lat, lon)
        assert dist < 1e-4

    def test_haversine_known_displacement(self):
        """Tests displacement of roughly ~111 meters (0.001 degree latitude delta)."""
        lat1, lon1 = -8.095469, 112.170750
        lat2, lon2 = -8.096469, 112.170750  # ~111 meters south
        dist = calculate_haversine_distance(lat1, lon1, lat2, lon2)
        assert 105.0 < dist < 120.0

    def test_geofence_within_radius_success(self):
        """Tests that client within 50m of school is accepted."""
        # PAUD LAB UM: -8.095469, 112.170750
        # Small offset within ~40 meters
        client_lat = -8.095469 + 0.0003
        client_lon = 112.170750 + 0.0001
        is_valid, dist, msg = validate_office_geofence(
            client_lat=client_lat,
            client_lon=client_lon,
            client_accuracy=10.0,
            radius_meters=100.0
        )
        assert is_valid is True
        assert dist <= 100.0
        assert "valid" in msg.lower()

    def test_geofence_outside_radius_rejected(self):
        """Tests that client 300m away is rejected with informative distance."""
        client_lat = -8.095469 + 0.003  # ~330m away
        client_lon = 112.170750
        is_valid, dist, msg = validate_office_geofence(
            client_lat=client_lat,
            client_lon=client_lon,
            client_accuracy=15.0,
            radius_meters=100.0
        )
        assert is_valid is False
        assert dist > 100.0
        assert "di luar radius" in msg.lower()

    def test_poor_gps_accuracy_rejected(self):
        """Tests that poor GPS accuracy (> 50m) is rejected even if coordinates are close."""
        is_valid, dist, msg = validate_office_geofence(
            client_lat=settings.DEFAULT_OFFICE_LATITUDE,
            client_lon=settings.DEFAULT_OFFICE_LONGITUDE,
            client_accuracy=85.0,  # Accuracy > 50m
            max_accuracy_meters=50.0
        )
        assert is_valid is False
        assert "akurasi sinyal gps" in msg.lower()


class TestAttendanceEndpoints:
    """Tests Attendance API endpoints via TestClient."""

    def test_check_in_outside_geofence_rejected(self):
        """POST /api/attendance/check-in outside radius returns 400."""
        app.dependency_overrides[get_current_user] = mock_auth_employee
        try:
            payload = {
                "image": "dummy_base64_data",
                "latitude": -8.0500,  # ~5 km away from school
                "longitude": 112.1500,
                "accuracy": 10.0
            }
            resp = client.post("/api/attendance/check-in", json=payload)
            assert resp.status_code == 400
            assert "di luar radius" in resp.json()["detail"].lower()
        finally:
            app.dependency_overrides.clear()

    def test_check_in_poor_gps_accuracy_rejected(self):
        """POST /api/attendance/check-in with accuracy > 50m returns 400."""
        app.dependency_overrides[get_current_user] = mock_auth_employee
        try:
            payload = {
                "image": "dummy_base64_data",
                "latitude": settings.DEFAULT_OFFICE_LATITUDE,
                "longitude": settings.DEFAULT_OFFICE_LONGITUDE,
                "accuracy": 120.0
            }
            resp = client.post("/api/attendance/check-in", json=payload)
            assert resp.status_code == 400
            assert "akurasi" in resp.json()["detail"].lower()
        finally:
            app.dependency_overrides.clear()

    def test_get_today_attendance_endpoint(self):
        """GET /api/attendance/today returns 200."""
        app.dependency_overrides[get_current_user] = mock_auth_employee
        try:
            resp = client.get("/api/attendance/today")
            assert resp.status_code == 200
            # Returns null or AttendanceRecordResponse
        finally:
            app.dependency_overrides.clear()

    def test_get_my_history_endpoint(self):
        """GET /api/attendance/me returns 200 and list of records."""
        app.dependency_overrides[get_current_user] = mock_auth_employee
        try:
            resp = client.get("/api/attendance/me?limit=10")
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)
        finally:
            app.dependency_overrides.clear()

    def test_get_monthly_summary_endpoint(self):
        """GET /api/attendance/summary/monthly returns 200 and summary object."""
        app.dependency_overrides[get_current_user] = mock_auth_employee
        try:
            resp = client.get("/api/attendance/summary/monthly")
            assert resp.status_code == 200
            data = resp.json()
            assert "month" in data
            assert "attendedDays" in data
            assert "onTimeCount" in data
            assert "lateCount" in data
        finally:
            app.dependency_overrides.clear()

    def test_organization_attendance_forbidden_for_employee(self):
        """GET /api/attendance/organization is forbidden (403) for non-admin."""
        app.dependency_overrides[get_current_user] = mock_auth_employee
        try:
            resp = client.get("/api/attendance/organization")
            assert resp.status_code == 403
        finally:
            app.dependency_overrides.clear()
