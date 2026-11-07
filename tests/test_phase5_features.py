"""
Unit & Integration Tests for Phase 5:
Leave Management, Excel Reporting, Organization Settings, and Audit Logs.
"""

import io
import pytest
import openpyxl
from fastapi.testclient import TestClient

from app.main import app
from app.core.firebase import initialize_firebase
from app.models.auth import TokenData
from app.models.user import UserRole
from app.security.auth import get_current_user, require_admin

initialize_firebase()
client = TestClient(app)

def mock_auth_admin():
    return TokenData(
        uid="usr_test_admin_p5",
        email="admin.p5@paudlabum.sch.id",
        role=UserRole.ADMIN,
        name="Admin Pengujian P5"
    )

def mock_auth_employee():
    return TokenData(
        uid="usr_test_guru_p5",
        email="guru.p5@paudlabum.sch.id",
        role=UserRole.EMPLOYEE,
        name="Guru Pengujian P5"
    )


class TestExcelReports:
    """Tests Excel report generation and analytics endpoint."""

    def test_export_attendance_excel_endpoint(self):
        """GET /api/reports/attendance/excel returns valid .xlsx workbook."""
        app.dependency_overrides[require_admin] = mock_auth_admin
        try:
            resp = client.get("/api/reports/attendance/excel")
            assert resp.status_code == 200
            assert "spreadsheetml.sheet" in resp.headers["content-type"]
            assert "attachment;" in resp.headers["content-disposition"]

            # Load bytes into openpyxl to verify workbook validity
            wb = openpyxl.load_workbook(io.BytesIO(resp.content))
            assert "Detail Presensi" in wb.sheetnames
            assert "Rekapitulasi Pegawai" in wb.sheetnames

            ws1 = wb["Detail Presensi"]
            # Verify PAUD LAB UM header exists
            assert "PAUD LABORATORIUM" in str(ws1["A1"].value)
        finally:
            app.dependency_overrides.clear()

    def test_attendance_analytics_endpoint(self):
        """GET /api/reports/attendance/analytics returns JSON breakdown."""
        app.dependency_overrides[require_admin] = mock_auth_admin
        try:
            resp = client.get("/api/reports/attendance/analytics")
            assert resp.status_code == 200
            data = resp.json()
            assert "month" in data
            assert "onTimeRate" in data
            assert "departments" in data
        finally:
            app.dependency_overrides.clear()


class TestLeaveManagement:
    """Tests leave submission, date validation, and admin review."""

    def test_submit_leave_invalid_dates_rejected(self):
        """Submitting leave where endDate < startDate returns 400."""
        app.dependency_overrides[get_current_user] = mock_auth_employee
        try:
            payload = {
                "leaveType": "sick",
                "startDate": "2026-11-10",
                "endDate": "2026-11-05",  # earlier than start
                "reason": "Sakit demam berdarah"
            }
            resp = client.post("/api/leaves", json=payload)
            assert resp.status_code == 400
            assert "tidak boleh lebih awal" in resp.json()["detail"].lower()
        finally:
            app.dependency_overrides.clear()

    def test_submit_and_review_leave_workflow(self):
        """Submits a leave request and then approves it via admin."""
        # 1. Submit
        app.dependency_overrides[get_current_user] = mock_auth_employee
        try:
            payload = {
                "leaveType": "annual_leave",
                "startDate": "2026-11-15",
                "endDate": "2026-11-17",
                "reason": "Keperluan keluarga di luar kota"
            }
            resp_sub = client.post("/api/leaves", json=payload)
            assert resp_sub.status_code == 200
            leave_data = resp_sub.json()
            assert leave_data["totalDays"] == 3
            assert leave_data["status"] == "pending"
            leave_id = leave_data["leaveId"]
        finally:
            app.dependency_overrides.clear()

        # 2. Review as Admin
        app.dependency_overrides[require_admin] = mock_auth_admin
        try:
            review_payload = {
                "status": "approved",
                "reviewerNotes": "Disetujui. Harap titipkan materi kelas kepada guru piket."
            }
            resp_rev = client.patch(f"/api/leaves/{leave_id}/review", json=review_payload)
            assert resp_rev.status_code == 200
            updated = resp_rev.json()
            assert updated["status"] == "approved"
            assert "guru piket" in updated["reviewerNotes"]
        finally:
            app.dependency_overrides.clear()


class TestSettingsAndAuditLogs:
    """Tests organization settings configuration and audit log inspection."""

    def test_get_organization_settings(self):
        """GET /api/settings/organization returns default school details."""
        app.dependency_overrides[get_current_user] = mock_auth_admin
        try:
            resp = client.get("/api/settings/organization")
            assert resp.status_code == 200
            data = resp.json()
            assert "PAUD" in data["organizationName"]
            assert data["latitude"] == -8.095469
            assert data["radiusMeters"] == 100.0
        finally:
            app.dependency_overrides.clear()

    def test_update_organization_settings_admin(self):
        """PATCH /api/settings/organization updates settings."""
        app.dependency_overrides[require_admin] = mock_auth_admin
        try:
            payload = {"gracePeriodMinutes": 20}
            resp = client.patch("/api/settings/organization", json=payload)
            assert resp.status_code == 200
            assert resp.json()["gracePeriodMinutes"] == 20
        finally:
            app.dependency_overrides.clear()

    def test_audit_logs_endpoint(self):
        """GET /api/audit returns list of system events."""
        app.dependency_overrides[require_admin] = mock_auth_admin
        try:
            resp = client.get("/api/audit?limit=20")
            assert resp.status_code == 200
            logs = resp.json()
            assert isinstance(logs, list)
        finally:
            app.dependency_overrides.clear()
