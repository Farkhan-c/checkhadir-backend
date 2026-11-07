import uuid
import pytest
from fastapi.testclient import TestClient
from firebase_admin import auth as fb_auth

from app.main import app
from app.core.firebase import initialize_firebase, get_firestore_client
from app.models.employee import EmployeeCreate, EmployeeUpdate, EmployeeResponse

initialize_firebase()
client = TestClient(app)

admin_headers = {"Authorization": "Bearer dev-token-admin01-admin"}
employee_headers = {"Authorization": "Bearer dev-token-emp01-employee"}


@pytest.fixture
def tracked_uids():
    """Tracks UIDs created during tests and cleans them from Firebase Auth & Firestore."""
    created_uids = []
    yield created_uids
    db = get_firestore_client()
    for uid in created_uids:
        if db is not None:
            try:
                db.collection("users").document(uid).delete()
            except Exception:
                pass
        try:
            fb_auth.delete_user(uid)
        except Exception:
            pass


def test_employee_pydantic_schemas():
    """Verify EmployeeCreate syncs niy/employeeId and enforces validation."""
    model = EmployeeCreate(
        niy="NIY-SCHEMA-01",
        name="Guru Uji Skema",
        email="skema@paudlabum.sch.id",
        password="Password123!",
        department="Tenaga Pendidik (Guru)",
        position="Guru Kelas",
    )
    assert model.niy == "NIY-SCHEMA-01"
    assert model.employeeId == "NIY-SCHEMA-01"
    assert model.isActive is True

    upd = EmployeeUpdate(position="Kepala Sekolah", isActive=False)
    assert upd.position == "Kepala Sekolah"
    assert upd.isActive is False


def test_admin_rbac_enforced_across_phase2_endpoints():
    """Verify all Phase 2 management endpoints reject unauthenticated and employee callers."""
    assert client.get("/api/employees").status_code == 401
    assert client.get("/api/employees", headers=employee_headers).status_code == 403
    assert client.get("/api/employees/some_uid", headers=employee_headers).status_code == 403
    assert (
        client.patch(
            "/api/employees/some_uid",
            json={"position": "Unauthorized"},
            headers=employee_headers,
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/employees",
            json={
                "niy": "NIY-UNAUTH",
                "name": "Unauthorized User",
                "email": "unauth@paudlabum.sch.id",
                "password": "Password123!",
                "department": "Tenaga Pendidik (Guru)",
                "position": "Staf",
            },
            headers=employee_headers,
        ).status_code
        == 403
    )


def test_create_get_search_paginate_and_patch_employee(tracked_uids):
    """End-to-end Phase 2 test against real Firebase Auth + Cloud Firestore."""
    tag = uuid.uuid4().hex[:6].upper()
    niy = f"NIY-P2-{tag}"
    email = f"guru.p2.{tag.lower()}@paudlabum.sch.id"

    # 1. POST /api/employees (provisions Firebase Auth + Firestore users/{uid})
    payload = {
        "niy": niy,
        "name": f"Ustadzah Fatimah {tag}",
        "email": email,
        "password": "PaudLabUser2026!",
        "phoneNumber": "081299887766",
        "department": "Tenaga Pendidik (Guru)",
        "position": "Guru Pendamping TK A",
        "isActive": True,
    }
    create_res = client.post("/api/employees", json=payload, headers=admin_headers)
    assert create_res.status_code == 201, create_res.text
    created = create_res.json()
    uid = created["uid"]
    tracked_uids.append(uid)

    parsed = EmployeeResponse(**created)
    assert parsed.niy == niy
    assert parsed.employeeId == niy
    assert parsed.faceRegistered is False
    assert parsed.isActive is True

    # Confirm real Firebase Auth user exists
    fb_user = fb_auth.get_user(uid)
    assert fb_user.email == email
    assert fb_user.disabled is False

    # Duplicate NIY must return 409
    dup_res = client.post("/api/employees", json=payload, headers=admin_headers)
    assert dup_res.status_code == 409

    # 2. GET /api/employees/{uid}
    detail_res = client.get(f"/api/employees/{uid}", headers=admin_headers)
    assert detail_res.status_code == 200
    assert detail_res.json()["email"] == email

    # Unknown UID returns 404
    assert client.get("/api/employees/uid_tidak_ada_99999", headers=admin_headers).status_code == 404

    # 3. GET /api/employees with search (q), department filter, and pagination
    search_res = client.get(
        "/api/employees",
        params={"q": niy, "department": "Tenaga Pendidik (Guru)", "skip": 0, "limit": 5},
        headers=admin_headers,
    )
    assert search_res.status_code == 200
    assert search_res.headers.get("x-total-count") == "1"
    items = search_res.json()
    assert len(items) == 1
    assert items[0]["uid"] == uid

    # Pagination offset beyond total returns empty slice while keeping X-Total-Count = 1
    empty_page = client.get(
        "/api/employees",
        params={"q": niy, "skip": 5, "limit": 5},
        headers=admin_headers,
    )
    assert empty_page.status_code == 200
    assert empty_page.headers.get("x-total-count") == "1"
    assert empty_page.json() == []

    # 4. PATCH /api/employees/{uid} — update profile & toggle active status
    patch_res = client.patch(
        f"/api/employees/{uid}",
        json={"position": "Kepala Tata Usaha", "department": "Tata Usaha & Administrasi", "isActive": False},
        headers=admin_headers,
    )
    assert patch_res.status_code == 200
    patched = patch_res.json()
    assert patched["position"] == "Kepala Tata Usaha"
    assert patched["department"] == "Tata Usaha & Administrasi"
    assert patched["isActive"] is False
    assert fb_auth.get_user(uid).disabled is True

    # Re-enable via PATCH toggle
    reenable_res = client.patch(
        f"/api/employees/{uid}",
        json={"isActive": True},
        headers=admin_headers,
    )
    assert reenable_res.status_code == 200
    assert reenable_res.json()["isActive"] is True
    assert fb_auth.get_user(uid).disabled is False


def test_deactivate_activate_and_reset_face_endpoints(tracked_uids):
    """Verify dedicated activate/deactivate/reset-face endpoints remain functional."""
    tag = uuid.uuid4().hex[:6].upper()
    create_res = client.post(
        "/api/employees",
        json={
            "niy": f"NIY-ACT-{tag}",
            "name": f"Dewi Sartika {tag}",
            "email": f"dewi.act.{tag.lower()}@paudlabum.sch.id",
            "password": "PaudLabUser2026!",
            "department": "Tenaga Pendidik (Guru)",
            "position": "Guru Kelas",
            "isActive": True,
        },
        headers=admin_headers,
    )
    assert create_res.status_code == 201
    uid = create_res.json()["uid"]
    tracked_uids.append(uid)

    res_deact = client.post(f"/api/employees/{uid}/deactivate", headers=admin_headers)
    assert res_deact.status_code == 200
    assert res_deact.json()["isActive"] is False

    res_act = client.post(f"/api/employees/{uid}/activate", headers=admin_headers)
    assert res_act.status_code == 200
    assert res_act.json()["isActive"] is True

    res_reset = client.post(f"/api/employees/{uid}/reset-face", headers=admin_headers)
    assert res_reset.status_code == 200
    assert res_reset.json()["faceRegistered"] is False


def test_list_departments():
    """Verify listing departments returns expected organization departments."""
    response = client.get("/api/employees/departments/list", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    dept_names = [d["name"] for d in data]
    assert "Tenaga Pendidik (Guru)" in dept_names
    assert "Tata Usaha & Administrasi" in dept_names
