"""
Enterprise Audit Trail & Activity Monitoring Service
Records immutable security and operational events into Cloud Firestore `audit_logs`
with standardized WIB timestamps, employee identity (Name & NIY), IP address, and device info.
"""

from typing import Any, Dict, Optional
from fastapi import Request
from firebase_admin import firestore
from app.core.timezone import now_wib
import logging

logger = logging.getLogger(__name__)

# Canonical Activity Types
ACTION_LOGIN_ATTEMPT = "LOGIN_ATTEMPT"
ACTION_FACE_ENROLLMENT = "FACE_ENROLLMENT"
ACTION_ATTENDANCE_CHECKIN = "ATTENDANCE_CHECKIN"
ACTION_ATTENDANCE_CHECKOUT = "ATTENDANCE_CHECKOUT"
ACTION_ATTENDANCE_REJECTED = "ATTENDANCE_REJECTED"
ACTION_SPOOF_ATTEMPT = "SPOOF_ATTEMPT"
ACTION_LEAVE_APPLICATION = "LEAVE_APPLICATION"

# Administrative & Account Lifecycle Activity Types
ACTION_EMPLOYEE_REGISTERED = "EMPLOYEE_REGISTERED"
ACTION_EMPLOYEE_CREATED = "EMPLOYEE_CREATED"
ACTION_EMPLOYEE_UPDATED = "EMPLOYEE_UPDATED"
ACTION_EMPLOYEE_DELETED = "EMPLOYEE_DELETED"
ACTION_FACE_RESET = "FACE_RESET"
ACTION_FACE_APPROVAL_ACCEPTED = "FACE_APPROVAL_ACCEPTED"
ACTION_FACE_APPROVAL_REJECTED = "FACE_APPROVAL_REJECTED"
ACTION_LEAVE_REVIEWED = "LEAVE_REVIEWED"
ACTION_SETTINGS_UPDATED = "SETTINGS_UPDATED"

LEGACY_ACTION_MAP: Dict[str, str] = {
    "attendance_check_in": ACTION_ATTENDANCE_CHECKIN,
    "attendance_check_out": ACTION_ATTENDANCE_CHECKOUT,
    "biometric_face_registered": ACTION_FACE_ENROLLMENT,
    "biometric_face_purged": ACTION_FACE_RESET,
    "face_enrollment_reset": ACTION_FACE_RESET,
    "leave_submitted": ACTION_LEAVE_APPLICATION,
    "leave_approved": ACTION_LEAVE_REVIEWED,
    "leave_rejected": ACTION_LEAVE_REVIEWED,
    "employee_self_registered": ACTION_EMPLOYEE_REGISTERED,
    "employee_created": ACTION_EMPLOYEE_CREATED,
    "employee_updated": ACTION_EMPLOYEE_UPDATED,
    "employee_activated": ACTION_EMPLOYEE_UPDATED,
    "employee_deactivated": ACTION_EMPLOYEE_UPDATED,
    "employee_deleted": ACTION_EMPLOYEE_DELETED,
    "attendance_settings_updated": ACTION_SETTINGS_UPDATED,
    "organization_settings_updated": ACTION_SETTINGS_UPDATED,
}


def normalize_audit_action(raw_action: Optional[str]) -> str:
    """Normalizes legacy lowercase audit action strings to canonical uppercase identifiers."""
    if not raw_action:
        return "SYSTEM_EVENT"
    cleaned = raw_action.strip()
    if cleaned in LEGACY_ACTION_MAP:
        return LEGACY_ACTION_MAP[cleaned]
    lower = cleaned.lower()
    if lower in LEGACY_ACTION_MAP:
        return LEGACY_ACTION_MAP[lower]
    return cleaned.upper()


def summarize_user_agent(user_agent: Optional[str]) -> str:
    """Parses raw User-Agent header into a concise browser + OS label."""
    if not user_agent:
        return "Tidak diketahui"
    ua = user_agent.strip()
    ua_lower = ua.lower()

    if "testclient" in ua_lower or "python-httpx" in ua_lower or "python-requests" in ua_lower:
        return "API Client"

    # Detect Browser
    if "edg/" in ua_lower or "edga/" in ua_lower or "edgios/" in ua_lower:
        browser = "Edge"
    elif "opr/" in ua_lower or "opera" in ua_lower:
        browser = "Opera"
    elif "chrome/" in ua_lower and "chromium" not in ua_lower:
        browser = "Chrome"
    elif "firefox/" in ua_lower or "fxios/" in ua_lower:
        browser = "Firefox"
    elif "safari/" in ua_lower and "chrome/" not in ua_lower:
        browser = "Safari"
    else:
        browser = "Browser"

    # Detect OS / Platform
    if "android" in ua_lower:
        platform = "Android"
    elif "iphone" in ua_lower or "ipad" in ua_lower or "ios" in ua_lower:
        platform = "iOS"
    elif "windows" in ua_lower:
        platform = "Windows"
    elif "mac os x" in ua_lower or "macintosh" in ua_lower:
        platform = "macOS"
    elif "linux" in ua_lower:
        platform = "Linux"
    else:
        platform = "Perangkat Web"

    return f"{browser} ({platform})"


def extract_client_metadata(request: Optional[Request]) -> Dict[str, str]:
    """Extracts real client IP address (proxy-aware) and User-Agent device summary."""
    if request is None:
        return {
            "ipAddress": "-",
            "deviceInfo": "Internal Server",
            "userAgent": "",
        }

    ip_addr = ""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        ip_addr = forwarded_for.split(",")[0].strip()
    if not ip_addr:
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            ip_addr = real_ip.strip()
    if not ip_addr and request.client and request.client.host:
        ip_addr = request.client.host

    if not ip_addr:
        ip_addr = "127.0.0.1"

    raw_ua = request.headers.get("user-agent", "")
    device_info = summarize_user_agent(raw_ua)

    return {
        "ipAddress": ip_addr,
        "deviceInfo": device_info,
        "userAgent": raw_ua[:300] if raw_ua else "",
    }


def record_audit_event(
    db: Any,
    *,
    action: str,
    actor_uid: str,
    status: str = "success",
    details: str = "",
    target_id: Optional[str] = None,
    employee_name: Optional[str] = None,
    niy: Optional[str] = None,
    department: Optional[str] = None,
    request: Optional[Request] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persists an immutable audit trail entry to Firestore `audit_logs`.
    Never raises exceptions so operational flows are never disrupted by logging errors.
    """
    if db is None:
        return

    try:
        resolved_name = employee_name
        resolved_niy = niy
        resolved_dept = department

        # Auto-resolve employee identity from Firestore `users/{actor_uid}` when not supplied
        if (
            actor_uid
            and actor_uid not in ("system", "anonymous", "unauthenticated", "-")
            and (not resolved_name or not resolved_niy)
        ):
            try:
                user_snap = db.collection("users").document(actor_uid).get()
                if user_snap.exists:
                    u_data = user_snap.to_dict() or {}
                    resolved_name = resolved_name or u_data.get("name")
                    resolved_niy = resolved_niy or u_data.get("niy") or u_data.get("employeeId")
                    resolved_dept = resolved_dept or u_data.get("department")
            except Exception:
                pass

        client_meta = extract_client_metadata(request)
        current_wib = now_wib()
        canonical_action = normalize_audit_action(action)
        normalized_status = "failed" if status.lower() in ("failed", "error", "rejected", "gagal") else "success"

        doc_payload: Dict[str, Any] = {
            "actorUid": actor_uid or "anonymous",
            "employeeName": resolved_name or "Pengguna",
            "niy": resolved_niy or "-",
            "department": resolved_dept or "-",
            "action": canonical_action,
            "targetId": target_id,
            "status": normalized_status,
            "details": details,
            "ipAddress": client_meta["ipAddress"],
            "deviceInfo": client_meta["deviceInfo"],
            "userAgent": client_meta["userAgent"],
            "workDateWib": current_wib.strftime("%Y-%m-%d"),
            "timestampWibIso": current_wib.isoformat(),
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
        if metadata:
            doc_payload["metadata"] = metadata

        db.collection("audit_logs").add(doc_payload)
    except Exception as exc:
        logger.warning(f"Non-fatal: failed to persist audit log ({action} for {actor_uid}): {exc}")
