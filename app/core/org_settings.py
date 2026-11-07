"""
Effective organization & dynamic attendance settings used by the attendance engine.

The Admin Settings page persists dynamic working hours, check-in/out windows, GPS geofence,
and feature toggles to Firestore `attendance_settings/default` (while keeping `settings/organization`
synchronized for backward compatibility).
"""

import time
import threading
from typing import Any, Dict
from app.core.config import settings
from app.core.firebase import get_firestore_client

_CACHE_TTL_SECONDS = 30
_cache: Dict[str, Any] = {"at": 0.0, "data": None}
_lock = threading.Lock()


def default_org_settings() -> Dict[str, Any]:
    return {
        "organizationName": settings.ORGANIZATION_NAME,
        "address": settings.DEFAULT_OFFICE_ADDRESS,
        "timezone": settings.DEFAULT_TIMEZONE,
        "workingDays": list(settings.DEFAULT_WORKING_DAYS),
        "checkInOpenTime": settings.DEFAULT_CHECK_IN_OPEN_TIME,
        "checkInCloseTime": settings.DEFAULT_CHECK_IN_CLOSE_TIME,
        "officialStartTime": settings.DEFAULT_SCHEDULE_START_TIME,
        "startTime": settings.DEFAULT_SCHEDULE_START_TIME,
        "gracePeriodMinutes": settings.DEFAULT_GRACE_PERIOD_MINUTES,
        "checkOutOpenTime": settings.DEFAULT_CHECK_OUT_OPEN_TIME,
        "checkOutCloseTime": settings.DEFAULT_CHECK_OUT_CLOSE_TIME,
        "officialEndTime": settings.DEFAULT_SCHEDULE_END_TIME,
        "endTime": settings.DEFAULT_SCHEDULE_END_TIME,
        "latitude": settings.DEFAULT_OFFICE_LATITUDE,
        "longitude": settings.DEFAULT_OFFICE_LONGITUDE,
        "radiusMeters": settings.DEFAULT_GEOFENCE_RADIUS_METERS,
        "maxGpsAccuracyMeters": settings.MAX_GPS_ACCURACY_METERS,
        "gpsEnabled": settings.DEFAULT_GPS_ENABLED,
        "faceThreshold": settings.FACE_SIMILARITY_THRESHOLD,
        "livenessEnabled": settings.REQUIRE_LIVENESS_DEFAULT,
        "emailNotificationEnabled": settings.DEFAULT_EMAIL_NOTIFICATION_ENABLED,
    }


def invalidate_org_settings_cache() -> None:
    with _lock:
        _cache["at"] = 0.0
        _cache["data"] = None


def get_effective_org_settings() -> Dict[str, Any]:
    """
    Firestore values layered over config defaults (`settings/organization` + authoritative
    `attendance_settings/default`), cached briefly to avoid repeated Firestore reads per request.
    """
    now = time.monotonic()
    with _lock:
        if _cache["data"] is not None and now - _cache["at"] < _CACHE_TTL_SECONDS:
            return dict(_cache["data"])

    merged = default_org_settings()
    db = get_firestore_client()
    if db is not None:
        try:
            org_doc = db.collection("settings").document("organization").get()
            if org_doc.exists:
                stored_org = org_doc.to_dict() or {}
                for k, v in stored_org.items():
                    if k in merged and v is not None:
                        merged[k] = v
                if "startTime" in stored_org and stored_org["startTime"]:
                    merged["officialStartTime"] = stored_org["startTime"]
                if "endTime" in stored_org and stored_org["endTime"]:
                    merged["officialEndTime"] = stored_org["endTime"]
        except Exception:
            pass

        try:
            att_doc = db.collection("attendance_settings").document("default").get()
            if att_doc.exists:
                stored_att = att_doc.to_dict() or {}
                for k, v in stored_att.items():
                    if k in merged and v is not None:
                        merged[k] = v
                # Keep startTime / endTime aliases synchronized with officialStartTime / officialEndTime
                if stored_att.get("officialStartTime"):
                    merged["startTime"] = stored_att["officialStartTime"]
                if stored_att.get("officialEndTime"):
                    merged["endTime"] = stored_att["officialEndTime"]
        except Exception:
            # Fall back to defaults rather than blocking attendance on a transient read error
            pass

    with _lock:
        _cache["at"] = now
        _cache["data"] = dict(merged)
    return merged
