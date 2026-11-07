"""
Settings API Router
Handles dynamic working hours, check-in/out windows, office geofence coordinates,
feature toggles (GPS, Liveness, Email Notification) in Firestore `attendance_settings/default`,
as well as organization identity in `settings/organization`.
"""

from typing import List
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from firebase_admin import firestore
from pydantic import ValidationError

from app.models.auth import TokenData
from app.models.settings import (
    OfficeLocationModel,
    WorkScheduleModel,
    AttendanceSettingsModel,
    DynamicAttendanceSettingsModel,
    DynamicAttendanceSettingsUpdate,
    OrganizationSettingsModel,
    OrganizationSettingsUpdate,
)
from app.security.auth import get_current_user, require_admin
from app.core.config import settings as app_settings
from app.core.timezone import now_wib
from app.core.firebase import get_firestore_client
from app.core.org_settings import default_org_settings, invalidate_org_settings_cache
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/settings", tags=["Settings"])

DEFAULT_ORG_SETTINGS = {
    "organizationName": app_settings.ORGANIZATION_NAME,
    "address": app_settings.DEFAULT_OFFICE_ADDRESS,
    "contactEmail": "paudlabum.blitar@um.ac.id",
    "contactPhone": "(0342) 801123",
    "latitude": app_settings.DEFAULT_OFFICE_LATITUDE,
    "longitude": app_settings.DEFAULT_OFFICE_LONGITUDE,
    "radiusMeters": app_settings.DEFAULT_GEOFENCE_RADIUS_METERS,
    "maxGpsAccuracyMeters": app_settings.MAX_GPS_ACCURACY_METERS,
    "startTime": app_settings.DEFAULT_SCHEDULE_START_TIME,
    "endTime": app_settings.DEFAULT_SCHEDULE_END_TIME,
    "gracePeriodMinutes": app_settings.DEFAULT_GRACE_PERIOD_MINUTES,
    "faceThreshold": app_settings.FACE_SIMILARITY_THRESHOLD,
    "livenessEnabled": True,
}


def _normalize_ts(ts):
    if hasattr(ts, "to_datetime"):
        return ts.to_datetime()
    if isinstance(ts, datetime):
        return ts
    return None


def _build_dynamic_attendance_defaults() -> dict:
    base = default_org_settings()
    return {
        "organizationName": base["organizationName"],
        "address": base["address"],
        "timezone": base["timezone"],
        "workingDays": base["workingDays"],
        "checkInOpenTime": base["checkInOpenTime"],
        "checkInCloseTime": base["checkInCloseTime"],
        "officialStartTime": base["officialStartTime"],
        "gracePeriodMinutes": base["gracePeriodMinutes"],
        "checkOutOpenTime": base["checkOutOpenTime"],
        "checkOutCloseTime": base["checkOutCloseTime"],
        "officialEndTime": base["officialEndTime"],
        "latitude": base["latitude"],
        "longitude": base["longitude"],
        "radiusMeters": base["radiusMeters"],
        "maxGpsAccuracyMeters": base["maxGpsAccuracyMeters"],
        "gpsEnabled": base["gpsEnabled"],
        "livenessEnabled": base["livenessEnabled"],
        "emailNotificationEnabled": base["emailNotificationEnabled"],
        "faceThreshold": base["faceThreshold"],
        "updatedAt": None,
    }


@router.get("/attendance", response_model=DynamicAttendanceSettingsModel)
async def get_dynamic_attendance_settings():
    """
    Public / Authenticated Endpoint:
    Returns the active dynamic working hours, check-in/out time windows, geofence radius,
    and feature toggles from Firestore `attendance_settings/default`.
    """
    merged = _build_dynamic_attendance_defaults()
    db = get_firestore_client()
    if db is not None:
        try:
            org_doc = db.collection("settings").document("organization").get()
            if org_doc.exists:
                org_d = org_doc.to_dict() or {}
                for k in (
                    "organizationName",
                    "address",
                    "latitude",
                    "longitude",
                    "radiusMeters",
                    "maxGpsAccuracyMeters",
                    "gracePeriodMinutes",
                    "faceThreshold",
                    "livenessEnabled",
                ):
                    if k in org_d and org_d[k] is not None:
                        merged[k] = org_d[k]
                if org_d.get("startTime"):
                    merged["officialStartTime"] = org_d["startTime"]
                if org_d.get("endTime"):
                    merged["officialEndTime"] = org_d["endTime"]

            att_doc = db.collection("attendance_settings").document("default").get()
            if att_doc.exists:
                att_d = att_doc.to_dict() or {}
                for k in merged.keys():
                    if k != "updatedAt" and k in att_d and att_d[k] is not None:
                        merged[k] = att_d[k]
                merged["updatedAt"] = _normalize_ts(att_d.get("updatedAt"))
        except Exception as exc:
            logger.warning(f"Failed to read attendance_settings/default: {exc}")

    return DynamicAttendanceSettingsModel(**merged)


@router.patch("/attendance", response_model=DynamicAttendanceSettingsModel)
async def update_dynamic_attendance_settings(
    payload: DynamicAttendanceSettingsUpdate,
    admin_user: TokenData = Depends(require_admin),
):
    """
    Admin-only Endpoint:
    Updates dynamic working schedule, check-in/out windows, geofence coordinates/radius,
    and feature toggles in Firestore `attendance_settings/default`.
    """
    db = get_firestore_client()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )

    current_model = await get_dynamic_attendance_settings()
    current_data = current_model.model_dump(exclude={"updatedAt"})

    update_dict = payload.model_dump(exclude_unset=True)
    if not update_dict:
        return current_model

    candidate = {**current_data, **update_dict}
    try:
        validated = DynamicAttendanceSettingsModel(**candidate)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ve.errors()[0].get("msg", "Parameter pengaturan jam kerja tidak valid."),
        )

    persist_payload = {
        **validated.model_dump(exclude={"updatedAt"}),
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    db.collection("attendance_settings").document("default").set(persist_payload, merge=True)

    # Synchronize shared fields into `settings/organization` for backward compatibility
    org_sync = {
        "organizationName": validated.organizationName,
        "address": validated.address,
        "latitude": validated.latitude,
        "longitude": validated.longitude,
        "radiusMeters": validated.radiusMeters,
        "maxGpsAccuracyMeters": validated.maxGpsAccuracyMeters,
        "startTime": validated.officialStartTime,
        "endTime": validated.officialEndTime,
        "gracePeriodMinutes": validated.gracePeriodMinutes,
        "faceThreshold": validated.faceThreshold,
        "livenessEnabled": validated.livenessEnabled,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    db.collection("settings").document("organization").set(org_sync, merge=True)

    invalidate_org_settings_cache()

    db.collection("audit_logs").add(
        {
            "actorUid": admin_user.uid,
            "action": "attendance_settings_updated",
            "targetId": "attendance_settings/default",
            "status": "success",
            "details": f"Dynamic attendance settings updated: {sorted(update_dict.keys())}",
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
    )

    current_wib = now_wib()
    return DynamicAttendanceSettingsModel(**{**validated.model_dump(exclude={"updatedAt"}), "updatedAt": current_wib})


@router.get("", response_model=AttendanceSettingsModel)
async def get_settings(current_user: TokenData = Depends(get_current_user)):
    """
    Returns the system-wide attendance settings summary.
    """
    dyn = await get_dynamic_attendance_settings()
    return AttendanceSettingsModel(
        timezone=dyn.timezone,
        defaultScheduleId="standard_schedule",
        geofenceEnabled=dyn.gpsEnabled,
        livenessEnabled=dyn.livenessEnabled,
        faceThreshold=dyn.faceThreshold,
        updatedAt=dyn.updatedAt,
    )


@router.get("/organization", response_model=OrganizationSettingsModel)
async def get_organization_settings(current_user: TokenData = Depends(get_current_user)):
    """
    Returns complete organization branding, geofence, and schedule settings.
    """
    merged = dict(DEFAULT_ORG_SETTINGS)
    db = get_firestore_client()
    if db is not None:
        doc = db.collection("settings").document("organization").get()
        if doc.exists:
            d = doc.to_dict() or {}
            merged.update({k: v for k, v in d.items() if k in merged and v is not None})
            merged["updatedAt"] = _normalize_ts(d.get("updatedAt"))

        att_doc = db.collection("attendance_settings").document("default").get()
        if att_doc.exists:
            att_d = att_doc.to_dict() or {}
            for k in (
                "organizationName",
                "address",
                "latitude",
                "longitude",
                "radiusMeters",
                "maxGpsAccuracyMeters",
                "gracePeriodMinutes",
                "faceThreshold",
                "livenessEnabled",
            ):
                if k in att_d and att_d[k] is not None:
                    merged[k] = att_d[k]
            if att_d.get("officialStartTime"):
                merged["startTime"] = att_d["officialStartTime"]
            if att_d.get("officialEndTime"):
                merged["endTime"] = att_d["officialEndTime"]
            if att_d.get("updatedAt"):
                merged["updatedAt"] = _normalize_ts(att_d.get("updatedAt"))

    return OrganizationSettingsModel(**merged)


@router.patch("/organization", response_model=OrganizationSettingsModel)
async def update_organization_settings(
    payload: OrganizationSettingsUpdate,
    admin_user: TokenData = Depends(require_admin),
):
    """
    Admin-only: Updates organization parameters, geofence coordinates, radius, or schedule.
    Also keeps `attendance_settings/default` synchronized.
    """
    db = get_firestore_client()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )

    doc_ref = db.collection("settings").document("organization")
    current_doc = doc_ref.get()
    current_data = current_doc.to_dict() if current_doc.exists else dict(DEFAULT_ORG_SETTINGS)

    update_dict = payload.model_dump(exclude_unset=True)
    if not update_dict:
        current_data["updatedAt"] = _normalize_ts(current_data.get("updatedAt"))
        return OrganizationSettingsModel(**{**DEFAULT_ORG_SETTINGS, **current_data})

    merged_data = {
        **DEFAULT_ORG_SETTINGS,
        **current_data,
        **update_dict,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    doc_ref.set(merged_data, merge=True)

    # Keep attendance_settings/default aligned on shared fields
    att_sync = {}
    for shared_key in (
        "organizationName",
        "address",
        "latitude",
        "longitude",
        "radiusMeters",
        "maxGpsAccuracyMeters",
        "gracePeriodMinutes",
        "faceThreshold",
        "livenessEnabled",
    ):
        if shared_key in update_dict:
            att_sync[shared_key] = update_dict[shared_key]
    if "startTime" in update_dict:
        att_sync["officialStartTime"] = update_dict["startTime"]
    if "endTime" in update_dict:
        att_sync["officialEndTime"] = update_dict["endTime"]
    if att_sync:
        att_sync["updatedAt"] = firestore.SERVER_TIMESTAMP
        db.collection("attendance_settings").document("default").set(att_sync, merge=True)

    invalidate_org_settings_cache()

    db.collection("audit_logs").add(
        {
            "actorUid": admin_user.uid,
            "action": "organization_settings_updated",
            "targetId": "organization",
            "status": "success",
            "details": f"Settings updated: {list(update_dict.keys())}",
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
    )

    current_wib = now_wib()
    return OrganizationSettingsModel(**{**merged_data, "updatedAt": current_wib})


@router.get("/office-locations", response_model=List[OfficeLocationModel])
async def list_office_locations(current_user: TokenData = Depends(get_current_user)):
    """
    Returns the active geofenced office locations.
    """
    dyn = await get_dynamic_attendance_settings()
    return [
        OfficeLocationModel(
            locationId="loc_paud_lab_um_blitar",
            officeName=dyn.organizationName,
            latitude=dyn.latitude,
            longitude=dyn.longitude,
            radiusMeters=dyn.radiusMeters,
            maxGpsAccuracyMeters=dyn.maxGpsAccuracyMeters,
            isActive=dyn.gpsEnabled,
        )
    ]


@router.get("/schedules", response_model=List[WorkScheduleModel])
async def list_work_schedules(current_user: TokenData = Depends(get_current_user)):
    """
    Returns configured working shifts.
    """
    dyn = await get_dynamic_attendance_settings()
    return [
        WorkScheduleModel(
            scheduleId="standard_schedule",
            name="Jadwal Dinamis PAUD LAB UM Kota Blitar",
            startTime=dyn.officialStartTime,
            endTime=dyn.officialEndTime,
            gracePeriodMinutes=dyn.gracePeriodMinutes,
            workingDays=dyn.workingDays,
            timezone=dyn.timezone,
            isActive=True,
        )
    ]
