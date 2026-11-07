"""
Audit Logs API Router
Provides real-time query endpoints for immutable employee activity & security audit trail inspection.
Complies with Indonesian UU PDP No. 27/2022 biometric governance and security auditing.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from firebase_admin import firestore
from pydantic import BaseModel
from app.models.auth import TokenData
from app.security.auth import require_admin
from app.core.firebase import get_firestore_client
from app.core.timezone import WIB, to_wib
from app.services.audit_logger import normalize_audit_action
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/audit", tags=["Audit Logs"])


class AuditLogEntry(BaseModel):
    logId: str
    actorUid: str
    employeeName: Optional[str] = None
    niy: Optional[str] = None
    department: Optional[str] = None
    action: str
    targetId: Optional[str] = None
    status: str
    details: Optional[str] = None
    ipAddress: Optional[str] = None
    deviceInfo: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    timestampWib: Optional[str] = None
    workDateWib: Optional[str] = None


def _parse_wib_timestamp(raw_ts: Any, raw_iso: Optional[str] = None) -> Optional[datetime]:
    if hasattr(raw_ts, "to_datetime"):
        return to_wib(raw_ts.to_datetime())
    if isinstance(raw_ts, datetime):
        return to_wib(raw_ts)
    if isinstance(raw_ts, str) and raw_ts:
        try:
            return to_wib(datetime.fromisoformat(raw_ts))
        except Exception:
            pass
    if raw_iso:
        try:
            return to_wib(datetime.fromisoformat(raw_iso))
        except Exception:
            pass
    return None


@router.get("", response_model=List[AuditLogEntry])
async def list_audit_logs(
    action: Optional[str] = Query(None, description="Filter tipe aktivitas (mis. ATTENDANCE_CHECKIN)"),
    status: Optional[str] = Query(None, description="Filter status log (success / failed)"),
    q: Optional[str] = Query(None, description="Cari nama karyawan, NIY, keterangan, atau IP"),
    date: Optional[str] = Query(None, description="Filter tanggal tunggal WIB (YYYY-MM-DD)"),
    startDate: Optional[str] = Query(None, description="Filter tanggal awal WIB (YYYY-MM-DD)"),
    endDate: Optional[str] = Query(None, description="Filter tanggal akhir WIB (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=500),
    admin_user: TokenData = Depends(require_admin),
):
    """
    Admin-only: Retrieves recent immutable employee activity & security audit events,
    sorted newest-first in WIB (`Asia/Jakarta`), with search by employee name/NIY and date filters.
    """
    db = get_firestore_client()
    if db is None:
        return []

    has_filters = any([action, status, q, date, startDate, endDate])
    scan_cap = min(800, max(limit * 5, 250)) if has_filters else limit

    try:
        docs_iter = (
            db.collection("audit_logs")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(scan_cap)
            .stream()
        )
        raw_docs = [(doc.id, doc.to_dict() or {}) for doc in docs_iter]
    except Exception as exc:
        logger.warning(f"Ordered audit_logs query fallback triggered: {exc}")
        raw_docs = [
            (doc.id, doc.to_dict() or {})
            for doc in db.collection("audit_logs").limit(scan_cap).stream()
        ]

    # Collect UIDs missing employeeName or NIY so we can enrich them in one lookup map
    missing_uids = {
        d.get("actorUid")
        for _, d in raw_docs
        if d.get("actorUid")
        and d.get("actorUid") not in ("system", "anonymous", "unauthenticated", "-")
        and (not d.get("employeeName") or not d.get("niy") or d.get("niy") == "-")
    }
    user_lookup: Dict[str, Dict[str, str]] = {}
    if missing_uids:
        try:
            for u_doc in db.collection("users").stream():
                if u_doc.id in missing_uids:
                    u_data = u_doc.to_dict() or {}
                    user_lookup[u_doc.id] = {
                        "name": u_data.get("name") or "",
                        "niy": u_data.get("niy") or u_data.get("employeeId") or "-",
                        "department": u_data.get("department") or "-",
                    }
        except Exception as exc:
            logger.warning(f"Could not enrich audit log user profiles: {exc}")

    wanted_action = normalize_audit_action(action) if action else None
    wanted_status = status.strip().lower() if status else None
    q_lower = q.strip().lower() if q else None
    eff_start = date or startDate
    eff_end = date or endDate

    entries: List[AuditLogEntry] = []
    min_dt_wib = datetime(1970, 1, 1, tzinfo=WIB)

    for doc_id, d in raw_docs:
        actor_uid = d.get("actorUid") or "system"
        raw_action = d.get("action") or "SYSTEM_EVENT"
        norm_action = normalize_audit_action(raw_action)

        if wanted_action and norm_action != wanted_action and raw_action != action:
            continue

        raw_status = str(d.get("status") or "success").lower()
        norm_status = "failed" if raw_status in ("failed", "error", "rejected", "gagal") else "success"
        if wanted_status and norm_status != wanted_status:
            continue

        ts_wib = _parse_wib_timestamp(d.get("timestamp"), d.get("timestampWibIso"))
        work_date_wib = d.get("workDateWib") or (ts_wib.strftime("%Y-%m-%d") if ts_wib else None)

        if eff_start and (not work_date_wib or work_date_wib < eff_start):
            continue
        if eff_end and (not work_date_wib or work_date_wib > eff_end):
            continue

        enriched = user_lookup.get(actor_uid, {})
        emp_name = d.get("employeeName") or enriched.get("name") or (
            "Sistem Otomatis" if actor_uid == "system" else actor_uid
        )
        emp_niy = d.get("niy")
        if not emp_niy or emp_niy == "-":
            emp_niy = enriched.get("niy") or "-"
        emp_dept = d.get("department")
        if not emp_dept or emp_dept == "-":
            emp_dept = enriched.get("department") or "-"

        details = d.get("details") or "-"
        ip_addr = d.get("ipAddress") or "-"
        device_info = d.get("deviceInfo") or "-"

        if q_lower:
            searchable = " ".join(
                [
                    emp_name,
                    emp_niy,
                    actor_uid,
                    norm_action,
                    raw_action,
                    details,
                    ip_addr,
                    device_info,
                    emp_dept,
                ]
            ).lower()
            if q_lower not in searchable:
                continue

        ts_display = ts_wib.strftime("%d %b %Y, %H:%M:%S WIB") if ts_wib else None

        entries.append(
            AuditLogEntry(
                logId=doc_id,
                actorUid=actor_uid,
                employeeName=emp_name,
                niy=emp_niy,
                department=emp_dept,
                action=norm_action,
                targetId=d.get("targetId"),
                status=norm_status,
                details=details,
                ipAddress=ip_addr,
                deviceInfo=device_info,
                metadata=d.get("metadata") if isinstance(d.get("metadata"), dict) else None,
                timestamp=ts_wib,
                timestampWib=ts_display,
                workDateWib=work_date_wib,
            )
        )

    entries.sort(key=lambda item: item.timestamp or min_dt_wib, reverse=True)
    return entries[:limit]
