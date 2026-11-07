#!/usr/bin/env python3
"""
Clean-Slate Production Database Initialization Utility — CheckHadir.Id
Target Institution: PAUD LAB UM KOTA BLITAR

Actions performed:
1. Provisions or updates the single Primary Administrator account in Firebase Auth
   with custom claims `{"role": "admin"}` and Firestore `users/{uid}` document.
2. Purges test attendance records (`attendance` & `attendances`), `audit_logs`,
   temporary leave test data (`leaves`), and temporary biometric vectors (`face_profiles`).
3. Deletes all dummy / non-admin employee users in Firestore `users` and Firebase Authentication
   (strictly preserving the primary Admin account).
4. Seeds authoritative default working hours & geofence policy in `attendance_settings/default`
   (active days Monday–Friday `[1,2,3,4,5]`, late tolerance `15` minutes, radius `100m`,
   PAUD LAB UM Kota Blitar coordinates `-8.095469, 112.170750`) and syncs `settings/organization`.

Usage Examples:
  # Dry-run preview (no destructive writes/deletes):
  python backend/scripts/init_production_db.py --admin-email admin@paudlabum.sch.id --dry-run

  # Live execution via CLI arguments:
  python backend/scripts/init_production_db.py \
      --admin-email admin@paudlabum.sch.id \
      --admin-password "<STRONG_PASSWORD>" \
      --admin-name "Administrator PAUD LAB UM" \
      --yes

  # Live execution via Environment Variables:
  # Set INIT_ADMIN_EMAIL, INIT_ADMIN_PASSWORD, INIT_ADMIN_NAME, INIT_ADMIN_NIY
  python backend/scripts/init_production_db.py --yes
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure `backend/` package root is importable regardless of current working directory
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from firebase_admin import auth as fb_auth, firestore  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.timezone import today_wib_str  # noqa: E402
from app.core.firebase import (  # noqa: E402
    get_firestore_client,
    initialize_firebase,
    is_firebase_ready,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("init_production_db")

# Collections purged during clean-slate reset (both `attendance` and `attendances` are included)
PURGE_COLLECTIONS: Tuple[str, ...] = (
    "attendance",
    "attendances",
    "audit_logs",
    "face_profiles",
    "leaves",
)

BATCH_LIMIT = 400


def build_default_attendance_settings(admin_uid: str) -> Dict[str, Any]:
    """
    Builds the canonical production working schedule & geofencing document
    for `attendance_settings/default` (Monday–Friday, 15m tolerance, 100m geofence).
    """
    return {
        "organizationName": settings.DEFAULT_OFFICE_NAME,
        "address": settings.DEFAULT_OFFICE_ADDRESS,
        "timezone": "Asia/Jakarta",
        "workingDays": [1, 2, 3, 4, 5],  # Senin - Jumat
        "checkInOpenTime": "06:30",
        "checkInCloseTime": "09:00",
        "officialStartTime": "08:00",
        "startTime": "08:00",
        "gracePeriodMinutes": 15,
        "checkOutOpenTime": "16:00",
        "checkOutCloseTime": "19:00",
        "officialEndTime": "17:00",
        "endTime": "17:00",
        "latitude": -8.095469,
        "longitude": 112.170750,
        "radiusMeters": 100.0,
        "maxGpsAccuracyMeters": 50.0,
        "gpsEnabled": True,
        "livenessEnabled": True,
        "emailNotificationEnabled": True,
        "faceThreshold": settings.FACE_SIMILARITY_THRESHOLD,
        "updatedBy": admin_uid,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }


def purge_collection(db: Any, collection_name: str, dry_run: bool = False) -> int:
    """
    Deletes all documents in a Firestore collection in batches of `BATCH_LIMIT`.
    Returns the number of documents matched/deleted.
    """
    docs = list(db.collection(collection_name).stream())
    total = len(docs)
    if total == 0:
        logger.info("Collection '%s' sudah bersih (0 dokumen).", collection_name)
        return 0

    if dry_run:
        logger.info("[DRY-RUN] Akan menghapus %d dokumen dari koleksi '%s'.", total, collection_name)
        return total

    deleted = 0
    for idx in range(0, total, BATCH_LIMIT):
        batch = db.batch()
        chunk = docs[idx : idx + BATCH_LIMIT]
        for doc in chunk:
            batch.delete(doc.reference)
        batch.commit()
        deleted += len(chunk)

    logger.info("Berhasil menghapus %d dokumen dari koleksi '%s'.", deleted, collection_name)
    return deleted


def ensure_primary_admin(
    db: Any,
    admin_email: str,
    admin_password: Optional[str],
    admin_name: str,
    admin_niy: str,
    admin_phone: str = "",
    dry_run: bool = False,
    auth_module: Any = fb_auth,
) -> str:
    """
    Ensures the single primary Admin account exists in Firebase Auth with `{"role": "admin"}`
    custom claims and writes/updates `users/{admin_uid}` in Firestore.
    Never logs passwords or credentials.
    """
    normalized_email = admin_email.strip().lower()
    if not normalized_email or "@" not in normalized_email:
        raise ValueError("Email Admin utama tidak valid.")

    existing_user = None
    try:
        existing_user = auth_module.get_user_by_email(normalized_email)
    except auth_module.UserNotFoundError:
        existing_user = None

    if existing_user is None:
        if not admin_password or len(admin_password) < 6:
            raise ValueError(
                "Akun Admin belum terdaftar di Firebase Auth. "
                "Harap sertakan --admin-password atau env INIT_ADMIN_PASSWORD (minimal 6 karakter)."
            )
        if dry_run:
            logger.info(
                "[DRY-RUN] Akan membuat akun Admin baru di Firebase Auth untuk email '%s'.",
                normalized_email,
            )
            return "dryrun_admin_uid"

        created = auth_module.create_user(
            email=normalized_email,
            password=admin_password,
            display_name=admin_name.strip(),
            email_verified=True,
            disabled=False,
        )
        admin_uid = created.uid
        logger.info("Akun Admin utama berhasil dibuat di Firebase Auth (uid=%s).", admin_uid)
    else:
        admin_uid = existing_user.uid
        if dry_run:
            logger.info(
                "[DRY-RUN] Akun Admin ditemukan (uid=%s, email=%s); akan memperbarui custom claims & dokumen Firestore.",
                admin_uid,
                normalized_email,
            )
            return admin_uid

        update_kwargs: Dict[str, Any] = {
            "display_name": admin_name.strip(),
            "email_verified": True,
            "disabled": False,
        }
        if admin_password:
            if len(admin_password) < 6:
                raise ValueError("Password Admin minimal harus 6 karakter.")
            update_kwargs["password"] = admin_password
        auth_module.update_user(admin_uid, **update_kwargs)
        logger.info("Akun Admin utama diperbarui di Firebase Auth (uid=%s).", admin_uid)

    # Enforce Firebase Custom Claims role='admin'
    auth_module.set_custom_user_claims(admin_uid, {"role": "admin"})

    # Upsert authoritative Admin profile in Firestore `users/{admin_uid}`
    now_date = today_wib_str()
    admin_doc = {
        "uid": admin_uid,
        "niy": admin_niy.strip(),
        "employeeId": admin_niy.strip(),
        "name": admin_name.strip(),
        "email": normalized_email,
        "phone": admin_phone.strip(),
        "phoneNumber": admin_phone.strip(),
        "department": "Pimpinan & Manajemen",
        "position": "Administrator Sistem",
        "employmentStatus": "Tetap",
        "joinDate": now_date,
        "role": "admin",
        "email_verified": True,
        "emailVerified": True,
        "is_active": True,
        "isActive": True,
        "face_registered": False,
        "faceRegistered": False,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    user_ref = db.collection("users").document(admin_uid)
    existing_doc = user_ref.get()
    if not existing_doc.exists:
        admin_doc["created_at"] = firestore.SERVER_TIMESTAMP
        admin_doc["createdAt"] = firestore.SERVER_TIMESTAMP

    user_ref.set(admin_doc, merge=True)
    logger.info("Dokumen Firestore users/%s dipastikan memiliki role='admin'.", admin_uid)
    return admin_uid


def purge_non_admin_users(
    db: Any,
    admin_uid: str,
    admin_email: str,
    dry_run: bool = False,
    auth_module: Any = fb_auth,
) -> Tuple[int, int]:
    """
    Deletes all dummy / non-admin users from Firestore `users` and Firebase Auth,
    strictly skipping `admin_uid` and `admin_email`.
    Returns `(deleted_firestore_users, deleted_auth_users)`.
    """
    normalized_admin_email = admin_email.strip().lower()

    # 1. Purge Firestore `users` documents except Primary Admin
    fs_docs = list(db.collection("users").stream())
    fs_targets = []
    for doc in fs_docs:
        data = doc.to_dict() or {}
        doc_email = str(data.get("email") or "").strip().lower()
        if doc.id == admin_uid or (normalized_admin_email and doc_email == normalized_admin_email):
            continue
        fs_targets.append(doc)

    if dry_run:
        logger.info(
            "[DRY-RUN] Akan menghapus %d dokumen user non-admin dari koleksi 'users'.",
            len(fs_targets),
        )
    else:
        for idx in range(0, len(fs_targets), BATCH_LIMIT):
            batch = db.batch()
            for doc in fs_targets[idx : idx + BATCH_LIMIT]:
                batch.delete(doc.reference)
            batch.commit()
        logger.info("Menghapus %d dokumen user dummy dari koleksi 'users'.", len(fs_targets))

    # 2. Purge Firebase Authentication accounts except Primary Admin
    auth_uids_to_delete: List[str] = []
    for user_rec in auth_module.list_users().iterate_all():
        rec_email = str(getattr(user_rec, "email", "") or "").strip().lower()
        if user_rec.uid == admin_uid or (normalized_admin_email and rec_email == normalized_admin_email):
            continue
        auth_uids_to_delete.append(user_rec.uid)

    if dry_run:
        logger.info(
            "[DRY-RUN] Akan menghapus %d akun user non-admin dari Firebase Authentication.",
            len(auth_uids_to_delete),
        )
    else:
        deleted_auth = 0
        # Firebase Admin SDK delete_users supports up to 1000 UIDs per call
        for idx in range(0, len(auth_uids_to_delete), 1000):
            chunk = auth_uids_to_delete[idx : idx + 1000]
            result = auth_module.delete_users(chunk)
            deleted_auth += result.success_count
            if result.failure_count:
                logger.warning(
                    "%d akun gagal dihapus dari Firebase Auth pada batch ini.",
                    result.failure_count,
                )
        logger.info("Menghapus %d akun karyawan dummy dari Firebase Authentication.", deleted_auth)
        return len(fs_targets), deleted_auth

    return len(fs_targets), len(auth_uids_to_delete)


def seed_default_attendance_settings(
    db: Any,
    admin_uid: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Writes default working hours & geofencing configuration to `attendance_settings/default`
    and keeps `settings/organization` synchronized.
    """
    payload = build_default_attendance_settings(admin_uid=admin_uid)
    if dry_run:
        logger.info(
            "[DRY-RUN] Akan menyimpan pengaturan jam kerja default ke 'attendance_settings/default' "
            "(Senin-Jumat, toleransi 15 menit, radius 100m)."
        )
        return payload

    db.collection("attendance_settings").document("default").set(payload, merge=True)
    db.collection("settings").document("organization").set(payload, merge=True)
    logger.info(
        "Pengaturan jam kerja default berhasil disimpan di 'attendance_settings/default' "
        "(Senin-Jumat, 08:00 WIB, toleransi 15m, radius 100m)."
    )
    return payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inisialisasi & Bersih-Bersih Database Produksi CheckHadir.Id (Clean Slate)."
    )
    parser.add_argument(
        "--admin-email",
        default=os.getenv("INIT_ADMIN_EMAIL", "admin@checkhadir.id"),
        help="Alamat email akun Admin utama (default: admin@checkhadir.id atau env INIT_ADMIN_EMAIL).",
    )
    parser.add_argument(
        "--admin-password",
        default=os.getenv("INIT_ADMIN_PASSWORD", ""),
        help="Password akun Admin utama minimal 6 karakter (atau env INIT_ADMIN_PASSWORD).",
    )
    parser.add_argument(
        "--admin-name",
        default=os.getenv("INIT_ADMIN_NAME", "Administrator PAUD LAB UM"),
        help="Nama tampilan Admin utama (default: 'Administrator PAUD LAB UM').",
    )
    parser.add_argument(
        "--admin-niy",
        default=os.getenv("INIT_ADMIN_NIY", "ADM-PAUD-001"),
        help="NIY / NIP Admin utama (default: 'ADM-PAUD-001').",
    )
    parser.add_argument(
        "--admin-phone",
        default=os.getenv("INIT_ADMIN_PHONE", ""),
        help="Nomor WhatsApp Admin utama opsional (atau env INIT_ADMIN_PHONE).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulasikan pembersihan dan tampilkan ringkasan tanpa menghapus data asli.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Lewati konfirmasi interaktif untuk eksekusi langsung.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.admin_email.strip():
        logger.error(
            "Email Admin utama wajib diisi melalui --admin-email atau variabel environment INIT_ADMIN_EMAIL."
        )
        return 1

    initialize_firebase()
    db = get_firestore_client()
    if db is None or not is_firebase_ready():
        logger.error(
            "Firebase Admin SDK belum terhubung ke project live (%s). "
            "Pastikan backend/serviceAccountKey.json tersedia.",
            settings.FIREBASE_PROJECT_ID,
        )
        return 1

    if not args.dry_run and not args.yes:
        print("\n" + "=" * 72)
        print("PERINGATAN: CLEAN SLATE DATABASE PRODUKSI — CHECKHADIR.ID")
        print(f"Firebase Project : {settings.FIREBASE_PROJECT_ID}")
        print(f"Primary Admin    : {args.admin_email.strip().lower()}")
        print("Seluruh data pengujian pada attendance/attendances, audit_logs,")
        print("face_profiles, leaves, dan akun karyawan non-admin AKAN DIHAPUS PERMANEN.")
        print("=" * 72)
        confirm = input("Ketik 'RESET' untuk melanjutkan: ").strip()
        if confirm != "RESET":
            logger.info("Dibatalkan oleh pengguna.")
            return 0

    try:
        # 1. Provision / ensure Primary Admin first so its UID is protected
        admin_uid = ensure_primary_admin(
            db=db,
            admin_email=args.admin_email,
            admin_password=args.admin_password or None,
            admin_name=args.admin_name,
            admin_niy=args.admin_niy,
            admin_phone=args.admin_phone,
            dry_run=args.dry_run,
        )

        # 2. Purge test attendance, biometric profiles, leaves, and old audit logs
        purged_counts: Dict[str, int] = {}
        for col_name in PURGE_COLLECTIONS:
            purged_counts[col_name] = purge_collection(db, col_name, dry_run=args.dry_run)

        # 3. Purge dummy non-admin users in Firestore & Firebase Auth
        deleted_fs_users, deleted_auth_users = purge_non_admin_users(
            db=db,
            admin_uid=admin_uid,
            admin_email=args.admin_email,
            dry_run=args.dry_run,
        )

        # 4. Seed default working schedule & geofencing settings
        seed_default_attendance_settings(db=db, admin_uid=admin_uid, dry_run=args.dry_run)

        # 5. Record initial clean-slate audit entry (when live)
        if not args.dry_run:
            db.collection("audit_logs").add(
                {
                    "actorUid": admin_uid,
                    "action": "production_clean_slate_initialized",
                    "targetId": "attendance_settings/default",
                    "status": "success",
                    "details": (
                        "Database diinisialisasi ulang untuk produksi (Clean Slate). "
                        f"Admin utama: {args.admin_email.strip().lower()}."
                    ),
                    "timestamp": firestore.SERVER_TIMESTAMP,
                }
            )

        logger.info("=== RINGKASAN INISIALISASI PRODUKSI (%s) ===", "DRY-RUN" if args.dry_run else "LIVE")
        logger.info("Admin Utama       : %s (uid=%s)", args.admin_email.strip().lower(), admin_uid)
        logger.info("Koleksi Dibersihkan: %s", purged_counts)
        logger.info(
            "User Dummy Dihapus: %d dokumen Firestore, %d akun Firebase Auth",
            deleted_fs_users,
            deleted_auth_users,
        )
        logger.info("Jadwal Default    : Senin-Jumat (08:00 WIB, toleransi 15m, radius 100m)")
        return 0

    except Exception as exc:
        logger.error("Inisialisasi produksi gagal: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
