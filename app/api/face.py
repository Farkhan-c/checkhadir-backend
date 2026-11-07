"""
Face Biometrics & Anti-Spoofing API Router
Provides comprehensive face quality validation, interactive liveness challenges,
biometric enrollment with Fernet (AES-128 + HMAC) encryption, 1:1 verification, and 1:N recognition.
Complies with Indonesian UU PDP No. 27/2022 biometric governance.
"""

import time
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from firebase_admin import firestore

from app.core.timezone import WIB, now_wib, to_wib
from app.services.audit_logger import (
    ACTION_FACE_APPROVAL_ACCEPTED,
    ACTION_FACE_APPROVAL_REJECTED,
    ACTION_FACE_ENROLLMENT,
    ACTION_FACE_RESET,
    ACTION_SPOOF_ATTEMPT,
    record_audit_event,
)

from app.models.auth import TokenData
from app.models.employee import resolve_face_verification_fields
from app.models.user import FaceVerificationStatus, UserRole
from app.models.face import (
    FaceApprovalActionResponse,
    FaceApprovalPendingItem,
    FaceApprovalRejectRequest,
    FaceQualityReport,
    LivenessChallengeStartResponse,
    LivenessVerifyRequest,
    LivenessVerifyResponse,
    FaceRegisterRequest,
    FaceRegisterResponse,
    FaceRecognizeRequest,
    FaceRecognizeResponse,
)
from app.security.auth import get_current_user, require_admin
from app.security.encryption import biometric_crypto
from app.security.rate_limit import rate_limit
from app.core.firebase import get_firestore_client
from app.core.config import settings
from app.core.org_settings import get_effective_org_settings
from app.face_recognition.engine import face_engine, FaceQualityError
from app.face_recognition.liveness import liveness_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/face", tags=["Face Biometrics"])
admin_face_approvals_router = APIRouter(
    prefix="/api/admin/face-approvals",
    tags=["Admin Face Approvals"],
)

# Face detection/matching is CPU-heavy and a brute-force vector: throttle per authenticated user
face_rate_limit = rate_limit("face", lambda: settings.RATE_LIMIT_FACE_PER_MINUTE)

class QualityCheckRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded frame from webcam")

@router.post("/quality-check", response_model=FaceQualityReport)
async def check_face_quality(
    payload: QualityCheckRequest,
    current_user: TokenData = Depends(face_rate_limit)
):
    """
    Evaluates real-time frame quality (sharpness/blur, illumination, single face detection)
    without persisting or extracting permanent biometric templates.
    """
    try:
        img = face_engine.decode_image(payload.image)
        report = face_engine.check_image_quality(img)
        return report
    except Exception as e:
        logger.error(f"Quality check failed: {e}")
        return FaceQualityReport(
            isQualityPassed=False,
            blurScore=0.0,
            isBlurry=True,
            brightnessScore=0.0,
            isTooDark=False,
            isTooBright=False,
            faceCount=0,
            message=f"Gagal memproses frame: {str(e)}"
        )

@router.post("/liveness/start", response_model=LivenessChallengeStartResponse)
async def start_liveness_challenge(
    preferred_challenge: Optional[str] = None,
    current_user: TokenData = Depends(face_rate_limit)
):
    """
    Issues a cryptographically secure, single-use anti-spoofing challenge session.
    Expires within 60 seconds.
    """
    try:
        challenge = liveness_service.create_challenge(
            uid=current_user.uid,
            preferred_challenge=preferred_challenge,
        )
        return LivenessChallengeStartResponse(**challenge)
    except Exception as e:
        logger.error(f"Failed to create liveness challenge: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal membuat sesi tantangan liveness."
        )

@router.post("/liveness/verify", response_model=LivenessVerifyResponse)
async def verify_liveness_challenge(
    payload: LivenessVerifyRequest,
    request: Request,
    current_user: TokenData = Depends(face_rate_limit)
):
    """
    Verifies user webcam interaction against issued liveness challenge.
    Prevents replay attacks through single-use session tokens and logs spoof attempts.
    """
    db = get_firestore_client()
    try:
        result = liveness_service.verify_liveness(
            session_id=payload.sessionId,
            baseline_input=payload.baselineImage,
            action_input=payload.actionImage,
            expected_uid=current_user.uid
        )
        msg_lower = str(result.get("message") or "").lower()
        if not result.get("passed") and (
            not result.get("passivePassed", True) or "spoofing" in msg_lower
        ):
            record_audit_event(
                db,
                action=ACTION_SPOOF_ATTEMPT,
                actor_uid=current_user.uid,
                target_id=payload.sessionId,
                status="failed",
                details=f"Liveness anti-spoofing menolak frame: {result.get('message')}",
                request=request,
                metadata={
                    "challengeType": result.get("challengeType"),
                    "passivePassed": result.get("passivePassed", False),
                    "confidence": result.get("confidence", 0.0),
                },
            )
        return LivenessVerifyResponse(**result)
    except FaceQualityError as fqe:
        if fqe.code == "SESSION_ALREADY_USED":
            record_audit_event(
                db,
                action=ACTION_SPOOF_ATTEMPT,
                actor_uid=current_user.uid,
                target_id=payload.sessionId,
                status="failed",
                details=f"Indikasi replay attack pada sesi liveness: {fqe.message}",
                request=request,
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=fqe.message)
    except Exception as e:
        logger.error(f"Liveness verification exception: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Verifikasi liveness gagal diproses.")

@router.post("/register", response_model=FaceRegisterResponse)
async def register_face_biometrics(
    payload: FaceRegisterRequest,
    request: Request,
    target_uid: Optional[str] = None,
    current_user: TokenData = Depends(face_rate_limit)
):
    """
    Enrolls employee biometric template from 3 quality samples:
    1. Validates UU PDP explicit consent.
    2. Validates image sharpness, illumination, and single-face presence across all samples.
    3. Verifies pairwise facial consistency between samples (guarantees same person).
    4. Computes normalized average 128-d ArcFace/SFace embedding vector.
    5. Encrypts template using Fernet (AES-128 + HMAC) (Fernet) before writing to Cloud Firestore.
    6. Raw images are strictly discarded from memory.
    """
    if not payload.consentAgreed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Persetujuan pemrosesan data biometrik (UU PDP) wajib disetujui sebelum mendaftarkan wajah."
        )

    # Resolve target UID (regular user can only register for self; admin can register for specified target)
    uid = current_user.uid
    if target_uid and target_uid != current_user.uid:
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Hanya Administrator yang dapat mendaftarkan wajah pegawai lain."
            )
        uid = target_uid

    db = get_firestore_client()
    if db is None:
        # Never report a successful enrollment that was not actually persisted
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service unavailable.")

    user_ref = db.collection("users").document(uid)
    user_snap = user_ref.get()
    u_data = (user_snap.to_dict() or {}) if user_snap.exists else {}
    current_face_info = resolve_face_verification_fields(u_data)
    current_face_status = current_face_info["status"]
    existing_face_doc = db.collection("face_profiles").document(uid).get()

    # Re-enrollment protection for non-admin callers:
    # Allowed when status is NOT_ENROLLED or REJECTED; blocked when PENDING_APPROVAL or APPROVED.
    if current_user.role != UserRole.ADMIN:
        if current_face_status == FaceVerificationStatus.PENDING_APPROVAL:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Perekaman wajah Anda sedang menunggu verifikasi dan persetujuan Administrator.",
            )
        if current_face_status == FaceVerificationStatus.APPROVED or (
            existing_face_doc.exists and current_face_status != FaceVerificationStatus.REJECTED
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Wajah Anda sudah terdaftar dan terverifikasi. Hubungi Administrator untuk mereset pendaftaran wajah sebelum mendaftar ulang.",
            )

    try:
        # Process and validate samples
        final_embedding, reports = face_engine.process_enrollment_samples(payload.samples)

        # Prevent duplicate enrollment: the same face must not be registered under two accounts.
        # The conflicting account is deliberately NOT disclosed to the caller.
        threshold = get_effective_org_settings()["faceThreshold"]
        for other in db.collection("face_profiles").stream():
            if other.id == uid:
                continue
            enc = (other.to_dict() or {}).get("encryptedEmbedding")
            if not enc:
                continue
            try:
                other_emb = biometric_crypto.decrypt_embedding(enc)
            except ValueError:
                logger.warning(f"Skipping undecryptable profile {other.id} during duplicate check")
                continue
            if face_engine.compute_similarity(final_embedding, other_emb) >= threshold:
                logger.warning(f"Duplicate face enrollment blocked for {uid} (matches existing profile)")
                record_audit_event(
                    db,
                    action=ACTION_FACE_ENROLLMENT,
                    actor_uid=uid,
                    target_id=uid,
                    status="failed",
                    details="Pendaftaran wajah ditolak: Wajah ini sudah terdaftar pada akun lain.",
                    request=request,
                )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Wajah ini sudah terdaftar pada akun lain. Hubungi Administrator."
                )

        # Encrypt embedding with Fernet (AES-128 + HMAC) and generate preview thumbnail for Admin review
        encrypted_embedding = biometric_crypto.encrypt_embedding(final_embedding)
        preview_thumbnail = face_engine.create_preview_thumbnail(payload.samples[0])

        enrolled_time = now_wib()
        is_admin_direct = current_user.role == UserRole.ADMIN and uid != current_user.uid
        target_verification_status = (
            FaceVerificationStatus.APPROVED
            if is_admin_direct
            else FaceVerificationStatus.PENDING_APPROVAL
        )
        is_face_registered = target_verification_status == FaceVerificationStatus.APPROVED

        # 1. Store encrypted profile + visual preview thumbnail in face_profiles/{uid}
        face_ref = db.collection("face_profiles").document(uid)
        face_ref.set({
            "uid": uid,
            "encryptedEmbedding": encrypted_embedding,
            "previewThumbnail": preview_thumbnail,
            "verificationStatus": target_verification_status.value,
            "keyVersion": biometric_crypto.version,
            "algorithm": "SFace-v1",
            "vectorDimension": 128,
            "sampleCount": len(payload.samples),
            "submittedAt": enrolled_time.isoformat(),
            "enrolledAt": firestore.SERVER_TIMESTAMP,
            "lastUpdated": firestore.SERVER_TIMESTAMP,
        })

        # 2. Update user biometric verification status in users/{uid}
        if user_snap.exists:
            user_ref.update({
                "faceRegistered": is_face_registered,
                "face_registered": is_face_registered,
                "faceVerificationStatus": target_verification_status.value,
                "face_verification_status": target_verification_status.value,
                "faceRejectionReason": None,
                "face_rejection_reason": None,
                "faceSubmittedAt": enrolled_time.isoformat(),
                "face_submitted_at": enrolled_time.isoformat(),
                "faceEnrolledAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            })

        # 3. Audit log
        actor_note = "oleh Admin (langsung disetujui)" if is_admin_direct else "mandiri (menunggu persetujuan Admin)"
        record_audit_event(
            db,
            action=ACTION_FACE_ENROLLMENT,
            actor_uid=uid,
            target_id=uid,
            employee_name=u_data.get("name"),
            niy=u_data.get("niy") or u_data.get("employeeId"),
            department=u_data.get("department"),
            status="success",
            details=(
                f"Perekaman profil biometrik wajah ({actor_note}) berhasil dengan "
                f"{len(payload.samples)} sampel (SFace-v1, enkripsi Fernet AES-128 + HMAC)."
            ),
            request=request,
            metadata={
                "samplesProcessed": len(payload.samples),
                "enrolledBy": current_user.uid,
                "faceVerificationStatus": target_verification_status.value,
            },
        )

        logger.info(
            f"Successfully enrolled biometrics for employee {uid} (status={target_verification_status.value})"
        )

        response_msg = (
            "Pendaftaran biometrik wajah karyawan oleh Administrator berhasil dan telah langsung aktif."
            if is_admin_direct
            else (
                "Perekaman Wajah Berhasil! Wajah Anda sedang menunggu verifikasi dan persetujuan "
                "dari Administrator sebelum dapat digunakan untuk presensi."
            )
        )

        return FaceRegisterResponse(
            success=True,
            uid=uid,
            samplesProcessed=len(payload.samples),
            algorithm="SFace-v1",
            enrolledAt=enrolled_time,
            faceRegistered=is_face_registered,
            face_registered=is_face_registered,
            faceVerificationStatus=target_verification_status,
            face_verification_status=target_verification_status,
            faceSubmittedAt=enrolled_time,
            face_submitted_at=enrolled_time,
            message=response_msg,
        )

    except HTTPException:
        raise
    except FaceQualityError as fqe:
        record_audit_event(
            db,
            action=ACTION_FACE_ENROLLMENT,
            actor_uid=uid,
            target_id=uid,
            status="failed",
            details=f"Perekaman biometrik wajah gagal kualitas: {fqe.message}",
            request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=fqe.message
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error during face enrollment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Terjadi kesalahan saat menyimpan biometrik."
        )

@router.post("/recognize", response_model=FaceRecognizeResponse)
async def recognize_face(
    payload: FaceRecognizeRequest,
    current_user: TokenData = Depends(face_rate_limit)
):
    """
    1:1 Face Verification:
    Compares live webcam frame against user's stored enrolled template.
    Returns match boolean, cosine similarity score, and confidence tier.
    Non-admin callers can only verify against their own template; probing another
    employee's template would make this endpoint a biometric oracle.
    """
    start_time = time.time()
    target_uid = payload.targetUid or current_user.uid
    if target_uid != current_user.uid and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Anda hanya dapat memverifikasi wajah Anda sendiri."
        )

    # Decode image and check quality
    try:
        img = face_engine.decode_image(payload.image)
        quality = face_engine.check_image_quality(img)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Gagal memproses gambar: {str(e)}")

    if not quality.isQualityPassed:
        proc_time = (time.time() - start_time) * 1000.0
        return FaceRecognizeResponse(
            matched=False,
            similarity=0.0,
            threshold=payload.similarityThreshold or settings.FACE_SIMILARITY_THRESHOLD,
            confidence="rejected",
            matchedUid=None,
            quality=quality,
            processingTimeMs=round(proc_time, 2)
        )

    db = get_firestore_client()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable."
        )

    # Verify approval status before allowing 1:1 matching for non-admin callers
    user_snap_for_match = db.collection("users").document(target_uid).get()
    if user_snap_for_match.exists and current_user.role != UserRole.ADMIN:
        match_face_info = resolve_face_verification_fields(user_snap_for_match.to_dict() or {})
        if match_face_info["status"] != FaceVerificationStatus.APPROVED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Status Biometrik wajah belum disetujui Administrator sehingga belum dapat digunakan untuk verifikasi.",
            )

    # Retrieve stored profile
    profile_doc = db.collection("face_profiles").document(target_uid).get()
    if not profile_doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template wajah belum didaftarkan untuk pengguna ini. Silakan lakukan pendaftaran wajah terlebih dahulu."
        )

    data = profile_doc.to_dict()
    encrypted_emb = data.get("encryptedEmbedding")
    if not encrypted_emb:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data template biometrik tidak valid atau rusak."
        )

    # Decrypt stored template
    try:
        enrolled_emb = biometric_crypto.decrypt_embedding(encrypted_emb)
    except Exception as e:
        logger.error(f"Decryption error for UID {target_uid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mendekripsi template biometrik."
        )

    # Extract live embedding
    try:
        live_emb = face_engine.extract_embedding(img, enforce_quality=False)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Ekstraksi fitur wajah gagal: {str(e)}")

    # Verify match
    applied_threshold = payload.similarityThreshold or settings.FACE_SIMILARITY_THRESHOLD
    is_match, similarity, confidence = face_engine.verify_match(
        live_embedding=live_emb,
        enrolled_embedding=enrolled_emb,
        threshold=applied_threshold
    )

    proc_time = (time.time() - start_time) * 1000.0

    # Fetch user name for response if matched
    matched_name = None
    if is_match:
        user_doc = db.collection("users").document(target_uid).get()
        if user_doc.exists:
            matched_name = user_doc.to_dict().get("name")

    return FaceRecognizeResponse(
        matched=is_match,
        similarity=similarity,
        threshold=applied_threshold,
        confidence=confidence,
        matchedUid=target_uid if is_match else None,
        matchedEmployeeName=matched_name,
        quality=quality,
        processingTimeMs=round(proc_time, 2)
    )

@router.post("/identify", response_model=FaceRecognizeResponse)
async def identify_face(
    payload: FaceRecognizeRequest,
    current_user: TokenData = Depends(require_admin)
):
    """
    1:N Face Identification (ADMIN ONLY):
    Compares live frame against ALL enrolled templates. Restricted to administrators because
    it reveals which employee a face belongs to.
    Finds the highest matching candidate exceeding the threshold.
    """
    start_time = time.time()

    try:
        img = face_engine.decode_image(payload.image)
        quality = face_engine.check_image_quality(img)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Gagal memproses gambar: {str(e)}")

    if not quality.isQualityPassed:
        proc_time = (time.time() - start_time) * 1000.0
        return FaceRecognizeResponse(
            matched=False,
            similarity=0.0,
            threshold=payload.similarityThreshold or settings.FACE_SIMILARITY_THRESHOLD,
            confidence="rejected",
            matchedUid=None,
            quality=quality,
            processingTimeMs=round(proc_time, 2)
        )

    db = get_firestore_client()
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable.")

    # Extract live embedding
    live_emb = face_engine.extract_embedding(img, enforce_quality=False)
    applied_threshold = payload.similarityThreshold or settings.FACE_SIMILARITY_THRESHOLD

    # Stream all face profiles
    profiles_stream = db.collection("face_profiles").stream()
    best_match_uid = None
    best_sim = -1.0
    best_confidence = "rejected"

    for p_doc in profiles_stream:
        p_data = p_doc.to_dict() or {}
        if p_data.get("verificationStatus") and str(p_data.get("verificationStatus")).upper() != "APPROVED":
            continue
        enc_emb = p_data.get("encryptedEmbedding")
        if not enc_emb:
            continue
        try:
            stored_emb = biometric_crypto.decrypt_embedding(enc_emb)
            sim = face_engine.compute_similarity(live_emb, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_match_uid = p_doc.id
        except Exception as e:
            logger.warning(f"Failed to decrypt profile {p_doc.id}: {e}")
            continue

    is_matched = best_sim >= applied_threshold
    matched_name = None

    if is_matched and best_match_uid:
        u_doc = db.collection("users").document(best_match_uid).get()
        if u_doc.exists:
            matched_name = u_doc.to_dict().get("name")
        best_confidence = "high" if best_sim >= 0.75 else "medium"

    proc_time = (time.time() - start_time) * 1000.0

    return FaceRecognizeResponse(
        matched=is_matched,
        similarity=round(best_sim, 4) if best_sim > 0 else 0.0,
        threshold=applied_threshold,
        confidence=best_confidence if is_matched else "rejected",
        matchedUid=best_match_uid if is_matched else None,
        matchedEmployeeName=matched_name,
        quality=quality,
        processingTimeMs=round(proc_time, 2)
    )

def _parse_wib_dt(val) -> Optional[datetime]:
    if val is None:
        return None
    if hasattr(val, "to_datetime"):
        return to_wib(val.to_datetime())
    if isinstance(val, datetime):
        return to_wib(val)
    if isinstance(val, str) and val.strip():
        try:
            return to_wib(datetime.fromisoformat(val.strip()))
        except Exception:
            return None
    return None


def _resolve_admin_display_name(db, admin_user: TokenData) -> str:
    if admin_user.name:
        return admin_user.name
    if db is not None:
        try:
            adm_snap = db.collection("users").document(admin_user.uid).get()
            if adm_snap.exists:
                adm_name = (adm_snap.to_dict() or {}).get("name")
                if adm_name:
                    return adm_name
        except Exception:
            pass
    return admin_user.email or "Administrator"


@router.get("/status/{uid}")
async def get_face_registration_status(
    uid: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Returns face enrollment & verification status and metadata (without sensitive biometric vectors).
    """
    if current_user.role != UserRole.ADMIN and current_user.uid != uid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Akses ditolak.")

    db = get_firestore_client()
    if db is None:
        return {
            "faceRegistered": False,
            "face_registered": False,
            "faceVerificationStatus": FaceVerificationStatus.NOT_ENROLLED.value,
            "face_verification_status": FaceVerificationStatus.NOT_ENROLLED.value,
            "faceRejectionReason": None,
            "face_rejection_reason": None,
            "faceSubmittedAt": None,
            "face_submitted_at": None,
            "enrolledAt": None,
        }

    u_doc = db.collection("users").document(uid).get()
    u_data = (u_doc.to_dict() or {}) if u_doc.exists else {}
    face_info = resolve_face_verification_fields(u_data)
    status_val = face_info["status"]
    is_registered = face_info["face_registered"]
    rejection_reason = face_info["rejection_reason"]
    submitted_dt = _parse_wib_dt(face_info["submitted_raw"])
    submitted_iso = submitted_dt.isoformat() if submitted_dt else None

    doc = db.collection("face_profiles").document(uid).get()
    if not doc.exists:
        return {
            "faceRegistered": False,
            "face_registered": False,
            "faceVerificationStatus": (
                status_val.value
                if status_val == FaceVerificationStatus.REJECTED
                else FaceVerificationStatus.NOT_ENROLLED.value
            ),
            "face_verification_status": (
                status_val.value
                if status_val == FaceVerificationStatus.REJECTED
                else FaceVerificationStatus.NOT_ENROLLED.value
            ),
            "faceRejectionReason": rejection_reason,
            "face_rejection_reason": rejection_reason,
            "faceSubmittedAt": submitted_iso,
            "face_submitted_at": submitted_iso,
            "enrolledAt": None,
        }

    data = doc.to_dict() or {}
    enrolled_dt = _parse_wib_dt(data.get("enrolledAt") or data.get("submittedAt"))
    enrolled_iso = enrolled_dt.isoformat() if enrolled_dt else submitted_iso

    return {
        "faceRegistered": is_registered,
        "face_registered": is_registered,
        "faceVerificationStatus": status_val.value,
        "face_verification_status": status_val.value,
        "faceRejectionReason": rejection_reason,
        "face_rejection_reason": rejection_reason,
        "faceSubmittedAt": submitted_iso or enrolled_iso,
        "face_submitted_at": submitted_iso or enrolled_iso,
        "algorithm": data.get("algorithm", "SFace-v1"),
        "sampleCount": data.get("sampleCount", 3),
        "enrolledAt": enrolled_iso,
    }


@router.delete("/profile/{uid}")
async def reset_face_profile(
    uid: str,
    request: Request,
    current_user: TokenData = Depends(require_admin)
):
    """
    Admin-only: Purges biometric vector from face_profiles and resets employee verification status.
    """
    db = get_firestore_client()
    if db is not None:
        db.collection("face_profiles").document(uid).delete()
        user_ref = db.collection("users").document(uid)
        u_snap = user_ref.get()
        u_data = (u_snap.to_dict() or {}) if u_snap.exists else {}
        if u_snap.exists:
            user_ref.update({
                "faceRegistered": False,
                "face_registered": False,
                "faceVerificationStatus": FaceVerificationStatus.NOT_ENROLLED.value,
                "face_verification_status": FaceVerificationStatus.NOT_ENROLLED.value,
                "faceRejectionReason": None,
                "face_rejection_reason": None,
                "faceSubmittedAt": None,
                "face_submitted_at": None,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            })

        record_audit_event(
            db,
            action=ACTION_FACE_RESET,
            actor_uid=current_user.uid,
            target_id=uid,
            employee_name=u_data.get("name"),
            niy=u_data.get("niy") or u_data.get("employeeId"),
            department=u_data.get("department"),
            status="success",
            details=f"Profil biometrik wajah {u_data.get('name') or uid} dihapus oleh Admin.",
            request=request,
        )

    return {"success": True, "message": f"Biometrik wajah untuk pengguna {uid} berhasil direset."}


@admin_face_approvals_router.get("", response_model=List[FaceApprovalPendingItem])
async def list_pending_face_approvals(
    admin_user: TokenData = Depends(require_admin),
):
    """
    Admin-only: Retrieves all employees currently in `PENDING_APPROVAL` status,
    enriched with profile details (Nama, NIY, Departemen) and visual face preview thumbnail.
    """
    db = get_firestore_client()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Layanan database Cloud Firestore tidak tersedia.",
        )

    pending_users: dict[str, dict] = {}
    for snap in (
        db.collection("users")
        .where("face_verification_status", "==", FaceVerificationStatus.PENDING_APPROVAL.value)
        .stream()
    ):
        pending_users[snap.id] = snap.to_dict() or {}

    for snap in (
        db.collection("users")
        .where("faceVerificationStatus", "==", FaceVerificationStatus.PENDING_APPROVAL.value)
        .stream()
    ):
        if snap.id not in pending_users:
            pending_users[snap.id] = snap.to_dict() or {}

    items: List[FaceApprovalPendingItem] = []
    min_wib = datetime(1970, 1, 1, tzinfo=WIB)

    for uid, u_data in pending_users.items():
        face_snap = db.collection("face_profiles").document(uid).get()
        f_data = (face_snap.to_dict() or {}) if face_snap.exists else {}

        submitted_dt = _parse_wib_dt(
            u_data.get("face_submitted_at")
            or u_data.get("faceSubmittedAt")
            or f_data.get("submittedAt")
            or f_data.get("enrolledAt")
        )
        submitted_wib_str = (
            submitted_dt.strftime("%d %b %Y, %H:%M WIB") if submitted_dt else "Baru saja"
        )
        resolved_niy = u_data.get("niy") or u_data.get("employeeId") or f"NIY-{uid[:7].upper()}"

        items.append(
            FaceApprovalPendingItem(
                uid=uid,
                name=u_data.get("name") or "Tanpa Nama",
                niy=resolved_niy,
                employeeId=u_data.get("employeeId") or resolved_niy,
                email=u_data.get("email"),
                phoneNumber=u_data.get("phoneNumber") or u_data.get("phone"),
                department=u_data.get("department") or "Umum",
                position=u_data.get("position") or "Staf",
                faceVerificationStatus=FaceVerificationStatus.PENDING_APPROVAL,
                face_verification_status=FaceVerificationStatus.PENDING_APPROVAL,
                faceSubmittedAt=submitted_dt,
                face_submitted_at=submitted_dt,
                faceSubmittedAtWib=submitted_wib_str,
                previewThumbnail=f_data.get("previewThumbnail"),
                sampleCount=int(f_data.get("sampleCount") or 3),
                algorithm=str(f_data.get("algorithm") or "SFace-v1"),
            )
        )

    items.sort(key=lambda item: item.faceSubmittedAt or min_wib, reverse=True)
    return items


@admin_face_approvals_router.post("/{uid}/approve", response_model=FaceApprovalActionResponse)
async def approve_employee_face(
    uid: str,
    request: Request,
    admin_user: TokenData = Depends(require_admin),
):
    """
    Admin-only: Approves an employee's recorded biometric face profile:
    - Sets `face_verification_status = 'APPROVED'` and `face_registered = true`.
    - Records immutable `FACE_APPROVAL_ACCEPTED` audit log with approving Admin's name.
    """
    db = get_firestore_client()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Layanan database Cloud Firestore tidak tersedia.",
        )

    user_ref = db.collection("users").document(uid)
    user_snap = user_ref.get()
    if not user_snap.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Karyawan dengan UID '{uid}' tidak ditemukan.",
        )

    face_ref = db.collection("face_profiles").document(uid)
    face_snap = face_ref.get()
    if not face_snap.exists or not (face_snap.to_dict() or {}).get("encryptedEmbedding"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Template biometrik wajah karyawan belum tersedia di database. Karyawan harus merekam wajah terlebih dahulu.",
        )

    u_data = user_snap.to_dict() or {}
    emp_name = u_data.get("name") or uid
    emp_niy = u_data.get("niy") or u_data.get("employeeId") or "-"
    emp_dept = u_data.get("department") or "-"
    admin_name = _resolve_admin_display_name(db, admin_user)
    approved_wib = now_wib()

    user_ref.update({
        "faceVerificationStatus": FaceVerificationStatus.APPROVED.value,
        "face_verification_status": FaceVerificationStatus.APPROVED.value,
        "faceRegistered": True,
        "face_registered": True,
        "faceRejectionReason": None,
        "face_rejection_reason": None,
        "faceApprovedAt": approved_wib.isoformat(),
        "face_approved_at": approved_wib.isoformat(),
        "faceApprovedBy": admin_name,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })

    face_ref.update({
        "verificationStatus": FaceVerificationStatus.APPROVED.value,
        "approvedAt": firestore.SERVER_TIMESTAMP,
        "approvedByUid": admin_user.uid,
        "approvedByName": admin_name,
        "lastUpdated": firestore.SERVER_TIMESTAMP,
    })

    record_audit_event(
        db,
        action=ACTION_FACE_APPROVAL_ACCEPTED,
        actor_uid=admin_user.uid,
        target_id=uid,
        employee_name=emp_name,
        niy=emp_niy,
        department=emp_dept,
        status="success",
        details=(
            f"Verifikasi wajah karyawan {emp_name} ({emp_niy}) disetujui oleh Admin {admin_name}. "
            "Profil biometrik kini aktif untuk presensi."
        ),
        request=request,
        metadata={
            "approvedByUid": admin_user.uid,
            "approvedByName": admin_name,
            "approvedAtWib": approved_wib.isoformat(),
        },
    )

    return FaceApprovalActionResponse(
        success=True,
        uid=uid,
        faceVerificationStatus=FaceVerificationStatus.APPROVED,
        face_verification_status=FaceVerificationStatus.APPROVED,
        faceRegistered=True,
        face_registered=True,
        faceRejectionReason=None,
        face_rejection_reason=None,
        message=f"Wajah karyawan {emp_name} ({emp_niy}) berhasil disetujui dan telah aktif untuk presensi.",
    )


@admin_face_approvals_router.post("/{uid}/reject", response_model=FaceApprovalActionResponse)
async def reject_employee_face(
    uid: str,
    payload: FaceApprovalRejectRequest,
    request: Request,
    admin_user: TokenData = Depends(require_admin),
):
    """
    Admin-only: Rejects an employee's submitted face verification:
    - Sets `face_verification_status = 'REJECTED'`, `face_registered = false`, and stores `face_rejection_reason`.
    - Purges temporary template & preview in `face_profiles/{uid}` so employee can re-enroll cleanly.
    - Records immutable `FACE_APPROVAL_REJECTED` audit log.
    """
    db = get_firestore_client()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Layanan database Cloud Firestore tidak tersedia.",
        )

    user_ref = db.collection("users").document(uid)
    user_snap = user_ref.get()
    if not user_snap.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Karyawan dengan UID '{uid}' tidak ditemukan.",
        )

    u_data = user_snap.to_dict() or {}
    emp_name = u_data.get("name") or uid
    emp_niy = u_data.get("niy") or u_data.get("employeeId") or "-"
    emp_dept = u_data.get("department") or "-"
    admin_name = _resolve_admin_display_name(db, admin_user)
    rejected_wib = now_wib()

    # 1. Update user document with REJECTED status and reason
    user_ref.update({
        "faceVerificationStatus": FaceVerificationStatus.REJECTED.value,
        "face_verification_status": FaceVerificationStatus.REJECTED.value,
        "faceRegistered": False,
        "face_registered": False,
        "faceRejectionReason": payload.reason,
        "face_rejection_reason": payload.reason,
        "faceRejectedAt": rejected_wib.isoformat(),
        "face_rejected_at": rejected_wib.isoformat(),
        "faceRejectedBy": admin_name,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })

    # 2. Delete temporary biometric template & preview in face_profiles/{uid}
    db.collection("face_profiles").document(uid).delete()

    # 3. Record immutable audit log
    record_audit_event(
        db,
        action=ACTION_FACE_APPROVAL_REJECTED,
        actor_uid=admin_user.uid,
        target_id=uid,
        employee_name=emp_name,
        niy=emp_niy,
        department=emp_dept,
        status="failed",
        details=(
            f"Verifikasi wajah karyawan {emp_name} ({emp_niy}) ditolak oleh Admin {admin_name}. "
            f"Alasan: {payload.reason}"
        ),
        request=request,
        metadata={
            "rejectedByUid": admin_user.uid,
            "rejectedByName": admin_name,
            "reason": payload.reason,
            "rejectedAtWib": rejected_wib.isoformat(),
        },
    )

    return FaceApprovalActionResponse(
        success=True,
        uid=uid,
        faceVerificationStatus=FaceVerificationStatus.REJECTED,
        face_verification_status=FaceVerificationStatus.REJECTED,
        faceRegistered=False,
        face_registered=False,
        faceRejectionReason=payload.reason,
        face_rejection_reason=payload.reason,
        message=f"Pengajuan wajah {emp_name} ({emp_niy}) telah ditolak dan template sementara dihapus.",
    )
