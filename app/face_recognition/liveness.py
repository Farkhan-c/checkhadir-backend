"""
Anti-Spoofing and Liveness Verification Service
Combines active interactive challenge-response with passive texture & color distribution checks.
Resistant to static photo prints, screen replays, and video re-capture.
"""

import time
import uuid
import threading
import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2

from app.core.config import settings
from app.face_recognition.engine import face_engine, FaceQualityError

logger = logging.getLogger(__name__)

SUPPORTED_CHALLENGES = {
    "turn_right": {
        "instruction": "Tolong tolehkan kepala sedikit ke kanan Anda, lalu kembali menatap kamera.",
        "description": "Head yaw rotation to the right"
    },
    "turn_left": {
        "instruction": "Tolong tolehkan kepala sedikit ke kiri Anda, lalu kembali menatap kamera.",
        "description": "Head yaw rotation to the left"
    },
    "smile": {
        "instruction": "Tersenyumlah secara wajar ke arah kamera.",
        "description": "Mouth horizontal widening / smile expansion"
    },
    "blink": {
        "instruction": "Kedipkan kedua mata Anda ke arah kamera.",
        "description": "Eye closure and reopening verification"
    },
    "nod": {
        "instruction": "Anggukkan kepala sedikit ke bawah lalu tatap kamera.",
        "description": "Pitch downward motion"
    },
    "stability_hold": {
        "instruction": "Wajah Terdeteksi, Menahan Posisi...",
        "description": "Hands-free oval face stability hold with passive & temporal anti-spoofing"
    }
}

class LivenessSession:
    def __init__(self, session_id: str, uid: str, challenge_type: str, ttl_seconds: int = 60):
        self.session_id = session_id
        self.uid = uid
        self.challenge_type = challenge_type
        self.issued_at = time.time()
        self.expires_at = self.issued_at + ttl_seconds
        self.used = False
        self.passed = False
        # A passed challenge authorises exactly ONE attendance action within this window.
        self.passed_valid_until = 0.0
        self.consumed = False
        # Identity of the person who performed the challenge (bound to the attendance frame later)
        self.embedding = None

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

class LivenessService:
    # Seconds a passed challenge stays valid for the attendance submission that follows it
    PASSED_VALIDITY_SECONDS = 120
    # Min similarity between the person who did the challenge and the attendance frame
    SESSION_BINDING_THRESHOLD = 0.50

    def __init__(self):
        self._sessions: Dict[str, LivenessSession] = {}
        self._attendance_lock = threading.Lock()

    def _cleanup_expired_sessions(self):
        """Removes expired challenge sessions from memory."""
        now = time.time()
        expired_ids = [
            sid for sid, sess in self._sessions.items()
            if max(sess.expires_at, sess.passed_valid_until) < now
        ]
        for sid in expired_ids:
            self._sessions.pop(sid, None)

    def create_challenge(self, uid: str, preferred_challenge: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a new signed single-use liveness challenge session.
        """
        self._cleanup_expired_sessions()

        if preferred_challenge and preferred_challenge in SUPPORTED_CHALLENGES:
            c_type = preferred_challenge
        else:
            # Pick deterministically or pseudo-randomly among turn_right, turn_left, smile
            c_type = np.random.choice(["turn_right", "turn_left", "smile", "blink"])

        session_id = f"live_{uuid.uuid4().hex[:16]}"
        ttl = settings.LIVENESS_CHALLENGE_TIMEOUT_SECONDS

        session = LivenessSession(session_id=session_id, uid=uid, challenge_type=c_type, ttl_seconds=ttl)
        self._sessions[session_id] = session

        return {
            "sessionId": session_id,
            "challengeType": c_type,
            "instruction": SUPPORTED_CHALLENGES[c_type]["instruction"],
            "expiresInSeconds": ttl
        }

    def check_passive_spoofing(self, image: np.ndarray, face_box: np.ndarray) -> Tuple[bool, str, float]:
        """
        Passive Anti-Spoofing:
        1. Color saturation distribution (reject black-and-white photocopies).
        2. Texture / high-frequency pattern analysis (screen moire / halftone print artifacts).
        """
        x, y, w, h = [int(v) for v in face_box[:4]]
        # Bound coordinates to image dimensions
        img_h, img_w = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        face_roi = image[y1:y2, x1:x2]
        if face_roi.size == 0:
            return False, "Face ROI invalid", 0.0

        # 1. Color saturation check in HSV
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mean_sat = float(np.mean(sat))

        if mean_sat < 15.0:
            # Extremely desaturated face - suspected grayscale photocopy or printout
            return False, "Deteksi spoofing: Gambar tidak memiliki saturasi warna alami (dugaan fotokopi hitam putih).", 0.1

        # 2. High-frequency / Moire detection via 2D Fourier Transform or Laplacian gradient
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

        # An authentic live webcam face typically has a natural Laplacian variance range
        if lap_var > 4500.0:
            # Unnaturally noisy or heavy halftone dot screen texture
            return False, "Deteksi spoofing: Pola tekstur frekuensi tinggi tidak wajar (dugaan cetakan kertas / layar piksel).", 0.3

        confidence = min(1.0, max(0.5, mean_sat / 100.0))
        return True, "Passive checks passed", confidence

    def consume_passed_session(
        self,
        session_id: Optional[str],
        uid: str,
        attendance_embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Atomically spends a liveness session that previously PASSED for this user.
        Returns True once; any replay, wrong user, expired, or failed session returns False.

        When `attendance_embedding` is given it must belong to the same person who performed
        the challenge. This stops "do the challenge live, then submit someone else's photo".
        """
        if not session_id:
            return False
        with self._attendance_lock:
            session = self._sessions.get(session_id)
            if (
                session is None
                or session.uid != uid
                or not session.passed
                or session.consumed
                or time.time() > session.passed_valid_until
            ):
                return False
            if attendance_embedding is not None and session.embedding is not None:
                sim = face_engine.compute_similarity(session.embedding, attendance_embedding)
                if sim < self.SESSION_BINDING_THRESHOLD:
                    logger.warning(f"Liveness binding mismatch for {uid}: sim={sim}")
                    return False
            session.consumed = True
            return True

    def verify_liveness(
        self,
        session_id: str,
        baseline_input: Union[str, bytes, np.ndarray],
        action_input: Union[str, bytes, np.ndarray],
        expected_uid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verifies active challenge compliance between baseline frame and action frame.
        Burn-after-reading: Session is marked used immediately.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise FaceQualityError("SESSION_NOT_FOUND", "Sesi liveness tidak ditemukan atau sudah kadaluarsa.")

        if session.is_expired:
            self._sessions.pop(session_id, None)
            raise FaceQualityError("SESSION_EXPIRED", "Sesi liveness telah kedaluwarsa. Silakan mulai kembali.")

        if session.used:
            raise FaceQualityError("SESSION_ALREADY_USED", "Token liveness sudah pernah digunakan (replay prevented).")

        if expected_uid and session.uid != expected_uid:
            raise FaceQualityError("UID_MISMATCH", "Sesi liveness tidak sesuai dengan identitas pengguna.")

        # Burn session to prevent replay attacks
        session.used = True

        # Decode frames
        img_base = face_engine.decode_image(baseline_input)
        img_action = face_engine.decode_image(action_input)

        # Detect faces in both frames
        faces_base = face_engine.detect_faces(img_base)
        faces_action = face_engine.detect_faces(img_action)

        if faces_base is None or len(faces_base) == 0:
            return {"passed": False, "challengeType": session.challenge_type, "confidence": 0.0, "message": "Wajah tidak terdeteksi pada frame awal.", "passivePassed": False}

        if faces_action is None or len(faces_action) == 0:
            return {"passed": False, "challengeType": session.challenge_type, "confidence": 0.0, "message": "Wajah tidak terdeteksi pada frame aksi.", "passivePassed": False}

        if len(faces_base) > 1 or len(faces_action) > 1:
            return {"passed": False, "challengeType": session.challenge_type, "confidence": 0.0, "message": "Terdeteksi lebih dari satu wajah.", "passivePassed": False}

        face_base = faces_base[0]
        face_action = faces_action[0]

        # Passive anti-spoofing check
        pass_base_ok, pass_base_msg, conf_base = self.check_passive_spoofing(img_base, face_base)
        pass_act_ok, pass_act_msg, conf_act = self.check_passive_spoofing(img_action, face_action)

        if not pass_base_ok or not pass_act_ok:
            reason = pass_base_msg if not pass_base_ok else pass_act_msg
            return {
                "passed": False,
                "challengeType": session.challenge_type,
                "confidence": 0.1,
                "message": reason,
                "passivePassed": False
            }

        # Identity consistency check: both frames must belong to the same person
        emb_base = face_engine.extract_embedding(img_base, enforce_quality=False)
        emb_action = face_engine.extract_embedding(img_action, enforce_quality=False)
        person_sim = face_engine.compute_similarity(emb_base, emb_action)

        if person_sim < 0.50:
            return {
                "passed": False,
                "challengeType": session.challenge_type,
                "confidence": 0.0,
                "message": f"Identitas wajah pada kedua frame tidak cocok (kesamaan: {person_sim:.2f}).",
                "passivePassed": True
            }

        # Active Challenge Geometry Evaluation using YuNet 5 landmarks
        # Landmarks: [x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm]
        c_type = session.challenge_type
        passed = False
        message = ""
        confidence = 0.85

        def extract_landmarks(face_arr):
            # face_arr has landmarks at indices 4..13
            pts = face_arr[4:14].reshape((5, 2))
            return {
                "right_eye": pts[0],
                "left_eye": pts[1],
                "nose": pts[2],
                "right_mouth": pts[3],
                "left_mouth": pts[4]
            }

        lm_base = extract_landmarks(face_base)
        lm_action = extract_landmarks(face_action)

        # Baseline metrics
        d_eyes_base = float(np.linalg.norm(lm_base["left_eye"] - lm_base["right_eye"]))
        d_eyes_action = float(np.linalg.norm(lm_action["left_eye"] - lm_action["right_eye"]))

        if d_eyes_base == 0 or d_eyes_action == 0:
            return {"passed": False, "challengeType": c_type, "confidence": 0.0, "message": "Dimensi mata tidak valid.", "passivePassed": True}

        if c_type in ["turn_right", "turn_left"]:
            # Yaw rotation estimation via relative nose position between eyes
            d_nose_r_base = float(np.linalg.norm(lm_base["nose"] - lm_base["right_eye"]))
            d_nose_l_base = float(np.linalg.norm(lm_base["nose"] - lm_base["left_eye"]))
            yaw_base = (d_nose_r_base - d_nose_l_base) / d_eyes_base

            d_nose_r_act = float(np.linalg.norm(lm_action["nose"] - lm_action["right_eye"]))
            d_nose_l_act = float(np.linalg.norm(lm_action["nose"] - lm_action["left_eye"]))
            yaw_act = (d_nose_r_act - d_nose_l_act) / d_eyes_action

            delta_yaw = yaw_act - yaw_base

            # Head rotation causes asymmetry between nose-eye distances.
            # Handle user mirror / webcam flipping: significant yaw shift indicates genuine head rotation.
            if abs(delta_yaw) >= 0.10:
                passed = True
                message = f"Gerakan toleh kepala ({c_type}) terverifikasi (pergeseran: {abs(delta_yaw):.3f})."
                confidence = min(0.98, 0.70 + abs(delta_yaw))
            else:
                passed = False
                message = f"Gerakan toleh kepala tidak terdeteksi cukup jelas (pergeseran: {abs(delta_yaw):.3f}, min: 0.10)."

        elif c_type == "smile":
            # Mouth horizontal width relative to eye distance
            mouth_w_base = float(np.linalg.norm(lm_base["left_mouth"] - lm_base["right_mouth"])) / d_eyes_base
            mouth_w_act = float(np.linalg.norm(lm_action["left_mouth"] - lm_action["right_mouth"])) / d_eyes_action
            ratio = mouth_w_act / mouth_w_base if mouth_w_base > 0 else 1.0

            if ratio >= 1.06:
                passed = True
                message = f"Ekspresi senyum terverifikasi (pelebaran mulut: {(ratio - 1.0)*100:.1f}%)."
                confidence = min(0.98, 0.75 + (ratio - 1.0))
            else:
                passed = False
                message = f"Ekspresi senyum belum mencukupi (pelebaran: {(ratio - 1.0)*100:.1f}%, min: 6%)."

        elif c_type == "blink":
            # Blinking causes variation in the eye region intensity or difference between open and closed state
            # Crop eye regions from both frames
            def get_eye_patch(img, eye_pt, size=24):
                ex, ey = int(eye_pt[0]), int(eye_pt[1])
                half = size // 2
                h, w = img.shape[:2]
                return img[max(0, ey-half):min(h, ey+half), max(0, ex-half):min(w, ex+half)]

            patch_b_re = get_eye_patch(img_base, lm_base["right_eye"])
            patch_a_re = get_eye_patch(img_action, lm_action["right_eye"])

            if patch_b_re.size > 0 and patch_a_re.size > 0:
                patch_b_re = cv2.resize(patch_b_re, (24, 24))
                patch_a_re = cv2.resize(patch_a_re, (24, 24))
                diff = float(np.mean(cv2.absdiff(patch_b_re, patch_a_re)))
                if diff >= 10.0:
                    passed = True
                    message = "Kedipan mata terverifikasi."
                    confidence = 0.90
                else:
                    passed = False
                    message = "Kedipan mata tidak terdeteksi jelas."
            else:
                passed = True
                message = "Uji kedipan mata selesai."

        elif c_type == "nod":
            # Vertical pitch motion: distance from eye midpoint to nose
            mid_eye_base = (lm_base["left_eye"] + lm_base["right_eye"]) / 2.0
            mid_eye_act = (lm_action["left_eye"] + lm_action["right_eye"]) / 2.0
            pitch_base = float(np.linalg.norm(mid_eye_base - lm_base["nose"])) / d_eyes_base
            pitch_act = float(np.linalg.norm(mid_eye_act - lm_action["nose"])) / d_eyes_action
            diff_pitch = abs(pitch_act - pitch_base)

            if diff_pitch >= 0.05:
                passed = True
                message = "Gerakan anggukan kepala terverifikasi."
                confidence = 0.88
            else:
                passed = False
                message = "Gerakan anggukan kepala tidak terdeteksi cukup jelas."

        elif c_type == "stability_hold":
            # Hands-free stability hold:
            # 1. Reject exact byte/pixel replay between t=0 and t=1.5s frames
            if isinstance(baseline_input, str) and isinstance(action_input, str) and baseline_input == action_input:
                return {
                    "passed": False,
                    "challengeType": c_type,
                    "confidence": 0.0,
                    "message": "Deteksi spoofing: Frame identik statis terdeteksi.",
                    "passivePassed": False,
                }

            h_b, w_b = img_base.shape[:2]
            h_a, w_a = img_action.shape[:2]
            bx, by, bw, bh = [int(v) for v in face_base[:4]]
            ax, ay, aw, ah = [int(v) for v in face_action[:4]]
            roi_b = img_base[max(0, by):min(h_b, by + bh), max(0, bx):min(w_b, bx + bw)]
            roi_a = img_action[max(0, ay):min(h_a, ay + ah), max(0, ax):min(w_a, ax + aw)]

            temporal_diff = 1.0
            if roi_b.size > 0 and roi_a.size > 0:
                roi_b_norm = cv2.resize(roi_b, (64, 64))
                roi_a_norm = cv2.resize(roi_a, (64, 64))
                temporal_diff = float(np.mean(cv2.absdiff(roi_b_norm, roi_a_norm)))

            # 2. Verify face remained stably positioned inside frame without erratic jump
            nose_shift = float(np.linalg.norm(lm_action["nose"] - lm_base["nose"])) / d_eyes_base
            scale_ratio = d_eyes_action / d_eyes_base

            if temporal_diff < 0.05:
                passed = False
                message = "Deteksi spoofing: Tidak ada variasi sensor kamera live."
            elif nose_shift <= 0.60 and 0.65 <= scale_ratio <= 1.45:
                passed = True
                message = "Kestabilan wajah & anti-spoofing pasif terverifikasi."
                confidence = min(0.96, max(0.82, 0.95 - (nose_shift * 0.2)))
            else:
                passed = False
                message = "Posisi wajah bergeser terlalu jauh saat menahan posisi. Tahan wajah tetap di dalam oval."

        session.passed = passed
        if passed:
            session.passed_valid_until = time.time() + self.PASSED_VALIDITY_SECONDS
            session.embedding = emb_action
        return {
            "passed": passed,
            "challengeType": c_type,
            "confidence": round(confidence, 3),
            "message": message,
            "passivePassed": True
        }

# Singleton instance
liveness_service = LivenessService()
