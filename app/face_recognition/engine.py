"""
Face Recognition Engine using OpenCV Zoo YuNet and SFace models.
Licensed under Apache 2.0 (commercially compliant).
Performs face detection, landmark extraction, quality checks, alignment, and 128-d feature extraction.
"""

import os
import io
import re
import base64
import logging
from typing import List, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image

from app.core.config import settings
from app.models.face import FaceQualityReport

logger = logging.getLogger(__name__)

# Image safety limits (applied to every client-supplied frame)
MAX_IMAGE_PIXELS = 25_000_000   # ~5000x5000; larger payloads are rejected before decoding
MAX_IMAGE_LONG_SIDE = 1920

class FaceQualityError(Exception):
    def __init__(self, code: str, message: str, report: Optional[FaceQualityReport] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.report = report

class FaceRecognitionEngine:
    """
    Core computer vision service for CheckHadir.Id.
    Integrates YuNet (face detector) and SFace (face recognizer) from OpenCV Zoo.
    """
    BLUR_THRESHOLD: float = 70.0
    MIN_BRIGHTNESS: float = 40.0
    MAX_BRIGHTNESS: float = 220.0
    MIN_FACE_SIZE: int = 60
    PAIRWISE_CONSISTENCY_THRESHOLD: float = 0.60

    def __init__(self):
        self._models_dir = os.path.join(os.path.dirname(__file__), "models")
        self._yunet_path = os.path.join(self._models_dir, "face_detection_yunet_2023mar.onnx")
        self._sface_path = os.path.join(self._models_dir, "face_recognition_sface_2021dec.onnx")

        self.detector = None
        self.recognizer = None
        self._initialize_models()

    def _initialize_models(self):
        """Initializes OpenCV YuNet and SFace neural networks."""
        if not os.path.exists(self._yunet_path) or not os.path.exists(self._sface_path):
            logger.warning(
                f"Face recognition models missing in {self._models_dir}. "
                "Attempting automated download of OpenCV Zoo models..."
            )
            self._download_models_fallback()

        try:
            # YuNet default input size (320, 320), score_thresh=0.6, nms_thresh=0.3
            self.detector = cv2.FaceDetectorYN.create(
                model=self._yunet_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=self._sface_path,
                config=""
            )
            logger.info("Face detector (YuNet) and recognizer (SFace) loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load OpenCV face recognition models: {e}")
            raise RuntimeError(f"Biometric engine initialization failed: {e}")

    def _download_models_fallback(self):
        """Downloads required ONNX weights if not present."""
        import urllib.request
        import ssl
        os.makedirs(self._models_dir, exist_ok=True)
        ctx = ssl.create_default_context()
        urls = {
            self._yunet_path: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            self._sface_path: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        }
        for dest, url in urls.items():
            if not os.path.exists(dest) or os.path.getsize(dest) < 100000:
                logger.info(f"Downloading {os.path.basename(dest)} from OpenCV Zoo...")
                req = urllib.request.Request(url, headers={"User-Agent": "CheckHadir-ModelFetcher/1.0"})
                with urllib.request.urlopen(req, context=ctx) as resp, open(dest, "wb") as f:
                    f.write(resp.read())
                logger.info(f"Saved {os.path.basename(dest)}")

    @staticmethod
    def decode_image(image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """
        Decodes base64 string, byte stream, or validates an existing numpy image.
        Returns a 3-channel BGR numpy array.
        """
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            return image_input

        if isinstance(image_input, str):
            # Strip data URL prefix if present (e.g. data:image/jpeg;base64,...)
            if "," in image_input:
                image_input = image_input.split(",", 1)[1]
            raw_bytes = base64.b64decode(image_input.strip())
        elif isinstance(image_input, (bytes, bytearray)):
            raw_bytes = bytes(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Guard against decompression bombs: read only the header first and cap total pixels
        try:
            with Image.open(io.BytesIO(raw_bytes)) as probe:
                w0, h0 = probe.size
        except Exception:
            raise ValueError("Format gambar tidak dikenali.")
        if w0 * h0 > MAX_IMAGE_PIXELS:
            raise ValueError("Resolusi gambar terlalu besar.")

        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback decoding with PIL for tricky formats (e.g. WEBP)
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if img is None or img.size == 0:
            raise ValueError("Failed to decode image from payload.")

        # Webcam frames never need more than ~FullHD for face matching; bounding the size keeps latency predictable
        long_side = max(img.shape[:2])
        if long_side > MAX_IMAGE_LONG_SIDE:
            scale = MAX_IMAGE_LONG_SIDE / long_side
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    def check_image_quality(self, image: np.ndarray) -> FaceQualityReport:
        """
        Evaluates blur (Laplacian variance), illumination (brightness), and image dimensions.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Blur check via Laplacian variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = blur_score < self.BLUR_THRESHOLD

        # 2. Brightness check
        brightness_score = float(np.mean(gray))
        is_too_dark = brightness_score < self.MIN_BRIGHTNESS
        is_too_bright = brightness_score > self.MAX_BRIGHTNESS

        # 3. Detect faces count
        faces = self.detect_faces(image)
        face_count = len(faces) if faces is not None else 0

        # Construct status summary
        messages = []
        if is_blurry:
            messages.append("Gambar terlalu buram, mohon jaga kestabilan kamera.")
        if is_too_dark:
            messages.append("Pencahayaan terlalu gelap, cari area yang lebih terang.")
        if is_too_bright:
            messages.append("Pencahayaan terlalu terang / silau.")
        if face_count == 0:
            messages.append("Tidak ada wajah terdeteksi.")
        elif face_count > 1:
            messages.append(f"Terdeteksi {face_count} wajah. Pastikan hanya ada 1 orang di depan kamera.")

        is_passed = (
            not is_blurry
            and not is_too_dark
            and not is_too_bright
            and face_count == 1
        )

        msg = "Kualitas gambar memenuhi standar verifikasi." if is_passed else "; ".join(messages)

        return FaceQualityReport(
            isQualityPassed=is_passed,
            blurScore=round(blur_score, 2),
            isBlurry=is_blurry,
            brightnessScore=round(brightness_score, 2),
            isTooDark=is_too_dark,
            isTooBright=is_too_bright,
            faceCount=face_count,
            message=msg
        )

    def detect_faces(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Executes YuNet face detector on image.
        Returns numpy array of detected faces or None.
        Each face row: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        """
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return None

        # Filter out tiny faces (artifacts/background noise)
        valid_faces = [f for f in faces if f[2] >= self.MIN_FACE_SIZE and f[3] >= self.MIN_FACE_SIZE]
        if not valid_faces:
            return None
        return np.array(valid_faces)

    def extract_embedding(self, image_input: Union[str, bytes, np.ndarray], enforce_quality: bool = True) -> np.ndarray:
        """
        Extracts 128-dimensional L2-normalized biometric embedding vector from image.
        Enforces quality standards and strictly ensures exactly 1 face is present.
        """
        img = self.decode_image(image_input)
        report = self.check_image_quality(img)

        if enforce_quality:
            if report.faceCount == 0:
                raise FaceQualityError("NO_FACE_DETECTED", "Wajah tidak terdeteksi dalam frame.", report)
            if report.faceCount > 1:
                raise FaceQualityError("MULTIPLE_FACES_DETECTED", f"Terdeteksi {report.faceCount} wajah. Pastikan Anda sendirian.", report)
            if report.isBlurry:
                raise FaceQualityError("IMAGE_BLURRY", f"Gambar buram (skor: {report.blurScore}). Mohon stabilkan kamera.", report)
            if report.isTooDark:
                raise FaceQualityError("IMAGE_TOO_DARK", f"Pencahayaan kurang (skor: {report.brightnessScore}).", report)
            if report.isTooBright:
                raise FaceQualityError("IMAGE_TOO_BRIGHT", f"Pencahayaan terlalu silau (skor: {report.brightnessScore}).", report)

        faces = self.detect_faces(img)
        if faces is None or len(faces) == 0:
            raise FaceQualityError("NO_FACE_DETECTED", "Wajah tidak terdeteksi dalam frame.", report)

        # Select primary face (highest detection confidence score)
        primary_face = faces[0]

        # Align and crop face using 5 facial landmarks
        aligned_face = self.recognizer.alignCrop(img, primary_face)

        # Extract 128-d feature
        raw_feat = self.recognizer.feature(aligned_face)

        # L2 Normalize feature vector
        norm = np.linalg.norm(raw_feat)
        if norm > 0:
            normalized_feat = raw_feat / norm
        else:
            normalized_feat = raw_feat

        return normalized_feat.flatten().astype(np.float32)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Computes cosine similarity between two face embeddings.
        Returns a float between -1.0 and 1.0 (1.0 indicates identical faces).
        """
        e1 = np.asarray(embedding1, dtype=np.float32).flatten()
        e2 = np.asarray(embedding2, dtype=np.float32).flatten()

        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = float(np.dot(e1, e2) / (norm1 * norm2))
        return round(similarity, 4)

    def verify_match(
        self,
        live_embedding: np.ndarray,
        enrolled_embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Compares live embedding against enrolled biometric template.
        Returns (is_match: bool, similarity: float, confidence_tier: str).
        """
        applied_threshold = threshold if threshold is not None else settings.FACE_SIMILARITY_THRESHOLD
        sim = self.compute_similarity(live_embedding, enrolled_embedding)
        is_match = sim >= applied_threshold

        if sim >= 0.75:
            confidence = "high"
        elif sim >= applied_threshold:
            confidence = "medium"
        elif sim >= applied_threshold - 0.10:
            confidence = "low"
        else:
            confidence = "rejected"

        return is_match, sim, confidence

    def process_enrollment_samples(self, samples: List[Union[str, bytes, np.ndarray]]) -> Tuple[np.ndarray, List[FaceQualityReport]]:
        """
        Processes 3-5 enrollment face samples:
        1. Validates each sample for quality (single face, blur, illumination).
        2. Extracts 128-d embeddings for each.
        3. Verifies pairwise consistency across all samples (guarantees same person).
        4. Computes normalized average embedding vector.
        """
        if len(samples) < 3:
            raise ValueError(f"Minimal 3 sampel wajah diperlukan untuk enrollment (diberikan: {len(samples)}).")

        extracted_embeddings = []
        quality_reports = []

        for idx, sample in enumerate(samples):
            img = self.decode_image(sample)
            report = self.check_image_quality(img)
            quality_reports.append(report)

            if not report.isQualityPassed:
                raise FaceQualityError(
                    "SAMPLE_QUALITY_FAILED",
                    f"Sampel #{idx + 1} tidak lolos uji kualitas: {report.message}",
                    report
                )

            embedding = self.extract_embedding(img, enforce_quality=False)
            extracted_embeddings.append(embedding)

        # Check pairwise consistency between all sample pairs
        n = len(extracted_embeddings)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_similarity(extracted_embeddings[i], extracted_embeddings[j])
                if sim < self.PAIRWISE_CONSISTENCY_THRESHOLD:
                    raise FaceQualityError(
                        "INCONSISTENT_SAMPLES",
                        f"Sampel #{i+1} dan #{j+1} memiliki tingkat kesamaan rendah ({sim:.3f}). Pastikan semua foto adalah orang yang sama.",
                    )

        # Compute average vector and L2 normalize
        mean_embedding = np.mean(extracted_embeddings, axis=0)
        norm = np.linalg.norm(mean_embedding)
        final_embedding = (mean_embedding / norm).astype(np.float32)

        return final_embedding, quality_reports

    def create_preview_thumbnail(
        self,
        sample: Union[str, bytes, np.ndarray],
        max_side: int = 360,
        jpeg_quality: int = 82,
    ) -> str:
        """
        Generates a compact Base64 JPEG preview thumbnail (`data:image/jpeg;base64,...`)
        from an enrollment sample for Administrator visual verification in `face_profiles/{uid}`.
        """
        img = self.decode_image(sample)
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_side and longest > 0:
            scale = max_side / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise ValueError("Gagal membuat thumbnail preview wajah.")
        b64_str = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64_str}"

# Singleton instance
face_engine = FaceRecognitionEngine()
