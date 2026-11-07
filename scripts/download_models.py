"""
Download official OpenCV Zoo ONNX models for face detection (YuNet) and face recognition (SFace).
License: Apache 2.0 (permissible for commercial use).

Files are verified against pinned SHA-256 checksums so a tampered or truncated download is rejected.
Run from anywhere:  python backend/scripts/download_models.py
"""
import hashlib
import os
import ssl
import sys
import urllib.request

MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "app", "face_recognition", "models")
)

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    },
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_models() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    ctx = ssl.create_default_context()
    failures = 0
    for filename, meta in MODELS.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path) and sha256_of(path) == meta["sha256"]:
            print(f"OK (verified): {filename}")
            continue
        print(f"Downloading {filename} ...")
        try:
            req = urllib.request.Request(meta["url"], headers={"User-Agent": "CheckHadir-ModelFetcher/1.0"})
            with urllib.request.urlopen(req, context=ctx) as resp, open(path, "wb") as out:
                out.write(resp.read())
            actual = sha256_of(path)
            if actual != meta["sha256"]:
                os.remove(path)
                print(f"CHECKSUM MISMATCH for {filename}: got {actual}")
                failures += 1
            else:
                print(f"Downloaded and verified: {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            failures += 1
    return failures


if __name__ == "__main__":
    sys.exit(1 if download_models() else 0)
