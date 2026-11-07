import sys
from pathlib import Path

# Daftarkan root direktori dan subdirektori app ke path sistem
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "app"))

from app.main import app

__all__ = ["app"]
