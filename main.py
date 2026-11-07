import os
import sys

# Dapatkan direktori saat ini
current_dir = os.path.dirname(os.path.abspath(__file__))

# Masukkan folder app ke sys.path
app_dir = os.path.join(current_dir, "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Import modul secara langsung dari dalam folder app
import main as app_module

app = getattr(app_module, "app")

__all__ = ["app"]
