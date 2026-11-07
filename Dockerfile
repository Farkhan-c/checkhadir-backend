FROM python:3.11-slim

# Install library sistem Linux untuk OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode ke /app
COPY . .

# Pastikan file __init__.py ada di dalam folder app agar dikenali sebagai modul
RUN touch /app/app/__init__.py

# Otomatis buat file serviceAccountKey.json dari Environment Variable jika aplikasi mencarinya berupa file fisik
RUN echo '#!/bin/sh\n\
if [ -n "$FIREBASE_SERVICE_ACCOUNT_JSON" ]; then\n\
  echo "$FIREBASE_SERVICE_ACCOUNT_JSON" > /app/serviceAccountKey.json\n\
  echo "$FIREBASE_SERVICE_ACCOUNT_JSON" > /app/app/serviceAccountKey.json\n\
fi\n\
python3 -c "import sys; sys.path.insert(0, '\''/app'\''); import app.main"\n\
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}\n' > /entrypoint.sh && chmod +x /entrypoint.sh

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["/entrypoint.sh"]
