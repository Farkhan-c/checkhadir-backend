FROM python:3.11-slim

# Dependensi sistem Linux untuk OpenCV headless & image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja utama
WORKDIR /app

# Salin requirements dan pasang library Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh repositori ke /app
COPY . /app/

# Pastikan Python dan Uvicorn mencari modul di /app
ENV PYTHONPATH="/app"
ENV PORT=8000

EXPOSE 8000

# Eksekusi via modul main yang baru dibuat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
