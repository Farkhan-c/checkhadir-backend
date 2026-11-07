FROM python:3.11-slim

# Install dependensi sistem Linux yang dibutuhkan OpenCV & SFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Salin requirements dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi folder proyek ke /app
COPY . .

# Set environment variable agar Python mengenali direktori root
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

# Jalankan uvicorn dengan path modul yang eksplisit
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
