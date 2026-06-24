# SViam voice interviewer — production image (Render, Docker runtime).
# The interview code is unchanged; this only provides the native libraries the
# stack needs at build + run time (aiortc/PyAV media, libsrtp, opus/vpx codecs,
# portaudio for the imported-but-deprecated pyaudio path, build tools for cffi).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libopus-dev \
        libvpx-dev \
        libsrtp2-dev \
        portaudio19-dev \
        build-essential \
        pkg-config \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (cached layer) — pipecat is pinned to 1.3.0 in requirements.
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code (interview logic + transport). .dockerignore keeps venv/.env/.git out.
COPY . .

# Render injects $PORT; server.py binds int(os.getenv("PORT", 8000)).
EXPOSE 8000
CMD ["python", "server.py"]
