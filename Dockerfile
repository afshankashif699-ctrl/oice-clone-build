FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN pip3 install --no-cache-dir --upgrade pip

# Install Torch individually (The biggest part)
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TTS with NO-CACHE to save RAM during build
RUN pip3 install --no-cache-dir TTS

# Install the rest
RUN pip3 install --no-cache-dir runpod pydub numpy soundfile scipy

COPY handler.py .

# Pre-download weights
RUN python3 -c 'from TTS.api import TTS; TTS("tts_models/multilingual/multi-dataset/xtts_v2")'

CMD ["python3", "-u", "handler.py"]