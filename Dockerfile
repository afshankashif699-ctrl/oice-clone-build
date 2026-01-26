FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install Torch (Large layer)
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# --- CRITICAL FIX START ---
# Downgrade transformers to fix the 'BeamSearchScorer' ImportError
RUN pip3 install --no-cache-dir transformers==4.33.0
# --- CRITICAL FIX END ---

# Install TTS
RUN pip3 install --no-cache-dir TTS

# Install remaining requirements
RUN pip3 install --no-cache-dir runpod pydub numpy soundfile scipy

COPY handler.py .

# Pre-download weights (This will now succeed)
RUN python3 -c 'from TTS.api import TTS; TTS("tts_models/multilingual/multi-dataset/xtts_v2")'

CMD ["python3", "-u", "handler.py"]