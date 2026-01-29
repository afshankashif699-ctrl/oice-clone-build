import runpod
import torch
from TTS.api import TTS
import base64
import os
import re
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects

# --- GLOBAL MODEL LOADING ---
tts = None

def init_model():
    global tts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS v2 on {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("✅ Model Loaded Successfully!")

def audio_prep(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        
        # Normalize and remove silence to help the model focus only on voice
        audio = effects.normalize(audio)
        
        # XTTS Accent Secret: Keep the reference between 6 to 10 seconds.
        # Too short = no accent. Too long = model gets confused.
        if len(audio) > 12000:
            audio = audio[:12000]
            
        audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Audio Prep Error: {e}")
        return False

def handler(job):
    global tts
    if tts is None:
        init_model()

    try:
        job_input = job["input"]
        text = job_input.get("text", "").strip()
        language = job_input.get("language", "en")
        speaker_wav_b64 = job_input.get("speaker_wav", "")
        
        # --- ACCENT OPTIMIZED SETTINGS ---
        
        # 1. Temperature: High (0.85) gives the model freedom to mimic the accent quirks.
        temperature = float(job_input.get("temperature", 0.85))
        
        # 2. Repetition Penalty: Low (1.45) is CRITICAL. 
        # High penalty (like 2.0 or 4.0) strips the accent and makes it robotic.
        repetition_penalty = float(job_input.get("repetition_penalty", 1.45))
        
        # 3. Top_P: Slightly lower (0.80) to keep the voice stable while accented.
        top_p = float(job_input.get("top_p", 0.80))
        
        speed = float(job_input.get("speed", 1.0))
        
        raw_path = "/tmp/raw_ref.wav"
        clean_path = "/tmp/clean_ref.wav"
        output_path = "/tmp/final_output.wav"

        if "," in speaker_wav_b64:
            speaker_wav_b64 = speaker_wav_b64.split(",")[1]
        
        with open(raw_path, "wb") as f:
            f.write(base64.b64decode(speaker_wav_b64))
        
        if not audio_prep(raw_path, clean_path):
            return {"status": "error", "message": "Audio preprocessing failed"}

        # Generate using the optimized accent parameters
        wav = tts.tts(
            text=text,
            speaker_wav=clean_path,
            language=language,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            speed=speed,
            split_sentences=True 
        )

        wav_norm = np.array(wav)
        sf.write(output_path, wav_norm, 24000)
        
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        return {"status": "success", "audio_base64": audio_base64}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})