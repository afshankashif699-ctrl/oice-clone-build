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
        # Normalize and strip silence to focus ONLY on the voice profile
        audio = effects.normalize(audio)
        
        # Accent ke liye 7-10 seconds ka clear audio best hai
        if len(audio) > 10000:
            audio = audio[:10000]
            
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
        
        # --- 100% ACCENT & FLOW SETTINGS ---
        # Temperature 0.90 tak le jane se accent 100% copy hota hai (quirks copy hote hain)
        temperature = float(job_input.get("temperature", 0.65)) 
        
        # Repetition Penalty ko 1.25 rakha hai taake speed aur flow natural rahe
        repetition_penalty = float(job_input.get("repetition_penalty", 1.25))
        
        # Speed: Reference voice ke flow ke liye 1.05 - 1.10 aksar behtar lagta hai
        speed = float(job_input.get("speed", 1.05))
        
        top_k = int(job_input.get("top_k", 50))
        top_p = float(job_input.get("top_p", 0.80))
        
        raw_path = "/tmp/raw_ref.wav"
        clean_path = "/tmp/clean_ref.wav"
        output_path = "/tmp/final_output.wav"

        if "," in speaker_wav_b64:
            speaker_wav_b64 = speaker_wav_b64.split(",")[1]
        
        with open(raw_path, "wb") as f:
            f.write(base64.b64decode(speaker_wav_b64))
        
        if not audio_prep(raw_path, clean_path):
            return {"status": "error", "message": "Audio preprocessing failed"}

        # --- FLOW FIX: CLEANING TEXT ---
        # XTTS breaks tab leta hai jab punctuation zyada ho. 
        # Hum unnecessary full stops aur commas ko manage karte hain.
        clean_text = text.replace("...", ".").replace("\n", " ")

        # --- GENERATION ---
        # split_sentences=False aur manual length_penalty se flow banta hai
        wav = tts.tts(
            text=clean_text,
            speaker_wav=clean_path,
            language=language,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=1.0, # Neutral length maintenance
            speed=speed,
            split_sentences=False # Isay False rakhne se sentences ke beech break khatam ho jayega
        )

        wav_norm = np.array(wav)
        sf.write(output_path, wav_norm, 24000)
        
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        return {"status": "success", "audio_base64": audio_base64}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})