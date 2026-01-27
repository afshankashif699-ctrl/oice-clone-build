import runpod
import torch
from TTS.api import TTS
import base64
import os
import re
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence

# --- GLOBAL MODEL LOADING ---
tts = None

def init_model():
    global tts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS v2 on {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("✅ Model Loaded Successfully!")

# --- NEW: AUTO PACING LOGIC ---
def calculate_pacing(audio_segment):
    """
    Analyzes the audio density to guess the speaker's speed.
    Returns a float multiplier for the speed parameter.
    """
    try:
        duration_sec = len(audio_segment) / 1000.0
        if duration_sec < 1.0:
            return 1.1  # Too short, safe default
        
        # Hum count karein gay k is audio me kitnay "bolnay walay chunks" hain
        # Choti silence settings taake words detect hon
        word_chunks = split_on_silence(
            audio_segment, 
            min_silence_len=100,    # 100ms gap means new word/syllable
            silence_thresh=-35      # Sensitivity
        )
        
        num_events = len(word_chunks)
        density = num_events / duration_sec  # Chunks per second
        
        print(f"DEBUG: Audio Density = {density:.2f} chunks/sec")

        # Logic: Zyada chunks/sec = Fast Speaker
        if density > 3.0:
            return 1.35  # Very Fast
        elif density > 2.2:
            return 1.25  # Fast
        elif density > 1.5:
            return 1.15  # Normal-Fast
        else:
            return 1.05  # Slow/Relaxed (XTTS default is usually slow, so we boost slightly)

    except Exception as e:
        print(f"Pacing Calc Error: {e}")
        return 1.1 # Fallback

def extreme_audio_prep(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = effects.normalize(audio)
        
        # --- CALCULATE PACING BEFORE CROPPING ---
        # Hum pehlay speed calculate karein gay puri audio se
        dynamic_speed = calculate_pacing(audio)
        
        nonsilent_chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
        if nonsilent_chunks:
            audio = sum(nonsilent_chunks)
        
        # Max 9 seconds to keep accent stable
        if len(audio) > 9000:
            audio = audio[:9000]
            
        audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)
        audio.export(output_path, format="wav")
        
        return True, dynamic_speed
    except Exception as e:
        print(f"Audio Prep Error: {e}")
        return False, 1.1

def handler(job):
    global tts
    if tts is None:
        init_model()

    try:
        job_input = job["input"]
        text = job_input.get("text", "").strip()
        language = job_input.get("language", "en")
        speaker_wav_b64 = job_input.get("speaker_wav", "")
        
        # --- SETTINGS ---
        temperature = float(job_input.get("temperature", 0.70)) 
        repetition_penalty = float(job_input.get("repetition_penalty", 1.2)) 
        top_p = float(job_input.get("top_p", 0.85))
        
        # User ki speed check karo, agar nahi di to None rakho taake hum auto calculate karein
        user_speed = job_input.get("speed")
        
        raw_path = "/tmp/raw_ref.wav"
        clean_path = "/tmp/clean_ref.wav"
        output_path = "/tmp/final_output.wav"

        if "," in speaker_wav_b64:
            speaker_wav_b64 = speaker_wav_b64.split(",")[1]
        
        with open(raw_path, "wb") as f:
            f.write(base64.b64decode(speaker_wav_b64))
        
        # --- PREP & GET SPEED ---
        success, calculated_speed = extreme_audio_prep(raw_path, clean_path)
        
        if not success:
            return {"status": "error", "message": "Audio preprocessing failed"}

        # Logic: Agar user ne speed di hai to wo use karo, warna auto-calculated use karo
        final_speed = float(user_speed) if user_speed else calculated_speed
        print(f"ℹ️ Final Applied Speed: {final_speed}")

        # --- TEXT SPLITTING LOGIC ---
        sentences = re.split(r'([.?!:;\n]+)', text)
        processed_sentences = []
        current = ""
        for s in sentences:
            current += s
            if len(current) > 80 or any(p in s for p in ".?!:;\n"):
                if current.strip():
                    processed_sentences.append(current.strip())
                current = ""
        if current.strip():
            processed_sentences.append(current.strip())

        combined_wav = []
        sample_rate = 24000 

        for sentence in processed_sentences:
            wav_chunk = tts.tts(
                text=sentence,
                speaker_wav=clean_path,
                language=language,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                speed=final_speed,  # Using dynamic speed here
                split_sentences=False 
            )
            combined_wav.extend(wav_chunk)

        sf.write(output_path, np.array(combined_wav), sample_rate)
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        return {"status": "success", "audio_base64": audio_base64, "applied_speed": final_speed}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})