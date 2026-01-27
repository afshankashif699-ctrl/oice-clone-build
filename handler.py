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

# --- PACING LOGIC ---
def calculate_pacing(audio_segment):
    try:
        duration_sec = len(audio_segment) / 1000.0
        if duration_sec < 1.0: return 1.1
        
        word_chunks = split_on_silence(audio_segment, min_silence_len=100, silence_thresh=-35)
        density = len(word_chunks) / duration_sec
        
        # Thora aggressive logic taake slow na ho
        if density > 3.0: return 1.35
        elif density > 2.2: return 1.25
        elif density > 1.5: return 1.15
        else: return 1.10 
    except:
        return 1.1

def extreme_audio_prep(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = effects.normalize(audio)
        dynamic_speed = calculate_pacing(audio)
        
        nonsilent_chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
        if nonsilent_chunks:
            audio = sum(nonsilent_chunks)
        
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
        
        # --- TUNED SETTINGS FOR VOICE MATCH ---
        # Temperature 0.75 accent k liye, lekin top_k glitch rokny k liye
        temperature = float(job_input.get("temperature", 0.75))
        # Penalty ko 1.1 kar diya taake awaaz ki "bhaari-pan" (timbre) na katay
        repetition_penalty = float(job_input.get("repetition_penalty", 1.1))
        top_p = float(job_input.get("top_p", 0.85))
        top_k = int(job_input.get("top_k", 50)) # Added safety layer
        
        user_speed = job_input.get("speed")
        
        raw_path = "/tmp/raw_ref.wav"
        clean_path = "/tmp/clean_ref.wav"
        output_path = "/tmp/final_output.wav"

        if "," in speaker_wav_b64:
            speaker_wav_b64 = speaker_wav_b64.split(",")[1]
        
        with open(raw_path, "wb") as f:
            f.write(base64.b64decode(speaker_wav_b64))
        
        success, calculated_speed = extreme_audio_prep(raw_path, clean_path)
        if not success:
            return {"status": "error", "message": "Audio preprocessing failed"}

        final_speed = float(user_speed) if user_speed else calculated_speed

        # --- SMART TEXT MERGING LOGIC (FIX FOR GLITCH) ---
        # 1. Pehlay basic splitting karo
        raw_sentences = re.split(r'([.?!:;\n]+)', text)
        
        # 2. Short sentences ko merge karo
        processed_sentences = []
        buffer_text = ""
        
        for part in raw_sentences:
            buffer_text += part
            # Agar buffer mein punctuation hai, check karo k length kitni hai
            if any(p in part for p in ".?!:;\n"):
                # Agar sentence 20 chars se chota hai (jaise "No."), to usay mat toro,
                # balkay aglay sentence k sath jor do.
                if len(buffer_text.strip()) > 20: 
                    processed_sentences.append(buffer_text.strip())
                    buffer_text = ""
                else:
                    # Chota sentence hai, space add kar k buffer mein rakho
                    buffer_text += " "
        
        # Agar end mein kuch bacha hai to add kar do
        if buffer_text.strip():
            processed_sentences.append(buffer_text.strip())

        combined_wav = []
        sample_rate = 24000 

        for sentence in processed_sentences:
            # Skip empty chunks
            if len(sentence) < 2: continue
            
            wav_chunk = tts.tts(
                text=sentence,
                speaker_wav=clean_path,
                language=language,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k, # Using top_k
                repetition_penalty=repetition_penalty,
                speed=final_speed,
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