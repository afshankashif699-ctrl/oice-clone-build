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
    # Using the standard model string ensures compatibility with the TTS library
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("✅ Model Loaded Successfully!")

def extreme_audio_prep(input_path, output_path):
    """
    Cleans reference audio: Normalizes, removes silence, and limits length
    to improve cloning stability.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio = effects.normalize(audio)
        # Remove silence longer than 500ms
        nonsilent_chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
        if nonsilent_chunks:
            audio = sum(nonsilent_chunks)
        
        # Limit to 12 seconds for stability (XTTS preference)
        if len(audio) > 12000:
            audio = audio[:12000]
        
        # Format for XTTS
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
        
        # --- INPUTS ---
        text = job_input.get("text", "").strip()
        language = job_input.get("language", "en")
        speaker_wav_b64 = job_input.get("speaker_wav", "")
        
        # --- SETTINGS (Merged from both files) ---
        speed = float(job_input.get("speed", 1.0))
        temperature = float(job_input.get("temperature", 0.75)) 
        repetition_penalty = float(job_input.get("repetition_penalty", 4.0)) # Higher as requested
        top_p = float(job_input.get("top_p", 0.85))
        top_k = int(job_input.get("top_k", 50))

        # --- PATHS ---
        raw_path = "/tmp/raw_ref.wav"
        clean_path = "/tmp/clean_ref.wav"
        output_path = "/tmp/final_output.wav"

        # --- PROCESS REFERENCE AUDIO ---
        if not speaker_wav_b64:
            return {"status": "error", "message": "No speaker_wav provided"}

        if "," in speaker_wav_b64:
            speaker_wav_b64 = speaker_wav_b64.split(",")[1]
        
        with open(raw_path, "wb") as f:
            f.write(base64.b64decode(speaker_wav_b64))
        
        if not extreme_audio_prep(raw_path, clean_path):
            return {"status": "error", "message": "Audio preprocessing failed"}

        # --- SMART STITCHING LOGIC ---
        # Splitting text helps prevent the "hallucination/stuttering" glitch on long runs
        chunks = re.split(r'([.?!:;\n]+)', text)
        sentences = []
        current_sent = ""
        for part in chunks:
            current_sent += part
            if len(current_sent) > 80 or re.search(r'[.?!:;\n]', part):
                if current_sent.strip():
                    sentences.append(current_sent.strip())
                current_sent = ""
        if current_sent.strip():
            sentences.append(current_sent.strip())

        combined_wav = []
        sample_rate = 24000 

        print(f"Generating {len(sentences)} segments...")
        for i, sentence in enumerate(sentences):
            if not sentence: continue
            
            wav_chunk = tts.tts(
                text=sentence,
                speaker_wav=clean_path,
                language=language,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                speed=speed,
                split_sentences=False # We handle splitting manually
            )
            combined_wav.extend(wav_chunk)
            
            # Add a slight natural pause (100ms) between segments
            if i < len(sentences) - 1:
                combined_wav.extend([0.0] * int(sample_rate * 0.1))

        # --- EXPORT & RETURN ---
        sf.write(output_path, np.array(combined_wav), sample_rate)

        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        return {"status": "success", "audio_base64": audio_base64}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})