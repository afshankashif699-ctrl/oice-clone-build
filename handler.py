# ... (imports and init_model remain the same) ...

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
        
        # --- OPTIMIZED CLONING SETTINGS ---
        # 1. Temperature: Lowered to 0.65.
        # Too high (0.8+) causes voice drift; too low (0.5-) sounds robotic.
        # 0.65 is the "Sweet Spot" for accent retention.
        temperature = float(job_input.get("temperature", 0.65)) 

        # 2. Repetition Penalty: Lowered to 1.8.
        # Your previous 4.0 was stripping the life and accent out of the voice.
        # 1.8 prevents loops but allows natural phonetic patterns.
        repetition_penalty = float(job_input.get("repetition_penalty", 1.8)) 

        # 3. Top P: Set to 0.8.
        # This forces the model to choose the most likely "accented" phonemes.
        top_p = float(job_input.get("top_p", 0.80))
        
        speed = float(job_input.get("speed", 1.0))
        top_k = int(job_input.get("top_k", 50))

        # ... (rest of paths and audio prep logic remains the same) ...

        # --- CORE TTS GENERATION ---
        wav = tts.tts(
            text=sentence,
            speaker_wav=clean_path,
            language=language,
            temperature=temperature,        # Using new 0.65
            top_p=top_p,                    # Using new 0.80
            top_k=top_k,
            repetition_penalty=repetition_penalty, # Using new 1.8
            speed=speed,
            split_sentences=False 
        )
        
# ... (rest of audio export remains the same) ...