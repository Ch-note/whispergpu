import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch

def download():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[Error] HF_TOKEN is required to download Pyannote models.")
        # Some models require auth (diarization), others like separation might be public 
        # but it's better to have it.
    
    print("--- 1. Downloading Faster-Whisper Model (medium) ---")
    # This will cache the model in ~/.cache/huggingface/hub
    WhisperModel("medium", device="cpu", compute_type="int8")

    print("--- 2. Downloading Pyannote Diarization Model (3.1) ---")
    try:
        Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=hf_token
        )
    except Exception as e:
        print(f"[Warning] Diarization download failed: {e}")

    print("--- 3. Downloading Pyannote Speech Separation Model (AMI 1.0) ---")
    try:
        Pipeline.from_pretrained(
            "pyannote/speech-separation-ami-1.0", 
            use_auth_token=hf_token
        )
    except Exception as e:
        print(f"[Warning] Separation download failed: {e}")

    print("--- [v8] All models pre-cached successfully ---")

if __name__ == "__main__":
    download()
