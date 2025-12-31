import os
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline, Model
from huggingface_hub import login

def download():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set. Some models might not download if they are private/gated.")
    else:
        login(token=hf_token)

    # 1. Download Whisper 'medium' model
    print("Downloading Whisper 'medium' model...")
    WhisperModel("medium", device="cpu", compute_type="int8") # Download to cache

    # 2. Download Pyannote models
    print("Downloading Pyannote speaker-diarization-3.1...")
    Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)


    print("Downloading Pyannote embedding model...")
    Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)


    print("Pre-download complete.")

if __name__ == "__main__":
    download()
