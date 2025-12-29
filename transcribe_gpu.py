from faster_whisper import WhisperModel
from pathlib import Path
import json
from config import MODEL_NAME, DEVICE, LANGUAGE

print(f"Whisper device = {DEVICE}")

model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "int8",
    num_workers=1
)

def transcribe_chunk(audio_path):
    segments, _ = model.transcribe(
        audio_path,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True
    )

    results = [
        {"start": s.start, "end": s.end, "text": s.text}
        for s in segments
    ]

    return results
