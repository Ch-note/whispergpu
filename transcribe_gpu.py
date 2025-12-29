from faster_whisper import WhisperModel
from pathlib import Path
import json
from config import MODEL_NAME, DEVICE, LANGUAGE

print(f"🚀 Whisper device = {DEVICE}")

model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "int8",
    num_workers=1
)

def transcribe_chunk(audio_path, out_dir):
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

    out_path = Path(out_dir) / (Path(audio_path).stem + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return out_path
