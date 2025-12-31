from faster_whisper import WhisperModel, BatchedInferencePipeline
from pathlib import Path
import json
from config import MODEL_NAME, DEVICE, LANGUAGE

print(f"Whisper device = {DEVICE}")

# 기초 모델 로드
_base_model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "int8",
    num_workers=1
)

# 4배 빠른 배청 처리를 위한 Pipeline 선언
model = BatchedInferencePipeline(_base_model)

def transcribe_chunk(audio_path):
    # BatchedInferencePipeline의 transcribe 호출
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
