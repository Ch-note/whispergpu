from faster_whisper import WhisperModel, BatchedInferencePipeline
from pathlib import Path
import json
from config import MODEL_NAME, DEVICE, LANGUAGE

print(f"Whisper device = {DEVICE}")

# Whisper 엔진 (Lazy loading을 위해 Singleton 관리)
_transcription_pipeline = None

def get_whisper_pipeline():
    """
    Whisper 모델을 지연 로딩하는 싱글톤 함수.
    최초 호출 시에만 모델을 메모리에 올립니다.
    """
    global _transcription_pipeline
    if _transcription_pipeline is None:
        print(f"[Engine] Loading Whisper Model ({MODEL_NAME}) on {DEVICE}...")
        # 기초 모델 로드
        base_model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type="float16" if DEVICE == "cuda" else "int8",
            num_workers=1
        )
        # 4배 빠른 배청 처리를 위한 Pipeline 선언
        _transcription_pipeline = BatchedInferencePipeline(base_model)
        print("[Engine] Whisper Model loaded successfully.")
    return _transcription_pipeline

def transcribe_chunk(audio_path):
    # BatchedInferencePipeline의 transcribe 호출
    pipeline = get_whisper_pipeline()
    
    segments, _ = pipeline.transcribe(
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
