from pathlib import Path
import torch
import numpy as np
import os
from pyannote.audio import Pipeline, Model, Inference, Audio
from pyannote.core import Segment
from huggingface_hub import login
from config import DEVICE

class Diarizer:
    """
    Hybrid Diarization + Overlap Awareness
    pyannote.audio 3.3.1 compatible
    """

    def __init__(self, hf_token: str):
        if DEVICE != "cuda":
            raise RuntimeError(f"Invalid DEVICE={DEVICE}. This pipeline requires CUDA.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available.")

        self.device = torch.device("cuda")

        # HF Login (Version-safe)
        login(token=hf_token)

        # Diarization Pipeline (3.3.1 uses 'token' instead of 'use_auth_token')
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.3",
            token=hf_token
        )
        
        if self.pipeline is None:
            raise RuntimeError("Failed to load the diarization pipeline.")
        
        self.pipeline.to(self.device)

        # Embedding model (Used for speaker identity)
        embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            token=hf_token
        )

        self.embedding_inference = Inference(
            embedding_model,
            window="whole",
        ).to(self.device)

        self.audio = Audio(sample_rate=16000, mono=True)

    def diarize(self, audio_path: str):
        """
        Returns diarization results with overlap awareness.
        """
        audio_path = str(Path(audio_path).resolve())
        waveform, sample_rate = self.audio(audio_path)
        audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

        diarization = self.pipeline(audio_dict)
        results = []

        # pyannote.audio 3.x tracks can overlap
        # We group them to detect multi-speaker segments
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.duration < 0.5:
                continue

            try:
                embedding = self.embedding_inference.crop(audio_dict, turn)
                if hasattr(embedding, "detach"):
                    embedding = embedding.detach().cpu().numpy()

                results.append({
                    "start": round(float(turn.start), 2),
                    "end": round(float(turn.end), 2),
                    "speaker": speaker,
                    "embedding": embedding,
                })
            except Exception as e:
                print(f"[WARN] Embedding error at {turn.start:.2f}s: {e}")
                continue

        return results

    def get_overlapping_segments(self, diarization_result):
        """
        [NEW] 식별된 화자들의 시간대를 분석하여 겹침 구간만 추출합니다.
        사후 음성 분리(Separation) 모델을 돌릴 대상을 선정하는 데 사용됩니다.
        """
        # (구현부: 겹침 구간 로직은 파이프라인 완성 후 구체화 예정)
        pass

def diarize_audio(audio_path: str):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set in environment")
    diarizer = Diarizer(hf_token=hf_token)
    return diarizer.diarize(audio_path)
