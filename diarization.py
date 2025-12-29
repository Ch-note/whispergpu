"""
diarization.py

- pyannote.audio 기반 화자 분리
- 청크 단위 diarization
- speaker turn + embedding 추출
"""

from pathlib import Path
import json
import numpy as np
import os
from pyannote.audio import Pipeline, Model


class Diarizer:
    def __init__(self, hf_token: str, device: str = "cuda"):
        """
        :param hf_token: HuggingFace access token
        :param device: "cuda" or "cpu"
        """
        if DEVICE != "cuda":
            raise RuntimeError(f"Invalid DEVICE={DEVICE}. This pipeline requires CUDA.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available")

        # 화자 분리 파이프라인
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token
        ).to(torch.device("cuda"))

        # speaker embedding 모델
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=hf_token
        ).to(torch.device("cuda"))

    def diarize(self, audio_path: str):
        """
        diarization 실행

        return:
        [
          {
            "start": float,
            "end": float,
            "local_speaker": "SPEAKER_00",
            "embedding": np.ndarray
          }
        ]
        """
        diarization = self.pipeline(audio_path)
        audio_path = Path(audio_path)

        results = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # 너무 짧은 발화는 제외 (embedding 불안정)
            if (turn.end - turn.start) < 0.3:
                continue

            # speaker embedding 추출
            embedding = self.embedding_model.crop(
                audio_path,
                turn
            ).detach().cpu().numpy()

            results.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "local_speaker": speaker,
                "embedding": embedding
            })

        return results

    def save(self, diarization_results, out_path: str):
        """
        embedding 제외하고 JSON 저장 (디버깅/로그용)
        """
        serializable = [
            {
                "start": r["start"],
                "end": r["end"],
                "local_speaker": r["local_speaker"]
            }
            for r in diarization_results
        ]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

# diarization.py (맨 아래)


def diarize_audio(audio_path: str, device: str = "cuda"):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")

    diarizer = Diarizer(
        hf_token=hf_token,
        device=device
    )
    return diarizer.diarize(audio_path)

