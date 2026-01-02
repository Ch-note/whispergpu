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
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        if self.pipeline is None:
            raise RuntimeError("Failed to load the diarization pipeline.")
        
        self.pipeline.to(self.device)

        # Embedding model (Used for speaker identity)
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=hf_token
        ).to(self.device)

        self.embedding_inference = Inference(
            self.embedding_model,
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

    def get_overlapping_segments(self, diar_results):
        """
        [NEW] 식별된 화자들의 시간대를 분석하여 겹침 구간만 추출합니다.
        알고리즘: 모든 시작/종료 시점을 정렬한 뒤, 각 구간별 활성 화자 수를 계산합니다.
        """
        if not diar_results:
            return []

        # 1. 모든 시점(Event) 수집
        events = []
        for d in diar_results:
            events.append((d["start"], 1, d["speaker"]))  # 시작
            events.append((d["end"], -1, d["speaker"]))   # 종료

        # 2. 시간순 정렬
        events.sort()

        overlaps = []
        active_speakers = set()
        last_time = events[0][0]

        # 3. 스윕 라인(Sweep-line) 알고리즘으로 겹침 구간 검색
        for time, kind, speaker in events:
            # 이전 시점부터 현재 시점까지 활성 화자가 2명 이상이면 겹침 구간
            if len(active_speakers) > 1 and time > last_time:
                overlaps.append({
                    "start": last_time,
                    "end": time,
                    "speakers": list(active_speakers)
                })
            
            if kind == 1:
                active_speakers.add(speaker)
            else:
                active_speakers.discard(speaker)
            
            last_time = time

        # 4. 인접한 동일 화자 겹침 구간 병합 (Optional, but cleaner)
        merged = []
        if overlaps:
            curr = overlaps[0]
            for i in range(1, len(overlaps)):
                nxt = overlaps[i]
                # 시간적으로 이어져 있고 참여 화자가 동일하면 병합
                if nxt["start"] == curr["end"] and set(nxt["speakers"]) == set(curr["speakers"]):
                    curr["end"] = nxt["end"]
                else:
                    merged.append(curr)
                    curr = nxt
            merged.append(curr)

        return merged

def diarize_audio(audio_path: str, diarizer: Diarizer = None):
    if diarizer is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is not set in environment")
        diarizer = Diarizer(hf_token=hf_token)
    return diarizer.diarize(audio_path)

