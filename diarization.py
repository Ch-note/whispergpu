from pathlib import Path
import torch
import numpy as np
import os

from pyannote.audio import Pipeline, Model, Inference, Audio
from pyannote.core import Segment

from config import DEVICE


class Diarizer:
    """
    Speaker diarization + embedding extractor
    pyannote.audio 3.x compatible
    """

    def __init__(self, hf_token: str):
        # ---- device policy: CUDA only ----
        if DEVICE != "cuda":
            raise RuntimeError(f"Invalid DEVICE={DEVICE}. This pipeline requires CUDA.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available.")

        self.device = torch.device("cuda")

        # ---- diarization pipeline ----
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            token=hf_token,
        ).to(self.device)

        # ---- embedding model (Inference wrapper is REQUIRED) ----
        embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            token=hf_token,
        )

        self.embedding_inference = Inference(
            embedding_model,
            window="whole",
        ).to(self.device)

        # ---- audio loader (explicit, version-safe) ----
        self.audio = Audio(sample_rate=16000, mono=True)

    def diarize(self, audio_path: str):
        """
        Run diarization and return segments with embeddings.

        Returns:
            List[dict]:
              {
                "start": float,
                "end": float,
                "speaker": str,
                "embedding": np.ndarray
              }
        """
        audio_path = str(Path(audio_path).resolve())

        # ---- load audio explicitly ----
        waveform, sample_rate = self.audio(audio_path)
        audio_dict = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

        # ---- run diarization ----
        diarization = self.pipeline(audio_dict)

        results = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # turn is pyannote.core.Segment

            embedding = self.embedding_inference.crop(audio_dict, turn)
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().numpy()

            results.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speaker,
                    "embedding": embedding,
                }
            )

        return results


def diarize_audio(audio_path: str):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_TOKEN is not set in environment")

    diarizer = Diarizer(hf_token=hf_token)
    return diarizer.diarize(audio_path)
