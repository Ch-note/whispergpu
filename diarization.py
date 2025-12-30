from pathlib import Path
import torch
import numpy as np
import os

from pyannote.audio import Pipeline, Model, Inference, Audio
from pyannote.core import Segment
from huggingface_hub import login, hf_hub_download
import huggingface_hub
import functools

from config import DEVICE
import sys

# ---- Global Monkey patch for huggingface_hub version compatibility ----
# More robust: it scans all currently loaded modules and replaces the reference.
_original_hf_hub_download = hf_hub_download

def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _original_hf_hub_download(*args, **kwargs)

# 1. Patch the main entry point
huggingface_hub.hf_hub_download = _patched_hf_hub_download

# 2. Patch all modules that have already imported 'hf_hub_download'
for name, module in list(sys.modules.items()):
    if hasattr(module, "hf_hub_download"):
        if getattr(module, "hf_hub_download") is _original_hf_hub_download:
            setattr(module, "hf_hub_download", _patched_hf_hub_download)

# 3. Explicitly patch key pyannote modules just to be sure
try:
    import pyannote.audio.core.pipeline
    pyannote.audio.core.pipeline.hf_hub_download = _patched_hf_hub_download
    import pyannote.audio.core.model
    pyannote.audio.core.model.hf_hub_download = _patched_hf_hub_download
except ImportError:
    pass


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

        # ---- global login for gated models (solves token/use_auth_token conflict) ----
        login(token=hf_token)

        # ---- diarization pipeline ----
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization"
        ).to(self.device)

        # ---- embedding model (Inference wrapper is REQUIRED) ----
        embedding_model = Model.from_pretrained(
            "pyannote/embedding"
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
            
            # ---- 너무 짧은 구간(0.5초 미만)은 임베딩 추출 시 에러 발생 가능하므로 제외 ----
            if turn.duration < 0.2:
                continue

            try:
                embedding = self.embedding_inference.crop(audio_dict, turn)
                if hasattr(embedding, "detach"):
                    embedding = embedding.detach().cpu().numpy()

                results.append(
                    {
                        "start": round(float(turn.start), 2),
                        "end": round(float(turn.end), 2),
                        "speaker": speaker,
                        "embedding": embedding,
                    }
                )
            except Exception as e:
                print(f"[WARN] Failed to extract embedding for segment {turn.start:.2f}-{turn.end:.2f}: {e}")
                continue

        return results


def diarize_audio(audio_path: str):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_TOKEN is not set in environment")

    diarizer = Diarizer(hf_token=hf_token)
    return diarizer.diarize(audio_path)
