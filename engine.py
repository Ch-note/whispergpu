import os
import threading
import asyncio
from websocket_manager import manager

class EngineManager:
    """
    Manages the lifecycle of ML models (Pyannote & Whisper).
    Handles background loading to prevent blocking the API server.
    """
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.shared_diarizer = None
        self.shared_separator = None
        self.engines_ready = False
        self._lock = threading.Lock()

    def load_engines(self, loop: asyncio.AbstractEventLoop = None):
        """
        Background engine initialization with Lazy Imports.
        """
        try:
            # Move heavy imports here to avoid blocking server startup
            from diarization import Diarizer
            from transcribe_gpu import get_whisper_pipeline
            from pyannote.audio import Pipeline
            import torch

            print("[Engine] Starting background engine initialization...")
            
            # 1. Diarizer 로드
            print("[Engine] Loading Diarizer (Pyannote)...")
            self.shared_diarizer = Diarizer(hf_token=self.hf_token)
            
            # 2. Speech Separator 로드 (v8)
            print("[Engine] [v8] Loading Speech Separator (AMI)...")
            try:
                self.shared_separator = Pipeline.from_pretrained(
                    "pyannote/speech-separation-ami-1.0",
                    use_auth_token=self.hf_token
                )
                if self.shared_separator and torch.cuda.is_available():
                    self.shared_separator.to(torch.device("cuda"))
            except Exception as se:
                print(f"[Engine] [v8] Separator load failed (Skipping): {se}")

            # 3. Whisper 로드 (Warm-up)
            print("[Engine] Loading Whisper (Faster-Whisper)...")
            get_whisper_pipeline()
            
            with self._lock:
                self.engines_ready = True
            print("[Engine] All engines loaded and ready.")
            
            # WebSocket으로 Ready 신호 방송
            if loop:
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast({"type": "status", "value": "ready"}),
                    loop
                )
        except Exception as e:
            print(f"[Engine] Failed to load engines: {e}")
            if loop:
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast({"type": "status", "value": "error", "message": str(e)}),
                    loop
                )

    def is_ready(self) -> bool:
        with self._lock:
            return self.engines_ready

    def get_diarizer(self):
        # Return type Diarizer would require import, using dynamic return
        return self.shared_diarizer

    def get_separator(self):
        return self.shared_separator

# global helper
engine_manager = None

def init_engine_manager(hf_token: str) -> EngineManager:
    global engine_manager
    if engine_manager is None:
        engine_manager = EngineManager(hf_token)
    return engine_manager
