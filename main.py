import json
import queue
import threading
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

# ----------------------------
# config
# ----------------------------
from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    MODEL_NAME,
    LANGUAGE,
    CHUNK_SEC,
    DEVICE,
)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Please set it via .env or environment variable.")

# ----------------------------
# pipeline modules
# ----------------------------
from diarization import diarize_audio
from speaker_linker import SpeakerRegistry
from speaker_assigner import assign_speakers
from transcribe_gpu import transcribe_chunk

# ----------------------------
# paths
# ----------------------------
INPUT_DIR = Path(INPUT_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTIAL_JSONL = OUTPUT_DIR / "partial_result.jsonl"
FINAL_JSON = OUTPUT_DIR / "final_result.json"

# ----------------------------
# global state (single meeting)
# ----------------------------
speaker_registry = SpeakerRegistry()
task_queue = queue.Queue()
meeting_ended = False

# ----------------------------
# worker
# ----------------------------
def worker_loop():
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            process_chunk(**task)
        finally:
            task_queue.task_done()


def process_chunk(chunk_index: int, wav_path: Path):
    """
    Process one wav chunk:
    diarization → speaker linking → STT → speaker assignment → JSONL append
    """

    # 1. diarization
    diar_segments = diarize_audio(
        wav_path,
        device=DEVICE
    )

    # 2. global speaker linking
    for d in diar_segments:
        spk_id, _ = speaker_registry.match_or_create(d["embedding"])
        d["speaker"] = spk_id

    # 3. STT
    stt_segments = transcribe_chunk(
        wav_path,
        model_name=MODEL_NAME,
        language=LANGUAGE,
        device=DEVICE
    )

    # 4. speaker assignment (diar ↔ STT)
    assigned_segments = assign_speakers(
        diar_segments=diar_segments,
        stt_segments=stt_segments,
        min_overlap_ratio=0.5
    )

    # 5. append JSONL (global timeline)
    with open(PARTIAL_JSONL, "a", encoding="utf-8") as f:
        for seg in assigned_segments:
            record = {
                "chunk": chunk_index,
                "speaker": seg["speaker"],
                "start": chunk_index * CHUNK_SEC + seg["start"],
                "end": chunk_index * CHUNK_SEC + seg["end"],
                "text": seg["text"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI()
worker_thread = threading.Thread(target=worker_loop, daemon=True)


@app.on_event("startup")
def startup():
    worker_thread.start()


@app.post("/chunk")
async def upload_chunk(
    chunkIndex: int = Form(...),
    file: UploadFile = File(...)
):
    global meeting_ended

    if meeting_ended:
        raise HTTPException(400, "Meeting already ended")

    if not file.filename.endswith(".wav"):
        raise HTTPException(400, "Only wav files are supported")

    wav_path = INPUT_DIR / f"chunk_{chunkIndex:05d}.wav"
    wav_path.write_bytes(await file.read())

    task_queue.put({
        "chunk_index": chunkIndex,
        "wav_path": wav_path
    })

    return {
        "status": "queued",
        "chunkIndex": chunkIndex
    }


@app.post("/end")
def end_meeting():
    global meeting_ended
    meeting_ended = True

    # stop worker
    task_queue.put(None)
    worker_thread.join()

    # merge JSONL → final JSON
    segments = []
    if PARTIAL_JSONL.exists():
        with open(PARTIAL_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                segments.append(json.loads(line))

    segments.sort(key=lambda x: x["start"])

    final_result = {
        "segments": segments
    }

    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    return {
        "status": "ended",
        "segments": len(segments),
        "output": str(FINAL_JSON)
    }


if __name__ == "__main__":
    main()
