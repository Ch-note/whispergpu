import json
import queue
import threading
import os
import asyncio
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from websocket_manager import manager

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
            print(f"[Worker] Starting process for chunk {task.get('chunk_index')}")
            process_chunk(**task)
        except Exception as e:
            import traceback
            print(f"[Worker] Error processing chunk: {str(e)}")
            traceback.print_exc()
        finally:
            task_queue.task_done()


def convert_to_wav(input_path: Path) -> Path:
    """
    Convert any audio to standard 16kHz Mono WAV for ML models.
    Requires ffmpeg to be installed on the system.
    """
    output_path = input_path.with_name(f"{input_path.stem}_converted.wav")
    
    # ffmpeg command: 16kHz, mono, wav
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1",
        str(output_path)
    ]
    
    try:
        print(f"[Worker] Converting {input_path.name} to 16kHz Mono WAV...")
        subprocess.run(cmd, check=True, capture_output=True)
        # 뱐환 완료 후 원본이 wav가 아니었다면 원본 삭제 (공간 절약)
        if input_path.suffix.lower() != ".wav":
            input_path.unlink()
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"[Worker] Audio conversion failed: {e.stderr.decode()}")
        return input_path # 실패 시 원본 그대로 시도


def process_chunk(chunk_index: int, wav_path: Path):
    """
    Process one wav chunk:
    diarization -> speaker linking -> STT -> speaker assignment -> JSONL append
    """

    # 0. 변환 (16kHz Mono WAV로 통일)
    wav_path = convert_to_wav(wav_path)

    # 1. diarization
    print(f"[Worker] Step 1: Diarizing {wav_path.name}...")
    diar_segments = diarize_audio(wav_path)


    # 2. 전역 화자 연결 (speaker -> global_speaker)
    for d in diar_segments:
        spk_id, _ = speaker_registry.match_or_create(d["embedding"])
        d["global_speaker"] = spk_id

    # 3. STT 수행
    print(f"[Worker] Step 2: Transcribing {wav_path.name}...")
    stt_segments = transcribe_chunk(
        wav_path
    )

    # 4. speaker assignment (diar <-> STT)
    print(f"[Worker] Step 3: Assigning speakers...")
    assigned_segments = assign_speakers(
        diar_segments=diar_segments,
        stt_segments=stt_segments,
        min_overlap_ratio=0.5
    )

    # 5. append JSONL (global timeline)
    print(f"[Worker] Step 4: Saving results to {PARTIAL_JSONL.name}...")
    records = []
    with open(PARTIAL_JSONL, "a", encoding="utf-8") as f:
        for seg in assigned_segments:
            global_start = round(chunk_index * CHUNK_SEC + seg["start"], 2)
            global_end = round(chunk_index * CHUNK_SEC + seg["end"], 2)
            record = {
                "chunk": chunk_index,
                "speaker": seg["speaker"],
                "start": global_start,
                "end": global_end,
                "text": seg["text"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    # 6. WebSocket 실시간 방송
    if loop:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({
                "type": "new_segments",
                "chunkIndex": chunk_index,
                "segments": records
            }),
            loop
        )


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI()
worker_thread = threading.Thread(target=worker_loop, daemon=True)
loop = None


@app.on_event("startup")
def startup():
    global loop
    loop = asyncio.get_event_loop()
    worker_thread.start()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """실시간 전사 결과를 수신하기 위한 WebSocket 엔드포인트"""
    await manager.connect(websocket)
    try:
        # 연결 유지 (클라이언트로부터의 메시지는 무시하거나 필요 시 처리)
        while True:
            await websocket.receive_text()
    except Exception:
        # 연결 종료 시 관리자에서 제거
        manager.disconnect(websocket)


@app.get("/result")
def get_result():
    records = []
    if PARTIAL_JSONL.exists():
        with open(PARTIAL_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    return records


@app.post("/chunk")
async def upload_chunk(
    chunkIndex: int = Form(...),
    file: UploadFile = File(...)
):
    global meeting_ended

    if meeting_ended:
        raise HTTPException(400, "Meeting already ended")

    # 브라우저 MediaRecorder는 보통 webm 등을 생성하므로 체크를 완화합니다.
    # pydub 등의 라이브러리가 있다면 여기서 wav로 변환하는 것이 안전하지만,
    # faster-whisper는 ffmpeg이 설치되어 있다면 대부분의 포맷을 처리할 수 있습니다.
    allowed_extensions = [".wav", ".webm", ".ogg", ".mp3", ".m4a"]
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions and not file.content_type.startswith("audio/"):
         raise HTTPException(400, f"Unsupported file type: {file.filename}")

    # 저장 시에는 chunkIndex를 사용하여 고유한 이름을 부여합니다.
    save_path = INPUT_DIR / f"chunk_{chunkIndex:05d}{ext}"
    save_path.write_bytes(await file.read())

    task_queue.put({
        "chunk_index": chunkIndex,
        "wav_path": save_path
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

    # merge JSONL -> final JSON
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
