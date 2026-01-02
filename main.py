import json
import queue
import threading
import os
import asyncio
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from websocket_manager import manager
from config import (
    INPUT_DIR, OUTPUT_DIR, CHUNK_SEC,
)
from speaker_linker import SpeakerRegistry
from engine import init_engine_manager
from processor import process_chunk

# ----------------------------
# Environment & Paths
# ----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
# For local debugging, you might need to set this in your terminal or .env
if not HF_TOKEN:
    print("[WARN] HF_TOKEN not found in environment variables.")

INPUT_DIR = Path(INPUT_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTIAL_JSONL = OUTPUT_DIR / "partial_result.jsonl"
FINAL_JSON = OUTPUT_DIR / "final_result.json"

# ----------------------------
# Global State
# ----------------------------
speaker_registry = SpeakerRegistry()
task_queue = queue.Queue()
meeting_ended = False
loop = None

# Initialize Engine
engine_mgr = init_engine_manager(HF_TOKEN)

# ----------------------------
# Worker Loop
# ----------------------------
def worker_loop():
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        # Wait until engines are ready
        while not engine_mgr.is_ready():
            print("[Worker] Engines not ready. Waiting 2s...")
            time.sleep(2)
            
        try:
            print(f"[Worker] Processing chunk {task.get('chunk_index')}")
            process_chunk(
                diarizer=engine_mgr.get_diarizer(),
                separator=engine_mgr.get_separator(),
                speaker_registry=speaker_registry,
                output_dir=OUTPUT_DIR,
                partial_jsonl=PARTIAL_JSONL,
                loop=loop,
                **task
            )
        except Exception as e:
            import traceback
            print(f"[Worker] Error processing chunk: {str(e)}")
            traceback.print_exc()
        finally:
            task_queue.task_done()

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

worker_thread = threading.Thread(target=worker_loop, daemon=True)

@app.get("/")
def read_root():
    """
    Azure Health Check Endpoint & Internal Status
    """
    status = "ready" if engine_mgr.is_ready() else "loading"
    return {
        "status": status,
        "message": "Whisper GPU API is running",
        "engines_ready": engine_mgr.is_ready()
    }

@app.on_event("startup")
def startup():
    global loop
    loop = asyncio.get_event_loop()
    
    # 1. Start background engine loading
    threading.Thread(target=engine_mgr.load_engines, args=(loop,), daemon=True).start()
    
    # 2. Start worker
    worker_thread.start()
    print("[Startup] API port 8000 opened. Engines loading in background...")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    status = "ready" if engine_mgr.is_ready() else "loading"
    await websocket.send_json({"type": "status", "value": status})
    try:
        while True:
            await websocket.receive_text()
    except Exception:
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

    ext = Path(file.filename).suffix.lower()
    save_path = INPUT_DIR / f"chunk_{chunkIndex:05d}{ext}"
    save_path.write_bytes(await file.read())

    task_queue.put({
        "chunk_index": chunkIndex,
        "wav_path": save_path
    })

    return {
        "status": "queued",
        "chunkIndex": chunkIndex,
        "engine_ready": engine_mgr.is_ready()
    }

@app.post("/reset")
def reset_meeting():
    global meeting_ended, worker_thread, speaker_registry, task_queue
    
    # 1. Reset states
    meeting_ended = False
    speaker_registry = SpeakerRegistry()
    
    # 2. Clear queue (should be empty anyway if ended)
    while not task_queue.empty():
        try:
            task_queue.get_nowait()
            task_queue.task_done()
        except queue.Empty:
            break
            
    # 3. Cleanup files
    if PARTIAL_JSONL.exists():
        PARTIAL_JSONL.unlink()
    if FINAL_JSON.exists():
        FINAL_JSON.unlink()
    # Optional: Clear INPUT_DIR chunks? 
    # for f in INPUT_DIR.glob("chunk_*"): f.unlink()

    # 4. Restart worker if dead
    if not worker_thread.is_alive():
        worker_thread = threading.Thread(target=worker_loop, daemon=True)
        worker_thread.start()
        print("[Reset] Worker thread restarted.")

    return {"status": "reset", "message": "Meeting state cleared, ready for new session."}

@app.post("/end")
def end_meeting():
    global meeting_ended, worker_thread
    if meeting_ended:
        return {"status": "already_ended"}
        
    meeting_ended = True
    task_queue.put(None)
    if worker_thread.is_alive():
        worker_thread.join()

    segments = []
    if PARTIAL_JSONL.exists():
        with open(PARTIAL_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                segments.append(json.loads(line))
    segments.sort(key=lambda x: x["start"])

    final_result = {"segments": segments}
    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    return {"status": "ended", "segments": len(segments), "output": str(FINAL_JSON)}

@app.post("/shutdown")
def shutdown():
    """백엔드 서버 및 컨테이너 종료"""
    print("👋 Shutdown requested. Closing server...")
    
    def kill_process():
        time.sleep(1)
        os._exit(0)
        
    threading.Thread(target=kill_process, daemon=True).start()
    return {"status": "shutting_down", "message": "Server process will exit in 1s."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
