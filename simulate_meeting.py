import os
import time
import requests
import subprocess
from pathlib import Path

# --- Configuration ---
API_BASE = "https://ieum-stt.livelymushroom-0e97085f.australiaeast.azurecontainerapps.io"
AUDIO_FILE = "./test.wav"  # Your local 1-hour audio file
CHUNK_SEC = 30
MAX_CHUNKS = 20  # Limit to 10 minutes (20 * 30s) for testing
TEMP_DIR = Path("./test_chunks")

def slice_audio(input_file):
    """Slices the input audio into 30s chunks using ffmpeg."""
    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir()
    
    # Cleanup previous chunks
    for f in TEMP_DIR.glob("chunk_*.wav"):
        f.unlink()

    print(f"[*] Splicing {input_file} into {CHUNK_SEC}s segments...")
    # We use -t to limit the total slicing time to save local disk space/time
    total_test_time = MAX_CHUNKS * CHUNK_SEC
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-t", str(total_test_time), 
        "-f", "segment", "-segment_time", str(CHUNK_SEC),
        "-c", "copy", str(TEMP_DIR / "chunk_%03d.wav")
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(list(TEMP_DIR.glob("chunk_*.wav")))

def stream_meeting():
    if not os.path.exists(AUDIO_FILE):
        print(f"[Error] {AUDIO_FILE} not found. Please rename your 1-hour file to '{AUDIO_FILE}'")
        return

    all_chunks = slice_audio(AUDIO_FILE)
    chunks = all_chunks[:MAX_CHUNKS]
    
    print(f"[*] Total chunks found: {len(all_chunks)}")
    print(f"[*] Limiting test to first {len(chunks)} chunks (~{len(chunks)*CHUNK_SEC/60:.1f} minutes)")
    
    # 1. Start Logging (Optional: You can add WebSocket here)
    print("[*] Starting Real-time Simulation...")
    
    try:
        for i, chunk_path in enumerate(chunks):
            start_time = time.time()
            print(f"\n[Chunk {i}] Uploading {chunk_path.name}...")
            
            # POST /chunk
            with open(chunk_path, "rb") as f:
                files = {"file": (chunk_path.name, f, "audio/wav")}
                data = {"chunkIndex": i}
                response = requests.post(f"{API_BASE}/chunk", files=files, data=data)
                
            if response.status_code == 200:
                print(f"[Chunk {i}] Success: {response.json()}")
            else:
                print(f"[Chunk {i}] Failed: {response.status_code} - {response.text}")

            # 2. Wait for the remainder of the 30 seconds
            elapsed = time.time() - start_time
            wait_time = max(0, CHUNK_SEC - elapsed)
            
            if i < len(chunks) - 1:
                print(f"[*] Waiting {wait_time:.1f}s for the next 'real-time' frame...")
                time.sleep(wait_time)
        
        # 3. End Meeting
        print("\n[*] All chunks sent. Ending meeting...")
        requests.post(f"{API_BASE}/end")
        print("[*] Meeting simulation finished. Check /result or Azure logs.")

    except KeyboardInterrupt:
        print("\n[!] Simulation stopped by user.")
    finally:
        # Cleanup
        # for f in TEMP_DIR.glob("chunk_*.wav"): f.unlink()
        pass

if __name__ == "__main__":
    stream_meeting()
