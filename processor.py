import json
import subprocess
import asyncio
from pathlib import Path
from websocket_manager import manager
from config import CHUNK_SEC
from refiner import Refiner

# 전역 Refiner 인스턴스 (맥락 유지를 위해 1개만 생성)
refiner = Refiner()

def convert_to_wav(input_path: Path) -> Path:
    """
    Convert any audio to standard 16kHz Mono WAV for ML models.
    """
    output_path = input_path.with_name(f"{input_path.stem}_converted.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1",
        str(output_path)
    ]
    try:
        print(f"[Processor] Converting {input_path.name} to 16kHz Mono WAV...")
        subprocess.run(cmd, check=True, capture_output=True)
        if input_path.suffix.lower() != ".wav":
            input_path.unlink()
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"[Processor] Audio conversion failed: {e.stderr.decode()}")
        return input_path 


def get_processing_regions(duration, overlaps):
    """
    Divide chunk into [start, end, type] segments based on overlaps.
    """
    if not overlaps:
        return [{"start": 0.0, "end": duration, "type": "clean"}]

    regions = []
    last_end = 0.0
    
    for ov in overlaps:
        # Pre-overlap clean region
        if ov["start"] > last_end:
            regions.append({"start": last_end, "end": ov["start"], "type": "clean"})
        
        # Overlap region
        regions.append({
            "start": ov["start"], 
            "end": ov["end"], 
            "type": "overlap", 
            "speakers": ov.get("speakers", [])
        })
        last_end = ov["end"]
    
    # Final clean region
    if last_end < duration:
        regions.append({"start": last_end, "end": duration, "type": "clean"})
        
    return regions


def process_chunk(
    diarizer, 
    separator,
    speaker_registry, 
    chunk_index: int, 
    wav_path: Path, 
    output_dir: Path,
    partial_jsonl: Path,
    loop: asyncio.AbstractEventLoop = None
):
    """
    Full audio processing pipeline for a single chunk (v8 Immediate Refinement).
    """
    from diarization import diarize_audio
    from transcribe_gpu import transcribe_chunk
    from speaker_assigner import assign_speakers

    wav_path = convert_to_wav(wav_path)

    # 1. Diarization
    print(f"[Processor] Step 1: Diarizing {wav_path.name}...")
    diar_segments = diarize_audio(wav_path, diarizer=diarizer)

    # [v8] Overlap Detection & Immediate Refinement
    overlaps = []
    if hasattr(diarizer, "get_overlapping_segments"):
        overlaps = diarizer.get_overlapping_segments(diar_segments)

    # 2. Transcription (STT) 
    print(f"[Processor] Step 2: Transcribing baseline {wav_path.name}...")
    stt_segments = transcribe_chunk(wav_path)

    # [v8] Speech Separation for Significant Overlaps
    if separator and overlaps:
        overlap_dir = output_dir / "overlaps"
        overlap_dir.mkdir(parents=True, exist_ok=True)
        
        for ov in overlaps:
            ov_duration = ov["end"] - ov["start"]
            if ov_duration >= 2.0:
                print(f"[Processor] [v8] Immediate Separation for {ov['start']}s ~ {ov['end']}s")
                
                # Slicing for separation
                ov_slice = overlap_dir / f"temp_ov_{chunk_index}_{ov['start']:.2f}.wav"
                slice_cmd = ["ffmpeg", "-y", "-i", str(wav_path), "-ss", str(ov["start"]), "-to", str(ov["end"]), str(ov_slice)]
                subprocess.run(slice_cmd, check=True, capture_output=True)

                try:
                    # 1. Separate
                    print(f"  - Running Separation model...")
                    # separator(ov_slice) returns a Multi-track Audios
                    # Note: separator expects filepath or waveform
                    res = separator(str(ov_slice))
                    
                    # 2. Transcribe Each Track
                    print(f"  - Transcribing separated tracks...")
                    for i, (track_name, track_audio) in enumerate(res.items()):
                        # Save track to temp wav
                        track_path = overlap_dir / f"temp_track_{i}.wav"
                        # pyannote.audio Result has .export() or similar if using specific pipelines
                        # But AMI-1.0 typically returns a Map of Audio
                        track_audio.write_audio(str(track_path))
                        
                        # Transcribe the single-speaker track
                        refined_segs = transcribe_chunk(track_path)
                        for rs in refined_segs:
                            # Adjust time to global chunk time
                            rs["start"] += ov["start"]
                            rs["end"] += ov["start"]
                            # Mark as refined to skip or handle specially in assigner if needed
                            rs["is_refined"] = True
                            stt_segments.append(rs)
                        
                        track_path.unlink(missing_ok=True)
                except Exception as ex:
                    print(f"[Processor] [v8] Separation/Refinement failed: {ex}")
                finally:
                    ov_slice.unlink(missing_ok=True)

    # 3. Speaker Linking
    for d in diar_segments:
        spk_id, _ = speaker_registry.match_or_create(d["embedding"])
        d["global_speaker"] = spk_id

    # 4. Speaker Assignment
    print(f"[Processor] Step 3: Assigning speakers...")
    assigned_segments = assign_speakers(
        diar_segments=diar_segments,
        stt_segments=stt_segments,
        min_overlap_ratio=0.5,
        overlaps=overlaps
    )

    # 4.5 LLM Refinement (v8 추가)
    print(f"[Processor] [v8] Step 3.5: Refining segments with LLM...")
    try:
        # 동기 환경에서 비동기 호출을 처리하기 위해 event loop 활용 (또는 refiner를 동기로 변경 가능하나 확장성 위해 유지)
        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(refiner.refine(assigned_segments, chunk_index), loop)
            assigned_segments = future.result(timeout=10) # 10초 타임아웃
        else:
            # 루프가 없으면 새 루프로 실행 (Worker 스레드 상황 대응)
            new_loop = asyncio.new_event_loop()
            assigned_segments = new_loop.run_until_complete(refiner.refine(assigned_segments, chunk_index))
            new_loop.close()
    except Exception as e:
        print(f"[Processor] [v8] Refinement failed, using raw segments: {e}")

    # 5. Save Results
    print(f"[Processor] Step 4: Saving results to {partial_jsonl.name}...")
    records = []
    with open(partial_jsonl, "a", encoding="utf-8") as f:
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

    # 6. WebSocket Broadcasting
    if loop:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({
                "type": "new_segments",
                "chunkIndex": chunk_index,
                "segments": records
            }),
            loop
        )
