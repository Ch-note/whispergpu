# v8 Speech Separation Implementation Plan (Proposed)

This document summarizes the research and strategy for implementing speech separation to handle overlapping segments in meeting environments.

## 1. Model Selection: `pyannote/speech-separation-ami-1.0`

After comparing various models, the Pyannote AMI model is selected as the primary candidate.

| Feature | Analysis |
| :--- | :--- |
| **Accuracy** | Highest for meeting environments (trained on AMI corpus). |
| **Stability** | Native compatibility with existing `pyannote.audio 3.x` backend. |
| **Resources** | Lightweight (~2GB VRAM), suitable for T4 GPU (16GB). |
| **Latency** | Fast inference, allowing for near-real-time processing within 30s chunks. |

---

## 2. Implementation Strategy: Immediate Refinement

Based on the decision to provide "fully refined results" every 30 seconds.

### Pipeline Workflow
1. **Diarization**: Identify speakers and detect overlap regions.
2. **Analysis**: If overlap > 2.0s, mark for separation.
3. **Separation**:
   - Slice the overlapping audio segment.
   - Run `pyannote/speech-separation-ami-1.0` to generate individual speaker tracks.
4. **Transcription**: Run `Whisper Medium` on each separated track independently.
5. **Merging**: Replace the original "combined" segment with individual speaker segments.
6. **Delivery**: Send the final, refined 30s block to the frontend via WebSocket.

### Safe-guards
- **Overlap Limit**: Process up to top 3-5 longest overlaps per chunk to guarantee 30s throughput.
- **Time Budget**: Fallback to standard labeling (`Speaker A & B`) if processing exceeds 25 seconds.

---

## 3. Alternative: Deferred Refinement (Post-Meeting)
If VRAM or Latency becomes an issue during multi-user sessions:
- Save overlap slices to `/output/overlaps/` during the meeting.
- Label as `(겹침 발화)` in real-time.
- Perform all separation and transcription in a batch after `POST /end`.

---

## 4. Hardware Considerations (T4 GPU / 16GB)
- **Current Load**: ~5-6GB (Whisper Medium + Diarizer).
- **Separation Addition**: +2GB (Total ~8GB).
- **Conclusion**: Ample VRAM available; focus should be on **Compute Speed** (ensuring 30s of processing fits within a 30s window).
