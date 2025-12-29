import json

def merge_chunks(json_files, chunk_sec, overlap_sec):
    merged = []
    current_offset = 0.0

    for idx, jf in enumerate(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            segments = json.load(f)

        for seg in segments:
            # overlap 앞부분 제거
            if idx > 0 and seg["start"] < overlap_sec:
                continue

            merged.append({
                "start": seg["start"] + current_offset,
                "end": seg["end"] + current_offset,
                "text": seg["text"]
            })

        # 다음 chunk 시작 시간
        current_offset += (chunk_sec - overlap_sec)

    return merged