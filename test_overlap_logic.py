def get_overlapping_segments(diar_results):
    """
    [Logic Copy from diarization.py]
    """
    if not diar_results:
        return []

    events = []
    for d in diar_results:
        events.append((d["start"], 1, d["speaker"]))
        events.append((d["end"], -1, d["speaker"]))

    events.sort()

    overlaps = []
    active_speakers = set()
    last_time = events[0][0]

    for time, kind, speaker in events:
        if len(active_speakers) > 1 and time > last_time:
            overlaps.append({
                "start": last_time,
                "end": time,
                "speakers": list(active_speakers)
            })
        
        if kind == 1:
            active_speakers.add(speaker)
        else:
            active_speakers.discard(speaker)
        
        last_time = time

    merged = []
    if overlaps:
        curr = overlaps[0]
        for i in range(1, len(overlaps)):
            nxt = overlaps[i]
            if nxt["start"] == curr["end"] and set(nxt["speakers"]) == set(curr["speakers"]):
                curr["end"] = nxt["end"]
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)

    return merged

def run_test():
    mock_results = [
        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_A"},
        {"start": 5.0, "end": 15.0, "speaker": "SPEAKER_B"},
        {"start": 12.0, "end": 18.0, "speaker": "SPEAKER_C"},
    ]

    overlaps = get_overlapping_segments(mock_results)

    print("--- Mock Diarization Results ---")
    for d in mock_results:
        print(f"{d['speaker']}: {d['start']} ~ {d['end']}")

    print("\n--- Detected Overlaps ---")
    for ov in overlaps:
        print(f"Overlap: {ov['start']}s ~ {ov['end']}s (Speakers: {ov['speakers']})")

    expected = [
        {"start": 5.0, "end": 10.0, "speakers": ["SPEAKER_A", "SPEAKER_B"]},
        {"start": 12.0, "end": 15.0, "speakers": ["SPEAKER_B", "SPEAKER_C"]}
    ]

    success = True
    for i, (o, e) in enumerate(zip(overlaps, expected)):
        if o["start"] != e["start"] or o["end"] != e["end"] or set(o["speakers"]) != set(e["speakers"]):
            success = False
            break

    if success and len(overlaps) == len(expected):
        print("\n✅ Logic verified!")
    else:
        print("\n❌ Logic failed.")

if __name__ == "__main__":
    run_test()
