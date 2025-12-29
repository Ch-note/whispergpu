"""
speaker_assigner.py

STT segment에 diarization + speaker_linker 결과를 기반으로
최종 speaker를 할당하는 모듈
"""

def _overlap(a_start, a_end, b_start, b_end):
    """
    두 시간 구간의 겹치는 길이 계산
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    stt_segments,
    diar_segments,
    min_overlap_ratio=0.5
):
    """
    STT 결과에 speaker 할당

    :param stt_segments: [
        { "start": float, "end": float, "text": str }
    ]

    :param diar_segments: [
        { "start": float, "end": float, "global_speaker": str }
    ]

    :param min_overlap_ratio: STT segment 대비 최소 겹침 비율

    :return: [
        { "start", "end", "speaker", "text" }
    ]
    """

    results = []

    for seg in stt_segments:
        seg_len = seg["end"] - seg["start"]
        overlap_map = {}

        for d in diar_segments:
            ov = _overlap(
                seg["start"], seg["end"],
                d["start"], d["end"]
            )

            if ov <= 0:
                continue

            spk = d["global_speaker"]
            overlap_map[spk] = overlap_map.get(spk, 0.0) + ov

        if not overlap_map:
            speaker = "UNKNOWN"
        else:
            best_speaker, best_overlap = max(
                overlap_map.items(),
                key=lambda x: x[1]
            )

            if best_overlap / seg_len >= min_overlap_ratio:
                speaker = best_speaker
            else:
                speaker = "UNKNOWN"

        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": seg["text"]
        })

    return results
