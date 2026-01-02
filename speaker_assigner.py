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
    min_overlap_ratio=0.5,
    overlaps=None
):
    """
    STT 결과에 speaker 할당 (v8 Overlap Awareness 포함)

    :param stt_segments: [
        { "start": float, "end": float, "text": str }
    ]

    :param diar_segments: [
        { "start": float, "end": float, "global_speaker": str }
    ]

    :param min_overlap_ratio: STT segment 대비 최소 겹침 비율 (겹침 발화가 없을 경우에만 사용)

    :param overlaps: [
        { "start": float, "end": float, "speakers": [str] }
    ] - 겹침 발화 구간 정보

    :return: [
        { "start", "end", "speaker", "text" }
    ]
    """

    results = []

    for seg in stt_segments:
        seg_len = seg["end"] - seg["start"]
        overlap_map = {}

        # 1. 겹침 발화(Overlap) 여부 먼저 확인
        involved_in_overlap = []
        if overlaps:
            for ov in overlaps:
                # STT 구간과 겹침 구간의 교지합이 2초 이상이거나 구간의 50% 이상이면 겹침으로 간주
                ov_len = _overlap(seg["start"], seg["end"], ov["start"], ov["end"])
                if ov_len >= 2.0 or (ov_len / seg_len >= 0.5):
                    involved_in_overlap = ov["speakers"]
                    break

        # 2. 일반적인 화자 매칭 (가장 높은 점유율)
        for d in diar_segments:
            ov = _overlap(
                seg["start"], seg["end"],
                d["start"], d["end"]
            )

            if ov <= 0:
                continue

            spk = d["global_speaker"]
            overlap_map[spk] = overlap_map.get(spk, 0.0) + ov

        if involved_in_overlap:
            # v8: 여러 화자가 동시에 말하는 것으로 표시
            # 화자들을 정렬하여 일관성 유지
            sorted_spk = sorted(list(set(involved_in_overlap)))
            speaker = " & ".join(sorted_spk)
            
            # 2초 이상의 유의미한 겹침일 경우 마커 추가
            # involved_in_overlap이 설정되었다는 것은 이미 내부 루프에서 임계값을 넘었다는 뜻
            # 여기서는 명시적으로 '(겹침 발화)' 문구 추가
            speaker += " (겹침 발화)"
        elif not overlap_map:
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
            "start": round(float(seg["start"]), 2),
            "end": round(float(seg["end"]), 2),
            "speaker": speaker,
            "text": seg["text"]
        })

    return results
