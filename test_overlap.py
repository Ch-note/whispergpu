import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

from diarization import Diarizer

def test_overlap_detection():
    # Mock 데이터 (화자 A, B, C가 섞인 상황)
    # A: 0~10s
    # B: 5~15s (5~10s는 A와 겹침)
    # C: 12~18s (12~15s는 B와 겹침)
    mock_results = [
        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_A"},
        {"start": 5.0, "end": 15.0, "speaker": "SPEAKER_B"},
        {"start": 12.0, "end": 18.0, "speaker": "SPEAKER_C"},
    ]

    # Diarizer 인스턴스 (실제 모델 로딩 없이 로직만 테스트하기 위해 Diarizer.__init__ 우회)
    class MockDiarizer(Diarizer):
        def __init__(self):
            pass

    diarizer = MockDiarizer()
    overlaps = diarizer.get_overlapping_segments(mock_results)

    print("--- Mock Diarization Results ---")
    for d in mock_results:
        print(f"{d['speaker']}: {d['start']} ~ {d['end']}")

    print("\n--- Detected Overlaps ---")
    for ov in overlaps:
        print(f"Overlap: {ov['start']}s ~ {ov['end']}s (Speakers: {ov['speakers']})")

    # 검증
    # 5.0 ~ 10.0 (A, B)
    # 12.0 ~ 15.0 (B, C)
    expected = [
        {"start": 5.0, "end": 10.0, "speakers": ["SPEAKER_A", "SPEAKER_B"]},
        {"start": 12.0, "end": 15.0, "speakers": ["SPEAKER_B", "SPEAKER_C"]}
    ]

    success = True
    if len(overlaps) != len(expected):
        success = False
    else:
        for i, (o, e) in enumerate(zip(overlaps, expected)):
            if o["start"] != e["start"] or o["end"] != e["end"] or set(o["speakers"]) != set(e["speakers"]):
                print(f"Mismatch at index {i}: Expected {e}, Got {o}")
                success = False

    if success:
        print("\n✅ Overlap detection logic verified successfully!")
    else:
        print("\n❌ Overlap detection logic failed.")

if __name__ == "__main__":
    test_overlap_detection()
