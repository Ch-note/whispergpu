"""
test_run.py
- FastAPI 없이 파이프라인 직접 테스트
- Colab / 로컬 공용
"""

from pathlib import Path
import json

from main import process_chunk
from config import INPUT_DIR, OUTPUT_DIR

# ----------------------------
# 설정
# ----------------------------
TEST_CHUNK_DIR = Path("test_chunks")  # 테스트용 wav 폴더
TEST_CHUNK_DIR.mkdir(exist_ok=True)

INPUT_DIR = Path(INPUT_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)

# ----------------------------
# 테스트 실행
# ----------------------------
def run_test():
    wav_files = sorted(TEST_CHUNK_DIR.glob("*.wav"))

    if not wav_files:
        raise RuntimeError("test_chunks 폴더에 wav 파일이 없습니다.")

    print(f"[INFO] {len(wav_files)} chunks found")

    for idx, wav_path in enumerate(wav_files):
        print(f"[INFO] processing chunk {idx}: {wav_path.name}")
        process_chunk(
            chunk_index=idx,
            wav_path=wav_path
        )

    print("[INFO] all chunks processed")

    # 결과 미리보기
    result_path = OUTPUT_DIR / "partial_result.jsonl"
    if result_path.exists():
        print("\n===== RESULT PREVIEW =====")
        with open(result_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(json.loads(line))
    else:
        print("[WARN] result file not found")


if __name__ == "__main__":
    run_test()
