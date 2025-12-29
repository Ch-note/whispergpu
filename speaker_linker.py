"""
speaker_linker.py

청크 단위 diarization speaker embedding을
전역 speaker ID로 통합하는 모듈

- sklearn 의존성 제거
- NumPy 기반 cosine similarity
- EMA 방식 embedding 업데이트
- 6인 회의 기준 안정 설계
"""

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    NumPy 기반 cosine similarity
    """
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    )


class SpeakerRegistry:
    def __init__(self, similarity_threshold=0.75, ema_alpha=0.8):
        """
        :param similarity_threshold: 같은 화자로 판단할 cosine similarity 기준
        :param ema_alpha: embedding EMA 업데이트 비율
        """
        self.speakers = {}  # { "SPK_0": {"embedding": np.ndarray, "count": int} }
        self.similarity_threshold = similarity_threshold
        self.ema_alpha = ema_alpha

    def match(self, embedding: np.ndarray):
        """
        기존 speaker 중 가장 유사한 speaker 찾기
        """
        best_id = None
        best_score = 0.0

        for spk_id, data in self.speakers.items():
            score = cosine_sim(embedding, data["embedding"])
            if score > best_score:
                best_score = score
                best_id = spk_id

        if best_score >= self.similarity_threshold:
            return best_id, best_score

        return None, best_score

    def register(self, embedding: np.ndarray):
        """
        신규 speaker 등록
        """
        new_id = f"SPK_{len(self.speakers)}"
        self.speakers[new_id] = {
            "embedding": embedding,
            "count": 1
        }
        return new_id

    def update(self, spk_id: str, new_embedding: np.ndarray):
        """
        기존 speaker embedding을 EMA 방식으로 업데이트
        """
        old = self.speakers[spk_id]["embedding"]
        alpha = self.ema_alpha

        updated = alpha * old + (1.0 - alpha) * new_embedding
        self.speakers[spk_id]["embedding"] = updated
        self.speakers[spk_id]["count"] += 1

    def match_or_create(self, embedding: np.ndarray, update: bool = True):
        """
        speaker 매칭 시도 -> 실패 시 신규 speaker 생성
        """
        spk_id, score = self.match(embedding)

        if spk_id is not None:
            if update:
                self.update(spk_id, embedding)
            return spk_id, score

        new_id = self.register(embedding)
        return new_id, None

