import os
import json
from typing import List, Dict
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

class Refiner:
    def __init__(self):
        # Azure OpenAI Setup
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

        # Azure AI Search Setup
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        
        self.search_client = None
        if self.search_endpoint and self.search_key and self.index_name:
            self.search_client = SearchClient(
                self.search_endpoint, 
                self.index_name, 
                AzureKeyCredential(self.search_key)
            )

        # Context History (맥락 유지용)
        self.history = []
        self.max_history = 5
        self.domain_terms = ""

    def _get_domain_knowledge(self, query: str):
        """RAG를 통해 도메인 지식(전문 용어 등)을 추출합니다."""
        if not self.search_client:
            return ""
        
        try:
            # 단순 텍스트 검색으로 용어 추출 (필요시 임베딩 추가 가능하나 성능 위해 일단 텍스트 처리)
            results = self.search_client.search(
                search_text=query,
                top=3,
                select=["content"]
            )
            terms = "\n".join([r["content"] for r in results])
            return terms[:1000] # 토큰 절약
        except Exception as e:
            print(f"[Refiner] RAG Search failed: {e}")
            return ""

    async def refine(self, segments: List[Dict], chunk_index: int) -> List[Dict]:
        """STT 세그먼트들을 LLM을 통해 정제합니다."""
        if not segments:
            return []

        raw_text = " ".join([seg["text"] for seg in segments])
        
        # 1. 도메인 지식 업데이트 (첫 청크이거나 중요 키워드 있을 때만 수행 권장이나 일단 매번 시도)
        if chunk_index % 5 == 0 or not self.domain_terms:
            self.domain_terms = self._get_domain_knowledge(raw_text)

        # 2. 프롬프트 구성
        context_history = "\n".join(self.history[-2:]) # 직전 2개 청크만 맥락으로 제공
        
        system_prompt = f"""당신은 전문 회의 속기사입니다. 회사 [이음]의 회의 전사 내용을 정제하세요.

[수정 원칙 - 중요]
1. 최소 수정 원칙: 문맥상 명백히 틀린 전문 용어, 고유 명사, 맞춤법 오류만 수정하세요.
2. 완전한 문장 보존: 의미가 명확하고 올바른 단어로 구성된 문장은 절대 건드리지 마세요.
3. 단어/어순 유지: 불필요한 미사여구를 추가하거나 문장 표현을 미화하지 마세요.
4. 출력 형식: 추가 설명(예: "이 문장은~") 없이 오직 JSON 배열만 반환하세요.

[참조 지식]
{self.domain_terms}

[직전 대화 맥락]
{context_history}
"""

        user_content = json.dumps(segments, ensure_ascii=False)

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            refined_data = json.loads(response.choices[0].message.content)
            # JSON 형태가 {"segments": [...]} 인지 [...] 인지 체크
            if isinstance(refined_data, dict) and "segments" in refined_data:
                refined_list = refined_data["segments"]
            elif isinstance(refined_data, list):
                refined_list = refined_data
            else:
                # 가끔 모델이 객체 안에 리스트를 담아줄 때가 있음
                val = list(refined_data.values())[0]
                refined_list = val if isinstance(val, list) else segments

            # 결과 업데이트 및 비교 로그 출력
            print(f"\n✨ [Refinement Log - Chunk {chunk_index}]")
            for i, seg in enumerate(segments):
                raw_text = seg["text"]
                if i < len(refined_list):
                    refined_text = refined_list[i].get("text", raw_text)
                    if raw_text != refined_text:
                        print(f"  [Changed] {seg['speaker']}:")
                        print(f"    - From: {raw_text}")
                        print(f"    - To:   {refined_text}")
                    seg["text"] = refined_text
            
            # 히스토리 업데이트 (최신 5개 제한)
            current_summary = " ".join([seg["text"] for seg in segments])
            self.history.append(current_summary)
            if len(self.history) > self.max_history:
                self.history.pop(0)

            return segments

        except Exception as e:
            print(f"[Refiner] Refinement failed: {e}")
            return segments # 실패 시 원본 반환
