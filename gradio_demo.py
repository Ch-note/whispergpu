import gradio as gr
import requests
import os
import time
import json
from pathlib import Path

# FastAPI 서버 주소 (VM 내부에서 돌릴 경우 localhost, 외부라면 IP)
SERVER_URL = "http://localhost:8000"

def process_audio(audio_path):
    if audio_path is None:
        return "오디오 파일이 없습니다."

    # 1. 파일 준비
    file_basename = os.path.basename(audio_path)
    files = {
        "file": (file_basename, open(audio_path, "rb"), "audio/wav")
    }
    
    # 2. 테스트용으로 chunkIndex=0으로 전송 (연속 테스트를 원하면 로직 추가 가능)
    data = {
        "chunkIndex": 0,
        "meetingId": "gradio_test_session"
    }

    try:
        # 3. /chunk API 호출
        resp = requests.post(f"{SERVER_URL}/chunk", files=files, data=data)
        if resp.status_code == 200:
            # 4. 결과 확인 (/result API 호출)
            time.sleep(2) # 분석 대기
            result_resp = requests.get(f"{SERVER_URL}/result")
            if result_resp.status_code == 200:
                results = result_resp.json()
                formatted_text = ""
                for r in results:
                    formatted_text += f"[{r['start']}s - {r['end']}s] {r['speaker']}: {r['text']}\n"
                return formatted_text
            else:
                return f"결과 조회 실패: {result_resp.text}"
        else:
            return f"업로드 실패: {resp.text}"
    except Exception as e:
        return f"에러 발생: {str(e)}"

# Gradio 인터페이스 구성
with gr.Blocks(title="Whisper Diarization Test") as demo:
    gr.Markdown("# 🎙️ Whisper Real-time Diarization Test")
    gr.Markdown("마이크로 녹음하거나 파일을 업로드하여 결과를 확인하세요.")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="음성 입력")
        
    with gr.Row():
        btn = gr.Button("분석 시작", variant="primary")
        
    output_text = gr.Textbox(label="전사 결과 (Speaker Diarization)", lines=10)
    
    btn.click(fn=process_audio, inputs=audio_input, outputs=output_text)

if __name__ == "__main__":
    # share=True를 설정하면 외부에서도 접속 가능한 public URL이 생성됩니다.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
