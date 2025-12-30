import asyncio
import websockets
import json

async def listen_realtime():
    # Azure VM의 공인 IP가 있다면 localhost 대신 IP를 넣으세요.
    uri = "ws://localhost:8000/ws"
    
    print(f"📡 {uri} 에 연결 시도 중...")
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket 연결 성공! 실시간 결과를 기다리는 중입니다...")
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "new_segments":
                    print(f"\n--- 🌊 Chunk {data['chunkIndex']} 분석 결과 도착 ---")
                    for seg in data["segments"]:
                        print(f"[{seg['start']}s - {seg['end']}s] {seg['speaker']}: {seg['text']}")
                else:
                    print(f"📩 수신 알림: {data}")
                    
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    asyncio.run(listen_realtime())
