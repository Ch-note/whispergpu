import asyncio
import websockets
import json

async def listen_realtime():
    # Azure 클라우드 서버 주소 (wss 사용)
    uri = "wss://ieum-stt.livelymushroom-0e97085f.australiaeast.azurecontainerapps.io/ws"
    
    print(f"[WS] Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("[WS] Connected! Waiting for results...")
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "new_segments":
                    print(f"\n--- Chunk {data['chunkIndex']} results ---")
                    for seg in data["segments"]:
                        print(f"[{seg['start']}s - {seg['end']}s] {seg['speaker']}: {seg['text']}")
                else:
                    print(f"[WS] Notification: {data}")
                    
    except Exception as e:
        print(f"[WS] Error: {e}")

if __name__ == "__main__":
    asyncio.run(listen_realtime())
