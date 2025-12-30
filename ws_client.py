import asyncio
import websockets
import json

async def listen_realtime():
    # Azure VM의 공인 IP가 있다면 localhost 대신 IP를 넣으세요.
    uri = "ws://localhost:8000/ws"
    
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
