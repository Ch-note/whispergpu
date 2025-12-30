from typing import List
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        # 현재 연결된 모든 WebSocket 세션을 관리
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # 클라이언트에게 준비 완료 신호 전송
        await websocket.send_json({"type": "server_ready"})
        print(f"[INFO] New WebSocket connection and sent ready signal. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[INFO] WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """연결된 모든 클라이언트에게 JSON 메시지 전송"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                # 연결이 끊긴 클라이언트는 자동으로 제외될 수 있도록 예외 처리
                print(f"[WARN] Failed to send message to a client: {e}")

# 싱글톤 인스턴스 생성
manager = ConnectionManager()
