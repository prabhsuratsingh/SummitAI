import json
import uuid
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional
import os
import uuid
from pathlib import Path
from pydantic import BaseModel
import aiofiles
from io import BytesIO
from elevenlabs.client import ElevenLabs
from google import genai
import socketio

load_dotenv()   

ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ai_client = genai.Client(api_key=GEMINI_API_KEY)
elevenlabs = ElevenLabs(
    api_key=ELEVEN_API_KEY,
)

app = FastAPI()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

socket_app = socketio.ASGIApp(sio)

app.mount("/socket.io", socket_app)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
# rooms: Dict[str, Dict[str, WebSocket]] = {}
rooms = {}

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def join(sid, data):
    room = data["room"]
    if room not in rooms:
        rooms[room] = set()
    for peer_sid in rooms[room]:
        await sio.emit("new-peer", {"peer": sid}, to=peer_sid)
        await sio.emit("existing-peer", {"peer": peer_sid}, to=sid)
    rooms[room].add(sid)
    print(f"{sid} joined {room}")

@sio.event
async def signal(sid, data):
    """Forward SDP or ICE messages."""
    target = data["target"]
    await sio.emit("signal", {"from": sid, "data": data["data"]}, to=target)

@sio.event
async def chat(sid, data):
    """Forward chat message to everyone in the room."""
    room = data["room"]
    msg = data["message"]
    for peer in rooms.get(room, []):
        await sio.emit("chat", {"from": sid, "message": msg}, to=peer)

@sio.event
async def disconnect(sid):
    for room, members in rooms.items():
        if sid in members:
            members.remove(sid)
            for peer_sid in members:
                await sio.emit("peer-left", {"peer": sid}, to=peer_sid)
            break
    print(f"Client disconnected: {sid}")

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]

    if room_id not in rooms:
        rooms[room_id] = {}

    room = rooms[room_id]

    # Reject if room full (2 clients max)
    if len(room) >= 2:
        await websocket.send_text(json.dumps({"type": "room_full"}))
        await websocket.close()
        return

    room[client_id] = websocket
    print(f"[{room_id}] client {client_id} connected (clients={len(room)})")

    try:
        await websocket.send_text(json.dumps({
            "type": "joined",
            "client_id": client_id,
            "peers": [cid for cid in room.keys() if cid != client_id]
        }))

        for cid, peer_ws in room.items():
            if cid != client_id:
                await peer_ws.send_text(json.dumps({
                    "type": "peer_joined",
                    "client_id": client_id
                }))

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get("type")
            to = message.get("to")

            if to:
                target_ws = room.get(to)
                if target_ws:
                    await target_ws.send_text(json.dumps(message))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "target_not_found"
                    }))
            else:
                for cid, peer_ws in room.items():
                    if cid != client_id:
                        await peer_ws.send_text(json.dumps(message))

    except WebSocketDisconnect:
        print(f"[{room_id}] client {client_id} disconnected")
    finally:
        try:
            del room[client_id]
        except Exception:
            pass
        for cid, peer_ws in list(room.items()):
            try:
                await peer_ws.send_text(json.dumps({
                    "type": "peer_left",
                    "client_id": client_id
                }))
            except Exception:
                pass
        if not room:
            del rooms[room_id]


# tempstore
STORAGE_DIR = Path("./meetings_data")
STORAGE_DIR.mkdir(exist_ok=True)

# Simple in-memory meeting index (use DB for production)
meetings = {}  # meeting_id -> { "chunks": [paths], "chats":[{...}], "participants":[], ... }


class ChatMessage(BaseModel):
    meeting_id: str
    sender: str
    text: str
    ts: float

@app.post("/ai/upload_chunk/")
async def upload_chunk(meeting_id: str = Form(...), chunk: UploadFile = File(...)):
    meeting_dir = STORAGE_DIR / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)
    chunk_id = str(uuid.uuid4()) + "_" + chunk.filename
    dest = meeting_dir / chunk_id
    async with aiofiles.open(dest, "wb") as f:
        content = await chunk.read()
        await f.write(content)

    meetings.setdefault(meeting_id, {}).setdefault("chunks", []).append(str(dest))
    return {"status": "ok", "path": str(dest)}

@app.post("/ai/log_chat/")
async def log_chat(msg: ChatMessage):
    meetings.setdefault(msg.meeting_id, {}).setdefault("chats", []).append(msg.dict())
    return {"status": "ok"}

@app.post("/ai/finalize/")
async def finalize_meeting(meeting_id: str, title: Optional[str] = None):
    """
    Called at meeting end. This will:
    - concatenate uploaded audio chunks into one file
    - call transcription (OpenAI)
    - call LLM summarization on transcript + chat log
    - return summary
    """
    if meeting_id not in meetings:
        raise HTTPException(status_code=404, detail="meeting not found")

    meeting = meetings[meeting_id]
    chunks = meeting.get("chunks", [])
    chats = meeting.get("chats", [])

    if not chunks:
        raise HTTPException(status_code=400, detail="no audio chunks uploaded")

    meeting_dir = STORAGE_DIR / meeting_id
    concatenated = meeting_dir / f"{meeting_id}_full.webm"  # assuming webm audio chunks
    # naive concatenation: for webm/ogg we should re-mux; for simplicity, we'll just join bytes for demo
    # PRODUCTION: use ffmpeg to concatenate properly
    with open(concatenated, "wb") as outfile:
        for p in chunks:
            with open(p, "rb") as infile:
                outfile.write(infile.read())

    transcript_text = await transcribe_with_elevenlabs(concatenated)

    prompt = build_summary_prompt(title or f"Meeting {meeting_id}", transcript_text, chats)

    summary = await summarize_with_gemini(prompt)

    meetings[meeting_id]["transcript"] = transcript_text
    meetings[meeting_id]["summary"] = summary

    return {"status": "ok", "summary": summary, "transcript": transcript_text}

async def transcribe_with_elevenlabs(audio_path: Path) -> str:
    """
    Use ElevenLabs Speech-to-Text for transcription.
    """

    with open(audio_path, "rb") as f:
        audio_data = BytesIO(f.read())
        transcription = elevenlabs.speech_to_text.convert(
            file=audio_data,
            model_id="scribe_v1", 
            tag_audio_events=True, 
            language_code="eng",
            diarize=True,
        )

    return transcription

async def summarize_with_gemini(prompt: str) -> str:
    """
    Summarize using Google Gemini.
    """
    response = ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.candidates[0].content.parts[0].text.strip()


def build_summary_prompt(title: str, transcript: str, chats: List[dict]) -> str:
    chat_text = "\n".join([f"[{c['ts']}] {c['sender']}: {c['text']}" for c in chats])
    prompt = f"""
Meeting Title: {title}

Transcript:
{transcript}

Chat messages:
{chat_text}

Please produce:
1) 3-5 sentence high-level summary.
2) Key decisions (bullet points).
3) Action items with owner (use 'Unknown' if not in chat) and due date if mentioned.
4) Important timestamps (if speaker mentioned a time) — include short quotes.
5) A short list of follow-up topics and suggested next steps.

Return the response in clear sections labeled exactly as:
SUMMARY, DECISIONS, ACTION ITEMS, TIMESTAMPS, NEXT STEPS.
"""
    return prompt



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)