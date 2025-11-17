from asyncore import loop
import datetime
import json
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional
import os
from pathlib import Path
from pydantic import BaseModel
import aiofiles
from io import BytesIO
from google import genai
import socketio
import asyncio
import subprocess
import whisper
import tempfile
import subprocess
from motor.motor_asyncio import AsyncIOMotorClient
from botocore.client import Config
import boto3
from botocore.auth import S3SigV4Auth
from botocore.awsrequest import AWSRequest
import requests
from datetime import datetime


load_dotenv()

# ---- MongoDB setup ----
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "summitai")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[MONGO_DB]

# ---- TEBI setup ----
TEBI_ACCESS_KEY = os.getenv("TEBI_ACCESS_KEY")
TEBI_SECRET_KEY = os.getenv("TEBI_SECRET_KEY")
TEBI_BUCKET = os.getenv("TEBI_BUCKET")
TEBI_ENDPOINT = os.getenv("TEBI_ENDPOINT")

s3 = boto3.client(
    "s3",
    aws_access_key_id=TEBI_ACCESS_KEY,
    aws_secret_access_key=TEBI_SECRET_KEY,
    endpoint_url=TEBI_ENDPOINT,
    config=Config(signature_version="s3v4")
)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ai_client = genai.Client(api_key=GEMINI_API_KEY)

print("Loading Whisper model...")
whisper_model = whisper.load_model("small")

app = FastAPI()

# Add CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=[])
socket_app = socketio.ASGIApp(sio, socketio_path="")

app.mount("/socket.io", socket_app)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

rooms = {}


async def upload_to_tebi(file_path: Path, dest_key: str) -> str:
    data = file_path.read_bytes()
    content_length = len(data)

    url = f"{TEBI_ENDPOINT.rstrip('/')}/{TEBI_BUCKET}/{dest_key}"

    # Create a signed request manually
    request = AWSRequest(
        method="PUT",
        url=url,
        data=data,
        headers={"Content-Length": str(content_length)}
    )
    S3SigV4Auth(
        credentials=s3._request_signer._credentials,
        service_name="s3",
        region_name="us-east-1" 
    ).add_auth(request)

    resp = requests.put(
        url,
        data=data,
        headers=dict(request.headers),
        timeout=60
    )

    if not resp.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Tebi upload failed ({resp.status_code}): {resp.text}"
        )

    return url



# ---------------- SOCKET.IO EVENTS ---------------- #

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
    target = data["target"]
    await sio.emit("signal", {"from": sid, "data": data["data"]}, to=target)

@sio.event
async def chat(sid, data):
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



def has_header(path: str) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0

# ---------------- WEBSOCKET HANDLER ---------------- #

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]

    if room_id not in rooms:
        rooms[room_id] = {}

    room = rooms[room_id]

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


# ---------------- STORAGE & MEETING LOGIC ---------------- #

STORAGE_DIR = Path("./meetings_data")
STORAGE_DIR.mkdir(exist_ok=True)
meetings = {}  # structure- meeting_id -> { "chunks": [paths], "chats": [...] }

class ChatMessage(BaseModel):
    meeting_id: str
    sender: str
    text: str
    ts: float


@app.post("/ai/upload_chunk/")
async def upload_chunk(meeting_id: str = Form(...), chunk: UploadFile = File(...)):
    meeting_dir = STORAGE_DIR / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)

    chunk_id = f"{uuid.uuid4()}_{chunk.filename}"
    local_path = meeting_dir / chunk_id

    async with aiofiles.open(local_path, "wb") as f:
        await f.write(await chunk.read())

    # Upload to Tebi
    s3_key = f"meetings/{meeting_id}/chunks/{chunk_id}"
    tebi_url = await upload_to_tebi(local_path, s3_key)

    # Save meeet data in MongoDB
    await db.chunks.insert_one({
        "meeting_id": meeting_id,
        "chunk_id": chunk_id,
        "s3_url": tebi_url,
        "created_at": datetime.utcnow()
    })

    # Link to meeting record
    await db.meetings.update_one(
        {"meeting_id": meeting_id},
        {"$push": {"chunks": {"chunk_id": chunk_id, "url": tebi_url}}},
        upsert=True
    )

    return {"status": "ok", "url": tebi_url}


@app.post("/ai/log_chat/")
async def log_chat(msg: ChatMessage):
    chat_doc = msg.dict()
    chat_doc["created_at"] = datetime.utcnow()

    await db.chats.insert_one(chat_doc)
    await db.meetings.update_one(
        {"meeting_id": msg.meeting_id},
        {"$push": {"chats": chat_doc}},
        upsert=True
    )

    return {"status": "ok"}


def trim_silence(input_wav, output_wav):
    # removes initial silence which breaks whisper
    subprocess.run([
        "ffmpeg","-y",
        "-i", str(input_wav),
        "-af", "silenceremove=start_periods=1:start_duration=0.3:start_threshold=-50dB",
        str(output_wav)
    ])



@app.post("/ai/finalize/")
async def finalize_meeting(meeting_id: str, title: Optional[str] = None):
    # ---- Fetch meeting data ----
    meeting = await db.meetings.find_one({"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="meeting not found")

    chunks = meeting.get("chunks", [])
    chats = meeting.get("chats", [])

    if not chunks:
        raise HTTPException(status_code=400, detail="no audio chunks uploaded")

    meeting_dir = STORAGE_DIR / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)

    # ---- Download chunks from Tebi ----
    downloaded_chunks = []
    for ch in chunks:
        chunk_id = ch["chunk_id"]
        local_path = meeting_dir / chunk_id
        s3.download_file(TEBI_BUCKET, f"meetings/{meeting_id}/chunks/{chunk_id}", str(local_path))
        downloaded_chunks.append(str(local_path))

    def ts(p):
        name = Path(p).name
        try:
            return int(name.split("_chunk_")[1].split(".")[0])
        except Exception:
            return 0

    downloaded_chunks = sorted(downloaded_chunks, key=ts)

    header = None
    bodies = []
    for p in downloaded_chunks:
        if header is None and has_header(p):
            header = p
        else:
            bodies.append(p)

    if header is None:
        raise HTTPException(500, "no valid header webm chunk found")

    joined_path = meeting_dir / f"{meeting_id}_joined.webm"
    with open(joined_path, "wb") as out:
        with open(header, "rb") as f:
            out.write(f.read())
        for p in bodies:
            with open(p, "rb") as f:
                out.write(f.read())

    wav_path = meeting_dir / f"{meeting_id}_final.wav"
    convert_cmd = [
        "ffmpeg", "-y",
        "-i", str(joined_path),
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz sample rate
        "-f", "wav",
        str(wav_path)
    ]
    subprocess.run(convert_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    trimmed = meeting_dir / f"{meeting_id}_trimmed.wav"
    trim_silence(wav_path, trimmed)
    print(f">>> Using trimmed WAV ({trimmed}): {os.path.getsize(trimmed)} bytes")

    transcript_text = await transcribe_with_whisper(trimmed)

    prompt = build_summary_prompt(title or f"Meeting {meeting_id}", transcript_text, chats)
    summary = await summarize_with_gemini(prompt)

    # ---- Upload processed audio to Tebi ----
    final_audio_key = f"meetings/{meeting_id}/final/{meeting_id}_final.wav"
    tebi_url = await upload_to_tebi(trimmed, final_audio_key)

    # ---- Update MongoDB record ----
    update_doc = {
        "meeting_id": meeting_id,
        "title": title or f"Meeting {meeting_id}",
        "transcript": transcript_text,
        "summary": summary,
        "final_audio_url": tebi_url,
        "updated_at": datetime.utcnow(),
        "status": "finalized"
    }
    await db.meetings.update_one(
        {"meeting_id": meeting_id},
        {"$set": update_doc},
        upsert=True
    )

    return {
        "status": "ok",
        "summary": summary,
        "transcript": transcript_text,
        "audio_url": tebi_url
    }



# ---------------- AI HELPERS ---------------- #

async def transcribe_with_whisper(audio_path: Path) -> str:
    """
    Transcribe audio using OpenAI Whisper (local).
    Supports .webm, .wav, .mp3, etc.
    """
    print(f"Transcribing {audio_path}...")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: whisper_model.transcribe(str(audio_path)))

    return result["text"].strip()


async def summarize_with_gemini(prompt: str) -> str:
    """
    Summarize using Google Gemini.
    """
    response = ai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.candidates[0].content.parts[0].text.strip()


def build_summary_prompt(title: str, transcript: str, chats: List[dict]) -> str:
    chat_text = "\n".join([f"[{c['ts']}] {c['sender']}: {c['text']}" for c in chats])
    return f"""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
