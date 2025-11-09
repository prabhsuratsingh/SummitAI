from asyncore import loop
import json
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
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
import time

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ai_client = genai.Client(api_key=GEMINI_API_KEY)

print("Loading Whisper model...")
whisper_model = whisper.load_model("small")

app = FastAPI()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio)

app.mount("/socket.io", socket_app)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

rooms = {}

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
    chunk_id = str(uuid.uuid4()) + "_" + chunk.filename
    dest = meeting_dir / chunk_id

    async with aiofiles.open(dest, "wb") as f:
        await f.write(await chunk.read())

    meetings.setdefault(meeting_id, {}).setdefault("chunks", []).append(str(dest))
    return {"status": "ok", "path": str(dest)}


@app.post("/ai/log_chat/")
async def log_chat(msg: ChatMessage):
    meetings.setdefault(msg.meeting_id, {}).setdefault("chats", []).append(msg.dict())
    return {"status": "ok"}


def trim_silence(input_wav, output_wav):
    # removes initial silence which breaks whisper
    subprocess.run([
        "ffmpeg","-y",
        "-i", str(input_wav),
        "-af", "silenceremove=start_periods=1:start_duration=0.3:start_threshold=-50dB",
        str(output_wav)
    ])

# Modified for benchmarking
@app.post("/ai/finalize/")
async def finalize_meeting(meeting_id: str, title: Optional[str] = None):
    if meeting_id not in meetings:
        raise HTTPException(status_code=404, detail="meeting not found")

    meeting = meetings[meeting_id]
    chunks = meeting.get("chunks", [])
    chats = meeting.get("chats", [])

    if not chunks:
        raise HTTPException(status_code=400, detail="no audio chunks uploaded")

    def ts(p):
        name = Path(p).name
        return int(name.split("_chunk_")[1].split(".")[0])

    chunks = sorted(chunks, key=ts)

    header = None
    bodies = []
    for p in chunks:
        if header is None and has_header(p):
            header = p
        else:
            bodies.append(p)

    if header is None:
        raise HTTPException(500, "no valid header webm chunk found")

    meeting_dir = STORAGE_DIR / meeting_id
    concatenated = meeting_dir / f"{meeting_id}_full.webm"

    # ---- Proper concatenation using ffmpeg ----
    concat_list = meeting_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for p in chunks:
            f.write(f"file '{Path(p).resolve()}'\n")

    joined_path = meeting_dir / f"{meeting_id}_joined.webm"

    with open(joined_path, "wb") as out:
        with open(header, "rb") as f:
            out.write(f.read())
        for p in bodies:
            with open(p, "rb") as f:
                out.write(f.read())

    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(concatenated)
    ]
    subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ---- Convert concatenated WebM → WAV (mono 16 kHz) ----
    wav_path = meeting_dir / f"{meeting_id}_final.wav"
    convert_cmd = [
        "ffmpeg", "-y",
        "-i", str(joined_path),
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(wav_path)
    ]
    subprocess.run(convert_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ---- Trim silence ----
    trimmed = meeting_dir / f"{meeting_id}_trimmed.wav"
    trim_silence(wav_path, trimmed)
    print(">>> using wav:", trimmed, os.path.getsize(trimmed))

    # ---- Measure internal timings ----
    timing_log = Path("./bench_results/internal_timings.csv")
    timing_log.parent.mkdir(exist_ok=True)
    if not timing_log.exists():
        with open(timing_log, "w") as f:
            f.write("meeting_id,phase,time_s\n")

    # FFmpeg already done above, but we can mark the timing:
    ffmpeg_t = 0.0  # optionally measure the concat/convert time if you wish

    start_whisper = time.perf_counter()
    transcript_text = await transcribe_with_whisper(trimmed)
    whisper_t = time.perf_counter() - start_whisper

    # ---- Build prompt BEFORE timing Gemini ----
    prompt = build_summary_prompt(title or f"Meeting {meeting_id}", transcript_text, chats)

    start_gemini = time.perf_counter()
    summary = await summarize_with_gemini(prompt)
    gemini_t = time.perf_counter() - start_gemini

    total_t = whisper_t + gemini_t

    print(f"[BENCH] Whisper={whisper_t:.2f}s Gemini={gemini_t:.2f}s Total={total_t:.2f}s")

    # Log to CSV
    with open(timing_log, "a") as f:
        f.write(f"{meeting_id},whisper,{whisper_t:.3f}\n")
        f.write(f"{meeting_id},gemini,{gemini_t:.3f}\n")
        f.write(f"{meeting_id},total,{total_t:.3f}\n")

    meetings[meeting_id]["transcript"] = transcript_text
    meetings[meeting_id]["summary"] = summary

    return {"status": "ok", "summary": summary, "transcript": transcript_text}


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
