import json
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from typing import Dict
import os
import uuid
import shutil
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import aiofiles
import requests

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

rooms: Dict[str, Dict[str, WebSocket]] = {}

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

    transcript_text = await transcribe_with_openai(concatenated)

    prompt = build_summary_prompt(title or f"Meeting {meeting_id}", transcript_text, chats)

    summary = await summarize_with_openai(prompt)

    meetings[meeting_id]["transcript"] = transcript_text
    meetings[meeting_id]["summary"] = summary

    return {"status": "ok", "summary": summary, "transcript": transcript_text}

async def transcribe_with_openai(audio_path: Path) -> str:
    """
    Upload audio file to OpenAI's transcription endpoint.
    This is a simplified example using requests; you can use official SDK.
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/webm")}
        data = {"model": "whisper-1", "language": "en"}
        resp = requests.post(url, headers=headers, files=files, data=data)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"transcription failed: {resp.text}")
    result = resp.json()
    # Common response: {"text": "transcribed text ..."}
    return result.get("text", "")

async def summarize_with_openai(prompt: str) -> str:
    """
    Call the LLM (chat completion) to produce meeting minutes.
    Adjust to the model you have access to.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini", 
        "messages": [
            {"role": "system", "content": "You are an assistant that converts raw meeting transcripts and chat logs into clear meeting minutes with action items."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.1
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"summary failed: {resp.text}")
    data = resp.json()
    # extract first assistant message
    return data["choices"][0]["message"]["content"].strip()

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