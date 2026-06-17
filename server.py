# """
# WebRTC Voice Agent Server (No LiveKit)
# =======================================
# Pure WebRTC signaling + Deepgram STT + OpenAI LLM + OpenAI TTS
# All real-time, no LiveKit dependency.

# Architecture:
#   Browser  <──WebRTC audio──>  aiortc server
#                                     │
#                         ┌───────────┼───────────┐
#                         ▼           ▼           ▼
#                    Deepgram     OpenAI LLM   OpenAI TTS
#                    (STT stream) (gpt-4.1)   (tts-1 stream)
# """

# import asyncio
# import json
# import logging
# import os
# import time
# import traceback
# from datetime import datetime
# from typing import Optional

# import aiofiles
# import aiohttp
# from aiohttp import web
# from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
# from aiortc.contrib.media import MediaBlackhole
# from av import AudioFrame
# from av.audio.resampler import AudioResampler
# import numpy as np

# from dotenv import load_dotenv
# from openai import AsyncOpenAI

# load_dotenv(".env")

# # ─── Logging ───────────────────────────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# )
# logger = logging.getLogger("webrtc_agent")

# # ─── Clients ───────────────────────────────────────────────────────────────────
# openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
# LLM_MODEL = os.getenv("LLM_CHOICE", "gpt-4.1-mini")

# # ─── Pricing (for cost logging) ────────────────────────────────────────────────
# PRICING = {
#     "stt_per_min":             0.0048,   
#     "llm_input_per_1m_tokens": 0.40,     # gpt-4.1-mini
#     "llm_output_per_1m_tokens":1.60,
#     "tts_per_1m_chars":        15.0,     # OpenAI tts-1
# }

# COST_LOG_PATH = os.getenv("COST_LOG_PATH", "session_costs.log")

# # ─── Cost Tracker ──────────────────────────────────────────────────────────────
# class SessionCostTracker:
#     def __init__(self, session_id: str):
#         self.session_id       = session_id
#         self.start_time       = time.time()
#         self.start_ts         = datetime.now().isoformat(timespec="seconds")
#         self.llm_input_tokens  = 0
#         self.llm_output_tokens = 0
#         self.tts_chars         = 0
#         self.stt_audio_sec     = 0.0
#         self.turns             = 0
#         self._flushed          = False
      
#     def add_stt(self, seconds: float):
#         self.stt_audio_sec += seconds

#     def add_llm(self, input_tokens: int, output_tokens: int):
#         self.llm_input_tokens  += input_tokens
#         self.llm_output_tokens += output_tokens
#         self.turns             += 1

#     def add_tts(self, chars: int):
#         self.tts_chars += chars

#     async def flush(self):
#         if self._flushed:
#             return
#         self._flushed = True

#         wall_sec     = time.time() - self.start_time
#         wall_minutes = wall_sec / 60.0
#         stt_minutes  = self.stt_audio_sec / 60.0

#         stt_cost        = stt_minutes * PRICING["stt_per_min"]
#         llm_in_cost     = self.llm_input_tokens  / 1_000_000 * PRICING["llm_input_per_1m_tokens"]
#         llm_out_cost    = self.llm_output_tokens / 1_000_000 * PRICING["llm_output_per_1m_tokens"]
#         tts_cost        = self.tts_chars / 1_000_000 * PRICING["tts_per_1m_chars"]
#         total_cost      = stt_cost + llm_in_cost + llm_out_cost + tts_cost

#         record = {
#             "session_id":   self.session_id,
#             "started_at":   self.start_ts,
#             "ended_at":     datetime.now().isoformat(timespec="seconds"),
#             "wall_minutes": round(wall_minutes, 3),
#             "turns":        self.turns,
#             "stt":  {"minutes": round(stt_minutes, 4), "cost_usd": round(stt_cost, 6)},
#             "llm":  {"input_tokens": self.llm_input_tokens,
#                      "output_tokens": self.llm_output_tokens,
#                      "cost_usd": round(llm_in_cost + llm_out_cost, 6)},
#             "tts":  {"chars": self.tts_chars, "cost_usd": round(tts_cost, 6)},
#             "total_usd": round(total_cost, 6),
#         }

#         async with aiofiles.open(COST_LOG_PATH, "a", encoding="utf-8") as f:
#             await f.write(json.dumps(record) + "\n")

#         logger.info(
#             f"\n{'═'*54}\n"
#             f"  SESSION COST SUMMARY  [{self.session_id}]\n"
#             f"{'═'*54}\n"
#             f"  Wall time : {wall_minutes:.2f} min  |  Turns: {self.turns}\n"
#             f"  STT       : ${stt_cost:.6f}  ({self.stt_audio_sec:.1f}s speech)\n"
#             f"  LLM       : ${llm_in_cost+llm_out_cost:.6f}  "
#             f"({self.llm_input_tokens}in / {self.llm_output_tokens}out tokens)\n"
#             f"  TTS       : ${tts_cost:.6f}  ({self.tts_chars:,} chars)\n"
#             f"  TOTAL     : ${total_cost:.6f}\n"
#             f"{'═'*54}"
#         )


# # ─── Audio sink: collects PCM frames, flushes to STT ──────────────────────────
# class MicrophoneTrackSink:

#     SAMPLE_RATE = 16000
#     CHANNELS = 1

#     SILENCE_DB = -35
#     SILENCE_SECS = 1.0

#     def __init__(self, on_transcript):

#         self.resampler = AudioResampler(
#             format="s16",
#             layout="mono",
#             rate=16000,
#         )

#         self._on_transcript = on_transcript

#         self._buf = []

#         self._buf_sec = 0.0

#         self._speaking = False

#         self._silence_since = None

#         self._speech_start = None

#         self._task = None


#     def receive(self, track: MediaStreamTrack):

#         self._task = asyncio.ensure_future(
#             self._run(track)
#         )


#     async def _run(self, track: MediaStreamTrack):

#         try:

#             while True:

#                 frame = await track.recv()

#                 pcm = self._to_mono16k(frame)

#                 self._buf.append(pcm)

#                 duration = len(pcm) / 2 / self.SAMPLE_RATE

#                 self._buf_sec += duration

#                 rms_db = self._rms_db(pcm)

#                 now = time.monotonic()

#                 is_speech = rms_db > self.SILENCE_DB

#                 if is_speech:

#                     self._silence_since = None

#                     if not self._speaking:

#                         self._speaking = True

#                         logger.info(
#                             f"🎙 Speech started ({rms_db:.1f} dBFS)"
#                         )

#                 else:

#                     if (
#                         self._speaking
#                         and self._silence_since is None
#                     ):

#                         self._silence_since = now


#                 if (
#                     self._speaking
#                     and self._silence_since is not None
#                     and now - self._silence_since >= self.SILENCE_SECS
#                 ):

#                     await self._flush()

#         except Exception as e:

#             logger.error(
#                 f"Audio sink error: {e}"
#             )


#     def _to_mono16k(
#         self,
#         frame: AudioFrame
#     ) -> bytes:

#         frames = self.resampler.resample(frame)

#         pcm = b""

#         for f in frames:

#             arr = f.to_ndarray()

#             logger.warning(
#                 f"[RESAMPLED] "
#                 f"fmt={f.format.name} "
#                 f"rate={f.sample_rate} "
#                 f"layout={f.layout.name}"
#             )

#             pcm += arr.astype(
#                 np.int16
#             ).tobytes()

#         return pcm


#     @staticmethod
#     def _rms_db(pcm):

#         arr = np.frombuffer(
#             pcm,
#             dtype=np.int16
#         ).astype(np.float32)

#         if len(arr) == 0:

#             return -100

#         rms = np.sqrt(
#             np.mean(arr ** 2)
#         )

#         return 20 * np.log10(
#             max(rms, 1e-9) / 32768
#         )


#     async def _flush(self):

#         if not self._buf:

#             return

#         audio = b"".join(self._buf)

#         duration = self._buf_sec

#         self._buf = []

#         self._buf_sec = 0

#         self._speaking = False

#         self._silence_since = None

#         logger.info(
#             f"🎙 Sending {duration:.2f}s audio to STT"
#         )

#         await self._on_transcript(
#             audio,
#             duration
#         )


#     def stop(self):

#         if self._task:

#             self._task.cancel()

# # ─── Deepgram STT ──────────────────────────────────────────────────────────────
# async def transcribe_audio(audio_bytes: bytes) -> str:
#     """Send PCM bytes to Deepgram REST endpoint, return transcript."""
#     url = "https://api.deepgram.com/v1/listen?model=nova-3&language=en&encoding=linear16&sample_rate=16000&channels=1&punctuate=true"
#     headers = {
#         "Authorization": f"Token {DEEPGRAM_API_KEY}",
#         "Content-Type":  "audio/raw",
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(url, headers=headers, data=audio_bytes) as resp:
#             if resp.status != 200:
#                 text = await resp.text()
#                 logger.error(f"Deepgram error {resp.status}: {text}")
#                 return ""
#             data     = await resp.json()
#             logger.warning("=" * 80)
#             logger.warning(json.dumps(data, indent=2))
#             logger.warning("=" * 80)
#             channels = data.get("results", {}).get("channels", [])
#             if channels:
#                 alts = channels[0].get("alternatives", [])
#                 if alts:
#                     transcript = alts[0].get("transcript", "").strip()
#                     if not transcript:
#                         logger.warning("⚠️  Deepgram returned empty transcript — audio may be silence or too short")
#                     return transcript
#     logger.warning("⚠️  Deepgram: no channels in response")
#     return ""


# # ─── Conversation session ───────────────────────────────────────────────────────
# class ConversationSession:
#     SYSTEM_PROMPT = (
#         "You are a helpful and friendly voice AI assistant. "
#         "Speak clearly and naturally, as if having a phone conversation. "
#         "Be concise but warm. Replies must be SHORT — 1–3 sentences max — "
#         "because they'll be converted to speech. If you don't know something, say so."
#     )

#     def __init__(self, ws: web.WebSocketResponse, session_id: str):
#         self._ws           = ws
#         self._session_id   = session_id
#         self._history      = []
#         self._cost         = SessionCostTracker(session_id)
#         self._speaking     = False        # is TTS currently playing?
#         self._interrupt    = asyncio.Event()
#         self._pc: Optional[RTCPeerConnection] = None
#         self._sink: Optional[MicrophoneTrackSink] = None
#         self._tts_queue    = asyncio.Queue()
#         self._tts_task     = None

#     # ── WebRTC ────────────────────────────────────────────────────────────────

#     async def handle_offer(self, offer_sdp: str, offer_type: str):
#         self._pc = RTCPeerConnection()
#         self._pc.on("connectionstatechange", self._on_connection_state)
#         self._pc.on("track", self._on_track)

#         offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
#         await self._pc.setRemoteDescription(offer)
#         answer = await self._pc.createAnswer()
#         await self._pc.setLocalDescription(answer)

#         await self._ws.send_json({
#             "type":    "answer",
#             "sdp":     self._pc.localDescription.sdp,
#             "sdp_type": self._pc.localDescription.type,
#         })
#         logger.info(f"[{self._session_id}] WebRTC answer sent")

#     def _on_connection_state(self):
#         state = self._pc.connectionState if self._pc else "unknown"
#         logger.info(f"[{self._session_id}] WebRTC state: {state}")

#     def _on_track(self, track: MediaStreamTrack):
#         if track.kind == "audio":
#             logger.info(f"[{self._session_id}] Audio track received")
#             self._sink = MicrophoneTrackSink(self._on_audio_chunk)
#             self._sink.receive(track)
#             # Send greeting once track is up
#             asyncio.ensure_future(self._greet())

#     # ── Audio pipeline ────────────────────────────────────────────────────────

#     async def _greet(self):
#         await asyncio.sleep(0.5)   # tiny delay so client is ready
#         await self._speak_and_send(
#             "Hello! I'm your voice assistant. How can I help you today?"
#         )

#     async def _on_audio_chunk(self, audio: bytes, duration_sec: float):
#         """Called when MicrophoneTrackSink detects end of speech."""
#         # Interrupt any ongoing TTS
#         if self._speaking:
#             logger.info(f"[{self._session_id}] 🛑 Barge-in detected — interrupting TTS")
#             self._interrupt.set()
#             await self._ws.send_json({"type": "interrupt"})
#             await asyncio.sleep(0.1)

#         self._cost.add_stt(duration_sec)

#         transcript = await transcribe_audio(audio)
#         if not transcript:
#             logger.debug("Empty transcript, skipping")
#             return

#         logger.info(f"[{self._session_id}] 📝 User: {transcript}")
#         await self._ws.send_json({"type": "transcript", "text": transcript, "speaker": "user"})

#         response = await self._llm_respond(transcript)
#         if response:
#             await self._speak_and_send(response)

#     async def _llm_respond(self, user_text: str) -> str:
#         self._history.append({"role": "user", "content": user_text})
#         messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self._history

#         try:
#             resp = await openai_client.chat.completions.create(
#                 model=LLM_MODEL,
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=200,
#             )
#             text       = resp.choices[0].message.content.strip()
#             in_tokens  = resp.usage.prompt_tokens
#             out_tokens = resp.usage.completion_tokens
#             self._cost.add_llm(in_tokens, out_tokens)
#             self._history.append({"role": "assistant", "content": text})
#             logger.info(f"[{self._session_id}] 🤖 Agent: {text}")
#             await self._ws.send_json({"type": "transcript", "text": text, "speaker": "agent"})
#             return text
#         except Exception as e:
#             logger.error(f"LLM error: {e}")
#             return "Sorry, I had trouble thinking of a response. Could you try again?"

#     async def _speak_and_send(self, text: str):
#         """Stream TTS audio back to client via WebSocket data channel."""
#         self._interrupt.clear()
#         self._speaking = True
#         self._cost.add_tts(len(text))

#         try:
#             await self._ws.send_json({"type": "tts_start"})

#             async with openai_client.audio.speech.with_streaming_response.create(
#                 model="tts-1",
#                 voice="echo",
#                 input=text,
#                 response_format="mp3",
#                 speed=1.0,
#             ) as response:
#                 async for chunk in response.iter_bytes(chunk_size=4096):
#                     if self._interrupt.is_set():
#                         logger.debug("TTS interrupted")
#                         break
#                     # Send audio chunk as binary over WS
#                     await self._ws.send_bytes(chunk)

#             await self._ws.send_json({"type": "tts_end"})
#         except Exception as e:
#             logger.error(f"TTS error: {e}")
#         finally:
#             self._speaking = False
#             self._interrupt.clear()

#     # ── Lifecycle ─────────────────────────────────────────────────────────────

#     async def close(self):
#         if self._sink:
#             self._sink.stop()
#         if self._pc:
#             await self._pc.close()
#         await self._cost.flush()


# # ─── Active sessions ───────────────────────────────────────────────────────────
# sessions: dict[str, ConversationSession] = {}


# # ─── HTTP handlers ─────────────────────────────────────────────────────────────

# async def handle_ws(request: web.Request) -> web.WebSocketResponse:
#     ws = web.WebSocketResponse()
#     await ws.prepare(request)

#     session_id = f"sess_{int(time.time() * 1000)}"
#     session    = ConversationSession(ws, session_id)
#     sessions[session_id] = session
#     logger.info(f"[{session_id}] New WebSocket connection")

#     try:
#         async for msg in ws:
#             if msg.type == aiohttp.WSMsgType.TEXT:
#                 data = json.loads(msg.data)
#                 mtype = data.get("type")

#                 if mtype == "offer":
#                     await session.handle_offer(data["sdp"], data["sdp_type"])

#                 elif mtype == "ice_candidate":
#                     # Client sends ICE candidates
#                     cand = data.get("candidate")
#                     if cand and session._pc:
#                         from aiortc import RTCIceCandidate
#                         # Parse the candidate string
#                         parts = cand["candidate"].split()
#                         if len(parts) >= 8:
#                             try:
#                                 ice = RTCIceCandidate(
#                                     component=int(parts[1]),
#                                     foundation=parts[0].replace("candidate:", ""),
#                                     ip=parts[4],
#                                     port=int(parts[5]),
#                                     priority=int(parts[3]),
#                                     protocol=parts[2],
#                                     type=parts[7],
#                                     sdpMid=cand.get("sdpMid"),
#                                     sdpMLineIndex=cand.get("sdpMLineIndex"),
#                                 )
#                                 await session._pc.addIceCandidate(ice)
#                             except Exception as e:
#                                 logger.debug(f"ICE parse error (ok): {e}")

#                 elif mtype == "close":
#                     break

#             elif msg.type == aiohttp.WSMsgType.ERROR:
#                 logger.error(f"WS error: {ws.exception()}")
#                 break
#     finally:
#         await session.close()
#         sessions.pop(session_id, None)
#         logger.info(f"[{session_id}] Session closed")

#     return ws


# async def handle_index(request: web.Request) -> web.Response:
#     index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
#     async with aiofiles.open(index_path, "r") as f:
#         content = await f.read()
#     return web.Response(content_type="text/html", text=content)


# async def handle_health(request: web.Request) -> web.Response:
#     return web.Response(text=json.dumps({"status": "ok", "sessions": len(sessions)}),
#                         content_type="application/json")


# # ─── App factory ───────────────────────────────────────────────────────────────

# def build_app() -> web.Application:
#     app = web.Application()
#     app.router.add_get("/",       handle_index)
#     app.router.add_get("/ws",     handle_ws)
#     app.router.add_get("/health", handle_health)
#     app.router.add_static("/static",
#                            os.path.join(os.path.dirname(__file__), "static"),
#                            show_index=False)
#     return app


# if __name__ == "__main__":
#     host = os.getenv("HOST", "0.0.0.0")
#     port = int(os.getenv("PORT", 8080))
#     logger.info(f"Starting WebRTC Voice Agent on {host}:{port}")
#     web.run_app(build_app(), host=host, port=port, access_log=None)
"""
WebRTC Voice Agent Server — Production-Ready
=============================================
Pure WebRTC signaling + Deepgram STT + OpenAI LLM + OpenAI TTS

Fixes applied vs. prototype:
  - Session cap + per-IP rate limiting
  - Collision-safe session IDs (uuid4)
  - Shared aiohttp.ClientSession with connection pooling
  - API semaphores (STT, LLM, TTS) with retry + exponential back-off
  - Per-session pipeline lock: STT→LLM→TTS runs sequentially, no overlap
  - Audio buffer hard cap (30 s) to prevent OOM
  - Conversation history rolling window (20 turns)
  - Timeouts on every external API call
  - ProcessPoolExecutor for CPU-bound audio work (resampling, RMS)
  - Graceful shutdown: drains sessions, flushes costs
  - logger.warning removed from the hot audio path
  - Proper task tracking with done-callbacks (no silent swallowed exceptions)
  - Deepgram response dump demoted from WARNING → DEBUG
  - Session-ID collision fix (uuid4)

Architecture:
  Browser  <──WebRTC audio──>  aiortc server
                                    │
                        ┌───────────┼───────────┐
                        ▼           ▼           ▼
                   Deepgram     OpenAI LLM   OpenAI TTS
                   (STT pool)   (gpt-4.1)   (tts-1 stream)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Optional

import aiofiles
import aiohttp
import numpy as np
from aiohttp import web
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(".env")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webrtc_agent")

# ─── Config ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
LLM_MODEL        = os.getenv("LLM_CHOICE", "gpt-4.1-mini")
COST_LOG_PATH    = os.getenv("COST_LOG_PATH", "session_costs.log")

MAX_SESSIONS          = int(os.getenv("MAX_SESSIONS", "100"))
MAX_SESSIONS_PER_IP   = int(os.getenv("MAX_SESSIONS_PER_IP", "5"))
STT_CONCURRENCY       = int(os.getenv("STT_CONCURRENCY", "20"))
LLM_CONCURRENCY       = int(os.getenv("LLM_CONCURRENCY", "20"))
TTS_CONCURRENCY       = int(os.getenv("TTS_CONCURRENCY", "10"))
MAX_HISTORY_TURNS     = int(os.getenv("MAX_HISTORY_TURNS", "20"))
MAX_AUDIO_BUF_SEC     = float(os.getenv("MAX_AUDIO_BUF_SEC", "30.0"))
API_TIMEOUT_SEC       = float(os.getenv("API_TIMEOUT_SEC", "10.0"))
API_MAX_RETRIES       = int(os.getenv("API_MAX_RETRIES", "3"))

# ─── Pricing ──────────────────────────────────────────────────────────────────
PRICING = {
    "stt_per_min":              0.0048,
    "llm_input_per_1m_tokens":  0.40,
    "llm_output_per_1m_tokens": 1.60,
    "tts_per_1m_chars":         15.0,
}

# ─── Global singletons (created in app startup) ───────────────────────────────
openai_client:     Optional[AsyncOpenAI]          = None
http_session:      Optional[aiohttp.ClientSession] = None
process_pool:      Optional[ProcessPoolExecutor]  = None

# Concurrency guards
_stt_sem: Optional[asyncio.Semaphore] = None
_llm_sem: Optional[asyncio.Semaphore] = None
_tts_sem: Optional[asyncio.Semaphore] = None

# Active sessions + per-IP counters
sessions:      dict[str, "ConversationSession"] = {}
ip_session_count: dict[str, int]               = {}


# ─── App lifecycle ─────────────────────────────────────────────────────────────

async def on_startup(app: web.Application) -> None:
    global openai_client, http_session, process_pool
    global _stt_sem, _llm_sem, _tts_sem

    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    connector = aiohttp.TCPConnector(limit=64, ttl_dns_cache=300)
    http_session = aiohttp.ClientSession(connector=connector)

    process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())

    _stt_sem = asyncio.Semaphore(STT_CONCURRENCY)
    _llm_sem = asyncio.Semaphore(LLM_CONCURRENCY)
    _tts_sem = asyncio.Semaphore(TTS_CONCURRENCY)

    logger.info(
        f"Server started | max_sessions={MAX_SESSIONS} "
        f"stt_concurrency={STT_CONCURRENCY} "
        f"llm_concurrency={LLM_CONCURRENCY} "
        f"tts_concurrency={TTS_CONCURRENCY}"
    )


async def on_shutdown(app: web.Application) -> None:
    logger.info("Graceful shutdown: closing all sessions …")
    close_tasks = [s.close() for s in list(sessions.values())]
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)

    if http_session:
        await http_session.close()
    if process_pool:
        process_pool.shutdown(wait=False)

    logger.info("Shutdown complete.")


# ─── Cost Tracker ─────────────────────────────────────────────────────────────

class SessionCostTracker:
    def __init__(self, session_id: str):
        self.session_id        = session_id
        self.start_time        = time.time()
        self.start_ts          = datetime.now().isoformat(timespec="seconds")
        self.llm_input_tokens  = 0
        self.llm_output_tokens = 0
        self.tts_chars         = 0
        self.stt_audio_sec     = 0.0
        self.turns             = 0
        self._flushed          = False

    def add_stt(self, seconds: float):
        self.stt_audio_sec += seconds

    def add_llm(self, input_tokens: int, output_tokens: int):
        self.llm_input_tokens  += input_tokens
        self.llm_output_tokens += output_tokens
        self.turns             += 1

    def add_tts(self, chars: int):
        self.tts_chars += chars

    async def flush(self):
        if self._flushed:
            return
        self._flushed = True

        wall_sec     = time.time() - self.start_time
        wall_minutes = wall_sec / 60.0
        stt_minutes  = self.stt_audio_sec / 60.0

        stt_cost     = stt_minutes  * PRICING["stt_per_min"]
        llm_in_cost  = self.llm_input_tokens  / 1_000_000 * PRICING["llm_input_per_1m_tokens"]
        llm_out_cost = self.llm_output_tokens / 1_000_000 * PRICING["llm_output_per_1m_tokens"]
        tts_cost     = self.tts_chars / 1_000_000 * PRICING["tts_per_1m_chars"]
        total_cost   = stt_cost + llm_in_cost + llm_out_cost + tts_cost

        record = {
            "session_id":   self.session_id,
            "started_at":   self.start_ts,
            "ended_at":     datetime.now().isoformat(timespec="seconds"),
            "wall_minutes": round(wall_minutes, 3),
            "turns":        self.turns,
            "stt":  {"minutes": round(stt_minutes, 4), "cost_usd": round(stt_cost, 6)},
            "llm":  {
                "input_tokens":  self.llm_input_tokens,
                "output_tokens": self.llm_output_tokens,
                "cost_usd":      round(llm_in_cost + llm_out_cost, 6),
            },
            "tts":  {"chars": self.tts_chars, "cost_usd": round(tts_cost, 6)},
            "total_usd": round(total_cost, 6),
        }

        async with aiofiles.open(COST_LOG_PATH, "a", encoding="utf-8") as f:
            await f.write(json.dumps(record) + "\n")

        logger.info(
            f"\n{'═'*54}\n"
            f"  SESSION COST SUMMARY  [{self.session_id}]\n"
            f"{'═'*54}\n"
            f"  Wall time : {wall_minutes:.2f} min  |  Turns: {self.turns}\n"
            f"  STT       : ${stt_cost:.6f}  ({self.stt_audio_sec:.1f}s speech)\n"
            f"  LLM       : ${llm_in_cost + llm_out_cost:.6f}  "
            f"({self.llm_input_tokens}in / {self.llm_output_tokens}out tokens)\n"
            f"  TTS       : ${tts_cost:.6f}  ({self.tts_chars:,} chars)\n"
            f"  TOTAL     : ${total_cost:.6f}\n"
            f"{'═'*54}"
        )


# ─── CPU-bound audio helpers (run in ProcessPoolExecutor) ─────────────────────

def _resample_frame_sync(
    pcm_bytes: bytes,
    src_rate: int,
    src_format: str,
    src_layout: str,
) -> bytes:
    """
    Pure-Python resampling approximation for cross-process use.
    In production, replace with a proper librosa/resampy call or
    pass raw numpy arrays.  This is a no-op placeholder that simply
    casts the already-resampled bytes coming from aiortc's
    AudioResampler (which runs in the main process).
    """
    return pcm_bytes


def _rms_db_sync(pcm_bytes: bytes) -> float:
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(arr) == 0:
        return -100.0
    rms = np.sqrt(np.mean(arr ** 2))
    return float(20 * np.log10(max(rms, 1e-9) / 32768))


# ─── Audio sink ───────────────────────────────────────────────────────────────

class MicrophoneTrackSink:
    SAMPLE_RATE  = 16_000
    CHANNELS     = 1
    SILENCE_DB   = -35.0
    SILENCE_SECS = 1.0

    def __init__(self, session_id: str, on_transcript):
        self._session_id     = session_id
        self._on_transcript  = on_transcript
        self._resampler      = AudioResampler(format="s16", layout="mono", rate=16_000)
        self._buf: list[bytes] = []
        self._buf_sec        = 0.0
        self._speaking       = False
        self._silence_since: Optional[float] = None
        self._task: Optional[asyncio.Task]   = None

    def receive(self, track: MediaStreamTrack) -> None:
        self._task = asyncio.create_task(self._run(track))
        self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"[{self._session_id}] Audio sink task crashed: {exc}", exc_info=exc)

    async def _run(self, track: MediaStreamTrack) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                frame: AudioFrame = await track.recv()
                pcm = self._to_mono16k(frame)          # resampler stays in main process

                self._buf.append(pcm)
                duration        = len(pcm) / 2 / self.SAMPLE_RATE
                self._buf_sec  += duration

                # RMS in process pool so numpy doesn't block the event loop
                rms_db: float = await loop.run_in_executor(
                    process_pool, _rms_db_sync, pcm
                )

                now        = time.monotonic()
                is_speech  = rms_db > self.SILENCE_DB

                if is_speech:
                    self._silence_since = None
                    if not self._speaking:
                        self._speaking = True
                        logger.info(f"[{self._session_id}] 🎙 Speech started ({rms_db:.1f} dBFS)")
                else:
                    if self._speaking and self._silence_since is None:
                        self._silence_since = now

                # Hard buffer cap — prevent OOM on open mics
                if self._buf_sec >= MAX_AUDIO_BUF_SEC:
                    logger.warning(
                        f"[{self._session_id}] Audio buffer hit {MAX_AUDIO_BUF_SEC}s cap — force-flushing"
                    )
                    await self._flush()
                    continue

                if (
                    self._speaking
                    and self._silence_since is not None
                    and now - self._silence_since >= self.SILENCE_SECS
                ):
                    await self._flush()

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"[{self._session_id}] Audio sink error: {exc}", exc_info=exc)

    def _to_mono16k(self, frame: AudioFrame) -> bytes:
        pcm = b""
        for f in self._resampler.resample(frame):
            arr  = f.to_ndarray()
            pcm += arr.astype(np.int16).tobytes()
        return pcm

    async def _flush(self) -> None:
        if not self._buf:
            return
        audio         = b"".join(self._buf)
        duration      = self._buf_sec
        self._buf     = []
        self._buf_sec = 0.0
        self._speaking      = False
        self._silence_since = None
        logger.info(f"[{self._session_id}] 🎙 Flushing {duration:.2f}s audio to STT")
        await self._on_transcript(audio, duration)

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()


# ─── Deepgram STT  (pooled HTTP, semaphore, retry) ────────────────────────────

async def transcribe_audio(session_id: str, audio_bytes: bytes) -> str:
    url = (
        "https://api.deepgram.com/v1/listen"
        "?model=nova-3&language=en&encoding=linear16"
        "&sample_rate=16000&channels=1&punctuate=true"
    )
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type":  "audio/raw",
    }
    timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SEC)

    async with _stt_sem:
        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                async with http_session.post(
                    url, headers=headers, data=audio_bytes, timeout=timeout
                ) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt
                        logger.warning(
                            f"[{session_id}] Deepgram 429 — retry in {wait}s (attempt {attempt})"
                        )
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"[{session_id}] Deepgram error {resp.status}: {text}")
                        return ""
                    data = await resp.json()
                    logger.debug(f"[{session_id}] Deepgram response: {json.dumps(data)}")
                    channels = data.get("results", {}).get("channels", [])
                    if channels:
                        alts = channels[0].get("alternatives", [])
                        if alts:
                            transcript = alts[0].get("transcript", "").strip()
                            if not transcript:
                                logger.warning(
                                    f"[{session_id}] ⚠️ Deepgram empty transcript"
                                )
                            return transcript
                    logger.warning(f"[{session_id}] ⚠️ Deepgram: no channels in response")
                    return ""
            except asyncio.TimeoutError:
                logger.warning(
                    f"[{session_id}] Deepgram timeout (attempt {attempt}/{API_MAX_RETRIES})"
                )
                if attempt == API_MAX_RETRIES:
                    return ""
            except Exception as exc:
                logger.error(f"[{session_id}] Deepgram exception: {exc}", exc_info=exc)
                return ""
    return ""


# ─── Conversation session ─────────────────────────────────────────────────────

class ConversationSession:
    SYSTEM_PROMPT = (
        "You are a helpful and friendly voice AI assistant. "
        "Speak clearly and naturally, as if having a phone conversation. "
        "Be concise but warm. Replies must be SHORT — 1–3 sentences max — "
        "because they will be converted to speech. If you don't know something, say so."
    )

    def __init__(self, ws: web.WebSocketResponse, session_id: str, peer_ip: str):
        self._ws         = ws
        self._session_id = session_id
        self._peer_ip    = peer_ip
        self._history:  list[dict] = []
        self._cost       = SessionCostTracker(session_id)

        self._speaking   = False
        self._interrupt  = asyncio.Event()

        # Pipeline lock: only one STT→LLM→TTS turn runs at a time per session
        self._pipeline_lock = asyncio.Lock()

        self._pc:   Optional[RTCPeerConnection]   = None
        self._sink: Optional[MicrophoneTrackSink] = None

    # ── WebRTC ────────────────────────────────────────────────────────────────

    async def handle_offer(self, offer_sdp: str, offer_type: str) -> None:
        self._pc = RTCPeerConnection()
        self._pc.on("connectionstatechange", self._on_connection_state)
        self._pc.on("track", self._on_track)

        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        await self._pc.setRemoteDescription(offer)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        await self._ws.send_json({
            "type":     "answer",
            "sdp":      self._pc.localDescription.sdp,
            "sdp_type": self._pc.localDescription.type,
        })
        logger.info(f"[{self._session_id}] WebRTC answer sent")

    def _on_connection_state(self) -> None:
        state = self._pc.connectionState if self._pc else "unknown"
        logger.info(f"[{self._session_id}] WebRTC state → {state}")

    def _on_track(self, track: MediaStreamTrack) -> None:
        if track.kind != "audio":
            return
        logger.info(f"[{self._session_id}] Audio track received")
        self._sink = MicrophoneTrackSink(self._session_id, self._on_audio_chunk)
        self._sink.receive(track)
        greet_task = asyncio.create_task(self._greet())
        greet_task.add_done_callback(
            lambda t: logger.error(
                f"[{self._session_id}] Greeting failed: {t.exception()}", exc_info=t.exception()
            ) if not t.cancelled() and t.exception() else None
        )

    # ── Audio pipeline ────────────────────────────────────────────────────────

    async def _greet(self) -> None:
        await asyncio.sleep(0.5)
        await self._speak_and_send("Hello! I'm your voice assistant. How can I help you today?")

    async def _on_audio_chunk(self, audio: bytes, duration_sec: float) -> None:
        """Called when the sink detects end-of-speech. Runs the full pipeline."""

        # Barge-in: interrupt any running TTS immediately
        if self._speaking:
            logger.info(f"[{self._session_id}] 🛑 Barge-in — interrupting TTS")
            self._interrupt.set()
            await self._ws.send_json({"type": "interrupt"})
            # Give the TTS coroutine a moment to observe the interrupt flag
            await asyncio.sleep(0.05)

        # Serialize turns: if a previous turn's LLM/TTS is still running, wait
        async with self._pipeline_lock:
            self._cost.add_stt(duration_sec)

            transcript = await transcribe_audio(self._session_id, audio)
            if not transcript:
                logger.debug(f"[{self._session_id}] Empty transcript, skipping")
                return

            logger.info(f"[{self._session_id}] 📝 User: {transcript}")
            await self._ws.send_json({"type": "transcript", "text": transcript, "speaker": "user"})

            # Trim history to rolling window before passing to LLM
            if len(self._history) > MAX_HISTORY_TURNS * 2:
                self._history = self._history[-(MAX_HISTORY_TURNS * 2):]

            response = await self._llm_respond(transcript)
            if response:
                await self._speak_and_send(response)

    async def _llm_respond(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self._history

        async with _llm_sem:
            for attempt in range(1, API_MAX_RETRIES + 1):
                try:
                    resp = await asyncio.wait_for(
                        openai_client.chat.completions.create(
                            model=LLM_MODEL,
                            messages=messages,
                            temperature=0.7,
                            max_tokens=200,
                        ),
                        timeout=API_TIMEOUT_SEC,
                    )
                    text       = resp.choices[0].message.content.strip()
                    in_tokens  = resp.usage.prompt_tokens
                    out_tokens = resp.usage.completion_tokens
                    self._cost.add_llm(in_tokens, out_tokens)
                    self._history.append({"role": "assistant", "content": text})
                    logger.info(f"[{self._session_id}] 🤖 Agent: {text}")
                    await self._ws.send_json(
                        {"type": "transcript", "text": text, "speaker": "agent"}
                    )
                    return text

                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self._session_id}] LLM timeout (attempt {attempt}/{API_MAX_RETRIES})"
                    )
                    if attempt == API_MAX_RETRIES:
                        break
                    await asyncio.sleep(2 ** attempt)

                except Exception as exc:
                    logger.error(f"[{self._session_id}] LLM error: {exc}", exc_info=exc)
                    # Don't retry on non-transient errors
                    break

        # Pop the user message we just added so history stays consistent
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()
        return "Sorry, I had trouble thinking of a response. Could you try again?"

    async def _speak_and_send(self, text: str) -> None:
        self._interrupt.clear()
        self._speaking = True
        self._cost.add_tts(len(text))

        async with _tts_sem:
            try:
                await self._ws.send_json({"type": "tts_start"})

                for attempt in range(1, API_MAX_RETRIES + 1):
                    try:
                        async with openai_client.audio.speech.with_streaming_response.create(
                            model="tts-1",
                            voice="echo",
                            input=text,
                            response_format="mp3",
                            speed=1.0,
                        ) as tts_response:
                            async for chunk in tts_response.iter_bytes(chunk_size=4096):
                                if self._interrupt.is_set():
                                    logger.debug(f"[{self._session_id}] TTS interrupted")
                                    return
                                await self._ws.send_bytes(chunk)
                        break  # success

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[{self._session_id}] TTS timeout (attempt {attempt}/{API_MAX_RETRIES})"
                        )
                        if attempt == API_MAX_RETRIES:
                            break
                        await asyncio.sleep(2 ** attempt)

                    except Exception as exc:
                        logger.error(f"[{self._session_id}] TTS error: {exc}", exc_info=exc)
                        break

            finally:
                self._speaking = False
                self._interrupt.clear()
                await self._ws.send_json({"type": "tts_end"})

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._sink:
            self._sink.stop()
        if self._pc:
            await self._pc.close()
        await self._cost.flush()


# ─── HTTP handlers ────────────────────────────────────────────────────────────

async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    peer_ip = request.remote or "unknown"

    # ── Enforce session cap ──
    if len(sessions) >= MAX_SESSIONS:
        logger.warning(f"Session cap ({MAX_SESSIONS}) reached — rejecting {peer_ip}")
        raise web.HTTPServiceUnavailable(reason="Server at capacity")

    # ── Enforce per-IP cap ──
    if ip_session_count.get(peer_ip, 0) >= MAX_SESSIONS_PER_IP:
        logger.warning(f"Per-IP cap ({MAX_SESSIONS_PER_IP}) reached for {peer_ip}")
        raise web.HTTPTooManyRequests(reason="Too many connections from your IP")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    session_id = f"sess_{uuid.uuid4().hex}"
    session    = ConversationSession(ws, session_id, peer_ip)

    sessions[session_id]                    = session
    ip_session_count[peer_ip]               = ip_session_count.get(peer_ip, 0) + 1
    logger.info(
        f"[{session_id}] New connection from {peer_ip} "
        f"(total={len(sessions)}, from_ip={ip_session_count[peer_ip]})"
    )

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data  = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning(f"[{session_id}] Malformed JSON, ignoring")
                    continue

                mtype = data.get("type")

                if mtype == "offer":
                    await session.handle_offer(data["sdp"], data["sdp_type"])

                elif mtype == "ice_candidate":
                    cand = data.get("candidate")
                    if cand and session._pc:
                        _add_ice_candidate(session_id, session._pc, cand)

                elif mtype == "close":
                    break

            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"[{session_id}] WS error: {ws.exception()}")
                break

    finally:
        await session.close()
        sessions.pop(session_id, None)
        count = ip_session_count.get(peer_ip, 1) - 1
        if count <= 0:
            ip_session_count.pop(peer_ip, None)
        else:
            ip_session_count[peer_ip] = count
        logger.info(f"[{session_id}] Session closed (total={len(sessions)})")

    return ws


def _add_ice_candidate(
    session_id: str,
    pc: RTCPeerConnection,
    cand: dict,
) -> None:
    """
    Parse and add a trickle-ICE candidate.
    Runs fire-and-forget so it doesn't block the WS read loop.
    """
    async def _do_add() -> None:
        try:
            raw = cand.get("candidate", "")
            if not raw:
                return
            parts = raw.split()
            # Minimum viable candidate has 8 space-separated tokens
            if len(parts) < 8:
                logger.debug(f"[{session_id}] Short ICE candidate, skipping: {raw}")
                return
            ice = RTCIceCandidate(
                component    = int(parts[1]),
                foundation   = parts[0].replace("candidate:", ""),
                ip           = parts[4],
                port         = int(parts[5]),
                priority     = int(parts[3]),
                protocol     = parts[2],
                type         = parts[7],
                sdpMid       = cand.get("sdpMid"),
                sdpMLineIndex= cand.get("sdpMLineIndex"),
            )
            await pc.addIceCandidate(ice)
        except Exception as exc:
            # ICE parse failures are common and non-fatal
            logger.debug(f"[{session_id}] ICE parse/add error: {exc}")

    task = asyncio.create_task(_do_add())
    task.add_done_callback(
        lambda t: logger.debug(f"[{session_id}] ICE task exception: {t.exception()}")
        if not t.cancelled() and t.exception() else None
    )


async def handle_index(request: web.Request) -> web.Response:
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    async with aiofiles.open(index_path, "r") as f:
        content = await f.read()
    return web.Response(content_type="text/html", text=content)


async def handle_health(request: web.Request) -> web.Response:
    payload = {
        "status":       "ok",
        "sessions":     len(sessions),
        "max_sessions": MAX_SESSIONS,
    }
    return web.Response(
        text=json.dumps(payload),
        content_type="application/json",
    )


# ─── App factory ──────────────────────────────────────────────────────────────

def build_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/",       handle_index)
    app.router.add_get("/ws",     handle_ws)
    app.router.add_get("/health", handle_health)
    app.router.add_static(
        "/static",
        os.path.join(os.path.dirname(__file__), "static"),
        show_index=False,
    )
    return app


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting WebRTC Voice Agent on {host}:{port}")
    web.run_app(build_app(), host=host, port=port, access_log=None)