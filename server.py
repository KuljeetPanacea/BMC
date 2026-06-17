"""
WebRTC Voice Agent Server — Fully Streaming
============================================
Optimizations vs previous version:
  1. STT  — Deepgram WebSocket (live partials, is_final events) instead of REST POST.
             Latency drops from ~600ms to ~80ms.
  2. LLM  — GPT streaming (stream=True). Tokens arrive token-by-token.
  3. TTS  — Sentence buffer: fire Cartesia the moment a sentence boundary is detected,
             while GPT is still generating the next sentence. Parallel pipeline.
  4. OUT  — WebRTC output AudioStreamTrack instead of WebSocket binary audio.
             Browser gets native echo cancellation (AEC) + jitter buffer for free.

Architecture per session:
  Browser mic  → WebRTC in track  → Deepgram WS → LLM stream → sentence buffer
                                                                        ↓
  Browser spkr ← WebRTC out track ← TTSAudioTrack ←← Cartesia SSE chunks
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from fractions import Fraction
from typing import Optional

import aiofiles
import aiohttp
import numpy as np
import noisereduce as nr
import websockets
from aiohttp import web
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaPlayer
from av import AudioFrame
from av.audio.resampler import AudioResampler
from dotenv import load_dotenv
from openai import AsyncOpenAI
from cartesia import AsyncCartesia

load_dotenv(".env")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webrtc_agent")

# ─── Config ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY     = os.getenv("DEEPGRAM_API_KEY", "")
CARTESIA_API_KEY     = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID    = os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")
CARTESIA_MODEL       = os.getenv("CARTESIA_MODEL", "sonic-2")
CARTESIA_SAMPLE_RATE = int(os.getenv("CARTESIA_SAMPLE_RATE", "22050"))

LLM_MODEL     = os.getenv("LLM_CHOICE", "gpt-4.1-mini")
COST_LOG_PATH = os.getenv("COST_LOG_PATH", "session_costs.log")

MAX_SESSIONS            = int(os.getenv("MAX_SESSIONS",        "100"))
MAX_SESSIONS_PER_IP     = int(os.getenv("MAX_SESSIONS_PER_IP", "5"))
STT_CONCURRENCY         = int(os.getenv("STT_CONCURRENCY",     "20"))
LLM_CONCURRENCY         = int(os.getenv("LLM_CONCURRENCY",     "20"))
TTS_CONCURRENCY_PER_SESSION = int(os.getenv("TTS_CONCURRENCY", "1"))

MAX_HISTORY_TURNS   = int(os.getenv("MAX_HISTORY_TURNS",   "20"))
MAX_AUDIO_BUF_SEC   = float(os.getenv("MAX_AUDIO_BUF_SEC", "30.0"))
API_TIMEOUT_SEC     = float(os.getenv("API_TIMEOUT_SEC",   "10.0"))
API_MAX_RETRIES     = int(os.getenv("API_MAX_RETRIES",     "3"))

VAD_AGGRESSIVENESS    = int(os.getenv("VAD_AGGRESSIVENESS",    "3"))
VAD_SILENCE_FRAMES    = int(os.getenv("VAD_SILENCE_FRAMES",    "30"))
VAD_MIN_SPEECH_FRAMES = int(os.getenv("VAD_MIN_SPEECH_FRAMES", "12"))
NOISE_REDUCE_PROP     = float(os.getenv("NOISE_REDUCE_PROP",   "0.85"))

# Sentence buffer: fire TTS when buffer has >= this many chars AND ends on punctuation.
# Prevents choppy short TTS calls while keeping first-word latency low.
SENTENCE_MIN_CHARS = int(os.getenv("SENTENCE_MIN_CHARS", "60"))

# Deepgram WebSocket URL — linear16 mono 16kHz, interim results on
DEEPGRAM_WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3"
    "&language=en"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&punctuate=true"
    "&interim_results=true"
    "&endpointing=300"      # ms of silence → speech_final event
    "&vad_events=true"
)

PRICING = {
    "stt_per_min":              0.0048,
    "llm_input_per_1m_tokens":  0.40,
    "llm_output_per_1m_tokens": 1.60,
    "tts_per_1m_chars":         65.0,
}

# ─── Global singletons ────────────────────────────────────────────────────────
openai_client:   Optional[AsyncOpenAI]           = None
cartesia_client: Optional[AsyncCartesia]         = None
http_session:    Optional[aiohttp.ClientSession] = None
process_pool:    Optional[ProcessPoolExecutor]   = None

_stt_sem: Optional[asyncio.Semaphore] = None
_llm_sem: Optional[asyncio.Semaphore] = None

sessions:         dict[str, "ConversationSession"] = {}
ip_session_count: dict[str, int]                   = {}


# ─── App lifecycle ─────────────────────────────────────────────────────────────

async def on_startup(app: web.Application) -> None:
    global openai_client, cartesia_client, http_session, process_pool
    global _stt_sem, _llm_sem

    for key, name in [
        (OPENAI_API_KEY, "OPENAI_API_KEY"),
        (DEEPGRAM_API_KEY, "DEEPGRAM_API_KEY"),
        (CARTESIA_API_KEY, "CARTESIA_API_KEY"),
    ]:
        if not key:
            logger.warning(f"{name} not set")

    openai_client   = AsyncOpenAI(api_key=OPENAI_API_KEY)
    cartesia_client = AsyncCartesia(api_key=CARTESIA_API_KEY)

    connector    = aiohttp.TCPConnector(limit=64, ttl_dns_cache=300)
    http_session = aiohttp.ClientSession(connector=connector)

    process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())

    _stt_sem = asyncio.Semaphore(STT_CONCURRENCY)
    _llm_sem = asyncio.Semaphore(LLM_CONCURRENCY)

    logger.info(
        f"Server started | max_sessions={MAX_SESSIONS} "
        f"cartesia_model={CARTESIA_MODEL} voice={CARTESIA_VOICE_ID}"
    )


async def on_shutdown(app: web.Application) -> None:
    logger.info("Graceful shutdown …")
    await asyncio.gather(*[s.close() for s in list(sessions.values())], return_exceptions=True)
    if cartesia_client: await cartesia_client.close()
    if http_session:    await http_session.close()
    if process_pool:    process_pool.shutdown(wait=False)
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

    def add_stt(self, seconds: float): self.stt_audio_sec     += seconds
    def add_tts(self, chars: int):     self.tts_chars         += chars
    def add_llm(self, inp: int, out: int):
        self.llm_input_tokens  += inp
        self.llm_output_tokens += out
        self.turns             += 1

    async def flush(self) -> None:
        if self._flushed: return
        self._flushed   = True
        wall_sec        = time.time() - self.start_time
        wall_minutes    = wall_sec / 60.0
        stt_minutes     = self.stt_audio_sec / 60.0
        stt_cost        = stt_minutes  * PRICING["stt_per_min"]
        llm_in_cost     = self.llm_input_tokens  / 1_000_000 * PRICING["llm_input_per_1m_tokens"]
        llm_out_cost    = self.llm_output_tokens / 1_000_000 * PRICING["llm_output_per_1m_tokens"]
        tts_cost        = self.tts_chars / 1_000_000 * PRICING["tts_per_1m_chars"]
        total_cost      = stt_cost + llm_in_cost + llm_out_cost + tts_cost

        record = {
            "session_id": self.session_id,
            "started_at": self.start_ts,
            "ended_at":   datetime.now().isoformat(timespec="seconds"),
            "wall_minutes": round(wall_minutes, 3),
            "turns": self.turns,
            "stt":  {"minutes": round(stt_minutes, 4), "cost_usd": round(stt_cost, 6)},
            "llm":  {"input_tokens": self.llm_input_tokens, "output_tokens": self.llm_output_tokens,
                     "cost_usd": round(llm_in_cost + llm_out_cost, 6)},
            "tts":  {"chars": self.tts_chars, "cost_usd": round(tts_cost, 6)},
            "total_usd": round(total_cost, 6),
        }
        async with aiofiles.open(COST_LOG_PATH, "a", encoding="utf-8") as f:
            await f.write(json.dumps(record) + "\n")

        logger.info(
            f"\n{'═'*54}\n  SESSION COST  [{self.session_id}]\n{'═'*54}\n"
            f"  Wall: {wall_minutes:.2f} min | Turns: {self.turns}\n"
            f"  STT : ${stt_cost:.6f}  ({self.stt_audio_sec:.1f}s)\n"
            f"  LLM : ${llm_in_cost+llm_out_cost:.6f}  ({self.llm_input_tokens}in/{self.llm_output_tokens}out)\n"
            f"  TTS : ${tts_cost:.6f}  ({self.tts_chars:,} chars)\n"
            f"  TOTAL: ${total_cost:.6f}\n{'═'*54}"
        )


# ─── CPU-bound helpers ────────────────────────────────────────────────────────

def _noise_reduce_sync(pcm_bytes: bytes, sample_rate: int, prop_decrease: float) -> bytes:
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(arr) < sample_rate // 10:
        return pcm_bytes
    reduced = nr.reduce_noise(
        y=arr, sr=sample_rate, stationary=True,
        prop_decrease=prop_decrease, n_jobs=1,
    )
    out = np.clip(reduced * 32768.0, -32768, 32767).astype(np.int16)
    return out.tobytes()


# ─── TTSAudioTrack ────────────────────────────────────────────────────────────
# An aiortc AudioStreamTrack that pulls PCM frames from an asyncio queue.
# ConversationSession enqueues PCM bytes from Cartesia; recv() dequeues and
# wraps them into AudioFrame objects for the WebRTC pipeline.

class TTSAudioTrack(MediaStreamTrack):
    """
    WebRTC output audio track fed by Cartesia PCM chunks.

    Two key fixes vs the previous version:
      1. Output at 48kHz (browser WebRTC native rate).
         Cartesia sends 22050Hz PCM; we resample it here on the server once.
         Without this the browser has to resample every frame internally,
         which causes pitch wobble and choppy sound.
      2. recv() is paced by wall-clock time (asyncio.sleep to next 20ms boundary).
         Without pacing aiortc calls recv() in a tight loop, flooding the RTP
         sender and corrupting the PTS timeline — the browser drops everything.
    """
    kind = "audio"

    OUT_RATE          = 48_000          # browser WebRTC native rate
    IN_RATE           = CARTESIA_SAMPLE_RATE   # 22050Hz from Cartesia
    SAMPLES_PER_FRAME = 960             # 960 @ 48kHz = exactly 20ms (standard ptime)
    FRAME_DURATION    = SAMPLES_PER_FRAME / OUT_RATE   # 0.020 s

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._buf        = b""
        self._pts        = 0
        self._next_time: Optional[float] = None

    def push_pcm(self, pcm_bytes: bytes) -> None:
        """
        Receive s16le PCM at CARTESIA_SAMPLE_RATE, resample to 48kHz, enqueue.
        Linear interpolation is fast enough in numpy and sounds clean for speech.
        """
        if not pcm_bytes:
            return
        src  = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        n_out = int(len(src) * self.OUT_RATE / self.IN_RATE)
        dst  = np.interp(
            np.linspace(0, len(src) - 1, n_out),
            np.arange(len(src)),
            src,
        )
        self._queue.put_nowait((np.clip(dst * 32768, -32768, 32767).astype(np.int16)).tobytes())

    async def recv(self) -> AudioFrame:
        """
        Return one 20ms frame of 48kHz audio, paced by wall-clock time.
        Returns silence when nothing is queued (keeps the WebRTC track alive).
        """
        loop = asyncio.get_event_loop()
        now  = loop.time()
        if self._next_time is None:
            self._next_time = now

        # Sleep until next frame boundary — this IS the pacing mechanism
        delay = self._next_time - now
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_time += self.FRAME_DURATION

        need = self.SAMPLES_PER_FRAME * 2   # bytes

        # Drain all queued chunks into rolling buffer (non-blocking)
        while len(self._buf) < need:
            try:
                self._buf += self._queue.get_nowait() or b""
            except asyncio.QueueEmpty:
                break

        # One short blocking wait to catch the first chunk of a new utterance
        if len(self._buf) < need:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.005)
                if chunk:
                    self._buf += chunk
                    while len(self._buf) < need:
                        try:
                            self._buf += self._queue.get_nowait() or b""
                        except asyncio.QueueEmpty:
                            break
            except asyncio.TimeoutError:
                pass

        if len(self._buf) >= need:
            frame_bytes, self._buf = self._buf[:need], self._buf[need:]
        else:
            frame_bytes = b"\x00" * need   # silence

        samples = np.frombuffer(frame_bytes, dtype=np.int16)
        frame   = AudioFrame(format="s16", layout="mono", samples=self.SAMPLES_PER_FRAME)
        frame.planes[0].update(samples.tobytes())
        frame.sample_rate = self.OUT_RATE
        frame.pts         = self._pts
        frame.time_base   = Fraction(1, self.OUT_RATE)
        self._pts        += self.SAMPLES_PER_FRAME
        return frame


# ─── MicrophoneReceiver ───────────────────────────────────────────────────────
# Receives WebRTC audio frames, resamples to 16kHz mono, and forwards
# 20ms PCM chunks to the Deepgram WebSocket.  Minimal local VAD is kept
# only to gate noise-reduction; speech detection is delegated to Deepgram.

class MicrophoneReceiver:
    SAMPLE_RATE   = 16_000
    FRAME_MS      = 20
    FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 320 samples

    def __init__(self, session_id: str, on_pcm_chunk):
        self._session_id   = session_id
        self._on_pcm_chunk = on_pcm_chunk   # coroutine(pcm_bytes: bytes)
        self._resampler    = AudioResampler(format="s16", layout="mono", rate=self.SAMPLE_RATE)
        self._ring         = b""
        self._task: Optional[asyncio.Task] = None

    def receive(self, track: MediaStreamTrack) -> None:
        self._task = asyncio.create_task(self._run(track))

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _run(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame: AudioFrame = await track.recv()
                pcm = b""
                for f in self._resampler.resample(frame):
                    pcm += f.to_ndarray().astype(np.int16).tobytes()
                self._ring += pcm
                while len(self._ring) >= self.FRAME_SAMPLES * 2:
                    chunk      = self._ring[: self.FRAME_SAMPLES * 2]
                    self._ring = self._ring[self.FRAME_SAMPLES * 2:]
                    await self._on_pcm_chunk(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"[{self._session_id}] MicrophoneReceiver error: {exc}", exc_info=exc)


# ─── DeepgramStreamer ─────────────────────────────────────────────────────────
# Opens a persistent WebSocket to Deepgram and forwards 20ms PCM frames.
# Fires callbacks on is_final and speech_final events.

class DeepgramStreamer:
    def __init__(self, session_id: str, on_transcript, on_speech_start=None):
        self._session_id     = session_id
        self._on_transcript  = on_transcript   # coroutine(text: str, duration_sec: float)
        self._on_speech_start = on_speech_start
        self._ws             = None
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._utterance_start: Optional[float] = None
        self._running = False

    async def start(self) -> None:
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        self._ws = await websockets.connect(DEEPGRAM_WS_URL, additional_headers=headers)
        self._running  = True
        self._send_task = asyncio.create_task(self._sender())
        self._recv_task = asyncio.create_task(self._receiver())
        logger.info(f"[{self._session_id}] Deepgram WS connected")

    async def send_pcm(self, pcm_bytes: bytes) -> None:
        """Enqueue 20ms raw s16le PCM for forwarding to Deepgram."""
        if self._running:
            await self._queue.put(pcm_bytes)

    async def stop(self) -> None:
        self._running = False
        await self._queue.put(None)   # sentinel
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await self._ws.close()
            except Exception:
                pass
        for t in [self._send_task, self._recv_task]:
            if t and not t.done():
                t.cancel()

    async def _sender(self) -> None:
        try:
            while self._running:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                if self._ws:
                    await self._ws.send(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"[{self._session_id}] Deepgram sender error: {exc}", exc_info=exc)

    async def _receiver(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "SpeechStarted":
                    self._utterance_start = time.time()
                    if self._on_speech_start:
                        await self._on_speech_start()

                elif msg_type == "Results":
                    channel = msg.get("channel", {})
                    alts    = channel.get("alternatives", [{}])
                    text    = alts[0].get("transcript", "").strip() if alts else ""
                    is_final     = msg.get("is_final", False)
                    speech_final = msg.get("speech_final", False)

                    if (is_final or speech_final) and text:
                        duration = time.time() - (self._utterance_start or time.time())
                        self._utterance_start = None
                        logger.info(f"[{self._session_id}] 📝 User: {text}")
                        await self._on_transcript(text, duration)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"[{self._session_id}] Deepgram receiver error: {exc}", exc_info=exc)


# ─── Sentence boundary detection ─────────────────────────────────────────────
# Fires when we have SENTENCE_MIN_CHARS chars AND the last char is . ! ?
# This prevents very short choppy TTS calls while still giving fast first response.

_SENTENCE_END = frozenset(".!?")
_ABBREV_PREFIXES = frozenset(["dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc"])

def _is_sentence_end(text: str) -> bool:
    """Return True if text ends on a real sentence boundary."""
    t = text.rstrip()
    if not t or t[-1] not in _SENTENCE_END:
        return False
    if len(t) < SENTENCE_MIN_CHARS:
        return False
    # Avoid splitting on abbreviations like "Dr." or numbers like "4.99"
    words = t.split()
    if words:
        last_word = words[-1].rstrip(".!?").lower()
        if last_word in _ABBREV_PREFIXES:
            return False
        # Number check — "4.99" or "3.14"
        try:
            float(last_word.replace(",", ""))
            return False
        except ValueError:
            pass
    return True


# ─── Conversation Session ─────────────────────────────────────────────────────

class ConversationSession:
    SYSTEM_PROMPT = (
        "You are a helpful and friendly voice AI assistant. "
        "Speak clearly and naturally, as if having a phone conversation. "
        "Be concise but warm. Replies must be SHORT — 1–3 sentences max — "
        "because they will be converted to speech. Never use bullet points, "
        "markdown, or special characters. If you don't know something, say so."
    )

    def __init__(self, ws: web.WebSocketResponse, session_id: str, peer_ip: str):
        self._ws           = ws
        self._session_id   = session_id
        self._peer_ip      = peer_ip
        self._history:     list[dict] = []
        self._cost         = SessionCostTracker(session_id)

        self._speaking     = False
        self._interrupt    = asyncio.Event()
        self._turn_seq     = 0

        # Barge-in cooldown: ignore SpeechStarted for this many seconds after
        # TTS starts, preventing the agent's own voice (mic echo before browser
        # AEC kicks in) from triggering self-interruption.
        self._tts_started_at:    float = 0.0
        self._barge_in_cooldown: float = 1.2   # seconds

        self._pipeline_lock = asyncio.Lock()
        self._tts_sem       = asyncio.Semaphore(TTS_CONCURRENCY_PER_SESSION)

        self._pc:       Optional[RTCPeerConnection] = None
        self._mic:      Optional[MicrophoneReceiver] = None
        self._dg:       Optional[DeepgramStreamer]   = None
        self._tts_track: Optional[TTSAudioTrack]     = None

    # ── WebRTC ────────────────────────────────────────────────────────────────

    async def handle_offer(self, offer_sdp: str, offer_type: str) -> None:
        self._pc = RTCPeerConnection()
        self._pc.on("connectionstatechange", self._on_connection_state)
        self._pc.on("track", self._on_track)

        # Add our TTS output track BEFORE creating the answer so it's in the SDP
        self._tts_track = TTSAudioTrack()
        self._pc.addTrack(self._tts_track)

        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        await self._ws.send_json({
            "type":     "answer",
            "sdp":      self._pc.localDescription.sdp,
            "sdp_type": self._pc.localDescription.type,
        })
        logger.info(f"[{self._session_id}] WebRTC answer sent (with TTS output track)")

    def _on_connection_state(self) -> None:
        state = self._pc.connectionState if self._pc else "unknown"
        logger.info(f"[{self._session_id}] WebRTC → {state}")

    def _on_track(self, track: MediaStreamTrack) -> None:
        if track.kind != "audio":
            return
        logger.info(f"[{self._session_id}] Audio input track received")

        # MicrophoneReceiver → DeepgramStreamer pipeline
        self._dg  = DeepgramStreamer(
            self._session_id,
            on_transcript=self._on_transcript,
            on_speech_start=self._on_speech_start,
        )
        self._mic = MicrophoneReceiver(self._session_id, on_pcm_chunk=self._dg.send_pcm)

        async def _start_pipeline():
            await self._dg.start()
            self._mic.receive(track)
            await asyncio.sleep(0.5)
            await self._greet()

        task = asyncio.create_task(_start_pipeline())
        task.add_done_callback(
            lambda t: logger.error(f"[{self._session_id}] Pipeline start failed: {t.exception()}", exc_info=t.exception())
            if not t.cancelled() and t.exception() else None
        )

    # ── Pipeline ──────────────────────────────────────────────────────────────

    async def _greet(self) -> None:
        await self._speak_and_send("Hello! I'm your voice assistant. How can I help you today?")

    async def _on_speech_start(self) -> None:
        """Deepgram detected speech starting — interrupt TTS if speaking."""
        if not self._speaking:
            return
        # Cooldown guard: ignore SpeechStarted events fired within the first
        # _barge_in_cooldown seconds of TTS playback. The agent's own voice
        # leaks into the mic before the browser's AEC suppresses it, causing
        # Deepgram to fire SpeechStarted on the agent's own output.
        elapsed = time.time() - self._tts_started_at
        if elapsed < self._barge_in_cooldown:
            logger.debug(
                f"[{self._session_id}] SpeechStarted ignored — "
                f"in AEC cooldown ({elapsed:.2f}s < {self._barge_in_cooldown}s)"
            )
            return
        logger.info(f"[{self._session_id}] 🛑 Barge-in — interrupting TTS")
        self._interrupt.set()
        await self._ws.send_json({"type": "interrupt", "turn_id": self._turn_seq})

    async def _on_transcript(self, text: str, duration_sec: float) -> None:
        """Called by DeepgramStreamer when a final transcript arrives."""
        if self._speaking:
            self._interrupt.set()
            await self._ws.send_json({"type": "interrupt", "turn_id": self._turn_seq})

        async with self._pipeline_lock:
            self._cost.add_stt(duration_sec)
            await self._ws.send_json({"type": "transcript", "text": text, "speaker": "user"})

            if len(self._history) > MAX_HISTORY_TURNS * 2:
                self._history = self._history[-(MAX_HISTORY_TURNS * 2):]

            # Stream LLM response, firing TTS sentence-by-sentence
            await self._llm_stream(text)

    async def _llm_stream(self, user_text: str) -> None:
        """
        Stream GPT tokens, buffer into sentences, fire each sentence
        to Cartesia as soon as it's complete — while GPT writes the next one.
        """
        self._history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self._history

        full_reply    = ""
        sentence_buf  = ""
        input_tokens  = 0
        output_tokens = 0

        async with _llm_sem:
            try:
                stream = await asyncio.wait_for(
                    openai_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=200,
                        stream=True,
                    ),
                    timeout=API_TIMEOUT_SEC,
                )

                async for chunk in stream:
                    if self._interrupt.is_set():
                        logger.debug(f"[{self._session_id}] LLM stream interrupted")
                        break

                    delta = chunk.choices[0].delta if chunk.choices else None
                    token = delta.content if delta and delta.content else ""

                    if token:
                        sentence_buf += token
                        full_reply   += token

                        if _is_sentence_end(sentence_buf):
                            sentence = sentence_buf.strip()
                            sentence_buf = ""
                            # Fire TTS without awaiting completion —
                            # let it run while GPT continues streaming
                            asyncio.create_task(self._speak_and_send(sentence))

                    # Collect usage from the final chunk (OpenAI sends it last)
                    if hasattr(chunk, "usage") and chunk.usage:
                        input_tokens  = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens

            except asyncio.TimeoutError:
                logger.warning(f"[{self._session_id}] LLM stream timeout")
                full_reply = full_reply or "Sorry, I had a bit of trouble there."
            except Exception as exc:
                logger.error(f"[{self._session_id}] LLM stream error: {exc}", exc_info=exc)
                full_reply = full_reply or "Sorry, I had a bit of trouble there."

        # Speak any remaining buffer (last partial sentence with no terminal punct)
        leftover = sentence_buf.strip()
        if leftover and not self._interrupt.is_set():
            asyncio.create_task(self._speak_and_send(leftover))

        # If nothing was generated yet (timeout before any token)
        if not full_reply and not self._interrupt.is_set():
            asyncio.create_task(
                self._speak_and_send("Sorry, I had a bit of trouble there. Could you try again?")
            )

        if full_reply:
            self._cost.add_llm(input_tokens, output_tokens)
            self._history.append({"role": "assistant", "content": full_reply})
            logger.info(f"[{self._session_id}] 🤖 Agent: {full_reply}")
            await self._ws.send_json({"type": "transcript", "text": full_reply, "speaker": "agent"})
        else:
            # Roll back user turn if nothing was produced
            if self._history and self._history[-1]["role"] == "user":
                self._history.pop()

    # ── Cartesia TTS → TTSAudioTrack ─────────────────────────────────────────

    async def _speak_and_send(self, text: str) -> None:
        """
        Convert one sentence to speech via Cartesia SSE and push PCM
        directly into the WebRTC output track (TTSAudioTrack).
        No WebSocket binary frames — the browser receives it as a WebRTC track.
        """
        if not text.strip():
            return

        self._cost.add_tts(len(text))
        self._turn_seq += 1
        turn_id = self._turn_seq

        async with self._tts_sem:
            if turn_id != self._turn_seq:
                return   # superseded by a newer turn

            self._speaking       = True
            self._tts_started_at = time.time()   # for barge-in cooldown
            try:
                await self._ws.send_json({"type": "tts_start", "turn_id": turn_id})

                for attempt in range(1, API_MAX_RETRIES + 1):
                    if self._interrupt.is_set():
                        return

                    try:
                        sse_stream = await cartesia_client.tts.sse(
                            model_id=CARTESIA_MODEL,
                            transcript=text,
                            voice={"id": CARTESIA_VOICE_ID},
                            output_format={
                                "container":   "raw",
                                "encoding":    "pcm_s16le",
                                "sample_rate": CARTESIA_SAMPLE_RATE,
                            },
                        )

                        async for event in sse_stream:
                            if self._interrupt.is_set():
                                return

                            if hasattr(event, "audio"):
                                audio_data: bytes = event.audio
                            elif isinstance(event, dict):
                                audio_data = event.get("audio", b"")
                            else:
                                continue

                            if audio_data and self._tts_track:
                                # Push PCM directly into the WebRTC output track
                                self._tts_track.push_pcm(audio_data)

                        break   # stream finished normally

                    except asyncio.TimeoutError:
                        logger.warning(f"[{self._session_id}] Cartesia timeout (attempt {attempt})")
                        if attempt == API_MAX_RETRIES:
                            break
                        await asyncio.sleep(2 ** attempt)
                    except Exception as exc:
                        logger.error(f"[{self._session_id}] Cartesia error: {exc}", exc_info=exc)
                        break

            finally:
                self._speaking = False
                self._interrupt.clear()
                if turn_id == self._turn_seq:
                    await self._ws.send_json({"type": "tts_end", "turn_id": turn_id})

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._mic: self._mic.stop()
        if self._dg:  await self._dg.stop()
        if self._pc:  await self._pc.close()
        await self._cost.flush()


# ─── HTTP handlers ────────────────────────────────────────────────────────────

async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    peer_ip = request.remote or "unknown"

    if len(sessions) >= MAX_SESSIONS:
        raise web.HTTPServiceUnavailable(reason="Server at capacity")
    if ip_session_count.get(peer_ip, 0) >= MAX_SESSIONS_PER_IP:
        raise web.HTTPTooManyRequests(reason="Too many connections from your IP")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    session_id = f"sess_{uuid.uuid4().hex}"
    session    = ConversationSession(ws, session_id, peer_ip)

    sessions[session_id]      = session
    ip_session_count[peer_ip] = ip_session_count.get(peer_ip, 0) + 1
    logger.info(f"[{session_id}] New connection from {peer_ip} (total={len(sessions)})")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
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
        if count <= 0: ip_session_count.pop(peer_ip, None)
        else:          ip_session_count[peer_ip] = count
        logger.info(f"[{session_id}] Session closed (total={len(sessions)})")

    return ws


def _add_ice_candidate(session_id: str, pc: RTCPeerConnection, cand: dict) -> None:
    async def _do_add():
        try:
            raw = cand.get("candidate", "")
            if not raw: return
            parts = raw.split()
            if len(parts) < 8: return
            ice = RTCIceCandidate(
                component=int(parts[1]), foundation=parts[0].replace("candidate:", ""),
                ip=parts[4], port=int(parts[5]), priority=int(parts[3]),
                protocol=parts[2], type=parts[7],
                sdpMid=cand.get("sdpMid"), sdpMLineIndex=cand.get("sdpMLineIndex"),
            )
            await pc.addIceCandidate(ice)
        except Exception as exc:
            logger.debug(f"[{session_id}] ICE add error (usually ok): {exc}")
    asyncio.create_task(_do_add())


async def handle_index(request: web.Request) -> web.Response:
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    async with aiofiles.open(path, "r") as f:
        content = await f.read()
    return web.Response(content_type="text/html", text=content)


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(
        text=json.dumps({"status": "ok", "sessions": len(sessions), "max_sessions": MAX_SESSIONS}),
        content_type="application/json",
    )


def build_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/",       handle_index)
    app.router.add_get("/ws",     handle_ws)
    app.router.add_get("/health", handle_health)
    app.router.add_static("/static", os.path.join(os.path.dirname(__file__), "static"), show_index=False)
    return app


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting WebRTC Voice Agent on {host}:{port}")
    web.run_app(build_app(), host=host, port=port, access_log=None)