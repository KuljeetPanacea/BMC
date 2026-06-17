"""
WebRTC Voice Agent Server — Cartesia TTS + Multi-User Fix
==========================================================
Changes from previous version:
  1. TTS provider switched from OpenAI → Cartesia (cartesia-ai SDK)
  2. Multi-user fix: _tts_sem moved from global to per-session, so one
     user's slow TTS never blocks another user's pipeline.
  3. _turn_seq, _speaking, _interrupt all remain per-session (were already
     correct) — no change needed there.
  4. .env: CARTESIA_API_KEY + CARTESIA_VOICE_ID + CARTESIA_MODEL are now
     the TTS config keys. OPENAI TTS keys removed.
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
from typing import Optional

import aiofiles
import aiohttp
import numpy as np
import noisereduce as nr
import webrtcvad
from aiohttp import web
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
)
from av import AudioFrame
from av.audio.resampler import AudioResampler
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Cartesia Python SDK  (pip install cartesia)
from cartesia import AsyncCartesia

load_dotenv(".env")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webrtc_agent")

# ─── Config ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY  = os.getenv("DEEPGRAM_API_KEY", "")
CARTESIA_API_KEY  = os.getenv("CARTESIA_API_KEY", "")
# Voice ID — find yours at https://play.cartesia.ai/voices
# Default: "a0e99841-438c-4a64-b679-ae501e7d6091"  (Barbershop Man)
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")
# Model: sonic-2 is Cartesia's latest (fastest + best quality)
CARTESIA_MODEL    = os.getenv("CARTESIA_MODEL", "sonic-2")

LLM_MODEL      = os.getenv("LLM_CHOICE", "gpt-4.1-mini")
COST_LOG_PATH  = os.getenv("COST_LOG_PATH", "session_costs.log")

MAX_SESSIONS        = int(os.getenv("MAX_SESSIONS",        "100"))
MAX_SESSIONS_PER_IP = int(os.getenv("MAX_SESSIONS_PER_IP", "5"))
STT_CONCURRENCY     = int(os.getenv("STT_CONCURRENCY",     "20"))
LLM_CONCURRENCY     = int(os.getenv("LLM_CONCURRENCY",     "20"))
# NOTE: TTS_CONCURRENCY is now a PER-SESSION limit (default 1 = one TTS
# stream at a time per user, which is all you ever need). The old global
# semaphore was the root cause of multi-user TTS blocking.
TTS_CONCURRENCY_PER_SESSION = int(os.getenv("TTS_CONCURRENCY", "1"))

MAX_HISTORY_TURNS   = int(os.getenv("MAX_HISTORY_TURNS",   "20"))
MAX_AUDIO_BUF_SEC   = float(os.getenv("MAX_AUDIO_BUF_SEC", "30.0"))
API_TIMEOUT_SEC     = float(os.getenv("API_TIMEOUT_SEC",   "10.0"))
API_MAX_RETRIES     = int(os.getenv("API_MAX_RETRIES",     "3"))

VAD_AGGRESSIVENESS    = int(os.getenv("VAD_AGGRESSIVENESS",   "3"))
VAD_SILENCE_FRAMES    = int(os.getenv("VAD_SILENCE_FRAMES",   "30"))
VAD_MIN_SPEECH_FRAMES = int(os.getenv("VAD_MIN_SPEECH_FRAMES","12"))
NOISE_REDUCE_PROP     = float(os.getenv("NOISE_REDUCE_PROP",  "0.85"))

# ─── Pricing ──────────────────────────────────────────────────────────────────
PRICING = {
    "stt_per_min":              0.0048,
    "llm_input_per_1m_tokens":  0.40,
    "llm_output_per_1m_tokens": 1.60,
    # Cartesia sonic-2: $0.000 per character (check your plan)
    # Update this if you're on a paid tier
    "tts_per_1m_chars":         65.0,   # Cartesia standard rate
}

# ─── Global singletons ────────────────────────────────────────────────────────
openai_client:   Optional[AsyncOpenAI]     = None
cartesia_client: Optional[AsyncCartesia]   = None
http_session:    Optional[aiohttp.ClientSession] = None
process_pool:    Optional[ProcessPoolExecutor]   = None

_stt_sem: Optional[asyncio.Semaphore] = None
_llm_sem: Optional[asyncio.Semaphore] = None
# ✅ NO global _tts_sem — TTS semaphore is now per ConversationSession

sessions:         dict[str, "ConversationSession"] = {}
ip_session_count: dict[str, int]                   = {}


# ─── App lifecycle ─────────────────────────────────────────────────────────────

async def on_startup(app: web.Application) -> None:
    global openai_client, cartesia_client, http_session, process_pool
    global _stt_sem, _llm_sem

    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — LLM calls will fail")
    if not DEEPGRAM_API_KEY:
        logger.warning("DEEPGRAM_API_KEY not set — STT calls will fail")
    if not CARTESIA_API_KEY:
        logger.warning("CARTESIA_API_KEY not set — TTS calls will fail")

    openai_client   = AsyncOpenAI(api_key=OPENAI_API_KEY)
    cartesia_client = AsyncCartesia(api_key=CARTESIA_API_KEY)

    connector    = aiohttp.TCPConnector(limit=64, ttl_dns_cache=300)
    http_session = aiohttp.ClientSession(connector=connector)

    process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())

    _stt_sem = asyncio.Semaphore(STT_CONCURRENCY)
    _llm_sem = asyncio.Semaphore(LLM_CONCURRENCY)

    logger.info(
        f"Server started | max_sessions={MAX_SESSIONS} "
        f"cartesia_model={CARTESIA_MODEL} voice={CARTESIA_VOICE_ID} "
        f"vad_aggressiveness={VAD_AGGRESSIVENESS}"
    )


async def on_shutdown(app: web.Application) -> None:
    logger.info("Graceful shutdown …")
    close_tasks = [s.close() for s in list(sessions.values())]
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)

    if cartesia_client:
        await cartesia_client.close()
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

    def add_stt(self, seconds: float):     self.stt_audio_sec     += seconds
    def add_tts(self, chars: int):         self.tts_chars         += chars
    def add_llm(self, inp: int, out: int):
        self.llm_input_tokens  += inp
        self.llm_output_tokens += out
        self.turns             += 1

    async def flush(self) -> None:
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


# ─── CPU-bound audio helpers ──────────────────────────────────────────────────

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


# ─── MicrophoneTrackSink ──────────────────────────────────────────────────────

class MicrophoneTrackSink:
    SAMPLE_RATE   = 16_000
    FRAME_MS      = 20
    FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

    def __init__(self, session_id: str, on_utterance, on_speech_confirmed=None):
        self._session_id          = session_id
        self._on_utterance        = on_utterance
        self._on_speech_confirmed = on_speech_confirmed

        self._resampler  = AudioResampler(format="s16", layout="mono", rate=self.SAMPLE_RATE)
        self._vad        = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._ring: bytes = b""

        self._utt_frames: list[bytes] = []
        self._speech_frames  = 0
        self._silence_frames = 0
        self._in_utterance   = False
        self._confirmed_sent = False
        self._utt_sec        = 0.0
        self._task: Optional[asyncio.Task] = None

    def receive(self, track: MediaStreamTrack) -> None:
        self._task = asyncio.create_task(self._run(track))
        self._task.add_done_callback(self._on_task_done)

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"[{self._session_id}] Audio sink crashed: {exc}", exc_info=exc)

    async def _run(self, track: MediaStreamTrack) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                frame: AudioFrame = await track.recv()
                pcm = self._to_mono16k(frame)
                self._ring += pcm
                while len(self._ring) >= self.FRAME_SAMPLES * 2:
                    frame_bytes  = self._ring[: self.FRAME_SAMPLES * 2]
                    self._ring   = self._ring[self.FRAME_SAMPLES * 2 :]
                    await self._process_frame(frame_bytes, loop)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"[{self._session_id}] Audio sink error: {exc}", exc_info=exc)

    async def _process_frame(self, frame_bytes: bytes, loop: asyncio.AbstractEventLoop) -> None:
        try:
            is_speech = self._vad.is_speech(frame_bytes, self.SAMPLE_RATE)
        except Exception:
            is_speech = False

        if is_speech:
            if not self._in_utterance:
                self._in_utterance   = True
                self._utt_frames     = []
                self._utt_sec        = 0.0
                self._speech_frames  = 0
                self._confirmed_sent = False
                logger.info(f"[{self._session_id}] 🎙 Speech start (VAD)")

            self._utt_frames.append(frame_bytes)
            self._utt_sec       += self.FRAME_MS / 1000
            self._speech_frames += 1
            self._silence_frames = 0

            if (
                not self._confirmed_sent
                and self._speech_frames >= VAD_MIN_SPEECH_FRAMES
                and self._on_speech_confirmed is not None
            ):
                self._confirmed_sent = True
                await self._on_speech_confirmed()

            if self._utt_sec >= MAX_AUDIO_BUF_SEC:
                logger.warning(f"[{self._session_id}] Max utterance length — force-flushing")
                await self._flush(loop)
        else:
            if self._in_utterance:
                self._utt_frames.append(frame_bytes)
                self._utt_sec        += self.FRAME_MS / 1000
                self._silence_frames += 1
                if self._silence_frames >= VAD_SILENCE_FRAMES:
                    await self._flush(loop)

    async def _flush(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self._utt_frames or self._speech_frames < VAD_MIN_SPEECH_FRAMES:
            if self._utt_frames:
                logger.debug(f"[{self._session_id}] Discarding short utterance")
            self._reset_utterance()
            return

        raw_pcm  = b"".join(self._utt_frames)
        duration = self._utt_sec
        self._reset_utterance()

        logger.info(f"[{self._session_id}] 🎙 Utterance end — {duration:.2f}s, noise-reducing")
        try:
            clean_pcm: bytes = await loop.run_in_executor(
                process_pool, _noise_reduce_sync, raw_pcm, self.SAMPLE_RATE, NOISE_REDUCE_PROP,
            )
        except Exception as exc:
            logger.warning(f"[{self._session_id}] Noise reduction failed: {exc}")
            clean_pcm = raw_pcm

        await self._on_utterance(clean_pcm, duration)

    def _reset_utterance(self) -> None:
        self._utt_frames     = []
        self._utt_sec        = 0.0
        self._speech_frames  = 0
        self._silence_frames = 0
        self._in_utterance   = False

    def _to_mono16k(self, frame: AudioFrame) -> bytes:
        pcm = b""
        for f in self._resampler.resample(frame):
            pcm += f.to_ndarray().astype(np.int16).tobytes()
        return pcm


# ─── Deepgram STT ─────────────────────────────────────────────────────────────

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
                        logger.warning(f"[{session_id}] Deepgram 429 — retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        logger.error(f"[{session_id}] Deepgram {resp.status}: {await resp.text()}")
                        return ""
                    data     = await resp.json()
                    channels = data.get("results", {}).get("channels", [])
                    if channels:
                        alts = channels[0].get("alternatives", [])
                        if alts:
                            return alts[0].get("transcript", "").strip()
                    return ""
            except asyncio.TimeoutError:
                logger.warning(f"[{session_id}] Deepgram timeout (attempt {attempt})")
                if attempt == API_MAX_RETRIES:
                    return ""
            except Exception as exc:
                logger.error(f"[{session_id}] Deepgram exception: {exc}", exc_info=exc)
                return ""
    return ""


# ─── Conversation Session ─────────────────────────────────────────────────────

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
        self._turn_seq   = 0

        self._pipeline_lock = asyncio.Lock()

        # ✅ Per-session TTS semaphore — completely isolated from other users.
        # Value of 1 means one TTS stream at a time per user (correct),
        # but doesn't starve other users the way the old global semaphore did.
        self._tts_sem = asyncio.Semaphore(TTS_CONCURRENCY_PER_SESSION)

        self._pc:   Optional[RTCPeerConnection]   = None
        self._sink: Optional[MicrophoneTrackSink] = None

    # ── WebRTC ────────────────────────────────────────────────────────────────

    async def handle_offer(self, offer_sdp: str, offer_type: str) -> None:
        self._pc = RTCPeerConnection()
        self._pc.on("connectionstatechange", self._on_connection_state)
        self._pc.on("track", self._on_track)

        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        )
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
        self._sink = MicrophoneTrackSink(
            self._session_id,
            on_utterance=self._on_utterance,
            on_speech_confirmed=self._on_speech_confirmed,
        )
        self._sink.receive(track)
        task = asyncio.create_task(self._greet())
        task.add_done_callback(
            lambda t: logger.error(
                f"[{self._session_id}] Greeting failed: {t.exception()}", exc_info=t.exception()
            ) if not t.cancelled() and t.exception() else None
        )

    # ── Audio pipeline ────────────────────────────────────────────────────────

    async def _greet(self) -> None:
        await asyncio.sleep(0.5)
        await self._speak_and_send("Hello! I'm your voice assistant. How can I help you today?")

    async def _on_speech_confirmed(self) -> None:
        if self._speaking:
            logger.info(f"[{self._session_id}] 🛑 Barge-in confirmed — interrupting TTS")
            self._interrupt.set()
            await self._ws.send_json({"type": "interrupt", "turn_id": self._turn_seq})
            await asyncio.sleep(0)

    async def _on_utterance(self, audio: bytes, duration_sec: float) -> None:
        if self._speaking:
            logger.info(f"[{self._session_id}] 🛑 Barge-in (end-of-utterance safety catch)")
            self._interrupt.set()
            await self._ws.send_json({"type": "interrupt", "turn_id": self._turn_seq})
            await asyncio.sleep(0)

        async with self._pipeline_lock:
            self._cost.add_stt(duration_sec)

            transcript = await transcribe_audio(self._session_id, audio)
            if not transcript:
                logger.debug(f"[{self._session_id}] Empty transcript — skipping turn")
                return

            logger.info(f"[{self._session_id}] 📝 User: {transcript}")
            await self._ws.send_json({"type": "transcript", "text": transcript, "speaker": "user"})

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
                    text    = resp.choices[0].message.content.strip()
                    in_tok  = resp.usage.prompt_tokens
                    out_tok = resp.usage.completion_tokens
                    self._cost.add_llm(in_tok, out_tok)
                    self._history.append({"role": "assistant", "content": text})
                    logger.info(f"[{self._session_id}] 🤖 Agent: {text}")
                    await self._ws.send_json({"type": "transcript", "text": text, "speaker": "agent"})
                    return text
                except asyncio.TimeoutError:
                    logger.warning(f"[{self._session_id}] LLM timeout (attempt {attempt})")
                    if attempt == API_MAX_RETRIES:
                        break
                    await asyncio.sleep(2 ** attempt)
                except Exception as exc:
                    logger.error(f"[{self._session_id}] LLM error: {exc}", exc_info=exc)
                    break

        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()
        return "Sorry, I had a bit of trouble there. Could you try again?"

    # ── Cartesia TTS ──────────────────────────────────────────────────────────

    async def _speak_and_send(self, text: str) -> None:
        self._cost.add_tts(len(text))

        self._turn_seq += 1
        turn_id = self._turn_seq

        async with self._tts_sem:   # ← now per-session, not global
            if turn_id != self._turn_seq:
                logger.debug(f"[{self._session_id}] Turn {turn_id} superseded — skipping TTS")
                return

            self._speaking = True
            try:
                await self._ws.send_json({"type": "tts_start", "turn_id": turn_id})

                for attempt in range(1, API_MAX_RETRIES + 1):
                    if self._interrupt.is_set() or turn_id != self._turn_seq:
                        logger.debug(f"[{self._session_id}] TTS turn {turn_id} aborted")
                        return

                    try:
                        seq_in_turn = 0

                        # Cartesia streaming TTS
                        # output_format: raw PCM → mp3 is fine for browser
                        # container="raw" + encoding="mp3" → streaming MP3 bytes
                        async for chunk in await cartesia_client.tts.sse(
                            model_id=CARTESIA_MODEL,
                            transcript=text,
                            voice={"id": CARTESIA_VOICE_ID},
                            output_format={
                                "container": "raw",
                                "encoding":  "mp3",
                                "sample_rate": 44100,
                            },
                            stream=True,
                        ):
                            if self._interrupt.is_set() or turn_id != self._turn_seq:
                                logger.debug(
                                    f"[{self._session_id}] TTS turn {turn_id} interrupted mid-stream"
                                )
                                return

                            audio_bytes: bytes = chunk.get("audio", b"")
                            if not audio_bytes:
                                continue

                            # Same 8-byte framing as before:
                            # [turn_id: uint32][seq_in_turn: uint32] big-endian
                            header = struct.pack(">II", turn_id, seq_in_turn)
                            seq_in_turn += 1
                            await self._ws.send_bytes(header + audio_bytes)

                        break   # stream finished normally

                    except asyncio.TimeoutError:
                        logger.warning(f"[{self._session_id}] Cartesia timeout (attempt {attempt})")
                        if attempt == API_MAX_RETRIES:
                            break
                        await asyncio.sleep(2 ** attempt)
                    except Exception as exc:
                        logger.error(f"[{self._session_id}] Cartesia TTS error: {exc}", exc_info=exc)
                        break

            finally:
                self._speaking = False
                self._interrupt.clear()
                if turn_id == self._turn_seq:
                    await self._ws.send_json({"type": "tts_end", "turn_id": turn_id})

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
    logger.info(
        f"[{session_id}] New connection from {peer_ip} "
        f"(total={len(sessions)}, from_ip={ip_session_count[peer_ip]})"
    )

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning(f"[{session_id}] Malformed JSON — ignoring")
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


def _add_ice_candidate(session_id: str, pc: RTCPeerConnection, cand: dict) -> None:
    async def _do_add() -> None:
        try:
            raw = cand.get("candidate", "")
            if not raw:
                return
            parts = raw.split()
            if len(parts) < 8:
                return
            ice = RTCIceCandidate(
                component     = int(parts[1]),
                foundation    = parts[0].replace("candidate:", ""),
                ip            = parts[4],
                port          = int(parts[5]),
                priority      = int(parts[3]),
                protocol      = parts[2],
                type          = parts[7],
                sdpMid        = cand.get("sdpMid"),
                sdpMLineIndex = cand.get("sdpMLineIndex"),
            )
            await pc.addIceCandidate(ice)
        except Exception as exc:
            logger.debug(f"[{session_id}] ICE add error (usually ok): {exc}")

    task = asyncio.create_task(_do_add())
    task.add_done_callback(
        lambda t: logger.debug(f"[{session_id}] ICE task exc: {t.exception()}")
        if not t.cancelled() and t.exception() else None
    )


async def handle_index(request: web.Request) -> web.Response:
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    async with aiofiles.open(index_path, "r") as f:
        content = await f.read()
    return web.Response(content_type="text/html", text=content)


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(
        text=json.dumps({
            "status": "ok",
            "sessions": len(sessions),
            "max_sessions": MAX_SESSIONS,
        }),
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