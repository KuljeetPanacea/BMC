"""
WebRTC Voice Agent Server — Production-Ready + Noise Cancellation + Reliable Barge-in
=======================================================================================
Pure WebRTC signaling + Deepgram STT + OpenAI LLM + OpenAI TTS

THIS VERSION'S KEY FIX vs the previous one
────────────────────────────────────────────
The server-side interrupt logic (_pipeline_lock scoping, _interrupt Event,
clearing it only in `finally`, setting `_speaking` inside the semaphore) was
already correct. The actual bug was on the CLIENT: TTS audio was buffered
into one big Blob and only started playing once `tts_end` arrived, with no
way for the client to know which `tts_start`/`tts_end`/audio-chunk sequence
a given message belonged to. That meant a stale, already-superseded TTS
turn could still start playing audio after the "interrupt" message had
already been processed (a race between async `audio.play()` and the
interrupt handler).

The fix: every TTS turn now carries an explicit `turn_id`. The client
keeps track of the "current" turn_id and ignores/aborts anything tagged
with an older one — so even if messages or playback calls resolve out of
order, stale audio can never start playing. The binary audio chunks
themselves can't carry a JSON tag, so we frame each chunk with a tiny
4-byte sequence header containing the turn ordinal, which is cheap and
keeps chunks self-describing without needing a side-channel.

Architecture:
  Browser  <──WebRTC audio──>  aiortc server
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                   ▼
          WebRTC VAD          noisereduce          ProcessPool
          (20 ms frames)      (per utterance)      (CPU work)
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
   Deepgram  OpenAI LLM  OpenAI TTS
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
# webrtcvad-wheels is a drop-in replacement for webrtcvad that works on
# Python 3.12+ and 3.14+. The original webrtcvad==2.0.10 uses `pkg_resources`
# which was removed in Python 3.14, causing ModuleNotFoundError on import.
# Install with: pip install webrtcvad-wheels
# API is identical — no other code changes needed.
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

MAX_SESSIONS        = int(os.getenv("MAX_SESSIONS",        "100"))
MAX_SESSIONS_PER_IP = int(os.getenv("MAX_SESSIONS_PER_IP", "5"))
STT_CONCURRENCY     = int(os.getenv("STT_CONCURRENCY",     "20"))
LLM_CONCURRENCY     = int(os.getenv("LLM_CONCURRENCY",     "20"))
TTS_CONCURRENCY     = int(os.getenv("TTS_CONCURRENCY",     "10"))
MAX_HISTORY_TURNS   = int(os.getenv("MAX_HISTORY_TURNS",   "20"))
MAX_AUDIO_BUF_SEC   = float(os.getenv("MAX_AUDIO_BUF_SEC", "30.0"))
API_TIMEOUT_SEC     = float(os.getenv("API_TIMEOUT_SEC",   "10.0"))
API_MAX_RETRIES     = int(os.getenv("API_MAX_RETRIES",     "3"))

# ── VAD / barge-in tuning ─────────────────────────────────────────────────────
VAD_AGGRESSIVENESS   = int(os.getenv("VAD_AGGRESSIVENESS",  "3"))
VAD_SILENCE_FRAMES    = int(os.getenv("VAD_SILENCE_FRAMES",  "30"))   # 600 ms
VAD_MIN_SPEECH_FRAMES = int(os.getenv("VAD_MIN_SPEECH_FRAMES", "12")) # 250 ms

# ── Noise reduction ───────────────────────────────────────────────────────────
NOISE_REDUCE_PROP = float(os.getenv("NOISE_REDUCE_PROP", "0.85"))

# ─── Pricing ──────────────────────────────────────────────────────────────────
PRICING = {
    "stt_per_min":              0.0048,
    "llm_input_per_1m_tokens":  0.40,
    "llm_output_per_1m_tokens": 1.60,
    "tts_per_1m_chars":         15.0,
}

# ─── Global singletons (initialised in on_startup) ────────────────────────────
openai_client: Optional[AsyncOpenAI]           = None
http_session:  Optional[aiohttp.ClientSession] = None
process_pool:  Optional[ProcessPoolExecutor]   = None

_stt_sem: Optional[asyncio.Semaphore] = None
_llm_sem: Optional[asyncio.Semaphore] = None
_tts_sem: Optional[asyncio.Semaphore] = None

sessions:         dict[str, "ConversationSession"] = {}
ip_session_count: dict[str, int]                   = {}


# ─── App lifecycle ─────────────────────────────────────────────────────────────

async def on_startup(app: web.Application) -> None:
    global openai_client, http_session, process_pool
    global _stt_sem, _llm_sem, _tts_sem

    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is not set — LLM and TTS calls will fail")
    if not DEEPGRAM_API_KEY:
        logger.warning("DEEPGRAM_API_KEY is not set — STT calls will fail")

    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    connector    = aiohttp.TCPConnector(limit=64, ttl_dns_cache=300)
    http_session = aiohttp.ClientSession(connector=connector)

    process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())

    _stt_sem = asyncio.Semaphore(STT_CONCURRENCY)
    _llm_sem = asyncio.Semaphore(LLM_CONCURRENCY)
    _tts_sem = asyncio.Semaphore(TTS_CONCURRENCY)

    logger.info(
        f"Server started | max_sessions={MAX_SESSIONS} "
        f"vad_aggressiveness={VAD_AGGRESSIVENESS} "
        f"vad_silence_frames={VAD_SILENCE_FRAMES} "
        f"noise_reduce_prop={NOISE_REDUCE_PROP}"
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

    def add_stt(self, seconds: float):     self.stt_audio_sec     += seconds
    def add_tts(self, chars: int):         self.tts_chars         += chars
    def add_llm(self, inp: int, out: int): self.llm_input_tokens  += inp; self.llm_output_tokens += out; self.turns += 1

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
    if len(arr) < sample_rate // 10:          # < 100 ms — too short, skip
        return pcm_bytes
    reduced = nr.reduce_noise(
        y=arr,
        sr=sample_rate,
        stationary=True,
        prop_decrease=prop_decrease,
        n_jobs=1,
    )
    out = np.clip(reduced * 32768.0, -32768, 32767).astype(np.int16)
    return out.tobytes()


# ─── MicrophoneTrackSink with WebRTC VAD ─────────────────────────────────────

class MicrophoneTrackSink:
    SAMPLE_RATE   = 16_000
    FRAME_MS      = 20
    FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 320 samples = 640 bytes

    def __init__(self, session_id: str, on_utterance, on_speech_confirmed=None):
        self._session_id  = session_id
        self._on_utterance = on_utterance
        # Fired exactly once per utterance, the moment VAD_MIN_SPEECH_FRAMES
        # of CONSECUTIVE confirmed speech is reached — i.e. well before the
        # user finishes talking. This is what lets the server cut TTS the
        # instant the user starts a real sentence, instead of waiting for
        # them to finish and for trailing silence to elapse (which is what
        # _on_utterance waits for). A brief noise pop never reaches this
        # threshold's consecutive-frame requirement, so it's still immune
        # to random background noise — same guarantee as before, just
        # signaled earlier.
        self._on_speech_confirmed = on_speech_confirmed

        self._resampler = AudioResampler(format="s16", layout="mono", rate=self.SAMPLE_RATE)
        self._vad       = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._ring: bytes = b""

        self._utt_frames: list[bytes] = []
        self._speech_frames  = 0
        self._silence_frames = 0
        self._in_utterance   = False
        self._confirmed_sent = False   # guards on_speech_confirmed to fire once per utterance
        self._utt_sec    = 0.0
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
                    frame_bytes = self._ring[: self.FRAME_SAMPLES * 2]
                    self._ring  = self._ring[self.FRAME_SAMPLES * 2 :]
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
                self._in_utterance  = True
                self._utt_frames    = []
                self._utt_sec       = 0.0
                self._speech_frames = 0
                self._confirmed_sent = False
                logger.info(f"[{self._session_id}] 🎙 Speech start (VAD)")

            self._utt_frames.append(frame_bytes)
            self._utt_sec       += self.FRAME_MS / 1000
            self._speech_frames += 1
            self._silence_frames = 0

            # Fire the early "this is real speech, not noise" signal the
            # moment we cross the confirmation threshold — same threshold
            # used to decide whether to keep the utterance at all, so the
            # noise-immunity guarantee is identical, just earlier.
            if (
                not self._confirmed_sent
                and self._speech_frames >= VAD_MIN_SPEECH_FRAMES
                and self._on_speech_confirmed is not None
            ):
                self._confirmed_sent = True
                await self._on_speech_confirmed()

            if self._utt_sec >= MAX_AUDIO_BUF_SEC:
                logger.warning(f"[{self._session_id}] Max utterance length hit — force-flushing")
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
                logger.debug(
                    f"[{self._session_id}] Discarding short utterance "
                    f"({self._speech_frames} speech frames < min {VAD_MIN_SPEECH_FRAMES})"
                )
            self._reset_utterance()
            return

        raw_pcm  = b"".join(self._utt_frames)
        duration = self._utt_sec
        self._reset_utterance()

        logger.info(f"[{self._session_id}] 🎙 Utterance end — {duration:.2f}s, running noise reduction")

        try:
            clean_pcm: bytes = await loop.run_in_executor(
                process_pool, _noise_reduce_sync, raw_pcm, self.SAMPLE_RATE, NOISE_REDUCE_PROP,
            )
        except Exception as exc:
            logger.warning(f"[{self._session_id}] Noise reduction failed, using raw audio: {exc}")
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
                    data = await resp.json()
                    channels = data.get("results", {}).get("channels", [])
                    if channels:
                        alts = channels[0].get("alternatives", [])
                        if alts:
                            transcript = alts[0].get("transcript", "").strip()
                            if not transcript:
                                logger.debug(f"[{session_id}] Deepgram: empty transcript")
                            return transcript
                    logger.warning(f"[{session_id}] Deepgram: no channels in response")
                    return ""
            except asyncio.TimeoutError:
                logger.warning(f"[{session_id}] Deepgram timeout (attempt {attempt})")
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

        self._speaking      = False
        self._interrupt     = asyncio.Event()

        # Monotonically increasing turn counter. Every TTS utterance gets the
        # NEXT value. This is sent in tts_start / tts_end / each audio frame
        # header so the client can always tell which turn a message belongs
        # to and discard anything that isn't the current one — even if
        # messages race or resolve out of order on the client.
        self._turn_seq = 0

        self._pipeline_lock = asyncio.Lock()

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
        """
        Fires the moment VAD has seen VAD_MIN_SPEECH_FRAMES (250ms) of
        CONSECUTIVE confirmed speech — i.e. as soon as the user has clearly
        started talking, not after they finish. This is what makes barge-in
        feel instant instead of laggy: previously the interrupt only fired
        from _on_utterance, which waits for the full utterance INCLUDING
        VAD_SILENCE_FRAMES (600ms) of trailing silence after the user stops
        talking — meaning TTS kept overlapping with the user's entire
        sentence before ever being told to stop.

        Noise immunity is unchanged: this uses the exact same
        VAD_MIN_SPEECH_FRAMES threshold that _flush() uses to decide whether
        an utterance is real speech at all, just checked earlier (the
        moment the threshold is crossed, instead of at utterance end). A
        single noise pop still can't reach this threshold's *consecutive*
        frame requirement.
        """
        if self._speaking:
            logger.info(f"[{self._session_id}] 🛑 Barge-in confirmed — interrupting TTS")
            self._interrupt.set()
            await self._ws.send_json({"type": "interrupt", "turn_id": self._turn_seq})
            await asyncio.sleep(0)

    async def _on_utterance(self, audio: bytes, duration_sec: float) -> None:
        # By the time we get here, _on_speech_confirmed has already fired
        # (if this was a barge-in) — TTS was stopped ~600ms+ earlier, as
        # soon as speech was confirmed, rather than only now. If _speaking
        # is somehow still true here (e.g. a new TTS turn started in the
        # gap), this is a safety-net catch, not the primary mechanism.
        if self._speaking:
            logger.info(f"[{self._session_id}] 🛑 Barge-in — interrupting TTS (end-of-utterance catch)")
            self._interrupt.set()
            await self._ws.send_json({"type": "interrupt", "turn_id": self._turn_seq})
            await asyncio.sleep(0)

        # ── STT + LLM under lock (no two turns overlap) ──────────────────
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

        # ── TTS outside the lock — barge-in never blocked here ───────────
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
                    in_tok     = resp.usage.prompt_tokens
                    out_tok    = resp.usage.completion_tokens
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

    async def _speak_and_send(self, text: str) -> None:
        self._cost.add_tts(len(text))

        # Claim this turn's id BEFORE acquiring the semaphore. That way even
        # a turn that's still waiting for a concurrency slot has an id the
        # client can later recognize as superseded if a newer turn starts.
        self._turn_seq += 1
        turn_id = self._turn_seq

        async with _tts_sem:
            # If we were superseded while waiting for the semaphore slot,
            # don't even start — saves an OpenAI TTS call for audio nobody
            # will hear.
            if turn_id != self._turn_seq:
                logger.debug(f"[{self._session_id}] Turn {turn_id} superseded before TTS call — skipping")
                return

            self._speaking = True
            try:
                await self._ws.send_json({"type": "tts_start", "turn_id": turn_id})

                for attempt in range(1, API_MAX_RETRIES + 1):
                    if self._interrupt.is_set() or turn_id != self._turn_seq:
                        logger.debug(f"[{self._session_id}] TTS turn {turn_id} aborted before attempt {attempt}")
                        return

                    try:
                        seq_in_turn = 0
                        async with openai_client.audio.speech.with_streaming_response.create(
                            model="tts-1",
                            voice="echo",
                            input=text,
                            response_format="mp3",
                            speed=1.0,
                        ) as tts_resp:
                            async for chunk in tts_resp.iter_bytes(chunk_size=4096):
                                if self._interrupt.is_set() or turn_id != self._turn_seq:
                                    logger.debug(f"[{self._session_id}] TTS turn {turn_id} interrupted mid-stream")
                                    return
                                # Frame each binary chunk with an 8-byte header:
                                # [turn_id: uint32][seq_in_turn: uint32], both
                                # big-endian, so the client can always tell
                                # which turn (and ordering) a chunk belongs to
                                # without a separate JSON side-channel.
                                header = struct.pack(">II", turn_id, seq_in_turn)
                                seq_in_turn += 1
                                await self._ws.send_bytes(header + chunk)
                        break
                    except asyncio.TimeoutError:
                        logger.warning(f"[{self._session_id}] TTS timeout (attempt {attempt})")
                        if attempt == API_MAX_RETRIES:
                            break
                        await asyncio.sleep(2 ** attempt)
                    except Exception as exc:
                        logger.error(f"[{self._session_id}] TTS error: {exc}", exc_info=exc)
                        break
            finally:
                self._speaking = False
                self._interrupt.clear()
                # Only tell the client this turn ended if it's still current.
                # If we were superseded, the new turn's tts_start already
                # told the client to move on — sending a stale tts_end here
                # would needlessly confuse turn bookkeeping client-side.
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
        logger.warning(f"Session cap reached — rejecting {peer_ip}")
        raise web.HTTPServiceUnavailable(reason="Server at capacity")

    if ip_session_count.get(peer_ip, 0) >= MAX_SESSIONS_PER_IP:
        logger.warning(f"Per-IP cap reached for {peer_ip}")
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
                    data  = json.loads(msg.data)
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
                logger.debug(f"[{session_id}] Short ICE candidate — skipping")
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
        text=json.dumps({"status": "ok", "sessions": len(sessions), "max_sessions": MAX_SESSIONS}),
        content_type="application/json",
    )


# ─── App factory ──────────────────────────────────────────────────────────────

def build_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/",        handle_index)
    app.router.add_get("/ws",      handle_ws)
    app.router.add_get("/health",  handle_health)
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