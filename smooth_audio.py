"""Smooth, glitch-free local audio output for Windows.

Why the stock transport stutters
--------------------------------
Pipecat's ``LocalAudioOutputTransport`` plays audio with *blocking* PyAudio
writes, and that blocking write IS the pacing mechanism (see
``base_output.py`` ``_audio_task_handler`` -> ``write_audio_frame``). So
playback is driven by an asyncio task that competes with STT, VAD and LLM
token streaming. On Windows' default MME host API (small device buffer),
whenever the event loop is briefly busy the device starves and you hear
"a few words, then silence, then a few more words".

The fix
-------
Open the output stream in **callback mode** and feed it from our own ring
buffer:

* PortAudio pulls audio on its *own* thread, so playback never stalls when the
  asyncio loop is busy.
* We pre-roll a small cushion (PREROLL_MS) before draining, so scheduling
  jitter is absorbed.
* ``write_audio_frame`` applies back-pressure (caps the buffer at MAX_BUFFER_MS)
  so latency stays bounded and Pipecat stays paced.
* On an ``InterruptionFrame`` (barge-in) we clear the buffer instantly so the
  bot stops talking right away.

This keeps the simple default device (mono, 24 kHz with MME format
conversion) and avoids WASAPI's shared-mode constraints (which would force a
24k->44.1k resample + mono->stereo upmix because Deepgram Aura can't emit
44.1 kHz).
"""

import asyncio
import threading

import pyaudio
from loguru import logger

from pipecat.frames.frames import Frame, InterruptionFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.local.audio import (
    LocalAudioOutputTransport,
    LocalAudioTransport,
)

PREROLL_MS = 150        # cushion before playback starts; absorbs scheduling jitter
MAX_BUFFER_MS = 400     # back-pressure cap. Larger than the 8B days because the Scout
                        # models add event-loop load; still low-latency, and flushed
                        # instantly on InterruptionFrame so barge-in stays responsive.


class SmoothLocalAudioOutputTransport(LocalAudioOutputTransport):
    """Callback-mode local audio output with a jitter-absorbing ring buffer."""

    async def start(self, frame: StartFrame):
        # Grandparent start (skip the stock blocking-write stream open).
        await BaseOutputTransport.start(self, frame)

        if self._out_stream:
            return

        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        self._channels = self._params.audio_out_channels
        self._bytes_per_frame = 2 * self._channels  # paInt16

        self._buf = bytearray()
        self._buf_lock = threading.Lock()
        self._prerolled = False
        self._preroll_bytes = self._ms_to_bytes(PREROLL_MS)
        self._max_bytes = self._ms_to_bytes(MAX_BUFFER_MS)
        self._underruns = 0
        self._mon_task = None

        cb_frames = int(self._sample_rate / 100)  # 10 ms callback granularity
        self._out_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._channels,
            rate=self._sample_rate,
            frames_per_buffer=cb_frames,
            output=True,
            output_device_index=self._params.output_device_index,
            stream_callback=self._out_callback,
        )
        self._out_stream.start_stream()

        logger.info(
            f"Smooth audio out (callback): {self._sample_rate} Hz x{self._channels}ch, "
            f"preroll={PREROLL_MS}ms, cap={MAX_BUFFER_MS}ms"
        )

        self._mon_task = asyncio.create_task(self._monitor())
        await self.set_transport_ready(frame)

    async def _monitor(self):
        """Log underrun count periodically (off the realtime callback thread)."""
        last = 0
        try:
            while True:
                await asyncio.sleep(3)
                if self._underruns != last:
                    logger.warning(f"audio underruns so far: {self._underruns}")
                    last = self._underruns
        except asyncio.CancelledError:
            pass

    async def cleanup(self):
        if self._mon_task:
            self._mon_task.cancel()
            self._mon_task = None
        await super().cleanup()

    def _ms_to_bytes(self, ms: int) -> int:
        return int(self._sample_rate * ms / 1000) * self._bytes_per_frame

    def _out_callback(self, in_data, frame_count, time_info, status):
        """Runs on PortAudio's thread — pull audio, immune to asyncio stalls."""
        need = frame_count * self._bytes_per_frame
        with self._buf_lock:
            if not self._prerolled:
                if len(self._buf) >= self._preroll_bytes:
                    self._prerolled = True
                else:
                    return (b"\x00" * need, pyaudio.paContinue)

            if len(self._buf) >= need:
                data = bytes(self._buf[:need])
                del self._buf[:need]
            else:
                # Underrun: emit what we have, pad with silence, rebuild cushion.
                data = bytes(self._buf) + b"\x00" * (need - len(self._buf))
                self._buf.clear()
                self._prerolled = False
                self._underruns += 1
        return (data, pyaudio.paContinue)

    async def write_audio_frame(self, frame) -> bool:
        """Feed the ring buffer; apply back-pressure to keep latency bounded."""
        # Back-pressure: wait while the buffer is full (this paces Pipecat,
        # replacing the role the stock blocking write played).
        while True:
            with self._buf_lock:
                if len(self._buf) < self._max_bytes:
                    self._buf.extend(frame.audio)
                    break
            await asyncio.sleep(0.005)
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Barge-in: drop buffered bot audio immediately so it stops talking.
        if isinstance(frame, InterruptionFrame):
            with self._buf_lock:
                self._buf.clear()
                self._prerolled = False
        await super().process_frame(frame, direction)


class SmoothLocalAudioTransport(LocalAudioTransport):
    """Drop-in LocalAudioTransport whose output uses the smooth callback path."""

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = SmoothLocalAudioOutputTransport(self._pyaudio, self._params)
        return self._output
