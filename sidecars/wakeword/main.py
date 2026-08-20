#!/usr/bin/env python3
"""
SpectreSTT — openWakeWord sidecar for "Hey Linus" detection.

IPC protocol (line-oriented, via stdin/stdout):

  stdout → Go:
    "DETECTED\n"  — wake phrase scored above threshold.
                    The audio stream is stopped *before* this line is written,
                    so the Go PortAudio capturer can open the device immediately.

  stdin ← Go:
    "PAUSE\n"     — suspend inference and keep the audio stream closed.
    "RESUME\n"    — reopen the audio stream and restart inference.

Wake word model:
  ⚠ STUB: "Hey Linus" requires a custom-trained openWakeWord model.
  Custom training uses openWakeWord's synthetic TTS pipeline and is scoped
  as a separate task. Until that model exists, this sidecar uses the closest
  available stock model ("hey_mycroft") as a placeholder.

  The model is selected by:
    1. --model <path>  (explicit path to an .onnx model file)
    2. Fallback: "hey_mycroft" stock model (bad accuracy for "Hey Linus",
       but lets the sidecar run and the pipeline be tested end-to-end).

  Detection threshold (--threshold, default 0.5):
    Tune against real usage. Lower = more sensitive (more false positives).
    The stock placeholder model's threshold has no bearing on the final
    "Hey Linus" model — retune after the real model is trained.

License note:
  openWakeWord pretrained models are CC-BY-NC-SA 4.0 (non-commercial).
  The library code itself is Apache 2.0.
  This is acceptable for SpectreSTT's personal/community-release scope.
  If SpectreSTT's distribution ever shifts toward anything monetised,
  replace the pretrained model with a custom-trained one or a different engine.
"""

import argparse
import sys
import threading
import time

import numpy as np
import sounddevice as sd

try:
    from openwakeword.model import Model as OWWModel
except ImportError:
    print("ERROR: openwakeword not installed. Run: pip install -r requirements.txt", flush=True)
    sys.exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000      # Hz — must match Aximo + Go VAD
CHUNK_SAMPLES = 1_280     # 80ms at 16kHz — openWakeWord's recommended chunk size
CHANNELS = 1

# ─── State ────────────────────────────────────────────────────────────────────

_paused = threading.Event()   # set = paused (inference suppressed)
_stop   = threading.Event()   # set = shutdown requested


def _stdin_reader() -> None:
    """
    Reads PAUSE / RESUME commands from Go (via stdin) in a background thread.
    This runs as a daemon so it exits automatically when the main thread exits.
    """
    for raw_line in sys.stdin:
        cmd = raw_line.strip()
        if cmd == "PAUSE":
            _paused.set()
        elif cmd == "RESUME":
            _paused.clear()
        elif cmd == "":
            pass  # blank line — ignore
        else:
            # Unknown command — log to stderr (not stdout, which is the IPC channel).
            print(f"[wakeword] unknown command: {cmd!r}", file=sys.stderr, flush=True)

    # stdin closed (Go process exited) — signal shutdown.
    _stop.set()
    _paused.clear()  # unblock main loop so it can exit cleanly


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SpectreSTT openWakeWord sidecar")
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Path to an openWakeWord-compatible .onnx model. "
            "If omitted, the 'hey_mycroft' stock placeholder is used. "
            "⚠ This placeholder has poor accuracy for 'Hey Linus' — "
            "train a real model and set this flag."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection score threshold (0–1). Default: 0.5. Tune against real usage.",
    )
    args = parser.parse_args()

    threshold: float = args.threshold
    model_path: str  = args.model

    # ── Load model ──────────────────────────────────────────────────────────
    if model_path:
        print(f"[wakeword] loading model: {model_path}", file=sys.stderr, flush=True)
        oww = OWWModel(wakeword_models=[model_path], inference_framework="onnx")
        model_key = list(oww.models.keys())[0]
    else:
        print(
            "[wakeword] ⚠ No --model specified. Using 'hey_mycroft' stock placeholder.\n"
            "          Accuracy for 'Hey Linus' will be poor until a real model is trained.",
            file=sys.stderr,
            flush=True,
        )
        oww = OWWModel(wakeword_models=["hey_mycroft"], inference_framework="onnx")
        model_key = "hey_mycroft"

    print(f"[wakeword] model key={model_key!r} threshold={threshold}", file=sys.stderr, flush=True)

    # ── Start stdin reader thread ────────────────────────────────────────────
    t = threading.Thread(target=_stdin_reader, daemon=True)
    t.start()

    # ── Audio capture + inference loop ──────────────────────────────────────
    audio_buffer = np.zeros(CHUNK_SAMPLES, dtype=np.int16)

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback — copies mic data into audio_buffer."""
        if status:
            print(f"[wakeword] audio status: {status}", file=sys.stderr, flush=True)
        audio_buffer[:] = indata[:, 0]

    print("[wakeword] opening audio stream", file=sys.stderr, flush=True)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback,
    ) as stream:

        print("[wakeword] listening for wake phrase", file=sys.stderr, flush=True)

        while not _stop.is_set():
            if _paused.is_set():
                # Pipeline is active — suspend and release audio device.
                if stream.active:
                    stream.stop()
                time.sleep(0.05)
                continue

            # Resume audio if we were paused.
            if not stream.active:
                stream.start()

            # Run inference on the latest chunk.
            prediction = oww.predict(audio_buffer)
            score = prediction.get(model_key, 0.0)

            if score >= threshold:
                # ── Detection ────────────────────────────────────────────────
                # 1. Stop audio stream FIRST to release the device before Go
                #    opens its own PortAudio stream.
                stream.stop()

                # 2. Set paused so the loop stays suspended until RESUME.
                _paused.set()

                # 3. Signal Go.
                print("DETECTED", flush=True)

                # Reset model state to avoid immediate re-triggering when
                # inference resumes after RESUME.
                oww.reset()

            # Small sleep to avoid spinning the CPU at full tilt between chunks.
            # sounddevice fills audio_buffer via callback; we just need to
            # check it at a reasonable rate.
            time.sleep(0.01)

    print("[wakeword] exiting", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
