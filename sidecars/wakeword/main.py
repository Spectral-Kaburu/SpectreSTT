#!/usr/bin/env python3
"""
SpectreSTT — openWakeWord sidecar for "Hey Linus" detection.

IPC protocol (line-oriented, via stdin/stdout):

  stdout → Go:
    "DETECTED\\n"  — wake phrase scored above threshold.
                    The audio stream is stopped *before* this line is written,
                    so the Go PortAudio capturer can open the device immediately.

  stdin ← Go:
    "PAUSE\\n"     — suspend inference and keep the audio stream closed.
    "RESUME\\n"    — reopen the audio stream and restart inference.

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
import logging
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import openwakeword.utils
openwakeword.utils.download_models()  # check current signature — may accept a framework or model-list arg

try:
    from openwakeword.model import Model as OWWModel
    import openwakeword.utils
    openwakeword.utils.download_models()  # check current signature — may accept a framework or model-list arg
except ImportError:
    # Print directly to stderr before the logger is set up — this is the very
    # first thing that can go wrong and we want it visible regardless.
    print(
        "\033[1;31mERROR\033[0m  openwakeword not installed. "
        "Run: pip install -r requirements.txt",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


INSTRUCT_LEVEL = 25
logging.addLevelName(INSTRUCT_LEVEL, "INSTRUCT")

def instruct(self, message, *args, **kws):
    if self.isEnabledFor(INSTRUCT_LEVEL):
        self._log(INSTRUCT_LEVEL, message, args, **kws)
logging.Logger.instruct = instruct

# ─── Colored logging ──────────────────────────────────────────────────────────

class _ColorFormatter(logging.Formatter):
    """
    A stderr formatter that prefixes each record with a fixed-width, ANSI-
    colored level indicator and the logger name (used as the component tag).

    Example output:
      20:03:51  INFO   wakeword  listening for wake phrase  model=hey_mycroft threshold=0.5
    """

    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"

    _LEVEL_STYLES: dict[int, str] = {
        logging.DEBUG:    "\033[36m",        # cyan
        logging.INFO:     "\033[32m",        # green
        INSTRUCT_LEVEL:   "\033[34m",        # blue
        logging.WARNING:  "\033[33m",        # yellow
        logging.ERROR:    "\033[31m",        # red
        logging.CRITICAL: "\033[1;31m",      # bold red
    }

    def format(self, record: logging.LogRecord) -> str:
        color  = self._LEVEL_STYLES.get(record.levelno, "")
        level  = f"{color}{record.levelname:<8}{self._RESET}"
        time_s = self.formatTime(record, datefmt="%H:%M:%S")
        msg    = record.getMessage()

        # Extra key=value pairs attached via the `extra` dict or logger.xxx(…, key=val).
        extras = ""
        skip = {"name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "taskName",
                "message"}
        kv_pairs = [
            f"\033[2m{k}=\033[0m{v!r}"
            for k, v in record.__dict__.items()
            if k not in skip
        ]
        if kv_pairs:
            extras = "  " + "  ".join(kv_pairs)

        tag = f"\033[2m{record.name}\033[0m"
        return f"{time_s}  {level}  {tag}  {msg}{extras}"


def _setup_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ColorFormatter())
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    return logging.getLogger("wakeword")


log = _setup_logging()


# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE    = 16_000   # Hz — must match Aximo + Go VAD
CHUNK_SAMPLES  = 1_280    # 80ms at 16kHz — openWakeWord's recommended chunk size
CHANNELS       = 1

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
            log.warning("unknown command from Go: %s", repr(cmd))

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
        log.info("loading model", extra={"path": model_path})
        oww = OWWModel(wakeword_models=[model_path], inference_framework="onnx")
        model_key = list(oww.models.keys())[0]
        # Just guessing the phrase from the filename roughly if it's a custom model
        log.instruct(f"model loaded, try saying '{model_key.replace('_', ' ')}' to wake the assistant")
    else:
        log.warning(
            "no --model specified — using 'hey_mycroft' stock placeholder; "
            "accuracy for 'Hey Linus' will be poor until a real model is trained"
        )
        oww = OWWModel(wakeword_models=["hey_mycroft"], inference_framework="onnx")
        model_key = "hey_mycroft"
        log.instruct("💡 SAY: 'Hey Mycroft' to wake the assistant")

    log.info("model ready", extra={"key": model_key, "threshold": threshold})

    # ── Start stdin reader thread ────────────────────────────────────────────
    t = threading.Thread(target=_stdin_reader, daemon=True)
    t.start()

    import queue
    audio_queue = queue.Queue()

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback — queues mic data."""
        if status:
            log.warning("audio status", extra={"status": str(status)})
        audio_queue.put(indata.copy())

    log.info("opening audio stream and listening for wake phrase")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback,
    ) as stream:

        while not _stop.is_set():
            if _paused.is_set():
                # Pipeline is active — suspend and release audio device.
                if stream.active:
                    stream.stop()
                # Clear out any stale audio
                while not audio_queue.empty():
                    audio_queue.get_nowait()
                time.sleep(0.05)
                continue

            # Resume audio if we were paused.
            if not stream.active:
                stream.start()

            try:
                # Get the next chunk of audio, blocking for up to 0.1s
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Run inference on the chunk.
            prediction = oww.predict(chunk[:, 0])
            score = prediction.get(model_key, 0.0)

            if score >= threshold:
                # ── Detection ────────────────────────────────────────────────
                # 1. Stop audio stream FIRST to release the device before Go
                #    opens its own PortAudio stream.
                stream.stop()

                # 2. Set paused so the loop stays suspended until RESUME.
                _paused.set()

                log.info("wake phrase detected", extra={"score": f"{score:.3f}"})

                # 3. Signal Go.
                print("DETECTED", flush=True)

                # Reset model state to avoid immediate re-triggering when
                # inference resumes after RESUME.
                oww.reset()

    log.info("exiting")


if __name__ == "__main__":
    main()
