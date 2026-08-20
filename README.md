# SpectreSTT

Privacy-first, local speech-to-text adapter. Listens for "Hey Linus," captures
the utterance, transcribes it via [Aximo](https://github.com/agent-axiom/aximo),
and pushes the transcript over a Unix socket to any connected client.

---

## Architecture

```
openWakeWord sidecar (Python)
    │  "Hey Linus" detected
    │  DETECTED → stdout
    ▼
Go pipeline
    │  PAUSE → sidecar stdin
    │  PortAudio capture starts
    ▼
WebRTC VAD (30ms frames, 800ms EOT threshold, 30s cap)
    │  trimmed PCM buffer
    ▼
Aximo HTTP  POST /v1/transcriptions  (audio/wav, 16kHz mono)
    │  2 attempts, 400ms delay; TTS → notify-send on failure
    ▼
Unix socket broadcast  →  Linus / any connected client
```

## System Requirements

| Dependency | Purpose | Install |
|---|---|---|
| Docker + docker compose | Aximo container lifecycle | [docs.docker.com](https://docs.docker.com/get-docker/) |
| libportaudio2 | PortAudio (Go audio capture) | `apt install libportaudio2` |
| libwebrtc-audio-processing | WebRTC VAD (cgo) | `apt install libwebrtc-audio-processing-dev` |
| Python ≥ 3.10 | openWakeWord sidecar | system or pyenv |
| notify-send | Desktop notification fallback | `apt install libnotify-bin` |

## Build

```bash
go build -o spectrestt ./cmd/spectrestt
```

## Python sidecar setup

```bash
cd sidecars/wakeword
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

```bash
cp spectrestt.example.json spectrestt.json
# Edit spectrestt.json — set aximo_compose_path and wakeword_sidecar_path at minimum
```

> [!NOTE]
> The example config uses `DisallowUnknownFields`, so the `_comment` key in
> `.example.json` will cause a parse error if you try to use it directly.
> Always copy to `spectrestt.json` first.

## Running

```bash
./spectrestt --config spectrestt.json
```

Transcripts are pushed as newline-delimited JSON to the configured `socket_path`:

```json
{"text":"turn off the lights","duration_ms":1420,"processing_ms":38,"timestamp":"2026-08-20T16:25:17Z"}
```

## Wake word — ⚠ Placeholder model

"Hey Linus" requires a **custom-trained openWakeWord model** (synthetic TTS
training pipeline). Until that model exists, the sidecar falls back to the
`hey_mycroft` stock model as a placeholder. Set `wakeword_model_path` in your
config once the real model is trained.

Custom training is scoped as a separate task. See `sidecars/wakeword/main.py`
for the model-loading logic.

## License notes

- SpectreSTT (this repo): MIT
- openWakeWord library: Apache 2.0
- openWakeWord **pretrained models**: CC-BY-NC-SA 4.0 (**non-commercial**)
  If SpectreSTT's distribution ever shifts toward anything monetised, the
  pretrained models must be replaced with a custom-trained model or a
  differently-licensed engine.

## Open parameters (flagged for review)

| Parameter | Value chosen | Flag |
|---|---|---|
| Startup readiness timeout | 40s (hard max 60s) | Tune if Aximo cold-starts slower |
| Readiness cache TTL | 30 min | Reset on any Transcribe failure |
| VAD silence threshold | 800ms | Increase for slower speech patterns |
| VAD frame size | 30ms | WebRTC VAD docs recommend this as most reliable |
| Max utterance cap | 30s | |
| Detection threshold | 0.5 | Tune after real model is trained |
| STT retry delay | 400ms | Within spec's 300–500ms range |
| STT encoding | audio/wav | Self-describing; carries sample-rate metadata |
