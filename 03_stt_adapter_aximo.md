# 03 — STT Adapter: Aximo Integration, Wake Word, and VAD

> Companion doc to `01_architecture.md`. This spec covers the local speech-to-text
> path: wake word detection, voice activity detection (VAD), and the STT provider
> that talks to Aximo. It supersedes any prior mention of whisper.cpp as the local
> STT engine — Aximo replaces whisper.cpp entirely.

---

## 1. What This Component Does

Aether needs to always be listening for its wake word without burning CPU, and
when triggered, capture a clean utterance and turn it into text — locally, with
zero raw audio leaving the machine in privacy mode (the default mode; see
`01_architecture.md` §2 for the privacy/speed mode split).

This spec covers three cooperating pieces:

1. **Wake word listener** — always-on, near-zero-CPU hot loop that detects
   "Hey Aether" and signals the pipeline to start capturing.
2. **VAD (Voice Activity Detection)** — once capture starts, trims silence from
   the start and end of the utterance so only actual speech is sent for
   transcription.
3. **STT provider** — sends the trimmed audio to **Aximo**, a self-hosted,
   CPU-only speech-to-text microservice, and returns clean text.

This path is used in **privacy mode only**. Speed mode sends raw audio directly
to the cloud LLM provider (Gemini Live) and does not touch this component at
all — see `01_architecture.md` §2.2.

---

## 2. What Is Aximo

Aximo (`agent-axiom/aximo` on GitHub, MIT licensed) is a self-hosted, CPU-only
speech-to-text microservice written in Rust. It runs local ONNX models
(Parakeet v3 for English) and exposes an HTTP + WebSocket API. It replaces
whisper.cpp as Aether's local STT engine.

Key facts a coding agent needs to know before implementing against it:

- Aximo is **not an in-process library**. It is a separate service, deployed
  as a Docker container, that Aether talks to over HTTP on `localhost`.
- Aximo's realtime WebSocket endpoint (`GET /v1/realtime`) exists but is
  **explicitly out of scope for this spec**. Its bundled models
  (`supports_native_streaming=false`) do bounded-buffered "realtime"
  transcription with lossy, best-effort partial results — not true
  incremental streaming. This is not needed for v1: privacy mode only ever
  needs one clean transcript per utterance, produced after VAD has already
  determined the utterance is complete. The realtime endpoint may be adopted
  in a future version; the `STTProvider` interface (§4) is deliberately
  designed so a WebSocket-based implementation can be added later without
  changing any calling code.
- Relevant endpoints for this spec:
  - `POST /v1/transcriptions` — one-shot transcription. Accepts `audio/wav`,
    `audio/mpeg`, `audio/flac`, `audio/mp4`, `audio/x-m4a`, `audio/pcm`, or
    `application/octet-stream`. Raw PCM must be `pcm_s16le`, 16kHz, mono.
    Returns JSON: `{"text": "...", "segments": [], "detected_language": null,
    "engine": "parakeet", "duration_ms": 1000, "processing_ms": 37}`.
  - `GET /health/ready` — readiness probe. Returns `200` when the service can
    accept inference work, `503` with a JSON `degraded` status if the engine
    has failed enough consecutive inferences to trip its internal
    degraded-state policy.
  - `GET /health/live` — liveness only (process is up, says nothing about
    whether inference works).
- **Known operational quirk**: if a client-side request to
  `POST /v1/transcriptions` times out, Aximo's backend inference thread may
  keep running — Rust cannot safely kill an in-flight blocking OS thread.
  The model execution slot (Aximo allows exactly one concurrent inference
  per loaded model) stays held until that orphaned call actually finishes.
  Practical implication: a single slow/stuck inference can block all
  subsequent transcription requests until it resolves. Aether's client
  needs its own timeout well under Aximo's configured
  `AXIMO_SHORT_INFERENCE_TIMEOUT_MS` (default 120000ms / 2 minutes) so the
  user isn't left waiting two minutes for an error.
- Model: Parakeet v3, English only, for v1. Aximo also supports a GigaAM
  (Russian) model but that is out of scope — Aether v1 is English-only.

---

## 3. Deployment Model

Aximo runs as a **Docker container**, managed via `docker compose`, on the
same machine as Aether, bound to `localhost` only (never exposed on a
non-loopback interface — Aximo's own docs warn it should not be exposed to
untrusted clients without an authenticated gateway, and Aether has no need to
expose it beyond localhost).

**Aether manages Aximo's lifecycle by shelling out to the `docker compose`
CLI** — not via the Docker Engine API. Concretely:

- Start: `docker compose -f <path-to-aximo-compose-file> up -d`
- Stop: `docker compose -f <path-to-aximo-compose-file> down`
- Readiness is determined **exclusively via HTTP** — poll
  `GET http://127.0.0.1:<aximo_port>/health/ready` — not via Docker's own
  container health-check machinery. Docker-level health is redundant with
  Aximo's own readiness endpoint and adds a second source of truth to keep
  in sync; don't implement it.

This keeps the STT component's dependency surface to "an HTTP client + a
subprocess call to `docker compose`," consistent with how other system
actions in this project shell out to CLI tools (see `01_architecture.md`,
`tools/actions/shell.py`).

Rationale for shell-out over the Docker Engine API Go client: avoids adding
a new Go dependency (`docker/docker/client` and its transitive graph) for
functionality that HTTP health polling already covers. If lifecycle needs
grow more complex later (e.g. needing structured container state, log
streaming), revisit this decision — it is not a permanent constraint.

### Startup sequence

1. Aether's audio component starts.
2. It shells out to bring the Aximo container up (`docker compose up -d`).
3. It polls `GET /health/ready` until `200` or a startup timeout is reached
   (exact timeout value: coding agent to propose, subject to review — this
   spec does not fix it).
4. Only after readiness is confirmed does the wake word listener + VAD +
   STT pipeline become "armed." Until then, Aether should not claim to be
   ready for voice input.

---

## 4. STT Provider Interface

Define an interface, not a concrete struct, so Aximo's short-audio HTTP path
can be swapped or supplemented by a future WebSocket-based realtime provider
without touching any code that calls it. This mirrors the existing
`BaseProvider` abstraction for LLM providers described in
`01_architecture.md` (no module calls an LLM provider directly — everything
goes through the communications manager). The same discipline applies here
for STT.

```go
package stt

import "context"

// Transcript is the result of a successful transcription.
type Transcript struct {
    Text       string // the transcribed text
    DurationMs int64  // duration of the input audio, as reported by the engine
    ProcessingMs int64 // time the engine took to transcribe, as reported by the engine
}

// Provider is implemented by any STT backend Aether can use to convert
// a captured utterance into text. AximoHTTPProvider is the only
// implementation in v1. A future AximoRealtimeProvider (WebSocket-backed)
// can implement this same interface without requiring changes to the
// audio pipeline that calls it.
type Provider interface {
    // Transcribe sends a complete utterance (already VAD-trimmed) for
    // transcription and returns the result. pcm must be pcm_s16le,
    // 16kHz, mono.
    Transcribe(ctx context.Context, pcm []byte) (Transcript, error)

    // Ready reports whether the provider is currently able to accept
    // transcription work. Returns nil if ready, or an error describing
    // why it is not (unreachable, degraded, etc).
    Ready(ctx context.Context) error
}
```

### AximoHTTPProvider — the only v1 implementation

Wraps `POST /v1/transcriptions` and `GET /health/ready`.

- `Transcribe`: encodes the PCM buffer as a WAV payload (or sends raw PCM
  with `content-type: audio/pcm` — coding agent's choice, document which was
  picked), POSTs to `/v1/transcriptions`, parses the JSON response into a
  `Transcript`. Applies a client-side timeout shorter than Aximo's
  configured inference timeout (see §2, operational quirk).
- `Ready`: performs `GET /health/ready`, returns `nil` on `200`, an error
  wrapping the response body on `503` or any transport failure.

### Failure handling — retry and notify

When the wake word fires and the pipeline is ready to call `Transcribe`:

1. Call `Provider.Ready(ctx)` first (or reuse a recently cached readiness
   state — coding agent's choice, but document the caching policy chosen;
   avoid polling `/health/ready` on every single utterance if a cheap
   cached check within the last few seconds is available and still valid).
2. If not ready, or if `Transcribe` itself fails (network error, non-200,
   `504 inference_timeout`, or client-side timeout): **retry once**, after
   a short delay (300–500ms) between the first and second attempt.
3. If the second attempt also fails: **do not retry a third time.** Surface
   a clear failure to the user — spoken via the SpectreTTS daemon (see
   `01_architecture.md` for the SpectreTTS Unix socket protocol) or via
   `notify-send`, coding agent's choice of which, but it must be one of the
   two, not a silent log-only failure. Suggested message: something to the
   effect of "Speech-to-text is unavailable right now."
4. This same two-retry-then-notify policy applies uniformly whether the
   failure happens on the very first call after startup, or mid-session
   after Aximo was previously healthy and then became unreachable
   (e.g. crashed, OOM'd). There is no separate "was previously healthy"
   code path — every failed utterance gets the same two-retry treatment
   before notifying.

---

## 5. Wake Word Listener

Unchanged from the original architecture: **pvporcupine**, detecting
"Hey Aether," running as a near-zero-CPU hot loop for as long as Aether is
running. On detection, it signals the audio pipeline to begin capture.

This is a separate concern from VAD (§6) — the wake word listener's only job
is deciding *when to start paying attention*. It does not do anything with
audio after the wake phrase is detected; that's VAD's job.

---

## 6. Voice Activity Detection (VAD)

**Library choice: WebRTC VAD**, not Silero VAD, for v1.

### Why WebRTC VAD over Silero VAD

This was a genuine trade-off, not an obvious pick — recorded here so it
isn't silently revisited without cause:

| | WebRTC VAD | Silero VAD |
|---|---|---|
| Method | Traditional signal processing (energy, zero-crossing, spectral features via GMM) | Small neural network (ONNX) |
| Accuracy | Lower; prone to false positives on background noise | Higher, especially in noisy environments |
| Go integration | Mature, simple cgo wrapper; no new runtime dependency | Requires an ONNX runtime Go binding — a second ONNX dependency alongside Aximo, embedded directly in the Aether binary |
| Resource cost | Negligible (pure DSP) | Also negligible in absolute terms (sub-millisecond per chunk), but non-zero setup complexity |

**Decision rationale**: In this specific pipeline, VAD's job is narrowly
scoped — trim silence around an utterance the wake word listener has *already*
gated. It is not responsible for detecting speech in an open, ungated room,
which is the scenario where WebRTC VAD's higher false-positive rate is most
costly. Given that narrower scope, the cost of a false positive here (VAD
keeps recording slightly too long, capturing a bit of trailing silence or
noise, marginally worse transcript or one wasted round-trip) is low, and it
does not justify adding a second ONNX runtime dependency to the Go binary
when Aximo is already the project's one ONNX/Rust dependency.

**Documented upgrade path**: if false positives prove to be a real problem
in practice (users report VAD failing to close utterances promptly, or
capturing significant garbage audio), Silero VAD is the recommended
upgrade — this is a known, accepted trade-off, not a rejected option.

### VAD's job in this pipeline

1. Wake word fires → capture starts.
2. VAD runs continuously on incoming audio frames (WebRTC VAD operates on
   10/20/30ms frames — coding agent to pick one, 20ms or 30ms are the more
   commonly used defaults per WebRTC VAD's own documentation).
3. VAD determines end-of-utterance (a sustained silence period after speech
   was detected — exact silence-duration threshold is an implementation
   parameter, coding agent to propose a starting value, e.g. 700ms–1s, and
   flag it as tunable).
4. Once end-of-utterance is detected, capture stops, the buffered audio
   (trimmed of leading/trailing silence) is handed to `Provider.Transcribe`.

---

## 7. Pipeline Wiring

```
wakeword.Listener (always running)
    │  triggers on "Hey Aether"
    ▼
audio capture starts (sounddevice equivalent — see 01_architecture.md)
    │
    ▼
vad.Detector (WebRTC VAD, running per-frame on captured audio)
    │  determines end-of-utterance
    ▼
trimmed PCM buffer (pcm_s16le, 16kHz, mono)
    │
    ▼
stt.Provider.Transcribe(ctx, pcm)  →  AximoHTTPProvider
    │  (2 retries w/ 300-500ms delay, then notify on failure — see §4)
    ▼
clean transcript text
    │
    ▼
Scrubber (privacy/scrubber.go — see 01_architecture.md §3)
    │
    ▼
Communications Manager → LLM Provider
```

No stage in this chain calls Aximo directly except `AximoHTTPProvider`. No
stage calls the wake word or VAD libraries except their own wrapper
packages. This preserves the microservices-style isolation described in
`01_architecture.md` §"Microservices Philosophy" — each piece can be tested
and swapped independently.

---

## 8. Suggested File Structure

```
aether/
  audio/
    wakeword/
      listener.go          ← pvporcupine wrapper, always-on hot loop
    vad/
      detector.go           ← WebRTC VAD wrapper, frame-by-frame speech/silence
    stt/
      provider.go            ← Provider interface, Transcript struct
      aximo_http.go            ← AximoHTTPProvider (implements Provider)
      aximo_client.go            ← low-level HTTP client: POST /v1/transcriptions, GET /health/ready
      aximo_lifecycle.go          ← docker compose up/down wrapper, startup readiness polling
    pipeline.go                    ← wires wakeword → capture → VAD → Provider.Transcribe → scrubber
  config/
    aether.json                      ← add: aximo_port, aximo_compose_path, stt retry/timeout settings
```

---

## 9. Explicitly Out of Scope for This Spec (v1)

- Aximo's `GET /v1/realtime` WebSocket endpoint and any streaming/partial
  transcript UX. The `Provider` interface supports adding this later.
- Silero VAD. Documented as the known upgrade path if WebRTC VAD's
  false-positive rate proves problematic in practice.
- GigaAM (Russian) model support in Aximo.
- Docker Engine API–based lifecycle management (using `docker compose` CLI
  shell-out instead — see §3 for rationale).
- Any change to speed mode. Speed mode continues to bypass local STT
  entirely and stream raw audio directly to Gemini Live, per
  `01_architecture.md` §2.2. Aximo is not part of the speed-mode path.

---

## 10. Open Parameters for the Implementing Agent to Propose

These are intentionally left as implementation choices, not fixed by this
spec — the coding agent should propose values and flag them for review
rather than treating them as already decided:

- Aximo container startup readiness-poll timeout (§3, step 3).
- WAV vs raw-PCM content-type for the `Transcribe` HTTP call (§4).
- Readiness-check caching window, if any (§4, step 1).
- WebRTC VAD frame size: 10/20/30ms (§6).
- End-of-utterance silence threshold (§6, step 3).
