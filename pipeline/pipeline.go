// Package pipeline wires together the wake word listener, audio capturer,
// VAD detector, STT provider, and socket server into a single coordinated
// capture-transcribe loop.
//
// # State machine
//
//	WAITING  ──DETECTED──▶  CAPTURING  ──EOT/cap──▶  TRANSCRIBING  ──▶  WAITING
//
// WAITING:      Wake word listener active. Audio device owned by Python sidecar.
// CAPTURING:    PortAudio open. VAD running. Sidecar PAUSED.
// TRANSCRIBING: VAD done. Audio sent to Aximo. Retry logic applied here.
//
// # Retry + notification policy (per spec §4)
//
// Each utterance gets exactly two attempts (Ready + Transcribe).
// If both fail, the user is notified via TTS → notify-send fallback.
// The pipeline then resumes the wakeword listener and re-enters WAITING —
// it does not block or exit on STT failure.
package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/Spectral-Kaburu/SpectreSTT/audio"
	"github.com/Spectral-Kaburu/SpectreSTT/config"
	"github.com/Spectral-Kaburu/SpectreSTT/notify"
	"github.com/Spectral-Kaburu/SpectreSTT/socket"
	"github.com/Spectral-Kaburu/SpectreSTT/stt"
	"github.com/Spectral-Kaburu/SpectreSTT/vad"
	"github.com/Spectral-Kaburu/SpectreSTT/wakeword"
)

// Pipeline coordinates the full voice-to-transcript flow.
type Pipeline struct {
	cfg       *config.Config
	listener  *wakeword.Listener
	provider  *stt.AximoHTTPProvider
	lifecycle *stt.AximoLifecycle
	sockSrv   *socket.Server
	notifier  *notify.Notifier
}

// New constructs a Pipeline from the given config.
// All components are created here; Start() activates them.
func New(cfg *config.Config) *Pipeline {
	// Low-level Aximo HTTP client (shared by lifecycle and provider).
	client := stt.NewAximoClient(cfg.AximoBaseURL(), cfg.AximoInferenceTimeout())

	provider := stt.NewAximoHTTPProvider(
		cfg.AximoBaseURL(),
		cfg.AximoInferenceTimeout(),
		cfg.AximoReadinessCacheTTL(),
	)

	lifecycle := stt.NewAximoLifecycle(
		cfg.AximoComposePath,
		client,
		cfg.AximoStartupTimeout(),
		cfg.AximoStartupPollInterval(),
	)

	listener := wakeword.New(
		cfg.WakeWordSidecarPath,
		cfg.WakeWordModelPath,
		cfg.WakeWordThreshold,
	)

	sockSrv := socket.New(cfg.SocketPath)
	notifier := notify.New(cfg.TTSSocketPath)

	return &Pipeline{
		cfg:       cfg,
		listener:  listener,
		provider:  provider,
		lifecycle: lifecycle,
		sockSrv:   sockSrv,
		notifier:  notifier,
	}
}

// Start brings up all subsystems and runs the capture-transcribe loop until
// ctx is cancelled.
//
// Startup order:
//  1. Unix socket server starts accepting connections.
//  2. Aximo container ensured running (or flagged unavailable — non-fatal).
//  3. Wake word listener starts (Python sidecar spawned).
//  4. Main loop waits for detections.
func (p *Pipeline) Start(ctx context.Context) error {
	// 1. Start the socket server.
	if err := p.sockSrv.Start(ctx); err != nil {
		return fmt.Errorf("pipeline: socket server: %w", err)
	}
	defer p.sockSrv.Close()

	// 2. Ensure Aximo is running. Non-fatal on timeout — pipeline continues
	//    in unarmed state and retries readiness per-utterance.
	if err := p.lifecycle.EnsureRunning(ctx); err != nil {
		slog.Warn("pipeline: Aximo not ready at startup; continuing unarmed", "err", err)
		p.notifier.Send("Speech recognition is starting up. Please wait a moment.") //nolint:errcheck
	}

	// 3. Start the wake word listener.
	detections, err := p.listener.Start(ctx)
	if err != nil {
		return fmt.Errorf("pipeline: wake word listener: %w", err)
	}
	defer p.listener.Stop() //nolint:errcheck

	slog.Info("pipeline: armed — listening for wake word")

	// 4. Main loop.
	for {
		select {
		case <-ctx.Done():
			return nil

		case _, ok := <-detections:
			if !ok {
				// Sidecar crashed.
				return fmt.Errorf("pipeline: wake word sidecar exited unexpectedly")
			}
			p.handleDetection(ctx)
		}
	}
}

// handleDetection runs one full CAPTURING → TRANSCRIBING cycle.
// It always ends by calling listener.Resume() so the pipeline returns to WAITING.
func (p *Pipeline) handleDetection(ctx context.Context) {
	// The sidecar already paused itself before writing DETECTED.
	// Send PAUSE explicitly as belt-and-suspenders (idempotent on sidecar side).
	if err := p.listener.Pause(); err != nil {
		slog.Warn("pipeline: could not send PAUSE to wakeword sidecar", "err", err)
	}

	defer func() {
		if err := p.listener.Resume(); err != nil {
			slog.Warn("pipeline: could not send RESUME to wakeword sidecar", "err", err)
		}
		slog.Info("pipeline: WAITING — listening for wake word")
	}()

	slog.Info("pipeline: CAPTURING")

	// Build VAD detector for this utterance.
	vadDet, err := vad.New(vad.Config{
		SampleRate:       p.cfg.AudioSampleRate,
		FrameDurationMs:  p.cfg.VADFrameMs,
		Mode:             p.cfg.VADMode,
		SilenceThreshold: p.cfg.VADSilenceThreshold(),
		MaxUtterance:     p.cfg.VADMaxUtterance(),
	})
	if err != nil {
		slog.Error("pipeline: vad init failed", "err", err)
		return
	}

	// Open audio capture. frameSize = samples per frame (not bytes).
	frameSize := p.cfg.AudioSampleRate * p.cfg.VADFrameMs / 1000
	capturer := audio.New(p.cfg.AudioSampleRate, frameSize)

	captureCtx, cancelCapture := context.WithTimeout(ctx, p.cfg.VADMaxUtterance()+2*time.Second)
	defer cancelCapture()

	frames, err := capturer.Capture(captureCtx)
	if err != nil {
		slog.Error("pipeline: audio capture failed", "err", err)
		return
	}

	// Run VAD until end-of-utterance or max cap.
	pcm, err := vadDet.ProcessUtterance(captureCtx, frames)
	cancelCapture() // stop capture now that VAD is done

	if err != nil {
		slog.Error("pipeline: VAD error", "err", err)
		return
	}

	if len(pcm) == 0 {
		slog.Info("pipeline: empty utterance (all silence) — discarding")
		return
	}

	slog.Info("pipeline: TRANSCRIBING", "pcm_bytes", len(pcm))
	p.transcribeWithRetry(ctx, pcm)
}

// transcribeWithRetry attempts transcription up to 2 times.
// On both failures, the user is notified; the pipeline then resumes normally.
func (p *Pipeline) transcribeWithRetry(ctx context.Context, pcm []byte) {
	const maxAttempts = 2

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if attempt > 1 {
			slog.Info("pipeline: retrying transcription", "attempt", attempt)
			select {
			case <-ctx.Done():
				return
			case <-time.After(p.cfg.STTRetryDelay()):
			}
		}

		transcript, err := p.attemptTranscribe(ctx, pcm)
		if err == nil {
			p.broadcast(transcript)
			return
		}

		slog.Warn("pipeline: transcription attempt failed",
			"attempt", attempt,
			"err", err,
		)
	}

	// Both attempts exhausted — notify the user.
	slog.Error("pipeline: transcription failed after all retries; notifying user")
	if notifyErr := p.notifier.Send("Speech-to-text is unavailable right now."); notifyErr != nil {
		slog.Error("pipeline: notification also failed", "err", notifyErr)
	}
}

// attemptTranscribe performs a single Ready + Transcribe attempt.
func (p *Pipeline) attemptTranscribe(ctx context.Context, pcm []byte) (stt.Transcript, error) {
	// Check readiness (cached; see aximo_http.go for caching policy).
	if err := p.provider.Ready(ctx); err != nil {
		return stt.Transcript{}, fmt.Errorf("not ready: %w", err)
	}

	return p.provider.Transcribe(ctx, pcm)
}

// broadcast sends the transcript to all connected socket clients and logs it.
func (p *Pipeline) broadcast(t stt.Transcript) {
	slog.Info("pipeline: transcript ready",
		"text", t.Text,
		"duration_ms", t.DurationMs,
		"processing_ms", t.ProcessingMs,
	)

	payload := socket.TranscriptPayload{
		Text:         t.Text,
		DurationMs:   t.DurationMs,
		ProcessingMs: t.ProcessingMs,
		Timestamp:    time.Now().UTC(),
	}

	if err := p.sockSrv.Broadcast(payload); err != nil {
		slog.Error("pipeline: socket broadcast failed", "err", err)
	}
}
