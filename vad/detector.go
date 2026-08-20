// Package vad wraps the WebRTC VAD library (go-webrtcvad) for use in the
// SpectreSTT audio pipeline.
//
// Design rationale (from spec §6): WebRTC VAD was chosen over Silero VAD for
// this pipeline because:
//   - Its job here is narrow: trim silence around an utterance the wake word
//     listener has already gated, not detect speech in an open, noisy room.
//   - It requires no ONNX runtime in the Go binary (Aximo already owns that
//     dependency).
//   - The cgo wrapper is mature and simple.
//
// Known trade-off: higher false-positive rate in noisy environments compared
// to Silero VAD. Documented upgrade path: if false positives prove to be a
// real problem (VAD fails to close utterances promptly), replace this package
// with a Silero VAD wrapper without changing any calling code — the Detector
// interface is unchanged.
package vad

import (
	"bytes"
	"context"
	"fmt"
	"time"

	webrtcvad "github.com/maxhawkins/go-webrtcvad"
)

// Detector wraps WebRTC VAD and manages the end-of-utterance state machine
// for a single capture session.
type Detector struct {
	vad             *webrtcvad.VAD
	sampleRate      int
	frameDurationMs int // must be 10, 20, or 30
	frameBytes      int // bytes per frame = sampleRate * frameDurationMs/1000 * 2 (int16)

	// silenceThreshold is the consecutive silence duration that signals
	// end-of-utterance. Default: 800ms.
	silenceThreshold time.Duration

	// maxUtterance is the hard cap on total capture duration.
	// Default: 30s. Prevents infinite stalls on broken VAD detection.
	maxUtterance time.Duration
}

// Config holds Detector construction parameters.
type Config struct {
	SampleRate       int           // must be 16000
	FrameDurationMs  int           // 10, 20, or 30; default 30
	Mode             int           // 0–3 VAD aggressiveness; default 3
	SilenceThreshold time.Duration // default 800ms
	MaxUtterance     time.Duration // default 30s
}

// New constructs a Detector from the given config and validates parameters.
func New(cfg Config) (*Detector, error) {
	if cfg.SampleRate != 16000 {
		return nil, fmt.Errorf("vad: sample rate must be 16000, got %d", cfg.SampleRate)
	}
	if cfg.FrameDurationMs != 10 && cfg.FrameDurationMs != 20 && cfg.FrameDurationMs != 30 {
		return nil, fmt.Errorf("vad: frame duration must be 10, 20, or 30ms, got %d", cfg.FrameDurationMs)
	}
	if cfg.Mode < 0 || cfg.Mode > 3 {
		return nil, fmt.Errorf("vad: mode must be 0–3, got %d", cfg.Mode)
	}

	v, err := webrtcvad.New()
	if err != nil {
		return nil, fmt.Errorf("vad: init: %w", err)
	}
	if err := v.SetMode(cfg.Mode); err != nil {
		return nil, fmt.Errorf("vad: set mode %d: %w", cfg.Mode, err)
	}

	samplesPerFrame := cfg.SampleRate * cfg.FrameDurationMs / 1000
	frameBytes := samplesPerFrame * 2 // int16 = 2 bytes per sample

	return &Detector{
		vad:              v,
		sampleRate:       cfg.SampleRate,
		frameDurationMs:  cfg.FrameDurationMs,
		frameBytes:       frameBytes,
		silenceThreshold: cfg.SilenceThreshold,
		maxUtterance:     cfg.MaxUtterance,
	}, nil
}

// FrameBytes returns the exact number of bytes expected per audio frame.
// Audio capture must produce frames of exactly this size.
// At 16kHz, 30ms: 480 samples × 2 bytes = 960 bytes.
func (d *Detector) FrameBytes() int {
	return d.frameBytes
}

// ProcessUtterance reads PCM frames from frames, runs VAD per frame, and
// returns the trimmed utterance as a contiguous []byte (pcm_s16le, 16kHz,
// mono) when end-of-utterance is detected.
//
// End-of-utterance conditions (whichever comes first):
//  1. Consecutive silence ≥ d.silenceThreshold after speech was detected.
//  2. Total captured duration ≥ d.maxUtterance.
//
// Leading silence is discarded: buffering begins only after the first speech
// frame is detected. Trailing silence is also discarded.
//
// The caller must close the frames channel when the audio source stops, or
// cancel ctx to abort early. ProcessUtterance returns an error only on ctx
// cancellation; a timeout via maxUtterance returns the audio collected so far.
func (d *Detector) ProcessUtterance(ctx context.Context, frames <-chan []byte) ([]byte, error) {
	var (
		buf            bytes.Buffer
		speechDetected bool
		silenceFrames  int
		totalFrames    int

		// How many consecutive silence frames constitute end-of-utterance.
		silenceFrameThreshold = int(d.silenceThreshold.Milliseconds()) / d.frameDurationMs

		// How many total frames before we hit the hard cap.
		maxFrames = int(d.maxUtterance.Milliseconds()) / d.frameDurationMs
	)

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()

		case frame, ok := <-frames:
			if !ok {
				// Channel closed — audio source stopped.
				return buf.Bytes(), nil
			}

			active, err := d.vad.Process(d.sampleRate, frame)
			if err != nil {
				// Treat a VAD processing error as silence rather than
				// aborting the utterance entirely.
				active = false
			}

			totalFrames++

			switch {
			case active:
				speechDetected = true
				silenceFrames = 0
				buf.Write(frame)

			case speechDetected:
				// We're in the post-speech silence window.
				silenceFrames++
				// Buffer trailing frames — we'll discard them if they turn
				// out to be the final silence, but they may contain the
				// very end of the phoneme.
				buf.Write(frame)

				if silenceFrames >= silenceFrameThreshold {
					// Trim the trailing silence we buffered.
					trimmed := trimTrailingSilence(buf.Bytes(), d.frameBytes, silenceFrames)
					return trimmed, nil
				}

			default:
				// Leading silence before any speech: discard.
			}

			if totalFrames >= maxFrames {
				// Hard cap reached. Return whatever we have (trimmed).
				if !speechDetected {
					// No speech at all within the cap — return empty.
					return []byte{}, nil
				}
				trimmed := trimTrailingSilence(buf.Bytes(), d.frameBytes, silenceFrames)
				return trimmed, nil
			}
		}
	}
}

// trimTrailingSilence removes the last silenceFrames worth of audio from buf.
// This discards the silence frames that were buffered while waiting for the
// silence threshold to trigger.
func trimTrailingSilence(buf []byte, frameBytes, silenceFrames int) []byte {
	trim := silenceFrames * frameBytes
	if trim >= len(buf) {
		return []byte{}
	}
	return buf[:len(buf)-trim]
}
