// Package stt defines the Provider interface and Transcript type for
// speech-to-text backends used by SpectreSTT.
//
// AximoHTTPProvider is the only v1 implementation. The interface is
// deliberately narrow so a future AximoRealtimeProvider (WebSocket-backed
// streaming endpoint) can be added without changing any calling code in the
// pipeline.
package stt

import "context"

// Transcript is the result of a successful transcription.
type Transcript struct {
	// Text is the transcribed utterance, as returned by the STT engine.
	Text string `json:"text"`

	// DurationMs is the duration of the input audio as reported by Aximo.
	DurationMs int64 `json:"duration_ms"`

	// ProcessingMs is the time the engine spent on inference, as reported
	// by Aximo. Useful for latency monitoring.
	ProcessingMs int64 `json:"processing_ms"`
}

// Provider is implemented by any STT backend SpectreSTT can use to convert
// a captured utterance into text.
//
// AximoHTTPProvider is the only implementation in v1. A future
// AximoRealtimeProvider (WebSocket-backed) can implement this same interface
// without requiring changes to the audio pipeline that calls it.
type Provider interface {
	// Transcribe sends a complete utterance (already VAD-trimmed) for
	// transcription and returns the result.
	//
	// pcm must be raw PCM audio: pcm_s16le, 16 kHz, mono.
	// The call blocks until transcription completes or ctx is cancelled.
	//
	// Errors include network failures, non-200 HTTP responses, client-side
	// timeouts, and Aximo-reported inference timeouts (HTTP 504).
	// The caller (pipeline) is responsible for retry logic.
	Transcribe(ctx context.Context, pcm []byte) (Transcript, error)

	// Ready reports whether the provider can currently accept transcription
	// work. Returns nil if ready; an error describing the failure otherwise
	// (service unreachable, degraded state, etc.).
	//
	// AximoHTTPProvider caches the last successful Ready result for a
	// configurable TTL (default 30 min). Any Transcribe failure invalidates
	// the cache immediately, forcing a fresh check on the next call.
	Ready(ctx context.Context) error
}
