// Package stt — AximoHTTPProvider implements Provider using
// POST /v1/transcriptions and GET /health/ready.
//
// Readiness caching policy:
//
//   A successful Ready() result is cached for AximoReadinessCacheTTL
//   (default 30 minutes). Within the TTL, subsequent Ready() calls return
//   the cached nil without hitting the network. This avoids polling
//   /health/ready on every utterance while still detecting a degraded Aximo
//   promptly after a failure:
//
//   • Any Transcribe() failure immediately invalidates the cache, forcing
//     a fresh /health/ready check on the next call.
//   • A failed Ready() result is never cached — every not-ready call checks
//     the network so recovery is detected as soon as Aximo comes back up.
//
// Retry logic lives in pipeline.go, not here. AximoHTTPProvider makes
// exactly one attempt per call; the pipeline wraps it in a two-attempt loop.
package stt

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// AximoHTTPProvider implements Provider via Aximo's one-shot HTTP endpoints.
type AximoHTTPProvider struct {
	client *AximoClient

	// ─── Readiness cache ──────────────────────────────────────────────────────
	cacheMu    sync.RWMutex
	cacheTime  time.Time
	cacheValid bool // true iff the last check was healthy
	cacheTTL   time.Duration
}

// NewAximoHTTPProvider creates an AximoHTTPProvider.
// baseURL: e.g. "http://127.0.0.1:8080"
// inferenceTimeout: client-side deadline for POST /v1/transcriptions.
// readinessCacheTTL: how long a healthy Ready() result is considered valid.
func NewAximoHTTPProvider(baseURL string, inferenceTimeout, readinessCacheTTL time.Duration) *AximoHTTPProvider {
	return &AximoHTTPProvider{
		client:   NewAximoClient(baseURL, inferenceTimeout),
		cacheTTL: readinessCacheTTL,
	}
}

// Ready implements Provider. Returns nil if Aximo can accept inference work.
//
// Cache behaviour:
//   - If the cached result is healthy and within TTL, returns nil immediately.
//   - Otherwise, calls GET /health/ready and updates the cache on success.
//   - A not-ready result is not cached; the next call will check again.
func (p *AximoHTTPProvider) Ready(ctx context.Context) error {
	p.cacheMu.RLock()
	if p.cacheValid && time.Since(p.cacheTime) < p.cacheTTL {
		p.cacheMu.RUnlock()
		return nil
	}
	p.cacheMu.RUnlock()

	err := p.client.GetHealthReady(ctx)
	if err == nil {
		p.cacheMu.Lock()
		p.cacheValid = true
		p.cacheTime = time.Now()
		p.cacheMu.Unlock()
	}
	// Failures are not cached — next call will probe again.
	return err
}

// InvalidateReadinessCache immediately invalidates the cached readiness
// result. Called by the pipeline on any Transcribe failure.
func (p *AximoHTTPProvider) InvalidateReadinessCache() {
	p.cacheMu.Lock()
	p.cacheValid = false
	p.cacheMu.Unlock()
}

// Transcribe implements Provider. Sends pcm (pcm_s16le, 16kHz, mono) to
// POST /v1/transcriptions and returns the result.
//
// Makes exactly ONE attempt. The pipeline is responsible for retry.
// On any failure, the readiness cache is invalidated so the next Ready()
// call probes Aximo's actual state rather than returning a stale healthy result.
func (p *AximoHTTPProvider) Transcribe(ctx context.Context, pcm []byte) (Transcript, error) {
	resp, err := p.client.PostTranscription(ctx, pcm)
	if err != nil {
		p.InvalidateReadinessCache()
		return Transcript{}, fmt.Errorf("aximo: transcribe: %w", err)
	}
	return Transcript{
		Text:         resp.Text,
		DurationMs:   resp.DurationMs,
		ProcessingMs: resp.ProcessingMs,
	}, nil
}
