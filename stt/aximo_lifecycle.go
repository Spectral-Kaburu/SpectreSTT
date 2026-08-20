// Package stt — AximoLifecycle manages the Aximo Docker container lifecycle
// via docker compose CLI shell-out.
//
// Rationale for shell-out over Docker Engine API (from spec §3):
// Avoids adding docker/docker/client and its transitive dependency graph to
// the Go binary for functionality that HTTP health polling already covers.
// Revisit if lifecycle needs grow more complex (log streaming, structured
// container state, etc.).
//
// Startup sequence:
//  1. Check if Aximo is already responding on /health/ready. If so, skip
//     docker compose up — the service was already running.
//  2. If not running: exec `docker compose -f <path> up -d`.
//  3. Poll /health/ready at 1s intervals until 200 or the startup timeout.
//  4. If timeout: signal the caller so it can mark STT as unavailable and
//     retry in the background (the pipeline does not block on this).
package stt

import (
	"context"
	"fmt"
	"os/exec"
	"time"
)

// AximoLifecycle manages the Aximo container via docker compose CLI.
type AximoLifecycle struct {
	composePath string
	client      *AximoClient // reuses the health-check HTTP client

	startupTimeout  time.Duration
	pollInterval    time.Duration
}

// NewAximoLifecycle creates an AximoLifecycle.
// composePath must be the absolute path to the docker-compose file.
// client is used exclusively for GET /health/ready polling.
func NewAximoLifecycle(composePath string, client *AximoClient, startupTimeout, pollInterval time.Duration) *AximoLifecycle {
	return &AximoLifecycle{
		composePath:    composePath,
		client:         client,
		startupTimeout: startupTimeout,
		pollInterval:   pollInterval,
	}
}

// EnsureRunning brings the Aximo container up if it is not already running,
// then waits for it to become /health/ready.
//
// Step 1: If Aximo is already responding (a previous run left it up), the
// docker compose up step is skipped entirely.
//
// Step 2: docker compose -f <composePath> up -d
//
// Step 3: Poll /health/ready at pollInterval until 200 or startupTimeout.
//
// Returns nil if Aximo is ready. Returns an error if the startup timeout
// is exceeded — the caller should mark STT as unavailable and retry in the
// background; SpectreSTT continues in an unarmed state.
func (l *AximoLifecycle) EnsureRunning(ctx context.Context) error {
	// Step 1: Check if already running.
	probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()

	if err := l.client.GetHealthReady(probeCtx); err == nil {
		// Already up and healthy — nothing to do.
		return nil
	}

	// Step 2: Bring the container up.
	if err := l.composeUp(ctx); err != nil {
		return fmt.Errorf("aximo_lifecycle: compose up: %w", err)
	}

	// Step 3: Poll for readiness.
	return l.waitReady(ctx)
}

// Stop brings the Aximo container down cleanly.
// docker compose -f <composePath> down
func (l *AximoLifecycle) Stop(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "docker", "compose", "-f", l.composePath, "down")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("aximo_lifecycle: compose down: %w\noutput: %s", err, out)
	}
	return nil
}

// composeUp runs docker compose up -d and returns any execution error.
func (l *AximoLifecycle) composeUp(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "docker", "compose", "-f", l.composePath, "up", "-d")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("exit error: %w\noutput: %s", err, out)
	}
	return nil
}

// waitReady polls GET /health/ready at l.pollInterval until it returns 200
// or the startup timeout (l.startupTimeout) is reached.
func (l *AximoLifecycle) waitReady(ctx context.Context) error {
	deadline := time.Now().Add(l.startupTimeout)
	ticker := time.NewTicker(l.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()

		case t := <-ticker.C:
			if t.After(deadline) {
				return fmt.Errorf(
					"aximo_lifecycle: Aximo did not become ready within %s; "+
						"continuing in unarmed state — will retry in the background",
					l.startupTimeout,
				)
			}

			probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
			err := l.client.GetHealthReady(probeCtx)
			cancel()

			if err == nil {
				return nil // Ready.
			}
			// Not ready yet; keep polling.
		}
	}
}
