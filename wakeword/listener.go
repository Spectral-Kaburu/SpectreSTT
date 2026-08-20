// Package wakeword manages the openWakeWord Python sidecar process and
// translates its stdout events into Go channel signals.
//
// # IPC protocol (stdin/stdout, line-oriented)
//
// Go → Python (written to the sidecar's stdin):
//
//	"PAUSE\n"  — suspend wake-word inference and release the audio device.
//	             Sent immediately when a DETECTED event is received, before
//	             the Go Capturer opens its own PortAudio stream.
//	"RESUME\n" — restart inference and reopen the audio device.
//	             Sent when the pipeline finishes the capture/transcribe cycle.
//
// Python → Go (read from the sidecar's stdout):
//
//	"DETECTED\n" — the wake phrase "Hey Linus" was detected above threshold.
//	               The sidecar autonomously pauses its audio stream before
//	               writing this line (see sidecars/wakeword/main.py), so by
//	               the time Go reads it the audio device is already free.
//
// # State suppression
//
// When the pipeline is in CAPTURING or TRANSCRIBING state it sends PAUSE
// so that "Linus" spoken mid-utterance (e.g. "…tell Linus about it…") does
// not re-fire the wake word and corrupt the active session. RESUME is sent
// only after the full pipeline cycle (capture → VAD → STT → broadcast)
// completes.
package wakeword

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
)

// Listener spawns and manages the openWakeWord Python sidecar.
type Listener struct {
	sidecarPath string
	modelPath   string // may be "" — sidecar uses its placeholder model
	threshold   float64

	mu     sync.Mutex
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	paused bool
}

// New creates a Listener. Start() must be called to launch the sidecar.
//
// sidecarPath: absolute path to sidecars/wakeword/main.py.
// modelPath: path to an openWakeWord-compatible .onnx model file.
//            Pass "" to use the sidecar's built-in placeholder.
// threshold: detection score threshold (0–1). Default 0.5.
func New(sidecarPath, modelPath string, threshold float64) *Listener {
	return &Listener{
		sidecarPath: sidecarPath,
		modelPath:   modelPath,
		threshold:   threshold,
	}
}

// Start launches the Python sidecar and returns a channel that receives a
// signal whenever the wake phrase "Hey Linus" is detected.
//
// The returned channel is unbuffered; the pipeline should process detections
// promptly to avoid blocking the sidecar's stdout reader goroutine.
//
// Start returns an error if the sidecar process cannot be started. Once
// running, sidecar crashes are surfaced by closing the returned channel.
// Callers should handle a closed channel as a fatal sidecar failure.
func (l *Listener) Start(ctx context.Context) (<-chan struct{}, error) {
	args := []string{l.sidecarPath}
	if l.modelPath != "" {
		args = append(args, "--model", l.modelPath)
	}
	args = append(args, "--threshold", fmt.Sprintf("%.4f", l.threshold))

	cmd := exec.CommandContext(ctx, "python3", args...)

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("wakeword: stdin pipe: %w", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("wakeword: stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("wakeword: start sidecar %q: %w", l.sidecarPath, err)
	}

	l.mu.Lock()
	l.cmd = cmd
	l.stdin = stdinPipe
	l.paused = false
	l.mu.Unlock()

	detections := make(chan struct{})

	go l.readLoop(stdoutPipe, detections, cmd)

	return detections, nil
}

// readLoop reads lines from the sidecar's stdout and forwards DETECTED
// events to the detections channel. Closes the channel when the sidecar exits.
func (l *Listener) readLoop(r io.Reader, detections chan<- struct{}, cmd *exec.Cmd) {
	defer close(detections)
	defer cmd.Wait() //nolint:errcheck — sidecar exit is handled by channel close

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "DETECTED" {
			// Non-blocking send: if the pipeline is not ready to receive
			// (e.g. already in CAPTURING state after a race), the detection
			// is dropped. The sidecar has already paused itself, so RESUME
			// must still be sent when the current cycle completes.
			select {
			case detections <- struct{}{}:
			default:
			}
		}
		// Other lines (e.g. debug output from the sidecar) are silently ignored.
	}
}

// Pause instructs the sidecar to suspend inference and release the audio
// device. Must be called before the Go Capturer opens PortAudio to avoid
// device conflicts.
//
// Note: the sidecar autonomously pauses before writing "DETECTED\n", so
// Pause() is a belt-and-suspenders call for cases where the pipeline sends
// PAUSE after receiving the signal (both are safe because the sidecar checks
// its paused state idempotently).
func (l *Listener) Pause() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.stdin == nil {
		return fmt.Errorf("wakeword: sidecar not running")
	}
	l.paused = true
	_, err := fmt.Fprintln(l.stdin, "PAUSE")
	return err
}

// Resume instructs the sidecar to reopen its audio stream and restart inference.
// Call this after the pipeline's capture/transcribe cycle is fully complete.
func (l *Listener) Resume() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.stdin == nil {
		return fmt.Errorf("wakeword: sidecar not running")
	}
	l.paused = false
	_, err := fmt.Fprintln(l.stdin, "RESUME")
	return err
}

// Stop terminates the sidecar process.
func (l *Listener) Stop() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.stdin != nil {
		l.stdin.Close()
		l.stdin = nil
	}
	if l.cmd != nil && l.cmd.Process != nil {
		return l.cmd.Process.Kill()
	}
	return nil
}
