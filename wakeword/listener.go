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
//
// # External usage
//
// The Listener type is designed for use both by the internal pipeline and by
// external systems. Construct via [New] with a [Config], call [Listener.Start]
// to begin detection, and use [Listener.Pause]/[Listener.Resume] to coordinate
// audio device ownership with other components.
package wakeword

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"strings"
	"sync"
)

// ErrNotRunning is returned when Pause, Resume, or other control methods are
// called on a Listener whose sidecar process is not currently running.
var ErrNotRunning = errors.New("wakeword: sidecar not running")

// Config holds the parameters needed to construct a [Listener].
type Config struct {
	// PythonPath is the absolute path to the Python executable
	// (e.g., in a venv) or "python3".
	PythonPath string

	// SidecarPath is the absolute path to sidecars/wakeword/main.py.
	SidecarPath string

	// ModelPath is the path to an openWakeWord-compatible .onnx model file.
	// Pass "" to use the sidecar's built-in placeholder.
	ModelPath string

	// Threshold is the detection score threshold (0–1). Default 0.5.
	Threshold float64
}

// Listener spawns and manages the openWakeWord Python sidecar.
type Listener struct {
	pythonPath  string
	sidecarPath string
	modelPath   string // may be "" — sidecar uses its placeholder model
	threshold   float64

	mu        sync.Mutex
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	paused    bool
	stderrBuf bytes.Buffer // accumulates sidecar stderr for crash diagnostics
}

// New creates a Listener from the given [Config].
// [Listener.Start] must be called to launch the sidecar.
func New(cfg Config) *Listener {
	return &Listener{
		pythonPath:  cfg.PythonPath,
		sidecarPath: cfg.SidecarPath,
		modelPath:   cfg.ModelPath,
		threshold:   cfg.Threshold,
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

	cmd := exec.CommandContext(ctx, l.pythonPath, args...)

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("wakeword: stdin pipe: %w", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("wakeword: stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("wakeword: stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("wakeword: start sidecar %q: %w", l.sidecarPath, err)
	}

	l.mu.Lock()
	l.cmd = cmd
	l.stdin = stdinPipe
	l.paused = false
	l.stderrBuf.Reset()
	l.mu.Unlock()

	detections := make(chan struct{})

	go l.readLoop(stdoutPipe, stderrPipe, detections, cmd)

	return detections, nil
}

// IsPaused reports whether the sidecar is currently in a paused state
// (inference suspended, audio device released). This is safe for concurrent
// use from any goroutine.
func (l *Listener) IsPaused() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.paused
}

// readLoop reads lines from the sidecar's stdout and forwards DETECTED
// events to the detections channel. Closes the channel when the sidecar exits.
//
// stderrPipe is drained concurrently so the sidecar never blocks on stderr
// writes. On an unexpected exit the captured stderr and process exit error are
// logged via slog so the developer sees the actual Python traceback.
func (l *Listener) readLoop(stdout io.Reader, stderr io.Reader, detections chan<- struct{}, cmd *exec.Cmd) {
	defer close(detections)

	// Drain stderr in a background goroutine so the sidecar can never block
	// trying to write to it. Output is forwarded line-by-line through slog so
	// it appears in the main process log alongside Go messages.
	go l.drainStderr(stderr)

	scanner := bufio.NewScanner(stdout)
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

	// stdout EOF — sidecar process is done. Wait to collect the exit status.
	exitErr := cmd.Wait()
	if exitErr != nil {
		l.logCrash(exitErr)
	}
}

// drainStderr reads lines from the sidecar's stderr, forwards them through
// slog, and buffers them for crash diagnostics.
func (l *Listener) drainStderr(stderr io.Reader) {
	sc := bufio.NewScanner(stderr)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		// Forward sidecar stderr as INFO; errors inside the sidecar will
		// surface through the exit-code path below.
		slog.Info("wakeword sidecar", "msg", line)

		// Also buffer for the crash-diagnostic path.
		l.mu.Lock()
		l.stderrBuf.WriteString(line)
		l.stderrBuf.WriteByte('\n')
		l.mu.Unlock()
	}
}

// logCrash logs the sidecar exit error together with any captured stderr
// for post-mortem diagnostics.
func (l *Listener) logCrash(exitErr error) {
	l.mu.Lock()
	captured := strings.TrimSpace(l.stderrBuf.String())
	l.mu.Unlock()

	if captured != "" {
		slog.Error("wakeword sidecar crashed",
			"exit_error", exitErr,
			"stderr", captured,
		)
	} else {
		slog.Error("wakeword sidecar crashed", "exit_error", exitErr)
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
//
// Returns [ErrNotRunning] if the sidecar process is not active.
func (l *Listener) Pause() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.stdin == nil {
		return ErrNotRunning
	}
	l.paused = true
	_, err := fmt.Fprintln(l.stdin, "PAUSE")
	return err
}

// Resume instructs the sidecar to reopen its audio stream and restart inference.
// Call this after the pipeline's capture/transcribe cycle is fully complete.
//
// Returns [ErrNotRunning] if the sidecar process is not active.
func (l *Listener) Resume() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.stdin == nil {
		return ErrNotRunning
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
