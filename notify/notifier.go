// Package notify delivers error notices to the user when the STT pipeline
// fails after exhausting its retry budget.
//
// Delivery strategy (in order):
//  1. SpectreTTS Unix socket — spoken audio feedback.
//  2. notify-send(1) — desktop notification fallback, used when the TTS
//     daemon is unreachable or its socket path is unconfigured.
//
// The caller should not care which delivery method succeeded; both are
// best-effort. Errors from both paths are logged but do not propagate — a
// notification failure must never crash or hang the pipeline.
package notify

import (
	"encoding/json"
	"fmt"
	"net"
	"os/exec"
	"time"
)

// Notifier sends user-facing notices via SpectreTTS → notify-send fallback.
type Notifier struct {
	// ttsSocketPath is the Unix socket path of the SpectreTTS daemon.
	// If empty, the TTS step is skipped and notify-send is used directly.
	ttsSocketPath string
}

// New creates a Notifier.
// ttsSocketPath may be empty — the TTS step will be skipped if so.
func New(ttsSocketPath string) *Notifier {
	return &Notifier{ttsSocketPath: ttsSocketPath}
}

// Send delivers msg to the user. It tries SpectreTTS first; if that fails
// (or is not configured), it shells out to notify-send.
//
// Send never returns a user-facing error — both paths are best-effort.
// The returned error is informational for callers that want to log it.
func (n *Notifier) Send(msg string) error {
	if n.ttsSocketPath != "" {
		if err := n.sendTTS(msg); err == nil {
			return nil
		}
		// TTS failed — fall through to notify-send.
	}
	return n.sendDesktopNotification(msg)
}

// ttsMessage is the JSON payload sent to the SpectreTTS daemon.
// Protocol assumption: connect → send JSON → close.
// ⚠ The exact SpectreTTS wire protocol is not finalised; adjust once the
// SpectreTTS socket spec is published.
type ttsMessage struct {
	Text string `json:"text"`
}

func (n *Notifier) sendTTS(msg string) error {
	conn, err := net.DialTimeout("unix", n.ttsSocketPath, 2*time.Second)
	if err != nil {
		return fmt.Errorf("notify: tts dial %q: %w", n.ttsSocketPath, err)
	}
	defer conn.Close()

	conn.SetDeadline(time.Now().Add(3 * time.Second))

	payload, err := json.Marshal(ttsMessage{Text: msg})
	if err != nil {
		return fmt.Errorf("notify: tts marshal: %w", err)
	}
	payload = append(payload, '\n')

	if _, err := conn.Write(payload); err != nil {
		return fmt.Errorf("notify: tts write: %w", err)
	}
	return nil
}

func (n *Notifier) sendDesktopNotification(msg string) error {
	cmd := exec.Command("notify-send",
		"--urgency=normal",
		"--app-name=SpectreSTT",
		"SpectreSTT",
		msg,
	)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("notify: notify-send: %w", err)
	}
	return nil
}
