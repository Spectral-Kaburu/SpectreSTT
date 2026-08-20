// Package config defines the runtime configuration for SpectreSTT.
// All tunable parameters live here; no values are hard-coded elsewhere.
//
// Configuration is loaded from a JSON file (path passed via --config flag).
// Missing fields fall back to the defaults returned by Default().
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// Config holds the complete runtime configuration for SpectreSTT.
type Config struct {
	// ─── Aximo service ────────────────────────────────────────────────────────

	// AximoHost is the hostname Aximo is bound to. Almost always "127.0.0.1".
	AximoHost string `json:"aximo_host"`

	// AximoPort is the port Aximo listens on.
	AximoPort int `json:"aximo_port"`

	// AximoComposePath is the absolute path to the docker-compose file that
	// manages the Aximo container. Must be set explicitly — there is no
	// default, because the correct path is deployment-specific.
	AximoComposePath string `json:"aximo_compose_path"`

	// AximoStartupTimeoutMs is how long (ms) to wait for Aximo to become
	// /health/ready after docker compose up. The service is considered
	// unavailable (but SpectreSTT continues in an unarmed state) if this
	// deadline is exceeded.
	//
	// Tuning guidance: 10 000–40 000ms is the expected range; 60 000ms is the
	// hard upper bound — anything longer suggests a broken container.
	AximoStartupTimeoutMs int `json:"aximo_startup_timeout_ms"`

	// AximoStartupPollIntervalMs is how often (ms) to poll /health/ready
	// during the startup sequence.
	AximoStartupPollIntervalMs int `json:"aximo_startup_poll_interval_ms"`

	// AximoInferenceTimeoutMs is the client-side HTTP timeout for
	// POST /v1/transcriptions. Must be well under Aximo's own
	// AXIMO_SHORT_INFERENCE_TIMEOUT_MS (default 120 000ms) so the user
	// isn't held for two minutes on a stuck inference.
	AximoInferenceTimeoutMs int `json:"aximo_inference_timeout_ms"`

	// AximoReadinessCacheTTLMs is how long (ms) a successful /health/ready
	// result is considered valid without re-querying the endpoint.
	// Any Transcribe failure immediately invalidates the cache regardless
	// of the remaining TTL.
	//
	// Default: 30 minutes (1 800 000ms). Set to 0 to disable caching.
	AximoReadinessCacheTTLMs int `json:"aximo_readiness_cache_ttl_ms"`

	// ─── STT retry ────────────────────────────────────────────────────────────

	// STTRetryDelayMs is the delay (ms) between the first and second
	// transcription attempt when the first fails. Range 300–500ms per spec.
	STTRetryDelayMs int `json:"stt_retry_delay_ms"`

	// ─── VAD ──────────────────────────────────────────────────────────────────

	// VADFrameMs is the WebRTC VAD frame duration in milliseconds.
	// Valid values: 10, 20, 30. Default 30ms (most reliable per WebRTC VAD docs).
	VADFrameMs int `json:"vad_frame_ms"`

	// VADMode controls WebRTC VAD aggressiveness.
	// 0 = quality (fewest false positives), 3 = very aggressive.
	// Default 3 — in this pipeline VAD only runs after the wake word has
	// already gated the stream, so false positives here merely extend
	// recording slightly; too-low aggressiveness misses the end of utterance.
	VADMode int `json:"vad_mode"`

	// VADSilenceThresholdMs is how many consecutive milliseconds of silence
	// signal end-of-utterance after speech has been detected.
	// Default: 800ms. Tunable — increase for slower speakers, decrease for
	// faster command-style usage.
	VADSilenceThresholdMs int `json:"vad_silence_threshold_ms"`

	// VADMaxUtteranceMs is the hard cap (ms) on how long a single utterance
	// capture session can last before VAD forcibly closes it and sends what
	// it has. Prevents the pipeline from stalling on a broken VAD end-detection.
	// Default: 30 000ms (30s).
	VADMaxUtteranceMs int `json:"vad_max_utterance_ms"`

	// ─── Wake word ────────────────────────────────────────────────────────────

	// WakeWordPythonPath is the path to the Python executable.
	// Can be set to a virtual environment python, e.g., "/opt/spectrestt/venv/bin/python".
	// Defaults to "python3" (system python).
	WakeWordPythonPath string `json:"wakeword_python_path"`

	// WakeWordSidecarPath is the path to the Python sidecar script.
	// Example: "/opt/spectrestt/sidecars/wakeword/main.py"
	WakeWordSidecarPath string `json:"wakeword_sidecar_path"`

	// WakeWordModelPath is passed to the Python sidecar as the openWakeWord
	// model to load. If empty, the sidecar uses its built-in placeholder
	// (hey_mycroft stock model — see sidecars/wakeword/main.py).
	//
	// ⚠ STUB: "Hey Linus" requires a custom-trained openWakeWord model.
	// Custom training is a separate task (see sidecars/wakeword/main.py).
	// Until the real model exists, leave this empty or point at any
	// openWakeWord-compatible .onnx model file.
	WakeWordModelPath string `json:"wakeword_model_path"`

	// WakeWordThreshold is the per-frame score (0–1) above which the
	// wake word is considered detected. Lower = more sensitive (more false
	// positives); higher = less sensitive (may miss soft utterances).
	// Default: 0.5. Tune against real usage.
	WakeWordThreshold float64 `json:"wakeword_threshold"`

	// ─── Unix socket output ───────────────────────────────────────────────────

	// SocketPath is the filesystem path of the Unix domain socket on which
	// SpectreSTT pushes transcript payloads to connected clients.
	// Example: "/run/spectrestt/transcripts.sock"
	SocketPath string `json:"socket_path"`

	// ─── Notification ─────────────────────────────────────────────────────────

	// TTSSocketPath is the Unix domain socket of the SpectreTTS daemon.
	// SpectreSTT attempts to deliver spoken error notices here first; if the
	// socket is unreachable, it falls back to notify-send(1).
	// Leave empty to skip TTS and go straight to notify-send.
	TTSSocketPath string `json:"tts_socket_path"`

	// ─── Audio capture ────────────────────────────────────────────────────────

	// AudioSampleRate is the microphone sample rate in Hz.
	// Must be 16000 — Aximo (Parakeet v3) and WebRTC VAD both require 16kHz.
	AudioSampleRate int `json:"audio_sample_rate"`
}

// Default returns a Config populated with safe, well-documented starting
// values. Callers should override with values from Load() where present.
func Default() *Config {
	return &Config{
		AximoHost:                  "127.0.0.1",
		AximoPort:                  8080,
		AximoComposePath:           "", // must be set explicitly
		AximoStartupTimeoutMs:      40_000,
		AximoStartupPollIntervalMs: 1_000,
		AximoInferenceTimeoutMs:    30_000, // 30s client-side; Aximo's own limit is 120s
		AximoReadinessCacheTTLMs:   1_800_000, // 30 minutes
		STTRetryDelayMs:            400,
		VADFrameMs:                 30,
		VADMode:                    3,
		VADSilenceThresholdMs:      800,
		VADMaxUtteranceMs:          30_000,
		WakeWordPythonPath:         "python3",
		WakeWordSidecarPath:        "",
		WakeWordModelPath:          "",
		WakeWordThreshold:          0.5,
		SocketPath:                 "/run/spectrestt/transcripts.sock",
		TTSSocketPath:              "",
		AudioSampleRate:            16_000,
	}
}

// Load reads the JSON file at path, merges it over Default(), and returns
// the resulting Config. Returns an error if the file cannot be read or parsed.
func Load(path string) (*Config, error) {
	cfg := Default()

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("config: open %q: %w", path, err)
	}
	defer f.Close()

	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(cfg); err != nil {
		return nil, fmt.Errorf("config: parse %q: %w", path, err)
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("config: validation: %w", err)
	}

	return cfg, nil
}

// Validate returns an error if the configuration contains values that would
// cause runtime failures. Call after Load() or after manual construction.
func (c *Config) Validate() error {
	if c.AximoComposePath == "" {
		return fmt.Errorf("aximo_compose_path must be set")
	}
	if c.AximoPort <= 0 || c.AximoPort > 65535 {
		return fmt.Errorf("aximo_port %d out of range", c.AximoPort)
	}
	if c.AximoStartupTimeoutMs > 60_000 {
		return fmt.Errorf("aximo_startup_timeout_ms %d exceeds hard max of 60 000ms", c.AximoStartupTimeoutMs)
	}
	if c.AximoInferenceTimeoutMs >= 120_000 {
		return fmt.Errorf("aximo_inference_timeout_ms %d must be < 120 000ms (Aximo's own limit)", c.AximoInferenceTimeoutMs)
	}
	if c.VADFrameMs != 10 && c.VADFrameMs != 20 && c.VADFrameMs != 30 {
		return fmt.Errorf("vad_frame_ms must be 10, 20, or 30; got %d", c.VADFrameMs)
	}
	if c.VADMode < 0 || c.VADMode > 3 {
		return fmt.Errorf("vad_mode must be 0–3; got %d", c.VADMode)
	}
	if c.AudioSampleRate != 16_000 {
		return fmt.Errorf("audio_sample_rate must be 16000; got %d", c.AudioSampleRate)
	}
	if c.WakeWordPythonPath == "" {
		return fmt.Errorf("wakeword_python_path must be set")
	}
	if c.WakeWordSidecarPath == "" {
		return fmt.Errorf("wakeword_sidecar_path must be set")
	}
	if c.SocketPath == "" {
		return fmt.Errorf("socket_path must be set")
	}
	if c.STTRetryDelayMs < 300 || c.STTRetryDelayMs > 500 {
		return fmt.Errorf("stt_retry_delay_ms %d out of 300–500ms range", c.STTRetryDelayMs)
	}
	return nil
}

// ─── Convenience duration accessors ──────────────────────────────────────────

func (c *Config) AximoStartupTimeout() time.Duration {
	return time.Duration(c.AximoStartupTimeoutMs) * time.Millisecond
}

func (c *Config) AximoStartupPollInterval() time.Duration {
	return time.Duration(c.AximoStartupPollIntervalMs) * time.Millisecond
}

func (c *Config) AximoInferenceTimeout() time.Duration {
	return time.Duration(c.AximoInferenceTimeoutMs) * time.Millisecond
}

func (c *Config) AximoReadinessCacheTTL() time.Duration {
	return time.Duration(c.AximoReadinessCacheTTLMs) * time.Millisecond
}

func (c *Config) STTRetryDelay() time.Duration {
	return time.Duration(c.STTRetryDelayMs) * time.Millisecond
}

func (c *Config) VADSilenceThreshold() time.Duration {
	return time.Duration(c.VADSilenceThresholdMs) * time.Millisecond
}

func (c *Config) VADMaxUtterance() time.Duration {
	return time.Duration(c.VADMaxUtteranceMs) * time.Millisecond
}

func (c *Config) AximoBaseURL() string {
	return fmt.Sprintf("http://%s:%d", c.AximoHost, c.AximoPort)
}
