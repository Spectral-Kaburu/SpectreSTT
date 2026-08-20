// SpectreSTT — self-hosted, privacy-first speech-to-text adapter.
//
// Usage:
//
//	spectrestt --config /path/to/spectrestt.json
//
// SpectreSTT:
//  1. Brings up the Aximo STT container (docker compose).
//  2. Spawns the openWakeWord Python sidecar.
//  3. Listens for "Hey Linus".
//  4. On detection: captures audio, runs VAD, transcribes via Aximo.
//  5. Pushes the transcript JSON over a Unix socket to any connected client.
//
// See spectrestt.example.json for all configurable parameters.
package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/lmittmann/tint"

	"github.com/Spectral-Kaburu/SpectreSTT/config"
	"github.com/Spectral-Kaburu/SpectreSTT/pipeline"
)

func main() {
	configPath := flag.String("config", "spectrestt.json", "path to config file")
	flag.Parse()

	// Colored, human-readable structured logging to stderr.
	// Switch to slog.NewJSONHandler when piping to a log aggregator.
	logger := slog.New(tint.NewHandler(os.Stderr, &tint.Options{
		Level:      slog.LevelInfo,
		TimeFormat: time.TimeOnly,
	}))
	slog.SetDefault(logger)

	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.Error("failed to load config", "path", *configPath, "err", err)
		os.Exit(1)
	}

	slog.Info("SpectreSTT starting",
		"aximo_host", cfg.AximoHost,
		"aximo_port", cfg.AximoPort,
		"socket_path", cfg.SocketPath,
		"wakeword_threshold", cfg.WakeWordThreshold,
	)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	p := pipeline.New(cfg)

	if err := p.Start(ctx); err != nil {
		slog.Error("pipeline exited with error", "err", err)
		os.Exit(1)
	}

	slog.Info("SpectreSTT stopped cleanly")
}
