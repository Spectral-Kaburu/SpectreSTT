// Package audio handles microphone capture via PortAudio.
//
// The Capturer opens the default input device, reads PCM frames of a fixed
// size (matching what the VAD detector expects), and sends them on a channel.
// It is activated only after the wake word fires; the Python wakeword sidecar
// owns the audio device during the idle hot-loop phase and releases it (by
// pausing its own stream) before signalling DETECTED — after which the Go
// Capturer can open without device conflicts.
package audio

import (
	"context"
	"encoding/binary"
	"fmt"

	"github.com/gordonklaus/portaudio"
)

// Capturer reads PCM frames from the default microphone via PortAudio.
type Capturer struct {
	sampleRate int
	frameSize  int // number of samples per frame (not bytes)
}

// New creates a Capturer. frameSize must match the VAD detector's FrameBytes()/2
// (i.e. number of int16 samples per frame).
//
// Example for 30ms at 16kHz: frameSize = 480 samples.
func New(sampleRate, frameSize int) *Capturer {
	return &Capturer{
		sampleRate: sampleRate,
		frameSize:  frameSize,
	}
}

// Capture opens the default input device, starts reading PCM frames, and
// pushes them on the returned channel. The channel is closed when ctx is
// cancelled or when an unrecoverable read error occurs.
//
// Each frame on the channel is exactly frameSize*2 bytes of pcm_s16le,
// 16kHz, mono. The VAD detector consumes these frames directly.
//
// Callers must drain the channel or cancel ctx to avoid goroutine leaks.
func (c *Capturer) Capture(ctx context.Context) (<-chan []byte, error) {
	if err := portaudio.Initialize(); err != nil {
		return nil, fmt.Errorf("audio: portaudio init: %w", err)
	}

	samples := make([]int16, c.frameSize)
	stream, err := portaudio.OpenDefaultStream(
		1,               // input channels (mono)
		0,               // output channels
		float64(c.sampleRate),
		len(samples),
		&samples,
	)
	if err != nil {
		portaudio.Terminate()
		return nil, fmt.Errorf("audio: open default stream: %w", err)
	}

	if err := stream.Start(); err != nil {
		stream.Close()
		portaudio.Terminate()
		return nil, fmt.Errorf("audio: start stream: %w", err)
	}

	ch := make(chan []byte, 8) // small buffer to absorb scheduling jitter

	go func() {
		defer func() {
			close(ch)
			stream.Stop()  //nolint:errcheck
			stream.Close() //nolint:errcheck
			portaudio.Terminate()
		}()

		for {
			if ctx.Err() != nil {
				return
			}

			if err := stream.Read(); err != nil {
				// PortAudio returns an error on device disconnection or
				// driver failure. Log via the returned channel closing;
				// the pipeline will treat the closed channel as end-of-stream.
				return
			}

			frame := int16ToBytes(samples)

			select {
			case <-ctx.Done():
				return
			case ch <- frame:
			}
		}
	}()

	return ch, nil
}

// int16ToBytes converts a slice of int16 samples to a pcm_s16le byte slice.
func int16ToBytes(samples []int16) []byte {
	buf := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(s))
	}
	return buf
}
