// Package stt — low-level Aximo HTTP client.
//
// AximoClient handles all raw HTTP communication with the Aximo service,
// keeping transport concerns isolated from the Provider business logic in
// aximo_http.go.
package stt

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// transcriptionRequest is the HTTP request to POST /v1/transcriptions.
// Aximo accepts audio/wav; we always encode as WAV (carries sample-rate and
// bit-depth metadata, unambiguous for the Parakeet v3 model).
//
// Encoding choice documented per spec §4: WAV was chosen over raw PCM
// (audio/pcm) because the WAV header encodes sample rate, channel count,
// and bit depth, making the payload self-describing. Raw PCM requires the
// server to trust the client's implicit assumptions about those parameters.

// transcriptionResponse is the JSON body returned by POST /v1/transcriptions
// on success.
type transcriptionResponse struct {
	Text             string            `json:"text"`
	Segments         []json.RawMessage `json:"segments"`
	DetectedLanguage *string           `json:"detected_language"`
	Engine           string            `json:"engine"`
	DurationMs       int64             `json:"duration_ms"`
	ProcessingMs     int64             `json:"processing_ms"`
}

// healthReadyResponse is the JSON body returned by GET /health/ready on 503.
type healthReadyResponse struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

// AximoClient wraps the Aximo HTTP API.
type AximoClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewAximoClient creates an AximoClient.
// timeout is applied as the HTTP client's overall deadline (connect + write +
// read). For POST /v1/transcriptions this should be well under Aximo's own
// AXIMO_SHORT_INFERENCE_TIMEOUT_MS (default 120 000ms).
func NewAximoClient(baseURL string, timeout time.Duration) *AximoClient {
	return &AximoClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

// PostTranscription sends pcm (pcm_s16le, 16kHz, mono) encoded as WAV to
// POST /v1/transcriptions and returns the parsed response.
func (c *AximoClient) PostTranscription(ctx context.Context, pcm []byte) (*transcriptionResponse, error) {
	wav := encodeWAV(pcm, 16000, 1, 16)

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		c.baseURL+"/v1/transcriptions",
		bytes.NewReader(wav),
	)
	if err != nil {
		return nil, fmt.Errorf("aximo_client: build request: %w", err)
	}
	req.Header.Set("Content-Type", "audio/wav")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("aximo_client: POST /v1/transcriptions: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("aximo_client: read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("aximo_client: POST /v1/transcriptions status %d: %s",
			resp.StatusCode, body)
	}

	var result transcriptionResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("aximo_client: decode response: %w", err)
	}
	return &result, nil
}

// GetHealthReady calls GET /health/ready and returns nil on 200, or an error
// on 503 / transport failure.
func (c *AximoClient) GetHealthReady(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health/ready", nil)
	if err != nil {
		return fmt.Errorf("aximo_client: build health request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("aximo_client: GET /health/ready: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		return nil
	}

	body, _ := io.ReadAll(resp.Body)
	return fmt.Errorf("aximo_client: /health/ready status %d: %s", resp.StatusCode, body)
}

// ─── WAV encoding ─────────────────────────────────────────────────────────────

// encodeWAV wraps pcm (pcm_s16le) in a standard RIFF WAV container.
// sampleRate: 16000, channels: 1, bitsPerSample: 16.
func encodeWAV(pcm []byte, sampleRate, channels, bitsPerSample int) []byte {
	dataSize := uint32(len(pcm))
	byteRate := uint32(sampleRate * channels * bitsPerSample / 8)
	blockAlign := uint16(channels * bitsPerSample / 8)

	buf := new(bytes.Buffer)
	buf.Grow(44 + int(dataSize))

	// RIFF chunk
	buf.WriteString("RIFF")
	binary.Write(buf, binary.LittleEndian, uint32(36+dataSize)) // chunk size
	buf.WriteString("WAVE")

	// fmt sub-chunk
	buf.WriteString("fmt ")
	binary.Write(buf, binary.LittleEndian, uint32(16))            // sub-chunk size (PCM)
	binary.Write(buf, binary.LittleEndian, uint16(1))             // audio format = PCM
	binary.Write(buf, binary.LittleEndian, uint16(channels))
	binary.Write(buf, binary.LittleEndian, uint32(sampleRate))
	binary.Write(buf, binary.LittleEndian, byteRate)
	binary.Write(buf, binary.LittleEndian, blockAlign)
	binary.Write(buf, binary.LittleEndian, uint16(bitsPerSample))

	// data sub-chunk
	buf.WriteString("data")
	binary.Write(buf, binary.LittleEndian, dataSize)
	buf.Write(pcm)

	return buf.Bytes()
}
