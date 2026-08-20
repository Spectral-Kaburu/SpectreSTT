// Package socket implements the Unix domain socket server through which
// SpectreSTT pushes transcript payloads to connected clients.
//
// Any process on the same machine (e.g. Linus orchestrator, other services)
// can connect to the socket path configured in spectrestt.json and receive
// newline-delimited JSON transcript messages.
//
// Wire format: each transcript is a single JSON object followed by '\n'.
// Clients should read line-by-line. Example:
//
//	{"text":"Turn off the lights","duration_ms":1420,"processing_ms":38,"timestamp":"2026-08-20T16:25:17Z"}
//
// Slow or disconnected clients are dropped without blocking the broadcast.
package socket

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"sync"
	"time"
)

// TranscriptPayload is the JSON message pushed to every connected client
// when a transcription is complete.
type TranscriptPayload struct {
	Text         string    `json:"text"`
	DurationMs   int64     `json:"duration_ms"`
	ProcessingMs int64     `json:"processing_ms"`
	Timestamp    time.Time `json:"timestamp"`
}

// Server is a Unix domain socket server that broadcasts TranscriptPayload
// messages to all currently connected clients.
type Server struct {
	path     string
	listener net.Listener

	mu      sync.RWMutex
	clients map[net.Conn]struct{}
}

// New creates a Server that will listen on path.
// Start() must be called to begin accepting connections.
func New(path string) *Server {
	return &Server{
		path:    path,
		clients: make(map[net.Conn]struct{}),
	}
}

// Start begins accepting client connections. It removes any stale socket
// file at path before binding. Runs until ctx is cancelled.
func (s *Server) Start(ctx context.Context) error {
	// Remove stale socket from a previous run.
	if err := os.Remove(s.path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("socket: remove stale %q: %w", s.path, err)
	}

	ln, err := net.Listen("unix", s.path)
	if err != nil {
		return fmt.Errorf("socket: listen %q: %w", s.path, err)
	}
	s.listener = ln

	go s.acceptLoop(ctx)
	return nil
}

// acceptLoop accepts connections until the listener is closed.
func (s *Server) acceptLoop(ctx context.Context) {
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			// Listener was closed (normal shutdown).
			return
		}
		s.mu.Lock()
		s.clients[conn] = struct{}{}
		s.mu.Unlock()

		// Monitor the connection so we can evict it when it closes.
		go s.monitorConn(conn)
	}
}

// monitorConn blocks until conn is closed by the client, then removes it
// from the active set.
func (s *Server) monitorConn(conn net.Conn) {
	buf := make([]byte, 1)
	// A zero-byte read unblocks when the client disconnects.
	for {
		conn.SetReadDeadline(time.Now().Add(30 * time.Second))
		_, err := conn.Read(buf)
		if err != nil {
			s.remove(conn)
			return
		}
	}
}

// remove evicts a connection from the active set and closes it.
func (s *Server) remove(conn net.Conn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.clients, conn)
	conn.Close()
}

// Broadcast serialises payload as JSON and writes it (with a trailing '\n')
// to every connected client. Slow or broken clients are dropped without
// blocking; writes are given a short deadline to avoid head-of-line blocking.
func (s *Server) Broadcast(payload TranscriptPayload) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("socket: marshal payload: %w", err)
	}
	data = append(data, '\n')

	s.mu.RLock()
	conns := make([]net.Conn, 0, len(s.clients))
	for c := range s.clients {
		conns = append(conns, c)
	}
	s.mu.RUnlock()

	const writeTimeout = 200 * time.Millisecond
	for _, c := range conns {
		c.SetWriteDeadline(time.Now().Add(writeTimeout))
		if _, err := c.Write(data); err != nil {
			// Client is too slow or disconnected — evict it.
			s.remove(c)
		}
	}
	return nil
}

// Close shuts down the server and closes all connected clients.
func (s *Server) Close() error {
	if s.listener != nil {
		s.listener.Close()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for c := range s.clients {
		c.Close()
		delete(s.clients, c)
	}
	os.Remove(s.path)
	return nil
}
