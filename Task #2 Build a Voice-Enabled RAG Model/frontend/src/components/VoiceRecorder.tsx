"use client";

import { useState, useRef, useEffect, useCallback } from "react";

interface VoiceRecorderProps {
  onAudioReady: (blob: Blob) => void;
  isLoading: boolean;
}

const MAX_RECORDING_SECONDS = 30;

export default function VoiceRecorder({ onAudioReady, isLoading }: VoiceRecorderProps) {
  const [state, setState] = useState<"idle" | "requesting" | "recording" | "processing">("idle");
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [permissionDenied, setPermissionDenied] = useState(false);

  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopTimer();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const startRecording = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Your browser doesn't support microphone access. Please use Chrome or Firefox.");
      return;
    }

    setError(null);
    setPermissionDenied(false);
    setState("requesting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
      });
      streamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/ogg";

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorder.current = recorder;
      chunks.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: mimeType });
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        if (blob.size > 100) {
          onAudioReady(blob);
        } else {
          setError("Recording was too short. Please try again.");
          setState("idle");
        }
      };

      recorder.start(100); // Collect in 100ms chunks
      setState("recording");
      setSeconds(0);

      timerRef.current = setInterval(() => {
        setSeconds((s) => {
          if (s + 1 >= MAX_RECORDING_SECONDS) {
            stopRecording();
            return s + 1;
          }
          return s + 1;
        });
      }, 1000);
    } catch (err: any) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setPermissionDenied(true);
        setError("Microphone access denied. Please allow microphone access in your browser settings.");
      } else {
        setError(`Could not access microphone: ${err.message}`);
      }
      setState("idle");
    }
  }, [onAudioReady]);

  const stopRecording = useCallback(() => {
    stopTimer();
    if (mediaRecorder.current?.state === "recording") {
      setState("processing");
      mediaRecorder.current.stop();
    }
  }, []);

  const isRecording = state === "recording";
  const isProcessing = state === "processing" || isLoading;

  return (
    <div style={{ textAlign: "center" }}>
      {/* ── Mic button ──────────────────────────────────────── */}
      <div style={{ position: "relative", display: "inline-block", marginBottom: 24 }}>
        {/* Animated rings when recording */}
        {isRecording && (
          <>
            <div className="recording-ring" style={{ animationDelay: "0s" }} />
            <div className="recording-ring" style={{ animationDelay: "0.5s" }} />
          </>
        )}

        <button
          id="mic-button"
          className={`mic-btn ${isRecording ? "recording" : ""}`}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isProcessing || state === "requesting"}
          aria-label={isRecording ? "Stop recording" : "Start recording"}
          style={{
            background: isRecording
              ? "linear-gradient(135deg, #ef4444, #dc2626)"
              : "var(--gradient-mic)",
          }}
        >
          {isProcessing ? (
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" strokeWidth="2"/>
              <path d="M12 2a10 10 0 0 1 10 10" stroke="white" strokeWidth="2" strokeLinecap="round">
                <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
              </path>
            </svg>
          ) : isRecording ? (
            <svg width="32" height="32" viewBox="0 0 24 24" fill="white">
              <rect x="6" y="6" width="12" height="12" rx="2"/>
            </svg>
          ) : (
            <svg width="36" height="36" viewBox="0 0 24 24" fill="white">
              <path d="M12 1a4 4 0 0 1 4 4v7a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" stroke="white" strokeWidth="2" fill="none" strokeLinecap="round"/>
              <line x1="12" y1="19" x2="12" y2="23" stroke="white" strokeWidth="2" strokeLinecap="round"/>
              <line x1="8" y1="23" x2="16" y2="23" stroke="white" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          )}
        </button>
      </div>

      {/* ── Waveform bars when recording ────────────────────── */}
      {isRecording && (
        <div style={{
          display: "flex",
          gap: 4,
          justifyContent: "center",
          alignItems: "center",
          height: 40,
          marginBottom: 16,
        }}>
          {Array.from({ length: 12 }).map((_, i) => (
            <div
              key={i}
              className="waveform-bar"
              style={{
                animationDelay: `${i * 0.07}s`,
                animationDuration: `${0.6 + Math.random() * 0.4}s`,
              }}
            />
          ))}
        </div>
      )}

      {/* ── Status text ──────────────────────────────────────── */}
      <div style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginBottom: 8 }}>
        {state === "requesting" && "Requesting microphone access…"}
        {isRecording && (
          <span>
            <span style={{ color: "#ef4444", fontWeight: 600 }}>● REC</span>
            {" "}
            <span style={{ fontVariantNumeric: "tabular-nums", fontFamily: "JetBrains Mono, monospace" }}>
              {String(Math.floor(seconds / 60)).padStart(2, "0")}:
              {String(seconds % 60).padStart(2, "0")}
            </span>
            {" / "}
            <span style={{ color: "var(--text-muted)" }}>{MAX_RECORDING_SECONDS}s max</span>
          </span>
        )}
        {state === "processing" && "Transcribing audio…"}
        {isLoading && "Processing your query…"}
        {state === "idle" && !isLoading && "Tap to speak — any Indian language works"}
      </div>

      {/* ── Permission denied helper ──────────────────────────── */}
      {permissionDenied && (
        <div className="glass-card" style={{
          display: "inline-block",
          padding: "12px 20px",
          marginTop: 12,
          borderColor: "rgba(251, 191, 36, 0.3)",
          background: "rgba(251, 191, 36, 0.05)",
          maxWidth: 420,
          textAlign: "left",
        }}>
          <div style={{ fontWeight: 600, color: "#fbbf24", marginBottom: 6, fontSize: "0.875rem" }}>
            📵 Microphone blocked
          </div>
          <div style={{ color: "var(--text-secondary)", fontSize: "0.8rem" }}>
            Click the 🔒 icon in your browser's address bar → allow microphone → refresh the page.
            Or use the <strong>Text</strong> mode below.
          </div>
        </div>
      )}

      {/* ── Other errors ──────────────────────────────────────── */}
      {error && !permissionDenied && (
        <div style={{ color: "#fb7185", fontSize: "0.85rem", marginTop: 8 }}>{error}</div>
      )}
    </div>
  );
}
