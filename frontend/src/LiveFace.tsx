import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { FaceMatch } from "./types";

const WS_URL = "ws://localhost:8000/ws/recognize";
const CAPTURE_INTERVAL_MS = 300;

const COLOR_CLEAR = "#34d399";
const COLOR_STOP = "#f87171";
const COLOR_SPOOF = "#e879f9";

type Phase = "connecting" | "live" | "offline" | "error";

const STATUS_LABEL: Record<Phase, string> = {
  connecting: "Menghubungkan kamera…",
  live: "LIVE",
  offline: "Gate offline — server terputus",
  error: "Gate error — koneksi gagal",
};

export interface LiveFaceHandle {
  captureFrame(): Promise<Blob | null>;
}

interface Props {
  onFaces?: (faces: FaceMatch[]) => void;
}

const LiveFace = forwardRef<LiveFaceHandle, Props>(({ onFaces }, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef(document.createElement("canvas"));
  const wsRef = useRef<WebSocket | null>(null);
  const [phase, setPhase] = useState<Phase>("connecting");

  useImperativeHandle(ref, () => ({
    captureFrame() {
      const video = videoRef.current;
      if (!video || video.videoWidth === 0) return Promise.resolve(null);

      const capture = captureCanvasRef.current;
      capture.width = video.videoWidth;
      capture.height = video.videoHeight;
      capture.getContext("2d")!.drawImage(video, 0, 0);
      return new Promise((resolve) => capture.toBlob(resolve, "image/jpeg", 0.9));
    },
  }));

  useEffect(() => {
    let stream: MediaStream;

    async function start() {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => setPhase("live");
      ws.onclose = () => setPhase("offline");
      ws.onerror = () => setPhase("error");
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data) as { faces: FaceMatch[] };
        drawResults(data.faces);
        onFaces?.(data.faces);
      };

      const interval = setInterval(sendFrame, CAPTURE_INTERVAL_MS);
      return () => clearInterval(interval);
    }

    function sendFrame() {
      const video = videoRef.current;
      const ws = wsRef.current;
      if (!video || !ws || ws.readyState !== WebSocket.OPEN || video.videoWidth === 0) return;

      const capture = captureCanvasRef.current;
      capture.width = video.videoWidth;
      capture.height = video.videoHeight;
      const ctx = capture.getContext("2d")!;
      ctx.drawImage(video, 0, 0);
      capture.toBlob((blob) => blob && ws.send(blob), "image/jpeg", 0.7);
    }

    function drawResults(faces: FaceMatch[]) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.font = "600 15px 'IBM Plex Mono', monospace";

      // The <video> is mirrored via CSS for a natural selfie view, but the bbox coordinates
      // from the backend are in the raw (unmirrored) frame. The canvas itself stays unmirrored
      // (so label text reads normally) -- mirror just the box's x-coordinates to line up.
      for (const face of faces) {
        const [rawX1, y1, rawX2, y2] = face.bbox;
        const x1 = canvas.width - rawX2;
        const x2 = canvas.width - rawX1;
        const known = face.person_id !== null;
        const color = !face.live ? COLOR_SPOOF : known ? COLOR_CLEAR : COLOR_STOP;
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const label = !face.live
          ? `SPOOF terdeteksi · ${(face.liveness_score * 100).toFixed(0)}%`
          : `${face.name ?? "Tidak dikenal"} · ${(face.similarity * 100).toFixed(0)}%`;
        ctx.fillStyle = color;
        ctx.fillText(label, x1, y1 > 20 ? y1 - 6 : y1 + 16);
      }
    }

    setPhase("connecting");
    const cleanupPromise = start().catch(() => setPhase("error"));

    return () => {
      cleanupPromise.then((cleanup) => cleanup?.());
      stream?.getTracks().forEach((t) => t.stop());
      wsRef.current?.close();
    };
  }, []);

  return (
    <div className="scanner">
      <div className="scanner__frame">
        <video ref={videoRef} muted playsInline />
        <canvas ref={canvasRef} />
        <span className="scanner__corner scanner__corner--tl" aria-hidden />
        <span className="scanner__corner scanner__corner--tr" aria-hidden />
        <span className="scanner__corner scanner__corner--bl" aria-hidden />
        <span className="scanner__corner scanner__corner--br" aria-hidden />
      </div>
      <p className={`scanner__status scanner__status--${phase}`}>
        <span className="scanner__dot" aria-hidden />
        {STATUS_LABEL[phase]}
      </p>
    </div>
  );
});

export default LiveFace;
