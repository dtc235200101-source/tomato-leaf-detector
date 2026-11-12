import { useCallback, useEffect, useRef, useState } from "react";

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [state, setState] = useState<"idle" | "starting" | "active" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setState("idle");
  }, []);

  const start = useCallback(async (): Promise<boolean> => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Trình duyệt không hỗ trợ webcam.");
      setState("error");
      return false;
    }
    stop();
    setError(null);
    setState("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      streamRef.current = stream;
      if (!videoRef.current) throw new Error("Video element is unavailable");
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setState("active");
      return true;
    } catch (reason) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setError(reason instanceof Error ? reason.message : "Không thể mở webcam.");
      setState("error");
      return false;
    }
  }, [stop]);

  useEffect(() => stop, [stop]);
  return { videoRef, state, error, start, stop };
}
