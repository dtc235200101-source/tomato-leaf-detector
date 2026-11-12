import { useCallback, useEffect, useRef, useState } from "react";

import { DetectionSocket } from "../api/wsClient";
import { isApiError, type StreamConfig, type StreamResult } from "../api/types";
import { encodeFramePacket } from "../lib/frameEncoder";

export function useDetectionStream(config: StreamConfig, onResult?: (result: StreamResult) => void) {
  const socketRef = useRef<DetectionSocket | null>(null);
  const configRef = useRef(config);
  const frameIdRef = useRef(0);
  const lastRenderedRef = useRef(-1);
  const inFlightRef = useRef(false);
  const [state, setState] = useState<"idle" | "connecting" | "connected" | "closed" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StreamResult | null>(null);

  useEffect(() => {
    configRef.current = config;
    socketRef.current?.sendConfig(config);
  }, [config]);

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    inFlightRef.current = false;
    setState("closed");
  }, []);

  const connect = useCallback(() => {
    disconnect();
    setError(null);
    const socket = new DetectionSocket(
      (message) => {
        if (isApiError(message)) {
          inFlightRef.current = false;
          setError(`${message.code}: ${message.message}`);
          return;
        }
        inFlightRef.current = false;
        if (message.frame_id > lastRenderedRef.current) {
          lastRenderedRef.current = message.frame_id;
          setResult(message);
          onResult?.(message);
        }
      },
      (nextState) => {
        setState(nextState);
        if (nextState === "connected") socket.sendConfig(configRef.current);
      },
    );
    socketRef.current = socket;
    socket.connect();
  }, [disconnect, onResult]);

  const sendJpeg = useCallback((jpeg: ArrayBuffer): boolean => {
    if (inFlightRef.current) return false;
    const frameId = frameIdRef.current++;
    const sent = socketRef.current?.sendFrame(encodeFramePacket(frameId, jpeg)) ?? false;
    if (sent) inFlightRef.current = true;
    return sent;
  }, []);

  const hasInFlight = useCallback(() => inFlightRef.current, []);

  useEffect(() => disconnect, [disconnect]);
  return { state, error, result, connect, disconnect, sendJpeg, hasInFlight };
}
