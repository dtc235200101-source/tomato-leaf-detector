import { isApiError, isStreamResult, type StreamConfig, type StreamMessage } from "./types";
import { websocketApiUrl } from "./endpoints";

export function websocketUrl(path = "/ws/detect"): string {
  return websocketApiUrl(path);
}

export class DetectionSocket {
  private socket: WebSocket | null = null;

  constructor(
    private readonly onMessage: (message: StreamMessage) => void,
    private readonly onState: (state: "connecting" | "connected" | "closed" | "error") => void,
  ) {}

  connect(url = websocketUrl()): void {
    this.close();
    this.onState("connecting");
    const socket = new WebSocket(url);
    this.socket = socket;
    socket.onopen = () => this.onState("connected");
    socket.onerror = () => this.onState("error");
    socket.onclose = () => this.onState("closed");
    socket.onmessage = (event: MessageEvent<string>) => {
      try {
        const value: unknown = JSON.parse(event.data);
        if (isStreamResult(value) || isApiError(value)) this.onMessage(value);
        else this.onMessage({ code: "INVALID_RESPONSE", message: "Server returned an invalid response" });
      } catch {
        this.onMessage({ code: "INVALID_RESPONSE", message: "Server returned invalid JSON" });
      }
    };
  }

  sendConfig(config: StreamConfig): boolean {
    if (!this.ready()) return false;
    this.socket?.send(JSON.stringify(config));
    return true;
  }

  sendFrame(packet: ArrayBuffer): boolean {
    if (!this.ready()) return false;
    this.socket?.send(packet);
    return true;
  }

  close(): void {
    if (this.socket && this.socket.readyState < WebSocket.CLOSING) this.socket.close();
    this.socket = null;
  }

  private ready(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }
}
