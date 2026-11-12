export function encodeFramePacket(frameId: number, jpeg: ArrayBuffer): ArrayBuffer {
  if (!Number.isInteger(frameId) || frameId < 0 || frameId > 0xffffffff) {
    throw new RangeError("frameId must be a uint32");
  }
  const packet = new Uint8Array(4 + jpeg.byteLength);
  new DataView(packet.buffer).setUint32(0, frameId, false);
  packet.set(new Uint8Array(jpeg), 4);
  return packet.buffer;
}

export async function encodeVideoFrame(
  video: HTMLVideoElement,
  maxEdge: number,
  quality = 0.7,
): Promise<Blob> {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  if (sourceWidth <= 0 || sourceHeight <= 0) throw new Error("Video dimensions are unavailable");
  const scale = Math.min(1, maxEdge / Math.max(sourceWidth, sourceHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(sourceWidth * scale);
  canvas.height = Math.round(sourceHeight * scale);
  canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
  if (!blob) throw new Error("JPEG encoding failed");
  return blob;
}
