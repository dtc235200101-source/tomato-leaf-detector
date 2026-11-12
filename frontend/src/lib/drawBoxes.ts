import type { Detection } from "../api/types";

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

function channelLuminance(value: number): number {
  const channel = value / 255;
  return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
}

/** Pick near-white or near-black label text with the higher contrast against a hex fill. */
export function readableTextColor(background: string): string {
  const match = /^#([0-9a-f]{6})$/i.exec(background.trim());
  if (!match) return "#101613";
  const hex = match[1];
  const [r, g, b] = [0, 2, 4].map((offset) => channelLuminance(Number.parseInt(hex.slice(offset, offset + 2), 16)));
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const contrastOnWhite = 1.05 / (luminance + 0.05);
  const contrastOnDark = (luminance + 0.05) / 0.05;
  return contrastOnWhite >= contrastOnDark ? "#ffffff" : "#101613";
}

export function containRect(containerWidth: number, containerHeight: number, imageWidth: number, imageHeight: number): Rect {
  const scale = Math.min(containerWidth / imageWidth, containerHeight / imageHeight);
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  return { x: (containerWidth - width) / 2, y: (containerHeight - height) / 2, width, height };
}

export function mapNormalizedBox(box: Detection["box"], rect: Rect): Rect {
  const [x1, y1, x2, y2] = box;
  return { x: rect.x + x1 * rect.width, y: rect.y + y1 * rect.height, width: (x2 - x1) * rect.width, height: (y2 - y1) * rect.height };
}

export function drawDetections(
  context: CanvasRenderingContext2D,
  container: { width: number; height: number },
  image: { width: number; height: number },
  detections: Detection[],
  colorForClass: (className: string) => string,
  labelForClass?: (className: string) => string,
): void {
  const rect = containRect(container.width, container.height, image.width, image.height);
  context.clearRect(0, 0, container.width, container.height);
  for (const detection of detections) {
    const box = mapNormalizedBox(detection.box, rect);
    const color = colorForClass(detection.class_name);
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.strokeRect(box.x, box.y, box.width, box.height);
    const className = labelForClass?.(detection.class_name) ?? detection.class_name;
    const label = `${className} ${(detection.score * 100).toFixed(0)}%`;
    context.font = "12px system-ui";
    const labelWidth = context.measureText(label).width + 8;
    context.fillStyle = color;
    context.fillRect(box.x, Math.max(rect.y, box.y - 18), labelWidth, 18);
    context.fillStyle = readableTextColor(color);
    context.fillText(label, box.x + 4, Math.max(rect.y + 13, box.y - 5));
  }
}
