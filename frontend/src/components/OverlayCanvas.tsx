import { useEffect, useRef, type RefObject } from "react";

import type { Detection } from "../api/types";
import { drawDetections } from "../lib/drawBoxes";

interface OverlayCanvasProps {
  containerRef: RefObject<HTMLElement | null>;
  image: { width: number; height: number } | null;
  detections: Detection[];
  colorForClass: (className: string) => string;
  labelForClass: (className: string) => string;
}

export function OverlayCanvas({ containerRef, image, detections, colorForClass, labelForClass }: OverlayCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas || !image) return;
    const render = () => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawDetections(context, rect, image, detections, colorForClass, labelForClass);
    };
    render();
    const observer = new ResizeObserver(render);
    observer.observe(container);
    return () => observer.disconnect();
  }, [colorForClass, containerRef, detections, image, labelForClass]);

  return <canvas ref={canvasRef} className="overlay-canvas" aria-label="Kết quả nhận diện" />;
}
