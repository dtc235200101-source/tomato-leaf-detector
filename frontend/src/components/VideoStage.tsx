import type { RefObject } from "react";

import type { Detection } from "../api/types";
import { OverlayCanvas } from "./OverlayCanvas";

interface VideoStageProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  stageRef: RefObject<HTMLDivElement | null>;
  active: boolean;
  image: { width: number; height: number } | null;
  detections: Detection[];
  colorForClass: (className: string) => string;
  labelForClass: (className: string) => string;
}

export function VideoStage({ videoRef, stageRef, active, image, detections, colorForClass, labelForClass }: VideoStageProps) {
  return (
    <div className="media-stage" ref={stageRef}>
      <video ref={videoRef} muted playsInline aria-label="Webcam preview" />
      {!active ? (
        <p className="stage-placeholder">Camera đang tắt</p>
      ) : !image ? (
        <p className="stage-placeholder">Đang chờ khung hình…</p>
      ) : null}
      <OverlayCanvas containerRef={stageRef} image={image} detections={detections} colorForClass={colorForClass} labelForClass={labelForClass} />
    </div>
  );
}
