import { describe, expect, it } from "vitest";

import { isDetectResult, isStreamResult } from "../src/api/types";

const detection = {
  class_id: 3,
  class_name: "late_blight",
  score: 0.81,
  box: [0.1, 0.2, 0.55, 0.77],
};

const assessment = {
  status: "accepted",
  message: "Đã xác minh lá cà chua",
  confidence: 0.93,
};

const recommendations = [{
  class_id: 3,
  class_name: "late_blight",
  title: "Mốc sương / úa muộn",
  actions: ["Kiểm tra toàn ruộng ngay."],
  caution: "Cần xác nhận thực địa.",
}];

describe("API response validation", () => {
  it("accepts the normalized detection response contract", () => {
    expect(isDetectResult({
      model: "yolo11s_coco",
      image: { width: 640, height: 480 },
      inference_ms: 18.7,
      detections: [detection],
      input_assessment: assessment,
      recommendations,
    })).toBe(true);
  });

  it("rejects malformed boxes and invalid WebSocket frame IDs", () => {
    expect(isDetectResult({
      model: "yolo11s_coco",
      image: { width: 640, height: 480 },
      inference_ms: 18.7,
      detections: [{ ...detection, box: [0, 0, 2, 1] }],
      input_assessment: assessment,
      recommendations,
    })).toBe(false);
    expect(isStreamResult({
      type: "result",
      frame_id: -1,
      model: "yolo11s_coco",
      image: { width: 640, height: 480 },
      inference_ms: 18.7,
      detections: [detection],
      input_assessment: assessment,
      recommendations,
    })).toBe(false);
  });
});
