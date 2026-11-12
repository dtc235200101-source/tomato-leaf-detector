export interface ClassInfo {
  id: number;
  slug: string;
  name_vi: string;
  group: string;
  color: string;
}

export interface ModelInfo {
  model_id: string;
  display_name: string;
  available: boolean;
  reason: string | null;
}

export interface Detection {
  class_id: number;
  class_name: string;
  score: number;
  box: [number, number, number, number];
}

export interface InputAssessment {
  status: "accepted" | "rejected" | "unverified";
  message: string;
  confidence: number | null;
}

export interface Recommendation {
  class_id: number;
  class_name: string;
  title: string;
  actions: string[];
  caution: string;
}

export interface DetectResult {
  model: string;
  image: { width: number; height: number };
  inference_ms: number;
  detections: Detection[];
  input_assessment: InputAssessment;
  recommendations: Recommendation[];
}

export interface StreamConfig {
  type: "config";
  model: string;
  conf: number;
  iou: number;
  imgsz: number;
  max_det: number;
  agnostic_nms: boolean;
}

export interface StreamResult extends DetectResult {
  type: "result";
  frame_id: number;
}

export interface ApiError {
  code: string;
  message: string;
}

export type StreamMessage = StreamResult | ApiError;

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : null;
}

function isImageInfo(value: unknown): value is DetectResult["image"] {
  const item = record(value);
  return item !== null && Number.isInteger(item.width) && (item.width as number) > 0 && Number.isInteger(item.height) && (item.height as number) > 0;
}

function isDetection(value: unknown): value is Detection {
  const item = record(value);
  if (item === null || !Number.isInteger(item.class_id) || typeof item.class_name !== "string" || typeof item.score !== "number" || item.score < 0 || item.score > 1 || !Array.isArray(item.box) || item.box.length !== 4) return false;
  return item.box.every((coordinate) => typeof coordinate === "number" && coordinate >= 0 && coordinate <= 1);
}

export function isClassInfo(value: unknown): value is ClassInfo {
  const item = record(value);
  return item !== null && Number.isInteger(item.id) && typeof item.slug === "string" && typeof item.name_vi === "string" && typeof item.group === "string" && typeof item.color === "string";
}

export function isModelInfo(value: unknown): value is ModelInfo {
  const item = record(value);
  return item !== null && typeof item.model_id === "string" && typeof item.display_name === "string" && typeof item.available === "boolean" && (typeof item.reason === "string" || item.reason === null);
}

export function isDetectResult(value: unknown): value is DetectResult {
  const item = record(value);
  const assessment = item === null ? null : record(item.input_assessment);
  const recommendations = item === null ? null : item.recommendations;
  return item !== null && typeof item.model === "string" && isImageInfo(item.image) && typeof item.inference_ms === "number" && item.inference_ms >= 0 && Array.isArray(item.detections) && item.detections.every(isDetection) && assessment !== null && (assessment.status === "accepted" || assessment.status === "rejected" || assessment.status === "unverified") && typeof assessment.message === "string" && (typeof assessment.confidence === "number" || assessment.confidence === null) && Array.isArray(recommendations) && recommendations.every((advice) => { const current = record(advice); return current !== null && Number.isInteger(current.class_id) && typeof current.class_name === "string" && typeof current.title === "string" && Array.isArray(current.actions) && current.actions.every((action) => typeof action === "string") && typeof current.caution === "string"; });
}

export function isApiError(value: unknown): value is ApiError {
  const item = record(value);
  return item !== null && typeof item.code === "string" && typeof item.message === "string";
}

export function isStreamResult(value: unknown): value is StreamResult {
  const item = record(value);
  if (item === null) return false;
  return (
    item.type === "result" &&
    Number.isInteger(item.frame_id) &&
    (item.frame_id as number) >= 0 &&
    isDetectResult(item)
  );
}
