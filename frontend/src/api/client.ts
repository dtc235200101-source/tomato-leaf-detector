import {
  isClassInfo,
  isDetectResult,
  isModelInfo,
  type ApiError,
  type ClassInfo,
  type DetectResult,
  type ModelInfo,
} from "./types";
import { apiUrl } from "./endpoints";

export class HttpApiError extends Error {
  readonly code: string;

  constructor(error: ApiError, readonly status: number) {
    super(error.message);
    this.code = error.code;
  }
}

async function requestJson<T>(
  path: string,
  validate: (payload: unknown) => payload is T,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(apiUrl(path), init);
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const error: ApiError =
      typeof payload === "object" && payload !== null && "code" in payload && "message" in payload
        ? (payload as ApiError)
        : { code: "HTTP_ERROR", message: `Request failed (${response.status})` };
    throw new HttpApiError(error, response.status);
  }
  if (!validate(payload)) {
    throw new HttpApiError({ code: "INVALID_RESPONSE", message: "Backend returned an invalid response" }, response.status);
  }
  return payload;
}

export interface SplitSummary {
  name: string;
  count: number;
  samples: string[];
}

export function datasetImage(split: string, imageId: string): string {
  const name = imageId.split("/").pop() ?? imageId;
  return apiUrl(`/api/dataset/image/${split}?name=${encodeURIComponent(name)}`);
}

export async function getDatasetSplits(): Promise<SplitSummary[]> {
  const payload = await fetchJsonUnknown("/api/dataset/splits");
  const value = payload as { splits?: unknown };
  if (typeof value !== "object" || value === null || !Array.isArray(value.splits)) {
    throw new HttpApiError({ code: "INVALID_RESPONSE", message: "Invalid dataset payload" }, 200);
  }
  return value.splits.filter(
    (item): item is SplitSummary =>
      typeof item === "object" && item !== null &&
      typeof (item as SplitSummary).name === "string" &&
      typeof (item as SplitSummary).count === "number" &&
      Array.isArray((item as SplitSummary).samples),
  );
}

async function fetchJsonUnknown(path: string): Promise<unknown> {
  const response = await fetch(apiUrl(path));
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const error = payload as ApiError | null;
    throw new HttpApiError(
      error && typeof error === "object" && "code" in error
        ? error
        : { code: "HTTP_ERROR", message: `Request failed (${response.status})` },
      response.status,
    );
  }
  return payload;
}

export function getClasses(): Promise<ClassInfo[]> {
  return requestJson("/api/classes", (payload): payload is ClassInfo[] => Array.isArray(payload) && payload.every(isClassInfo));
}

export function getModels(): Promise<ModelInfo[]> {
  return requestJson("/api/models", (payload): payload is ModelInfo[] => Array.isArray(payload) && payload.every(isModelInfo));
}

export function detectImage(
  file: File,
  options: { model: string; conf: number; iou: number; imgsz: number; maxDet: number },
): Promise<DetectResult> {
  const query = new URLSearchParams({
    model: options.model,
    conf: String(options.conf),
    iou: String(options.iou),
    imgsz: String(options.imgsz),
    max_det: String(options.maxDet),
  });
  const body = new FormData();
  body.append("file", file);
  return requestJson(`/api/detect?${query}`, isDetectResult, { method: "POST", body });
}
