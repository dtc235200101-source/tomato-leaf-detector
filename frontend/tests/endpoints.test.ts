import { describe, expect, it } from "vitest";

import { apiUrl, websocketApiUrl } from "../src/api/endpoints";

describe("deployment endpoint resolution", () => {
  it("keeps relative endpoints when no public API base is configured", () => {
    expect(apiUrl("/api/health")).toBe("/api/health");
    expect(websocketApiUrl("/ws/detect")).toMatch(/\/ws\/detect$/);
  });

  it("rejects endpoint paths without a leading slash", () => {
    expect(() => apiUrl("api/health")).toThrow("must start");
    expect(() => websocketApiUrl("ws/detect")).toThrow("must start");
  });
});
