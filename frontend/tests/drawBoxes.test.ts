import { describe, expect, it } from "vitest";

import { containRect, mapNormalizedBox, readableTextColor } from "../src/lib/drawBoxes";

describe("containRect", () => {
  it("adds horizontal letterboxing for portrait video in a landscape container", () => {
    expect(containRect(400, 200, 100, 200)).toEqual({ x: 150, y: 0, width: 100, height: 200 });
  });

  it("adds vertical letterboxing for landscape video in a portrait container", () => {
    expect(containRect(200, 400, 400, 200)).toEqual({ x: 0, y: 150, width: 200, height: 100 });
  });
});

describe("mapNormalizedBox", () => {
  it("maps corner and full-frame boxes inside the contained image region", () => {
    const rect = containRect(400, 200, 100, 200);
    expect(mapNormalizedBox([0, 0, 1, 1], rect)).toEqual({ x: 150, y: 0, width: 100, height: 200 });
    expect(mapNormalizedBox([0.1, 0.25, 0.6, 0.75], rect)).toEqual({ x: 160, y: 50, width: 50, height: 100 });
  });
});

describe("readableTextColor", () => {
  it("uses dark label text on light class colors", () => {
    expect(readableTextColor("#E9C46A")).toBe("#101613");
  });

  it("uses white label text on dark class colors", () => {
    expect(readableTextColor("#577590")).toBe("#ffffff");
  });
});
