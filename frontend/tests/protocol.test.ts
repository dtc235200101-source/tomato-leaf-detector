import { describe, expect, it } from "vitest";

import { encodeFramePacket } from "../src/lib/frameEncoder";

describe("encodeFramePacket", () => {
  it("writes a uint32 big-endian frame header followed by JPEG bytes", () => {
    const packet = new Uint8Array(encodeFramePacket(0x01020304, new Uint8Array([9, 8]).buffer));
    expect([...packet]).toEqual([1, 2, 3, 4, 9, 8]);
  });

  it("rejects invalid frame IDs", () => {
    expect(() => encodeFramePacket(-1, new ArrayBuffer(0))).toThrow(RangeError);
  });
});
