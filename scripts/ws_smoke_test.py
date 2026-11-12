"""Minimal WebSocket protocol smoke test for a running local server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from io import BytesIO

from PIL import Image


def _jpeg_bytes(path: str | None) -> bytes:
    if path:
        with open(path, "rb") as stream:
            return stream.read()
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "green").save(buffer, format="JPEG")
    return buffer.getvalue()


async def _smoke(args: argparse.Namespace) -> int:
    import websockets

    uri = f"ws://{args.host}:{args.port}/ws/detect"
    config = {
        "type": "config",
        "model": args.model,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "agnostic_nms": False,
    }
    frame = (1).to_bytes(4, byteorder="big") + _jpeg_bytes(args.image)
    async with websockets.connect(uri, max_size=None) as socket:
        await socket.send(json.dumps(config))
        await socket.send(frame)
        raw = await asyncio.wait_for(socket.recv(), timeout=args.timeout)
    response = json.loads(raw)
    print(json.dumps(response, indent=2, ensure_ascii=False))
    if response.get("type") == "result":
        return 0
    if args.allow_unavailable and response.get("code") == "MODEL_UNAVAILABLE":
        return 0
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="yolo11s_coco")
    parser.add_argument("--image", default=None, help="Optional local JPEG/PNG path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Treat MODEL_UNAVAILABLE as a successful graceful-degradation check",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(_smoke(args))
    except Exception as error:  # noqa: BLE001 - CLI must report connection/protocol errors clearly
        print(f"WebSocket smoke test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
