"""Check the SAM 3.1 client against the live Meta Model API (needs META_API_KEY in the env).

    python scripts/rig_sam_check.py photo.jpg "green circle" "black tape"
    python scripts/rig_sam_check.py --camera 0 "pen tip"
    python scripts/rig_sam_check.py --out runs/samcheck photo.jpg "red cap"

For each prompt it prints the parsed boxes and confidences, any parse diagnostics and the raw
output text, and writes the raw response body (no key in it) and an overlay PNG with the boxes
to --out. Use it to confirm the UNVERIFIED items in nomanual/rig/sam.py's docstring: the env
var, the PNG data URL, include_confidence, and whether the mask raster is the box's size.
Each image costs about $0.0025 (Meta's listed price, 2026-10-27).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomanual.rig.sam import SamClient, SamConfigError, SamError, extract_text  # noqa: E402


def load_image(args: argparse.Namespace) -> np.ndarray:
    if args.camera is not None:
        from nomanual.rig.camera import Camera, OpenCVSource

        with Camera(OpenCVSource(args.camera)) as cam:
            for _ in range(10):  # let exposure settle
                frame = cam.wait_for_frame(timeout=5.0)
            return frame.image
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    return rgb[:, :, ::-1].copy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="image file (omit with --camera)")
    ap.add_argument("prompts", nargs="+", help="short noun phrases, one concept each")
    ap.add_argument("--camera", type=int, help="grab a frame from this camera index instead")
    ap.add_argument("--out", type=Path, default=Path("runs/samcheck"))
    ap.add_argument("--no-confidence", action="store_true")
    ap.add_argument("--lossless", action="store_true", help="mask_encoding lossless instead of one_bit")
    args = ap.parse_args(argv)
    if args.camera is not None and args.image is not None:
        args.prompts.insert(0, args.image)
        args.image = None
    if args.camera is None and args.image is None:
        ap.error("give an image or --camera")

    try:
        sam = SamClient()
    except SamConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    img = load_image(args)
    args.out.mkdir(parents=True, exist_ok=True)
    overlay = Image.fromarray(img[:, :, ::-1].copy())
    draw = ImageDraw.Draw(overlay)
    print(f"{sam!r}; image {img.shape[1]}x{img.shape[0]}")
    status = 0
    for i, prompt in enumerate(args.prompts):
        try:
            res = sam.segment(img, prompt, include_confidence=not args.no_confidence,
                              mask_encoding="lossless" if args.lossless else "one_bit")
        except SamError as exc:
            print(f"[{prompt}] FAILED: {exc}")
            status = 1
            continue
        raw = sam.last_response or {}
        (args.out / f"response_{i}.json").write_text(json.dumps(raw, indent=2))
        print(f"\n[{prompt}] {len(res.segments)} match(es) in {res.elapsed_s:.2f} s, response {res.response_id}")
        for s in res.segments:
            x1, y1, x2, y2 = s.box
            mt = s.mask_token
            mask = f"mask {mt.rows}x{mt.cols} enc {mt.encoding!r} payload {len(mt.payload)} chars" if mt else "no mask"
            fits_box = mt is not None and (mt.rows, mt.cols) == (y2 - y1 + 1, x2 - x1 + 1)
            print(f"  object {s.object_id} frame {s.frame}: box ({x1},{y1})-({x2},{y2}) of {s.image_size}, "
                  f"c={s.confidence}, {mask}, mask is box-sized: {fits_box}")
            draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 255), width=2)
            draw.text((x1 + 2, max(0, y1 - 12)), f"{prompt} {s.confidence}", fill=(255, 0, 255))
        for d in res.diagnostics:
            print(f"  diagnostic: {d}")
        print(f"  raw text: {extract_text(raw)[:300]!r}")
    overlay.save(args.out / "overlay.png")
    print(f"\nwrote {args.out / 'overlay.png'} and the raw responses; calls made: {sam.calls}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
