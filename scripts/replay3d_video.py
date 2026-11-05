"""Render a `replay.html` from `scripts/replay3d.py` to a 1080p MP4 (headless Chrome with software
WebGL, then ffmpeg).

    python scripts/replay3d_video.py results/replay3d/demo_A_attempt2/replay.html --out demo_A.mp4 [--fps 30]

Needs Playwright for Python (`pip install playwright`; it drives the system Chrome given by
`--chrome`, no browser download), Chrome or Chromium, and ffmpeg. Frames are captured by seeking
the page's own timeline (`window.replay.seek`), so the video is deterministic and not limited
by how fast software WebGL draws. The page is opened with `?render=1`, which hides the controls;
the permanent "not hardware" label stays.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--no-sandbox"]


async def capture(html: Path, frames: Path, fps: int, width: int, height: int, theme: str, chrome: str) -> int:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=chrome, args=ARGS)
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(f"{html.resolve().as_uri()}?render=1&theme={theme}")
        await page.wait_for_function("window.replay !== undefined", timeout=60000)
        duration = await page.evaluate("window.replay.duration")
        n = int(duration * fps) + fps  # one second of the last frame
        for i in range(n):
            await page.evaluate(f"window.replay.seek({min(i / fps, duration)})")
            await page.screenshot(path=str(frames / f"{i:06d}.jpg"), type="jpeg", quality=92)
            if i % (fps * 10) == 0:
                print(f"  frame {i}/{n}", flush=True)
        await browser.close()
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("html", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--chrome", default=shutil.which("google-chrome") or shutil.which("chromium") or "chromium")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(tmp)
        n = asyncio.run(capture(args.html, frames, args.fps, args.width, args.height, args.theme, args.chrome))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(frames / "%06d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium", str(args.out)],
                       check=True)
    print(f"wrote {args.out} ({n} frames, {args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
