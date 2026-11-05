"""Rig setup and measurements (SPEC §3.1, §3.4). Every subcommand also runs on the fake rig
(`--fake`: the firmware controller on this laptop + a synthetic camera), for rehearsal.

    python scripts/rig_setup.py markers --board-mm 400 300 --marker-mm 30 --out rig/markers.png
    python scripts/rig_setup.py calibrate --camera 0 --board-mm 400 300 --marker-mm 30
    python scripts/rig_setup.py board --camera 0 --target "green circle" --hazard "black tape" --tip "red pen tip"
    python scripts/rig_setup.py repeatability --port COM5 --camera 0      # DIP 0: calibration mode
    python scripts/rig_setup.py latency --port COM5 --camera 0            # DIP 0
    python scripts/rig_setup.py dmax --port COM5 --camera 0 --label draw1 # DIP 1..8, once per draw
    python scripts/rig_setup.py dmax --sim                                # expected values, no rig
    python scripts/rig_setup.py dmax --combine rig/dmax_draw*.json --slope-max 0.95

Outputs go to rig/ (calibration.json, board.json, dmax_<label>.json, ...). The camera index,
resolution and the calibration file are shared by all subcommands. SAM needs META_API_KEY in
the environment; without it, ground the tip by hand with --tip-px X Y (pixel of the tip in the
first frame). Read docs/RIG_RUNBOOK.md for the order of steps on the day.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomanual.rig import procedures as P  # noqa: E402
from nomanual.rig.calibrate import Calibration, CalibrationError, MarkerLayout  # noqa: E402

OUT = Path("rig")


# --- building the rig --------------------------------------------------------------------------


def _sam(args):
    if getattr(args, "no_sam", False):
        return None
    from nomanual.rig.sam import SamClient, SamConfigError

    try:
        return SamClient()
    except SamConfigError:
        return None


def open_rig(args, draw_for_fake: int = 1):
    """(env, closer) for the real rig or the fake one."""
    if args.fake:
        from nomanual.rig.fake_rig import build_fake_rig

        rig = build_fake_rig(None, draw=args.draw if args.draw is not None else draw_for_fake,
                             time_scale=args.time_scale, heartbeat_s=0.1 / max(1.0, args.time_scale / 2),
                             run_dir=args.out / "frames_setup", log=_log(args))
        return rig.env, rig.close
    if not args.port or args.camera is None:
        raise SystemExit("need --port and --camera (or --fake)")
    from nomanual.rig.camera import Camera, OpenCVSource
    from nomanual.rig.env import RigEnv
    from nomanual.rig.serial_client import PySerialTransport, SerialClient
    from nomanual.rig.tracker import SpotModel, TipTracker

    cal = Calibration.load(args.calibration)
    camera = Camera(OpenCVSource(args.camera, args.width, args.height, args.fps)).start()
    client = SerialClient(lambda: PySerialTransport(args.port), log=_log(args))
    tracker = TipTracker(cal, colour=SpotModel(min_value=args.laser_min_value) if args.laser else None)
    if args.tip_px:
        frame = camera.wait_for_frame(timeout=5.0)
        tracker.ground(frame.image, px=tuple(args.tip_px), radius=args.tip_radius)
    sam = _sam(args)
    if sam is None and not args.tip_px and not args.laser:
        raise SystemExit("no META_API_KEY for SAM: ground the tip by hand with --tip-px X Y")
    env = RigEnv(client, camera, cal, tracker, sam=sam, run_dir=args.out / "frames_setup",
                 tip_prompt=args.tip, log=_log(args))

    def close():
        env.close()
        camera.stop()

    return env, close


def _log(args):
    return (lambda m: print(f"[rig] {m}")) if getattr(args, "verbose", False) else None


def _grab(args):
    """One frame from the camera (or the fake board) for calibration / board-from-words."""
    if args.fake:
        from nomanual.rig.camera import FakeSource

        markers = "dots" if getattr(args, "dots", False) else "aruco"
        mm, (bw, bh) = getattr(args, "marker_mm", 30.0), getattr(args, "board_mm", (400.0, 300.0))
        size = (mm / bw, mm / bh)  # a square marker on a non-square board, in board units
        src = FakeSource(markers=markers, marker_size=size, realtime=False, tip=lambda t: (0.3, 0.4),
                         corners=((70, 425), (575, 410), (550, 65), (95, 50)))
        return src.render(0.0), src
    from nomanual.rig.camera import Camera, OpenCVSource

    with Camera(OpenCVSource(args.camera, args.width, args.height, args.fps)) as cam:
        for _ in range(15):  # let auto exposure settle
            frame = cam.wait_for_frame(timeout=5.0)
    return frame.image, None


# --- subcommands -------------------------------------------------------------------------------


def cmd_markers(args) -> int:
    from nomanual.rig.calibrate import marker_sheet

    layout = MarkerLayout(args.board_mm[0], args.board_mm[1], args.marker_mm)
    path = marker_sheet(args.out / "markers.png", layout, dpi=args.dpi)
    print(f"wrote {path}: print at 100% scale, check one marker measures {args.marker_mm:g} mm, and stick "
          "each marker centred on its board corner, upright (top towards the far edge, +y)")
    return 0


def cmd_calibrate(args) -> int:
    from nomanual.rig.calibrate import calibrate_aruco, calibrate_dots
    from nomanual.rig.camera import Camera

    img, _ = _grab(args)
    try:
        if args.dots:
            cal = calibrate_dots(img)
        else:
            cal = calibrate_aruco(img, MarkerLayout(args.board_mm[0], args.board_mm[1], args.marker_mm))
    except CalibrationError as exc:
        Camera.save_still(img, args.out / "calibration_failed.jpg")
        print(f"calibration failed: {exc} (frame saved to {args.out / 'calibration_failed.jpg'})")
        return 1
    print(cal.report())
    cal.save(args.calibration)
    _overlay(img, cal, args.out / "calibration_check.png")
    print(f"wrote {args.calibration} and {args.out / 'calibration_check.png'} (a 0.1 grid drawn through the "
          "calibration: it must line up with the board)")
    return 0


def _overlay(img, cal: Calibration, path: Path) -> None:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img[:, :, ::-1].copy())
    d = ImageDraw.Draw(im)
    for k in range(11):
        a = k / 10
        for p, q in (((a, 0.0), (a, 1.0)), ((0.0, a), (1.0, a))):
            d.line([tuple(cal.to_pixel(p)), tuple(cal.to_pixel(q))], fill=(255, 0, 255), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def cmd_board(args) -> int:
    from nomanual.rig.sam import SamError, board_from_words

    img, src = _grab(args)
    cal = Calibration.load(args.calibration)
    if args.fake:
        from nomanual.rig.fake_rig import FakeSegmenter

        sam = FakeSegmenter(src, {args.target: (0.7, 0.7, 0.8, 0.8),
                                  **{h: (0.4, 0.2 + 0.1 * i, 0.5, 0.6) for i, h in enumerate(args.hazard)}})
    else:
        sam = _sam(args)
        if sam is None:
            raise SystemExit("board-from-words needs SAM: set META_API_KEY")
    start = None
    if args.laser:  # the start is where the dot rests, found by brightness (SAM reads the rest)
        from nomanual.rig.tracker import SpotModel, TipTracker

        fix = TipTracker(cal, colour=SpotModel(min_value=args.laser_min_value)).ground_spot(img)
        if fix is None:
            print("failed: no laser dot on the board (is the servo supply on and the exposure locked?)")
            return 1
        start = fix.pos
    try:
        board, results = board_from_words(sam, img, cal, args.target, args.hazard, tip=args.tip, start=start,
                                          min_confidence=args.min_confidence)
    except SamError as exc:
        print(f"failed: {exc}")
        return 1
    for k, r in results.items():
        print(f"{k}: {r.prompt!r} -> {len(r.segments)} match(es), confidences {[s.confidence for s in r.segments]}")
    args.board_out.parent.mkdir(parents=True, exist_ok=True)
    args.board_out.write_text(board.model_dump_json(indent=1))
    print(f"wrote {args.board_out}: target {board.target.center} r={board.target.r:.3f}, "
          f"{len(board.hazards)} hazard(s), start {board.start}. Check it against the board before use.")
    return 0


def cmd_repeatability(args) -> int:
    env, close = open_rig(args, draw_for_fake=0)
    try:
        env.reset()
        rep = P.measure_repeatability(env, P.grid_points(args.grid), repeats=args.repeats, still_frames=args.still_frames)
    finally:
        close()
    print(rep.text())
    lo, hi = rep.slopes()
    out = {"spread": rep.spread(), "jitter": rep.jitter, "jitter_px": rep.jitter_px, "slope_min": lo, "slope_max": hi,
           "points": rep.points, "fixes": {str(k): v for k, v in rep.fixes.items()}}
    _write(args.out / "repeatability.json", out)
    return 0


def cmd_latency(args) -> int:
    env, close = open_rig(args, draw_for_fake=0)
    try:
        env.reset()
        rep = P.measure_latency(env, n_ping=args.pings, n_moves=args.moves)
    finally:
        close()
    print(rep.text())
    _write(args.out / "latency.json", {"ping_s": rep.ping_s, "motion_s": rep.motion_s, "move_px": rep.move_px})
    return 0


def cmd_dmax(args) -> int:
    schedule = P.dmax_schedule(hold=args.hold, rounds=args.rounds, seed=args.seed)
    if args.limit:
        schedule = schedule[: args.limit]
    if args.combine:
        reps = [json.loads(Path(p).read_text()) for p in args.combine]
        observed = max(r["observed"] for r in reps)
        stopped = sum(r["stopped"] for r in reps)
        print("\n".join(f"{r['label']}: observed {r['observed']:.4f}, {r['stopped']} not done" for r in reps))
    elif args.sim:
        from nomanual.core.machine import MachineConfig

        exported = json.loads(args.configs.read_text())
        reps = []
        for d in exported["draws"]:
            rep = P.simulate_dmax(MachineConfig.model_validate(d["config"]), schedule, label=f"sim draw {d['draw']}")
            print(rep.summary())
            reps.append(rep)
        observed, stopped = max(r.observed for r in reps), 0
        _write(args.out / "dmax_sim.json", {"observed": observed, "draws": [r.to_json() for r in reps]})
    else:
        env, close = open_rig(args, draw_for_fake=1)
        try:
            env.reset()
            rep = P.measure_dmax(env, schedule, label=args.label, reset_every=args.reset_every)
        finally:
            close()
        print(rep.summary())
        _write(args.out / f"dmax_{args.label}.json", rep.to_json())
        observed, stopped = rep.observed, rep.stopped
    rec = P.recommend_dmax(observed, sigma=args.sigma, rel_margin=args.rel_margin, box_servo=args.box,
                           slope_max=args.slope_max, stopped=stopped)
    print(rec.text())
    return 0


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    print(f"wrote {path}")


# --- CLI ----------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fake", action="store_true", help="rehearse on the fake rig")
    common.add_argument("--draw", type=int, help="--fake: DIP draw to emulate")
    common.add_argument("--time-scale", type=float, default=4.0, help="--fake: firmware time per wall time")
    common.add_argument("--port")
    common.add_argument("--camera", type=int)
    common.add_argument("--width", type=int, default=1280)
    common.add_argument("--height", type=int, default=720)
    common.add_argument("--fps", type=float, default=30.0)
    common.add_argument("--calibration", type=Path, default=OUT / "calibration.json")
    common.add_argument("--out", type=Path, default=OUT)
    common.add_argument("--tip", default="pen tip", help="SAM prompt for the pointer tip")
    common.add_argument("--tip-px", type=float, nargs=2, metavar=("X", "Y"), help="ground the tip by hand")
    common.add_argument("--tip-radius", type=float, default=8.0)
    common.add_argument("--laser", action="store_true", help="the pointer is a laser dot, tracked by brightness")
    common.add_argument("--laser-min-value", type=float, default=200.0, help="dimmest dot pixel, 0-255")
    common.add_argument("--no-sam", action="store_true")
    common.add_argument("--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("markers", parents=[common], help="printable ArUco marker sheet")
    m.add_argument("--board-mm", type=float, nargs=2, default=(400.0, 300.0), metavar=("W", "H"))
    m.add_argument("--marker-mm", type=float, default=30.0)
    m.add_argument("--dpi", type=int, default=300)
    m.set_defaults(func=cmd_markers)

    c = sub.add_parser("calibrate", parents=[common], help="homography from the four markers")
    c.add_argument("--board-mm", type=float, nargs=2, default=(400.0, 300.0), metavar=("W", "H"))
    c.add_argument("--marker-mm", type=float, default=30.0)
    c.add_argument("--dots", action="store_true", help="coloured corner dots instead of ArUco")
    c.set_defaults(func=cmd_calibrate)

    b = sub.add_parser("board", parents=[common], help="board JSON from words via SAM")
    b.add_argument("--target", required=True)
    b.add_argument("--hazard", action="append", default=[])
    b.add_argument("--min-confidence", type=float, default=0.3)
    b.add_argument("--board-out", type=Path, default=OUT / "board.json")
    b.set_defaults(func=cmd_board)

    r = sub.add_parser("repeatability", parents=[common], help="calibration mode: jog grid repeatability")
    r.add_argument("--grid", type=int, default=3)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--still-frames", type=int, default=60)
    r.set_defaults(func=cmd_repeatability)

    la = sub.add_parser("latency", parents=[common], help="calibration mode: serial and command-to-motion latency")
    la.add_argument("--pings", type=int, default=50)
    la.add_argument("--moves", type=int, default=10)
    la.set_defaults(func=cmd_latency)

    d = sub.add_parser("dmax", parents=[common], help="SPEC 3.4 d_max measurement")
    d.add_argument("--label", default="draw")
    d.add_argument("--hold", type=float, default=2.0)
    d.add_argument("--rounds", type=int, default=2)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--limit", type=int, help="only the first N actions (rehearsal)")
    d.add_argument("--reset-every", type=int, help="reboot to the start every N actions")
    d.add_argument("--sim", action="store_true", help="expected values on the exported configs, no rig")
    d.add_argument("--combine", nargs="+", help="dmax_*.json files from several draws")
    d.add_argument("--configs", type=Path, default=Path("bench/configs_export.json"),
                   help="--sim: the configs flashed into the firmware (scripts/export_configs.py)")
    d.add_argument("--sigma", type=float, default=0.002, help="measured position sigma (repeatability)")
    d.add_argument("--rel-margin", type=float, default=0.25)
    d.add_argument("--box", type=float, default=0.20, help="FW_TRAVEL_BOX in the flashed firmware")
    d.add_argument("--slope-max", type=float, default=P.SIM_SLOPE_MAX, help="from `repeatability`")
    d.set_defaults(func=cmd_dmax)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
