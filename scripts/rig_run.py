"""Run the agent on the rig: the same `Agent` as in the simulator, with `RigEnv` in place of
`SimEnv` (SPEC §3.1). `--fake` rehearses the whole run on this laptop (the firmware controller
compiled for the host + a synthetic camera), with no hardware.

    python scripts/rig_run.py --fake --draw 1 --budget 30                     # rehearsal, no model
    python scripts/rig_run.py --port COM5 --camera 0 --board rig/board.json   # the real rig, enumeration arm
    python scripts/rig_run.py --port COM5 --camera 0 --arm model --provider baseten --d-max 0.20

The agent never sees the draw: `--draw` on the real rig is only written to the event log as the
hidden config id (set it to the DIP draw you chose, for scoring afterwards). Events, frames and
the manual go to `--out` (default runs/rig_<timestamp>). Ctrl-C sends `X` to the firmware before
exiting; the servo power cut-off stays the first stop (docs/HARDWARE.md). Read
docs/RIG_RUNBOOK.md first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomanual.core.agent import TASKS, Agent, AgentConfig  # noqa: E402
from nomanual.core.board import Board  # noqa: E402
from nomanual.core.events import EventWriter  # noqa: E402
from nomanual.core.shield import ShieldConfig  # noqa: E402

EXPORT = Path("bench/configs_export.json")


def _log(args):
    return (lambda m: print(f"[rig] {m}")) if args.verbose else None


def build_env(args, run_dir: Path):
    """(env, board, hidden config id, closer)."""
    if args.fake:
        from nomanual.rig.fake_rig import build_fake_rig

        export = json.loads(args.configs.read_text())
        draws = {d["draw"]: d for d in export["draws"]}
        draw = args.draw or 1
        if draw not in draws:
            raise SystemExit(f"--draw {draw}: the export has draws {sorted(draws)}")
        board = Board.model_validate(export["board"]) if args.board is None else _board(args.board)
        rig = build_fake_rig(None, draw=draw, time_scale=args.time_scale,
                             heartbeat_s=0.1 / max(1.0, args.time_scale / 2), run_dir=run_dir, log=_log(args))
        return rig.env, board, draws[draw]["config"]["id"], rig.close

    if not args.port or args.camera is None:
        raise SystemExit("need --port and --camera (or --fake)")
    if args.board is None:
        raise SystemExit("need --board (from `rig_setup.py board`, checked by hand)")
    from nomanual.rig.calibrate import Calibration
    from nomanual.rig.camera import Camera, OpenCVSource
    from nomanual.rig.env import RigEnv
    from nomanual.rig.serial_client import PySerialTransport, SerialClient
    from nomanual.rig.tracker import SpotModel, TipTracker

    board = _board(args.board)
    cal = Calibration.load(args.calibration)
    camera = Camera(OpenCVSource(args.camera, args.width, args.height, args.fps)).start()
    try:
        video = {}
        if args.record:
            camera.wait_for_frame(timeout=5.0)
            time.sleep(1.5)  # enough frames to measure the real frame rate; the webcam runs below nominal
            video = {"file": "camera.mp4", "file_fps": round(camera.measured_fps() or args.fps, 3)}
            camera.start_recording(run_dir / "camera.mp4", fps=video["file_fps"])
            video["started_wall"] = time.time()
        client = SerialClient(lambda: PySerialTransport(args.port), log=_log(args))
        tracker = TipTracker(cal, base_sigma_px=args.base_sigma_px,
                             colour=SpotModel(min_value=args.laser_min_value) if args.laser else None)
        if args.tip_px:
            frame = camera.wait_for_frame(timeout=5.0)
            tracker.ground(frame.image, px=tuple(args.tip_px), radius=args.tip_radius)
        sam = None
        if not args.no_sam:
            from nomanual.rig.sam import SamClient, SamConfigError

            try:
                sam = SamClient()
            except SamConfigError:
                sam = None
        if sam is None and not args.tip_px and not args.laser:
            raise SystemExit("no META_API_KEY for SAM: ground the tip by hand with --tip-px X Y")
        kw = {} if args.frame_lag is None else {"frame_lag_s": args.frame_lag}
        env = RigEnv(client, camera, cal, tracker, sam=sam, run_dir=run_dir, tip_prompt=args.tip,
                     log=_log(args), **kw)
    except BaseException:
        camera.stop()
        raise

    def close():
        if args.record:  # first, so a failure closing the port cannot lose the video
            recorded = camera.stop_recording()
            video.update(stopped_wall=time.time(), frames=recorded[1] if recorded else 0)
            (run_dir / "video_sync.json").write_text(json.dumps(video, indent=1))
        try:
            env.close()
        finally:
            camera.stop()

    hidden = f"draw{args.draw}" if args.draw is not None else ""
    return env, board, hidden, close


def _board(path: Path) -> Board:
    if not path.exists():
        raise SystemExit(f"{path} not found: run `rig_setup.py board` first")
    return Board.model_validate_json(path.read_text())


def _model(args):
    """(model, label, provider) or (None, 'no model', 'none') for the enumeration arm."""
    if args.arm == "enum":
        return None, "no model", "none"
    from nomanual.core.models import FakeModel, make_model, missing_env
    from nomanual.core.proposer import FAKE_LABEL, fake_responder

    if args.provider == "fake":
        return FakeModel(fake_responder(seed=args.seed)), FAKE_LABEL, "fake"
    missing = missing_env(args.provider)
    if missing:
        raise SystemExit(f"{args.provider} needs {', '.join(missing)} in the environment")
    m = make_model(args.provider)
    return m, f"{args.provider} {m.model}", args.provider


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="rehearse on the fake rig")
    ap.add_argument("--draw", type=int, help="--fake: draw to emulate (1..8); real rig: logged as the hidden id")
    ap.add_argument("--configs", type=Path, default=EXPORT, help="--fake: the exported draw table")
    ap.add_argument("--time-scale", type=float, default=4.0, help="--fake: firmware time per wall time")
    ap.add_argument("--port")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--calibration", type=Path, default=Path("rig/calibration.json"))
    ap.add_argument("--board", type=Path, help="board JSON (real rig: required; --fake: the export's board)")
    ap.add_argument("--tip", default="pen tip", help="SAM prompt for the pointer tip")
    ap.add_argument("--tip-px", type=float, nargs=2, metavar=("X", "Y"), help="ground the tip by hand")
    ap.add_argument("--tip-radius", type=float, default=8.0)
    ap.add_argument("--laser", action="store_true", help="the pointer is a laser dot, tracked by brightness")
    ap.add_argument("--laser-min-value", type=float, default=200.0, help="dimmest dot pixel, 0-255")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--record", action="store_true", help="save the camera to camera.mp4 in the run folder")
    ap.add_argument("--frame-lag", type=float, help="seconds; RigEnv.frame_lag_s (from `rig_setup.py latency`)")
    ap.add_argument("--base-sigma-px", type=float, default=0.5,
                    help="tracker position noise in pixels (from `rig_setup.py repeatability`)")
    ap.add_argument("--arm", choices=("enum", "model"), default="enum")
    ap.add_argument("--provider", help="--arm model: a provider from nomanual.core.models, or 'fake'")
    ap.add_argument("--seed", type=int, default=0, help="seed of the fake model")
    ap.add_argument("--budget", type=int, default=80, help="experiments per task")
    ap.add_argument("--tasks", nargs="+", choices=("reach", "park"), default=list(TASKS))
    ap.add_argument("--d-max", type=float, help="ShieldConfig.d_max from `rig_setup.py dmax --combine`")
    ap.add_argument("--no-verified-ground", action="store_true",
                    help="turn the shield's verified-ground rule off (the benchmark's shield_off arm); disclose it")
    ap.add_argument("--pause-between", action="store_true",
                    help="wait for Enter after each task but the last (Demo C: cut power, swap leads)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.arm == "model" and not args.provider:
        ap.error("--arm model needs --provider")

    run_dir = args.out or Path("runs") / time.strftime("rig_%Y%m%d_%H%M%S")
    model, label, provider = _model(args)
    env, board, hidden, close = build_env(args, run_dir)

    def between(i: int) -> None:
        if args.pause_between:
            input(f"task {i + 1} done. Cut servo power, change the rig, restore power, then press Enter: ")

    config = AgentConfig(hypotheses=args.arm, task_budget=args.budget,
                         shield=ShieldConfig(d_max=args.d_max, verified_ground=not args.no_verified_ground),
                         vision=model is not None)
    summary = None
    try:
        with EventWriter(run_dir) as writer:
            (run_dir / "events_started.txt").write_text(f"{time.time():.3f}\n")  # event t = seconds after this
            log = None
            proposer = None
            if model is not None:
                from nomanual.core.models import CallLog
                from nomanual.core.proposer import ModelProposer

                log = CallLog(run_dir / "model_calls.jsonl")
                proposer = ModelProposer(model, label, log)
            agent = Agent(env, board, writer, config, proposer=proposer, env_kind="rig",
                          config_id_hidden=hidden, model_name=getattr(model, "model", "none"),
                          provider=provider, frames_root=run_dir, prose_model=model, call_log=log,
                          between_tasks=between)
            summary = agent.run(tuple(args.tasks))
    except KeyboardInterrupt:
        print("interrupted: stopping the rig")
    finally:
        close()  # RigEnv.close sends X before closing the port

    if summary is None:
        return 130
    out = {
        "status": summary.status,
        "tasks": [{"name": t.name, "success": t.success, "experiments": t.experiments, "reason": t.reason}
                  for t in summary.tasks],
        "experiments": summary.experiments,
        "shield_interventions": summary.shield_interventions,
        "alarms": summary.alarms,
        "repairs_accepted": summary.repairs_accepted,
        "wall_s": round(summary.wall_s, 1),
        "laptop_stop": getattr(env, "stop_reason", None),
        "model": label,
    }
    (run_dir / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"events: {run_dir / 'events.jsonl'}")
    return 0 if summary.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
