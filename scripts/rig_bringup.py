"""First power-on of the rig, step by step (docs/HARDWARE.md "Bring-up checklist"), with a report.

    python scripts/rig_bringup.py calib --port COM5          # DIP all off (draw 0), servo power on
    python scripts/rig_bringup.py run --port COM5 --draw 1   # then DIP 1, reset
    python scripts/rig_bringup.py calib --fake --yes         # rehearse on the firmware controller

Each step says what it does, runs it, and checks what the firmware can show by itself (banner,
ping, refusals, the watchdog trip, statuses). Steps that only a person can judge (did both servos
move, which way is +u, no buzzing at a corner) ask y/n; `--yes` answers yes and the report marks
those answers "not checked by a person". The report goes to `rig/bringup_<phase>.json`.

Safety does not depend on this script or on any model: everything it sends goes through the
firmware's soft limits, rate cap, travel box and 300 ms watchdog. Keep the servo cut-off in reach.
The run phase only sends neutral actions (all channels 0), so nothing should move; the hidden draw
stays untouched. The cut-off and the Demo C signal swap are printed as manual procedures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomanual.core.env import Action  # noqa: E402
from nomanual.rig.serial_client import RigError, RigRefused, SerialClient  # noqa: E402

JOG_RATE = 0.25
"""Firmware jog/boot glide rate, servo units per second (guard.h)."""
WATCHDOG_S = 0.3

NEUTRAL = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

CUTOFF = """Manual check (cut-off): in run mode start `r 0 0 0 0 2` in rig_console.py, cut the servo
supply mid-action: the ESP32 keeps answering (`p`). Restore power with a hand on the cut-off."""
SWAP = """Manual procedure (Demo C signal swap): finish a task, cut the servo supply, swap the two servo
signal leads (GPIO 18 <-> 19), restore power with a hand on the cut-off. Each servo then jumps to
the other's pulse (|u - v| on both). Do not tell the agent."""


class Bringup:
    def __init__(self, client: SerialClient, yes: bool, device: Any = None) -> None:
        self.client = client
        self.yes = yes
        self.device = device
        self.steps: list[dict[str, Any]] = []

    def step(self, name: str, ok: bool, **info: Any) -> bool:
        self.steps.append({"step": name, "ok": bool(ok), **info})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {info}" if info else ""), flush=True)
        return ok

    def ask(self, name: str, question: str) -> bool:
        if self.yes:
            return self.step(name, True, person="not checked by a person (--yes)")
        ans = input(f"  ? {question} [y/n] ").strip().lower()
        return self.step(name, ans.startswith("y"), person=ans)

    def raw(self, line: str) -> str:
        """One protocol line, waiting for its reply (diagnostics; bypasses the client's checks
        but not the firmware's)."""
        return self.client._request(line, lambda t: t == line.split()[0] or t.startswith("E "))

    # --- phases ------------------------------------------------------------------------------

    def calib(self) -> None:
        c = self.client
        b = c.banner
        self.step("banner says calibration mode", b is not None and b.mode == "calib" and b.draw == 0,
                  banner=None if b is None else f"NM2 {b.fw} draw={b.draw} mode={b.mode}")
        rtt = [c.ping() * 1000 for _ in range(20)]
        self.step("ping round trip", max(rtt) < 50, median_ms=round(statistics.median(rtt), 1), max_ms=round(max(rtt), 1))
        u, v, _, _ = c.status()
        self.step("status inside soft limits", 0.05 <= u <= 0.95 and 0.05 <= v <= 0.95, u=u, v=v)
        print("  jog to the middle (0.5, 0.5) at 0.25 units/s ...", flush=True)
        u, v = c.jog(0.5, 0.5)
        self.step("jog reached the middle", abs(u - 0.5) < 1e-3 and abs(v - 0.5) < 1e-3, u=u, v=v)
        self.ask("both servos moved slowly to the middle", "Did both servos move, slowly and smoothly?")
        dirs: dict[str, str] = {}
        for axis, target in (("u", (0.6, 0.5)), ("v", (0.5, 0.6))):
            c.jog(*target)
            if self.yes:
                dirs[axis] = "unknown (--yes)"
            else:
                dirs[axis] = input(f"  ? Which way did the tip move on the board for +{axis}? [left/right/up/down] ").strip()
            c.jog(0.5, 0.5)
        self.step("directions noted", True, plus_u=dirs["u"], plus_v=dirs["v"])
        # 0.88, not 0.95: the rig's firmware stops there (firmware/src/soft_top.h).
        for corner in ((0.05, 0.05), (0.88, 0.05), (0.88, 0.88), (0.05, 0.88)):
            print(f"  corner {corner} ...", flush=True)
            c.jog(*corner)
        c.jog(0.5, 0.5)
        self.ask("corners: no buzzing, tip on the board and in view",
                 "At all four corners: no buzzing against an end stop, tip stayed on the board and in the camera view?")
        reply = self.raw("J 0.9900 0.5000")
        self.step("firmware refuses a jog past the soft limit", reply == "E range", reply=reply)
        reply = self.raw("R 0 0 0 0 200 1")
        self.step("firmware refuses actions in calibration mode", reply == "E denied", reply=reply)
        # Watchdog: start a long jog, then send nothing. The firmware must stop it ~0.3 s in.
        u0, _, _, _ = c.status()
        self.raw("J 0.8800 0.5000")
        t0 = time.perf_counter()
        time.sleep(1.2)
        u1, v1, _, _ = c.status()
        time.sleep(0.5)
        u2, _, _, _ = c.status()
        quiet = time.perf_counter() - t0
        expected = u0 + JOG_RATE * WATCHDOG_S
        self.step("watchdog stops a jog when the laptop goes quiet",
                  u1 < u0 + JOG_RATE * 0.6 and abs(u2 - u1) < 1e-3 and u1 < 0.88,
                  start_u=round(u0, 4), stopped_at_u=round(u1, 4), still_at_u=round(u2, 4),
                  expected_about=round(expected, 4), unattended_jog_would_reach=0.88,
                  note=f"no line for {quiet:.1f} s except the two S reads")
        c.jog(0.5, 0.5)
        print(CUTOFF)

    def run(self, draw: int) -> None:
        c = self.client
        b = c.banner
        self.step("banner says run mode, right draw", b is not None and b.mode == "run" and b.draw == draw,
                  banner=None if b is None else f"NM2 {b.fw} draw={b.draw} mode={b.mode}")
        self.ask("tip glided to the start mark", "Did the tip glide slowly to the start mark after the reset?")
        for line, name in (("S", "status"), ("J 0.5000 0.5000", "jog")):
            reply = self.raw(line)
            self.step(f"firmware refuses {name} in run mode", reply == "E denied", reply=reply)
        res = c.run(Action(channels=NEUTRAL, duration=0.2))
        self.step("neutral action 0.2 s", res.device_status == "done", status=res.device_status,
                  hold_ticks=res.hold_ticks, elapsed_s=round(res.elapsed, 3))
        res = c.run(Action(channels=NEUTRAL, duration=2.0))
        self.step("2 s neutral action with heartbeat is not cut", res.device_status == "done",
                  status=res.device_status, elapsed_s=round(res.elapsed, 3))
        hb = c.heartbeat_s
        c.heartbeat_s = None
        try:
            res = c.run(Action(channels=NEUTRAL, duration=1.0))
        finally:
            c.heartbeat_s = hb
        self.step("watchdog cuts an action without heartbeat", res.device_status == "watchdog",
                  status=res.device_status, elapsed_s=round(res.elapsed, 3))
        out: dict[str, Any] = {}
        th = threading.Thread(target=lambda: out.update(r=c.run(Action(channels=NEUTRAL, duration=2.0))))
        th.start()
        time.sleep(0.4)
        c.stop(wait=False)
        th.join(timeout=5)
        r = out.get("r")
        self.step("X stops a running action", r is not None and r.device_status == "stopped",
                  status=None if r is None else r.device_status, elapsed_s=None if r is None else round(r.elapsed, 3))
        self.ask("nothing moved during the neutral actions", "Did the tip stay still through all of the above?")
        print(CUTOFF)
        print(SWAP)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["calib", "run"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port")
    src.add_argument("--fake", action="store_true", help="the firmware controller on this laptop")
    ap.add_argument("--draw", type=int, default=1, help="run phase: the DIP draw that is set")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--yes", action="store_true", help="answer every person check with yes (rehearsal)")
    ap.add_argument("--out", type=Path, default=Path("rig"))
    args = ap.parse_args(argv)

    device = None
    if args.fake:
        from nomanual.rig.fake_device import FakeSerialDevice

        device = FakeSerialDevice(draw=0 if args.phase == "calib" else args.draw)
        opener = device.open
    else:
        from nomanual.rig.serial_client import PySerialTransport

        def opener():
            return PySerialTransport(args.port, args.baud)

    client = SerialClient(opener)
    bu = Bringup(client, args.yes, device)
    print(f"bring-up, {args.phase} phase ({'fake device' if args.fake else args.port})", flush=True)
    try:
        client.connect()
        if args.phase == "calib":
            bu.calib()
        else:
            bu.run(args.draw)
    except (RigError, RigRefused, OSError) as exc:
        bu.step("link", False, error=f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        client.stop(wait=False)
        bu.step("interrupted", False)
    finally:
        client.close()
        if device is not None:
            device.close()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bringup_{args.phase}{'_fake' if args.fake else ''}.json"
    ok = all(s["ok"] for s in bu.steps)
    path.write_text(json.dumps({"phase": args.phase, "fake": args.fake, "port": args.port, "ok": ok,
                                "t": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": bu.steps}, indent=2), encoding="utf-8")
    print(f"{'ALL PASS' if ok else 'SOME STEPS FAILED'}; report {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
