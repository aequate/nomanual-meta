"""Manual console for the rig: jog, run, stop, status, by hand over the serial protocol.

    python scripts/rig_console.py --port COM5            # the real ESP32
    python scripts/rig_console.py --fake --draw 3        # the firmware controller on this laptop
    python scripts/rig_console.py --fake --draw 0        # fake board in calibration mode

Commands (one per line; `help` lists them):
    r A B C D SECONDS   run an action, e.g. `r 1 0 1 0 0.5` (heartbeat sent automatically)
    x                   stop now (neutral)
    h                   hello (prints the banner)
    p                   ping (round trip time)
    s                   status u v mode led (calibration mode only: draw 0)
    j U V               jog the servos to (U, V) in calibration mode, waits until there
    truth               the fake device's hidden state (--fake only)
    q                   quit (sends X first)

Nothing here moves anything the firmware would not also accept from the agent: the firmware's
soft limits, rate cap, travel box and watchdog apply to every command.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomanual.core.env import CHANNELS, Action  # noqa: E402
from nomanual.rig.serial_client import RigError, SerialClient  # noqa: E402

HELP = __doc__.split("Commands (one per line; `help` lists them):\n", 1)[1].split("\n\n", 1)[0]


def execute(client: SerialClient, line: str, device=None) -> str | None:
    """Run one console command; returns the text to print, or None to quit."""
    parts = shlex.split(line.strip())
    if not parts:
        return ""
    cmd, args = parts[0].lower(), parts[1:]
    try:
        if cmd in ("q", "quit", "exit"):
            client.stop(wait=False)
            return None
        if cmd in ("help", "?"):
            return HELP
        if cmd == "r":
            if len(args) != 5:
                return "usage: r A B C D SECONDS"
            vals = [float(a) for a in args]
            action = Action(channels=dict(zip(CHANNELS, vals[:4], strict=True)), duration=vals[4])
            res = client.run(action)
            return f"{res.sent} -> {res.device_status} ({res.hold_ticks} hold ticks, {res.elapsed:.3f} s)"
        if cmd == "x":
            client.stop()
            return "X"
        if cmd == "h":
            b = client.hello()
            return f"NM2 {b.fw} draw={b.draw} mode={b.mode}"
        if cmd == "p":
            return f"round trip {client.ping() * 1000:.1f} ms"
        if cmd == "s":
            u, v, mode, led = client.status()
            return f"u={u:.4f} v={v:.4f} mode={mode} led={led}"
        if cmd == "j":
            if len(args) != 2:
                return "usage: j U V"
            u, v = client.jog(float(args[0]), float(args[1]))
            return f"at u={u:.4f} v={v:.4f}"
        if cmd == "truth":
            if device is None:
                return "truth is only available with --fake"
            st = device.query()
            return (f"out=({st.out_u:.4f}, {st.out_v:.4f}) engine=({st.eng_u:.4f}, {st.eng_v:.4f}) "
                    f"mode={st.mode} led={st.led} state={st.state}")
        return f"unknown command {cmd!r}; type help"
    except (ValueError, RigError) as exc:
        return f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyUSB0")
    src.add_argument("--fake", action="store_true", help="run the firmware controller on this laptop")
    ap.add_argument("--draw", type=int, default=1, help="--fake: DIP draw (0 = calibration mode)")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args(argv)

    device = None
    if args.fake:
        from nomanual.rig.fake_device import FakeSerialDevice

        device = FakeSerialDevice(draw=args.draw)
        opener = device.open
    else:
        from nomanual.rig.serial_client import PySerialTransport

        def opener():
            return PySerialTransport(args.port, args.baud)

    client = SerialClient(opener, log=lambda m: print(f"[client] {m}"))
    try:
        b = client.connect()
        print(f"connected: NM2 {b.fw} draw={b.draw} mode={b.mode}; type help")
        while True:
            try:
                line = input("rig> ")
            except EOFError:
                line = "q"
            out = execute(client, line, device)
            if out is None:
                break
            if out:
                print(out)
    except KeyboardInterrupt:
        client.stop(wait=False)
    finally:
        client.close()
        if device is not None:
            device.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
