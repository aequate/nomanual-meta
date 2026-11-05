"""A 3D replay of a logged run: one self-contained `replay.html`, built from the run's `events.jsonl` alone.

    .venv/bin/python scripts/replay3d.py runs/demo_A --out results/replay3d/demo_A
    .venv/bin/python scripts/replay3d.py runs/fresh-00 --out results/replay3d/fresh-00 --rewire-after-task 1
    (Windows: .venv/Scripts/python)

The page shows the rig of `docs/HARDWARE.md` in three.js (0.149.0, inlined with its MIT licence):
the 400 × 300 mm board with its corner markers, the hazards in red and the target in green as
logged in `run_start`, the pan-tilt bracket with two servos, a rod with a blue tip, and a front
panel with buttons A–D that light while an action holds them. The amber LED lights only when the
run logs a light reading.

- **The tip** is drawn at the logged before/after positions of every experiment, with a fading
  trail. Between them the motion is interpolated, and the page says so. Servo angles and the rod
  are derived from the tip position; they are cosmetic.
- **The side panel** is all from the log: experiment, task, action (planned → run), the
  planner's reason, the shield's verdict, the current leader in plain text, the hypothesis pool
  size, the alarm banner, whether the repair was accepted, and after an alarm the summary of
  `change_report.md`. The report is built here from the same events, **without** its simulator
  ground-truth section: no hidden truth reaches the page.
- **The rewire** (the operator swaps the two servo leads) is not in the agent's log, which by
  design never sees it. With `--rewire-after-task N` the run protocol is stated (the leads were
  swapped after task N) and the two signal wires cross at that task boundary, with a note. Without
  it no crossing is drawn.
- **A permanent label** says "Replay of a logged simulator run, not hardware", with the provider
  and the run id.

Controls: play/pause, speed, a scrub bar, and jump to the next alarm. It works at phone width,
in light and dark. `window.replay.seek(t)` and `window.replay.duration` let a headless browser
capture frames (`?render=1` hides the controls).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomanual.core.env import Action  # noqa: E402

VENDOR = Path(__file__).resolve().parent / "vendor"
THREE = VENDOR / "three-0.149.0.min.js"
THREE_LICENSE = VENDOR / "three-0.149.0-LICENSE.txt"
THREE_VERSION = "0.149.0"


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    """Raw events; a truncated last line (a crashed run) is dropped."""
    out = []
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def _short(action: dict[str, Any] | None) -> str:
    return Action.model_validate(action).short() if action else "–"


def _leader(items: list[dict[str, Any]]) -> str:
    for it in items:
        if it.get("state") != "contradicted":
            return it["text"]
    return "none survives"


def timeline(events: list[dict[str, Any]]) -> dict[str, Any]:
    """The run as the page needs it: one step per experiment, each with the state the log shows
    right after that experiment."""
    if not events or events[0]["kind"] != "run_start":
        raise ValueError("events.jsonl must start with run_start")
    start = events[0]["data"]
    out: dict[str, Any] = {
        "run": events[0].get("run", ""), "provider": start.get("provider", "?"), "model": start.get("model", "?"),
        "env": start.get("env", "?"), "board": start["board"], "steps": [], "tasks": [], "light": None,
        "proposals": [],
    }
    task: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    shield: dict[str, Any] | None = None
    action_seq: int | None = None
    leader, pool = "no hypotheses yet", 0
    first_pos: list[float] | None = None
    for e in events[1:]:
        k, d = e["kind"], e["data"]
        steps = out["steps"]
        if k == "observation" and first_pos is None:
            first_pos = d["pos"]
        elif k == "task":
            if d["status"] == "start":
                task = {"name": d["name"], "index": len(out["tasks"]), "start": d["experiments"], "end": None,
                        "status": "running"}
                out["tasks"].append(task)
            elif task is not None:
                task["end"], task["status"] = d["experiments"], d["status"]
        elif k == "plan":
            plan = d
        elif k == "shield":
            shield = d
        elif k == "action":
            action_seq = d["seq"]
        elif k == "result" and action_seq is not None:
            chosen = plan["candidates"][plan["chosen"]]["action"] if plan and plan.get("candidates") else None
            light = None
            if out["light"] and out["light"] in d["after"].get("extras", {}):
                light = d["after"]["extras"][out["light"]]
            steps.append({
                "n": action_seq, "task": task["name"] if task else "", "task_index": task["index"] if task else 0,
                "action": d["action"], "run": _short(d["action"]), "planned": _short(chosen),
                "why": plan["why"] if plan else "", "shield": (shield or {}).get("verdict", "–"),
                "shield_reason": (shield or {}).get("reason", ""),
                "before": d["before"]["pos"], "after": d["after"]["pos"], "status": d["status"], "light": light,
                "leader": leader, "pool": pool, "alarm": None, "repair": None, "notes": [],
            })
            plan = shield = None
            action_seq = None
        elif k == "hypotheses":
            leader, pool = _leader(d["items"]), int(d.get("total") or len(d["items"]))
            if steps:
                steps[-1]["leader"], steps[-1]["pool"] = leader, pool
        elif k == "alarm" and steps:
            steps[-1]["alarm"] = {"cause": d.get("cause", "drift"), "stat": d["stat"], "threshold": d["threshold"]}
        elif k == "repair" and steps:
            steps[-1]["repair"] = {"accepted": bool(d["accepted"]), "edits": list(d["edits"])}
        elif k == "observable_proposed" and steps:
            steps[-1]["notes"].append(f"Muse proposed a new observable from raw frames: \"{d['description']}\"")
        elif k == "observable_tracked":
            if d["ok"]:
                out["light"] = d["name"]
            if steps:
                steps[-1]["notes"].append(f"tracking {d['name']}: {'ok' if d['ok'] else 'failed'}")
        elif k == "proposal" and steps and d.get("source") == "muse":
            steps[-1]["notes"].append(f"new Muse proposal ({len(d.get('hypotheses', []))} hypotheses shown)")
    out["start"] = first_pos or (out["steps"][0]["before"] if out["steps"] else out["board"]["start"])
    return out


def change_summary(run_dir: Path) -> dict[int, list[str]]:
    """Experiment → the summary lines of `change_report.md` for the alarm after it, built from the
    events with no ground truth: the alarm, the fit-before verdict (second, so it stays in view on
    a short screen), before and after, then the controls table."""
    from nomanual.core.change_report import change_report

    text = change_report(run_dir, None)
    out: dict[int, list[str]] = {}
    for section in text.split("## Change after experiment ")[1:]:
        n = int(section.split("\n", 1)[0].strip())
        alarm, verdict, rest = [], [], []
        for line in section.splitlines()[1:]:
            line = line.strip()
            if line.startswith("- Alarm"):
                alarm.append(line[2:])
            elif line.startswith("- Before") or line.startswith("- After"):
                rest.append(line[2:])
            elif re.match(r"^\| [A-D] \|", line):
                cells = [c.strip() for c in line.strip("|").split("|")]
                rest.append(f"{cells[0]}: {cells[1]} → {cells[2]}")
            elif line.startswith("**Fit-before test**") or line.startswith("**After the alarm:**"):
                verdict.append(line.rstrip(":") + ("." if line.endswith(":") else ""))
        out[n] = [line.replace("**", "") for line in alarm + verdict + rest]
    return out


def build(run_dir: Path, rewire_after_task: int | None = None) -> dict[str, Any]:
    data = timeline(read_events(run_dir))
    data["changes"] = {str(k): v for k, v in change_summary(run_dir).items()}
    data["rewire_after_task"] = rewire_after_task
    data["three_version"] = THREE_VERSION
    return data


def render_html(data: dict[str, Any]) -> str:
    three = THREE.read_text(encoding="utf-8")
    licence = THREE_LICENSE.read_text(encoding="utf-8")
    page = TEMPLATE.replace("__THREE_LICENSE__", licence.replace("*/", "* /"))
    page = page.replace("__THREE__", three)
    return page.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rewire-after-task", type=int, default=None, metavar="N",
                    help="the run protocol: the operator swapped the servo leads after task N (1-based)")
    args = ap.parse_args(argv)
    data = build(args.run_dir, args.rewire_after_task)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "replay.html"
    path.write_text(render_html(data), encoding="utf-8")
    print(f"wrote {path} ({len(data['steps'])} experiments, {path.stat().st_size // 1024} KB)")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>NoManual replay</title>
<style>
:root { --bg:#f6f5f2; --fg:#1d1d1f; --muted:#5d5d63; --panel:#ffffff; --line:#d9d7d2; --accent:#2f6fdf;
        --alarm:#b3261e; --ok:#1f7a3d; --scene:#eceae5; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141416; --fg:#ececf0; --muted:#a1a1aa; --panel:#1e1e22; --line:#33333a; --accent:#7aa7ff;
          --alarm:#ff6b5e; --ok:#5fd38a; --scene:#202024; }
}
:root.light { --bg:#f6f5f2; --fg:#1d1d1f; --muted:#5d5d63; --panel:#ffffff; --line:#d9d7d2; --accent:#2f6fdf;
              --alarm:#b3261e; --ok:#1f7a3d; --scene:#eceae5; }
:root.dark { --bg:#141416; --fg:#ececf0; --muted:#a1a1aa; --panel:#1e1e22; --line:#33333a; --accent:#7aa7ff;
             --alarm:#ff6b5e; --ok:#5fd38a; --scene:#202024; }
* { box-sizing: border-box; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--fg);
             font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
#app { display:flex; flex-direction:column; height:100%; }
#label { padding:6px 12px; font-weight:600; border-bottom:1px solid var(--line); display:flex; gap:12px;
         flex-wrap:wrap; align-items:center; }
#label .tag { background:var(--alarm); color:#fff; border-radius:4px; padding:1px 8px; }
#label .meta { font-weight:400; color:var(--muted); }
#main { flex:1; display:flex; min-height:0; }
#view { position:relative; flex:1 1 60%; min-width:0; background:var(--scene); }
#view canvas { display:block; width:100%; height:100%; }
#hud { position:absolute; left:10px; bottom:10px; font-size:12px; color:var(--muted);
       background:color-mix(in srgb, var(--panel) 80%, transparent); padding:4px 8px; border-radius:6px; max-width:90%; }
#side { flex:0 0 360px; overflow:auto; padding:12px; border-left:1px solid var(--line); background:var(--panel); }
#side h2 { font-size:13px; margin:12px 0 4px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
#side .big { font-size:22px; font-weight:700; white-space:nowrap; }
#side .row { display:flex; justify-content:space-between; gap:8px; }
#side .v { text-align:right; }
#alarm { display:none; background:var(--alarm); color:#fff; padding:8px 10px; border-radius:6px; margin:8px 0; font-weight:600; }
#repair { margin:4px 0; font-weight:600; }
#change { display:none; border:1px solid var(--line); border-radius:6px; padding:8px; margin-top:6px; font-size:12px; }
#change li { margin:2px 0; }
#notes { font-size:12px; color:var(--muted); }
#controls { display:flex; gap:8px; align-items:center; padding:8px 12px; border-top:1px solid var(--line); flex-wrap:wrap; }
#controls button, #controls select { background:var(--panel); color:var(--fg); border:1px solid var(--line);
       border-radius:6px; padding:6px 10px; font:inherit; min-height:36px; }
#scrub { flex:1 1 200px; accent-color:var(--accent); min-height:36px; }
.render #controls { display:none; }
@media (max-width: 760px) {
  #main { flex-direction:column; }
  #view { flex:0 0 52vh; }
  #side { flex:1 1 auto; border-left:none; border-top:1px solid var(--line); }
  #hud { font-size:10px; padding:2px 6px; left:6px; bottom:6px; }
}
</style>
</head>
<body>
<div id="app">
  <div id="label"><span class="tag">Replay of a logged simulator run, not hardware</span>
    <span class="meta" id="meta"></span></div>
  <div id="main">
    <div id="view"><div id="hud"></div></div>
    <div id="side">
      <div class="row"><div><div class="big" id="exp">–</div><div id="task" class="meta"></div></div>
        <div class="v"><div id="pool" class="meta"></div><div id="status" class="meta"></div></div></div>
      <div id="alarm"></div><div id="repair"></div>
      <h2>Action</h2><div id="action"></div>
      <h2>Planner's reason</h2><div id="why"></div>
      <h2>Shield</h2><div id="shield"></div>
      <h2>Current leader</h2><div id="leader"></div>
      <div id="change"><b>Change report (written after the run from the whole log, so it counts later experiments too)</b><ul id="changelist"></ul></div>
      <h2>Log notes</h2><div id="notes"></div>
    </div>
  </div>
  <div id="controls">
    <button id="play" aria-label="play or pause">▶ Play</button>
    <select id="speed" aria-label="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option>
      <option value="2">2×</option><option value="4">4×</option></select>
    <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="scrub">
    <button id="nextalarm">Next alarm ⏭</button>
    <button id="theme" aria-label="toggle light or dark">◐</button>
  </div>
</div>
<script>
/*! three.js __THREE_VERSION__ — https://threejs.org — licence:
__THREE_LICENSE__ */
__THREE__
</script>
<script>
const DATA = __DATA__;
(function () {
const params = new URLSearchParams(location.search);
const RENDER = params.get("render") === "1";
if (RENDER) document.body.classList.add("render");
if (params.get("theme") === "dark") document.documentElement.classList.add("dark");
if (params.get("theme") === "light") document.documentElement.classList.add("light");
const $ = (id) => document.getElementById(id);
$("meta").textContent = `provider ${DATA.provider} · model ${DATA.model} · run ${DATA.run} · env ${DATA.env} · ` +
  `${DATA.steps.length} experiments · three.js ${DATA.three_version}`;

// --- geometry (mm): board 400 x 300, marker centres 20 mm in from the edges --------------------
const BW = 400, BH = 300, MX = 20, MY = 20, SW = BW - 2 * MX, SH = BH - 2 * MY;
const bx = (u) => MX + SW * u - BW / 2, by = (v) => MY + SH * v;   // board units -> scene mm
const PIVOT = new THREE.Vector3(0, 150, 300);                     // tilt axis, in front of the board

// --- timing: each experiment gets a slot; the move fills its first part ------------------------
const steps = DATA.steps;
const slots = []; let T = 0;
for (const s of steps) { const move = 0.25 + 0.25 * s.action.duration; const slot = move + 0.2;
  slots.push({ t0: T, move, slot }); T += slot; }
const DURATION = Math.max(T, 0.001);

// --- scene --------------------------------------------------------------------------------------
const view = $("view");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
view.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 1, 5000);
const target = new THREE.Vector3(30, 120, 150);
let orbit = { az: -0.5, el: 0.38, r: 820 };
function placeCamera() {
  camera.position.set(target.x + orbit.r * Math.sin(orbit.az) * Math.cos(orbit.el),
                      target.y + orbit.r * Math.sin(orbit.el),
                      target.z + orbit.r * Math.cos(orbit.az) * Math.cos(orbit.el));
  camera.lookAt(target);
}
placeCamera();
scene.add(new THREE.HemisphereLight(0xffffff, 0x666670, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 0.7); sun.position.set(-300, 600, 500); scene.add(sun);

const mat = (c, extra) => new THREE.MeshStandardMaterial(Object.assign({ color: c, roughness: 0.8 }, extra || {}));
const box = (w, h, d, m) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);

// desk and board
const desk = box(900, 10, 650, mat(0x9a8f80)); desk.position.set(0, -5, 200); scene.add(desk);
const board = box(BW, BH, 6, mat(0xf3f1ea)); board.position.set(0, BH / 2, -3); scene.add(board);
const boardZ = 0.5;
// markers: black squares with a white inner square, 30 mm, at the four corners
for (const [u, v] of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
  const m = box(30, 30, 1, mat(0x111111)); m.position.set(bx(u), by(v), boardZ); scene.add(m);
  const i = box(12, 12, 1.2, mat(0xffffff)); i.position.set(bx(u) + 4, by(v) - 3, boardZ + 0.3); scene.add(i);
}
function flat(shape, color, z) { const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape), mat(color));
  mesh.position.z = z; scene.add(mesh); return mesh; }
for (const hz of DATA.board.hazards) {
  const s = new THREE.Shape();
  const pts = hz.type === "rect" ? [[hz.x0, hz.y0], [hz.x1, hz.y0], [hz.x1, hz.y1], [hz.x0, hz.y1]] : hz.points;
  pts.forEach(([u, v], i) => i ? s.lineTo(bx(u), by(v)) : s.moveTo(bx(u), by(v)));
  flat(s, 0xd33a2c, boardZ + 0.6);
}
{ const t = DATA.board.target; const s = new THREE.Shape();
  s.absarc(bx(t.center[0]), by(t.center[1]), t.r * SW, 0, Math.PI * 2, false); flat(s, 0x2e9e4f, boardZ + 0.6); }
{ const st = DATA.board.start; const s = new THREE.Shape(); s.absarc(bx(st[0]), by(st[1]), 5, 0, Math.PI * 2, false);
  flat(s, 0x333333, boardZ + 0.7); }

// pan-tilt bracket: post, pan servo, U bracket, tilt servo, rod, blue tip
const post = box(30, PIVOT.y - 40, 30, mat(0x6b6b70)); post.position.set(0, (PIVOT.y - 40) / 2, PIVOT.z); scene.add(post);
const panServo = box(23, 23, 12, mat(0x2b4c9b)); panServo.position.set(0, PIVOT.y - 30, PIVOT.z); scene.add(panServo);
const panGroup = new THREE.Group(); panGroup.position.set(0, PIVOT.y - 18, PIVOT.z); scene.add(panGroup);
const yoke = box(40, 4, 26, mat(0x9aa0a8)); panGroup.add(yoke);
for (const sx of [-18, 18]) { const arm = box(4, 28, 26, mat(0x9aa0a8)); arm.position.set(sx, 14, 0); panGroup.add(arm); }
const tiltGroup = new THREE.Group(); tiltGroup.position.set(0, 18, 0); panGroup.add(tiltGroup);
const tiltServo = box(23, 12, 23, mat(0x2b4c9b)); tiltGroup.add(tiltServo);
const rod = new THREE.Mesh(new THREE.CylinderGeometry(2.5, 2.5, 1, 12), mat(0x3a3a3a));
rod.rotation.x = Math.PI / 2; tiltGroup.add(rod);
const tip = new THREE.Mesh(new THREE.SphereGeometry(6, 20, 14), mat(0x1f5fd6, { emissive: 0x0b2a66 }));
scene.add(tip);

// front panel with buttons A-D and the amber LED; an ESP32 box; wires
const panel = box(160, 22, 70, mat(0x2a2a2e)); panel.position.set(210, 11, 400); panel.rotation.x = -0.25; scene.add(panel);
const buttons = {}; const chans = ["A", "B", "C", "D"];
const btnColor = { A: 0xe8a33d, B: 0x4aa3df, C: 0x9b6dd6, D: 0x5bbf6a };
chans.forEach((c, i) => { const b = new THREE.Mesh(new THREE.CylinderGeometry(9, 9, 6, 20), mat(0x55555a));
  b.position.set(-54 + i * 32, 14, 6); panel.add(b); buttons[c] = b; });
const led = new THREE.Mesh(new THREE.SphereGeometry(5, 16, 12), mat(0x5a4310));
led.position.set(66, 14, -18); panel.add(led);
const esp = box(55, 8, 28, mat(0x1b5e3b)); esp.position.set(120, 4, 300); scene.add(esp);
function sprite(text, w, bgc) { w = w || 128; const c = document.createElement("canvas"); c.width = w; c.height = 64;
  const g = c.getContext("2d"); if (bgc) { g.fillStyle = bgc; g.fillRect(0, 0, w, 64); }
  g.fillStyle = "#ffffff"; g.font = "bold 40px system-ui, sans-serif";
  g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, w / 2, 32);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(22 * w / 128, 11, 1); return s; }
chans.forEach((c, i) => { const s = sprite(c); s.position.set(-54 + i * 32, 30, 22); panel.add(s); });
{ const s = sprite("LED"); s.position.set(66, 30, -6); panel.add(s); }
const world = (o) => o.getWorldPosition(new THREE.Vector3());
function curve(a, b, lift) { const m = a.clone().add(b).multiplyScalar(0.5); m.y += lift;
  return new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(a, m, b), 32, 1.6, 6, false); }
scene.updateMatrixWorld(true);
const espOut18 = new THREE.Vector3(105, 9, 290), espOut19 = new THREE.Vector3(115, 9, 290);
chans.forEach((c, i) => { const p = world(buttons[c]); p.y -= 10;
  scene.add(new THREE.Mesh(curve(p, new THREE.Vector3(130 + i * 5, 8, 312), 20), mat(btnColor[c]))); });
const panIn = new THREE.Vector3(10, PIVOT.y - 34, PIVOT.z - 8), tiltIn = new THREE.Vector3(12, PIVOT.y + 2, PIVOT.z - 8);
const wire18 = new THREE.Mesh(curve(espOut18, panIn, 60), mat(0xff6a00));
const wire19 = new THREE.Mesh(curve(espOut19, tiltIn, 90), mat(0xf2d230));
scene.add(wire18, wire19);
const l18 = sprite("18", 96, "#c05000"), l19 = sprite("19", 96, "#8a7a00");
l18.position.set(96, 26, 284); l19.position.set(124, 26, 284); scene.add(l18, l19);
const swapTag = sprite("leads swapped by the operator", 768, "#b3261e");
swapTag.position.set(60, PIVOT.y + 60, PIVOT.z); swapTag.visible = false; scene.add(swapTag);
let crossed = false;
function setCrossed(x) {
  if (x === crossed) return; crossed = x;
  wire18.geometry.dispose(); wire19.geometry.dispose();
  wire18.geometry = curve(espOut18, x ? tiltIn : panIn, x ? 110 : 60);
  wire19.geometry = curve(espOut19, x ? panIn : tiltIn, x ? 30 : 90);
  swapTag.visible = x;
}

// trail
const TRAIL = 160;
const trailGeo = new THREE.BufferGeometry();
const trailPos = new Float32Array(TRAIL * 3), trailCol = new Float32Array(TRAIL * 3);
trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
trailGeo.setAttribute("color", new THREE.BufferAttribute(trailCol, 3));
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ vertexColors: true }));
const trailDots = new THREE.Points(trailGeo, new THREE.PointsMaterial({ vertexColors: true, size: 5, sizeAttenuation: false }));
scene.add(trail, trailDots);

// --- state at time t ----------------------------------------------------------------------------
function stepAt(t) { let lo = 0, hi = slots.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (slots[mid].t0 <= t) lo = mid; else hi = mid - 1; }
  return lo; }
function posAt(t) {
  if (!steps.length) return DATA.start;
  const i = stepAt(t), s = steps[i], sl = slots[i];
  const f = Math.min(1, Math.max(0, (t - sl.t0) / sl.move));
  return [s.before[0] + (s.after[0] - s.before[0]) * f, s.before[1] + (s.after[1] - s.before[1]) * f, f < 1 ? i : -1];
}
function setTip(u, v) {
  const p = new THREE.Vector3(bx(u), by(v), 6); tip.position.copy(p);
  // cosmetic servo angles: pan about y, tilt about x, so the rod points from the pivot at the tip
  const d = p.clone().sub(new THREE.Vector3(0, PIVOT.y, PIVOT.z));
  panGroup.rotation.y = Math.atan2(-d.x, -d.z);
  tiltGroup.rotation.x = Math.atan2(d.y, Math.hypot(d.x, d.z));
  const len = d.length() - 6; rod.scale.y = len; rod.position.set(0, 0, -len / 2);
}
const bg = () => new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue("--scene").trim() || "#eceae5");

function update(t) {
  const i = steps.length ? stepAt(t) : 0, s = steps[i];
  const [u, v, moving] = posAt(t);
  setTip(u, v);
  // trail: sample the path backwards in time
  const back = bg(), blue = new THREE.Color(0x1f5fd6);
  for (let k = 0; k < TRAIL; k++) {
    const tk = Math.max(0, t - k * 0.06); const [pu, pv] = posAt(tk);
    trailPos.set([bx(pu), by(pv), 3], k * 3);
    const c = blue.clone().lerp(back, k / TRAIL); trailCol.set([c.r, c.g, c.b], k * 3);
  }
  trailGeo.attributes.position.needsUpdate = true; trailGeo.attributes.color.needsUpdate = true;
  // buttons light while the action holds them
  for (const c of chans) {
    const held = s && moving === i && Math.abs(s.action.channels[c] || 0) >= 0.05;
    buttons[c].material.color.setHex(held ? btnColor[c] : 0x55555a);
    buttons[c].material.emissive = new THREE.Color(held ? btnColor[c] : 0x000000);
    buttons[c].material.emissiveIntensity = held ? 0.6 : 0;
  }
  // the LED: only what the log reads
  let light = null; for (let k = i; k >= 0 && steps.length; k--) if (steps[k].light !== null) { light = steps[k].light; break; }
  led.material.color.setHex(light === null ? 0x5a4310 : (light >= 0.5 ? 0xffb000 : 0x6b5520));
  led.material.emissive = new THREE.Color(light !== null && light >= 0.5 ? 0xffa000 : 0x000000);
  led.material.emissiveIntensity = light !== null && light >= 0.5 ? 1.2 : 0;
  // the rewire, as stated by the run protocol
  const rw = DATA.rewire_after_task;
  setCrossed(rw !== null && s !== undefined && s.task_index >= rw);
  renderPanel(i, t, light, moving === i);
  scene.background = back;
  renderer.render(scene, camera);
}

function esc(x) { return String(x).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
let lastPanel = -2;
function renderPanel(i, t, light, moving) {
  const s = steps[i];
  $("hud").textContent = "Tip: logged before/after positions; motion between them is interpolated. " +
    "Servo angles are cosmetic." + (DATA.rewire_after_task !== null
      ? ` Run protocol: the operator swapped the servo leads after task ${DATA.rewire_after_task} (not in the agent's log).` : "");
  if (!s || i === lastPanel) return; lastPanel = i;
  $("exp").textContent = `Experiment ${s.n}`;
  const tk = DATA.tasks[s.task_index];
  // a task's outcome shows only once its last experiment has run, never ahead of time
  const done = tk && tk.end !== null && s.n >= tk.end - 1;
  $("task").textContent = `Task ${s.task_index + 1}: ${s.task}` + (tk ? ` (${done ? tk.status : "running"})` : "");
  $("pool").textContent = `${s.pool} hypotheses in the pool`;
  $("status").textContent = `result: ${s.status}` + (light === null ? "" : ` · light ${light >= 0.5 ? "on" : "off"}`);
  $("action").textContent = s.planned === s.run ? s.run : `${s.planned} → ${s.run} (shortened)`;
  $("why").textContent = s.why || "–";
  $("shield").textContent = `${s.shield}: ${s.shield_reason}`;
  $("leader").textContent = s.leader;
  // alarm and repair: from this experiment on, until the next task
  let alarmAt = -1; for (let k = i; k >= 0; k--) { if (steps[k].task_index !== s.task_index && alarmAt < 0) break;
    if (steps[k].alarm) { alarmAt = k; break; } }
  const a = alarmAt >= 0 ? steps[alarmAt] : null;
  $("alarm").style.display = a ? "block" : "none";
  if (a) $("alarm").textContent = `ALARM after experiment ${a.n}: ${a.alarm.cause}, statistic ${a.alarm.stat.toFixed(1)} ` +
    `against ${a.alarm.threshold.toFixed(1)}`;
  $("repair").textContent = a && a.repair ? (a.repair.accepted ? "Repair accepted: " : "Repair not accepted: ") +
    a.repair.edits.join("; ") : "";
  $("repair").style.color = a && a.repair ? (a.repair.accepted ? "var(--ok)" : "var(--alarm)") : "";
  const ch = a ? DATA.changes[String(a.n)] : null;
  $("change").style.display = ch ? "block" : "none";
  if (ch) $("changelist").innerHTML = ch.map((l) => `<li>${esc(l)}</li>`).join("");
  $("notes").innerHTML = steps.slice(Math.max(0, i - 4), i + 1).flatMap((x) => x.notes.map((n) => `#${x.n}: ${esc(n)}`))
    .map((n) => `<div>${n}</div>`).join("") || "–";
}

// --- sizing, controls, animation ----------------------------------------------------------------
function resize() { const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / Math.max(h, 1);
  orbit.r = camera.aspect < 1 ? 980 : 820; placeCamera(); camera.updateProjectionMatrix(); update(now); }
let now = 0, playing = !RENDER && params.get("autoplay") !== "0", speed = 1, last = null;
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "⏸ Pause" : "▶ Play"; };
$("play").textContent = playing ? "⏸ Pause" : "▶ Play";
$("speed").onchange = (e) => { speed = parseFloat(e.target.value); };
$("scrub").oninput = (e) => { now = DURATION * e.target.value / 1000; update(now); };
$("nextalarm").onclick = () => { const i = steps.length ? stepAt(now) : 0;
  for (let k = i + 1; k < steps.length; k++) if (steps[k].alarm) { now = slots[k].t0 + slots[k].move + 0.01; update(now); return; }
  $("nextalarm").textContent = "No later alarm"; setTimeout(() => { $("nextalarm").textContent = "Next alarm ⏭"; }, 1500); };
$("theme").onclick = () => { const r = document.documentElement;
  const dark = r.classList.contains("dark") || (!r.classList.contains("light") && matchMedia("(prefers-color-scheme: dark)").matches);
  r.classList.toggle("dark", !dark); r.classList.toggle("light", dark); lastPanel = -2; update(now); };
let drag = null;
renderer.domElement.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY, orbit.az, orbit.el]; });
window.addEventListener("pointerup", () => { drag = null; });
window.addEventListener("pointermove", (e) => { if (!drag) return;
  orbit.az = drag[2] - (e.clientX - drag[0]) * 0.006; orbit.el = Math.min(1.3, Math.max(0.05, drag[3] + (e.clientY - drag[1]) * 0.006));
  placeCamera(); update(now); });
function frame(ts) { if (last !== null && playing) { now = Math.min(DURATION, now + (ts - last) / 1000 * speed);
    if (now >= DURATION) { playing = false; $("play").textContent = "▶ Play"; }
    $("scrub").value = Math.round(1000 * now / DURATION); update(now); }
  last = ts; requestAnimationFrame(frame); }
window.addEventListener("resize", resize);
window.replay = { duration: DURATION, seek: (t) => { now = Math.max(0, Math.min(DURATION, t)); lastPanel = -2;
                    $("scrub").value = Math.round(1000 * now / DURATION); update(now); return now; },
                  steps: steps.length, slots };
resize(); requestAnimationFrame(frame);
})();
</script>
</body>
</html>
"""
TEMPLATE = TEMPLATE.replace("__THREE_VERSION__", THREE_VERSION)


if __name__ == "__main__":
    sys.exit(main())
