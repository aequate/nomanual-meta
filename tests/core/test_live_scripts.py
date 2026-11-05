"""The live scripts skip cleanly without keys (no network is touched)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from nomanual.core.models import PROVIDERS

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for spec in PROVIDERS.values():
        for var in spec.needs:
            monkeypatch.delenv(var, raising=False)


def test_live_smoke_skips_every_provider_and_runs_the_fake(no_keys: None, tmp_path: Path, capsys) -> None:
    assert _load("live_smoke").main(["--fake", "--log-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.count("SKIP") == len(PROVIDERS) and "fake: OK" in out
    assert (tmp_path / "model_calls.jsonl").exists()


def test_live_dev_skips_without_keys_and_refuses_without_a_price(no_keys: None, monkeypatch, capsys) -> None:
    live_dev = _load("live_dev")
    assert live_dev.main(["--provider", "baseten"]) == 0
    assert "SKIP" in capsys.readouterr().out
    monkeypatch.setenv("BASETEN_API_KEY", "not-a-real-key")
    monkeypatch.delenv("NOMANUAL_PRICE_BASETEN", raising=False)
    assert live_dev.main(["--provider", "baseten"]) == 2
    assert "refusing" in capsys.readouterr().out


def test_live_dev_takes_fresh_seeds(no_keys: None, monkeypatch, tmp_path: Path) -> None:
    # With --seeds the run set is `bench.fresh(seeds)`, not the dev set (no model is called: the
    # runner is replaced, and the cap check runs on what it returns).
    live_dev = _load("live_dev")
    monkeypatch.setenv("BASETEN_API_KEY", "not-a-real-key")
    monkeypatch.setenv("NOMANUAL_PRICE_BASETEN", "1.25,4.25")
    seen: list[list[str]] = []
    monkeypatch.setattr(live_dev, "run_set", lambda items, *a, **k: seen.append([it.name for it in items]) or [])
    out = tmp_path / "fresh.jsonl"
    out.write_text("", encoding="utf-8")
    monkeypatch.setattr(live_dev, "report", lambda *a, **k: "")
    assert live_dev.main(["--provider", "baseten", "--seeds", "7000", "7001", "--out", str(out)]) == 0
    assert seen == [["fresh-00", "fresh-01"]]
