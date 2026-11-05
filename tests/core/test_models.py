import json
import os

import httpx
import pytest

from nomanual.core.models import (
    MUSE_OPEN_ID,
    CallLog,
    FakeModel,
    InvalidReply,
    MissingConfig,
    ModelError,
    OpenAICompatible,
    available,
    extract_json,
    make_model,
    scripted,
)

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


def need_n(value: object) -> list[str]:
    return [] if isinstance(value, dict) and isinstance(value.get("n"), int) else ["`n` must be an integer"]


def chat(content: str, t_in: int = 10, t_out: int = 5, reasoning: str | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg}], "usage": {"prompt_tokens": t_in, "completion_tokens": t_out}}


def provider(handler, **kw) -> tuple[OpenAICompatible, list[float]]:
    slept: list[float] = []
    m = OpenAICompatible("test", "https://example.invalid/v1", "m", "k", client=httpx.Client(transport=httpx.MockTransport(handler)),
                         sleep=slept.append, **kw)
    return m, slept


def test_extract_json_tolerates_fences_and_prose() -> None:
    assert extract_json('{"n": 1}') == {"n": 1}
    assert extract_json('Sure:\n```json\n{"n": 2}\n```') == {"n": 2}
    assert extract_json('I think {"n": 3} fits; also {"n": 4}') == {"n": 3}
    assert extract_json("[1, 2]") == [1, 2]
    assert extract_json('{"a ": [{" n": 1}]}') == {"a": [{"n": 1}]}
    assert extract_json('Sure: {"n ": 5}') == {"n": 5}
    with pytest.raises(ValueError):
        extract_json("no json { here")


def test_extract_json_strips_nested_object_keys() -> None:
    assert extract_json('{"a ": [{" n": 1}]}') == {"a": [{"n": 1}]}
    assert extract_json('Answer: {"a ": [{" n": 1}]} done') == {"a": [{"n": 1}]}


def test_extract_json_keeps_only_complete_proposal_prefix() -> None:
    text = '{"reasoning":"x", "hypotheses": [{"a ":1},{"b":2},{"unfinished":['
    assert extract_json(text) == {"hypotheses": [{"a":1},{"b":2}]}
    # A complete envelope retains its original content for normal validation, even if a
    # program is invalid. Recovery never manufactures an unfinished object.
    assert extract_json('{"hypotheses":[{"a":1},{}]}') == {"hypotheses":[{"a":1},{}]}
    assert extract_json('{"hypotheses":[{"a":1}]} trailing') == {"hypotheses":[{"a":1}]}


def test_request_shape_usage_and_reasoning() -> None:
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append({"url": str(req.url), "auth": req.headers["authorization"], "body": json.loads(req.content)})
        return httpx.Response(200, json=chat('{"n": 7}', reasoning="because"))

    m, _ = provider(handler, price_in_per_mtok=1.0, price_out_per_mtok=2.0)
    r = m.complete([{"role": "user", "content": "count"}], schema=SCHEMA, images=[b"\xff\xd8jpegbytes"], validate=need_n)
    assert r.parsed == {"n": 7} and r.reasoning == "because" and r.attempts == 1 and not r.problems
    assert r.tokens_in == 10 and r.tokens_out == 5 and r.cost_usd == pytest.approx(20e-6) and r.cost_known
    body = seen[0]["body"]
    assert seen[0]["url"] == "https://example.invalid/v1/chat/completions" and seen[0]["auth"] == "Bearer k"
    assert body["messages"][0]["role"] == "system" and "JSON schema" in body["messages"][0]["content"]
    parts = body["messages"][-1]["content"]
    assert parts[0] == {"type": "text", "text": "count"}
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert body["response_format"]["type"] == "json_schema"


def test_retries_transient_errors_then_fails_cleanly() -> None:
    calls = {"n": 0}

    def flaky(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("slow")
        if calls["n"] == 2:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=chat("hello"))

    m, slept = provider(flaky)
    assert m.complete([{"role": "user", "content": "hi"}]).text == "hello"
    assert slept == [2.0, 4.0]

    m, slept = provider(lambda req: httpx.Response(429, text="rate"), retries=2)
    with pytest.raises(ModelError, match="failed after 3 tries"):
        m.complete([{"role": "user", "content": "hi"}])
    assert len(slept) == 2

    m, slept = provider(lambda req: httpx.Response(401, text="bad key"))
    with pytest.raises(ModelError, match="401"):
        m.complete([{"role": "user", "content": "hi"}])
    assert slept == []


def test_schema_rejected_falls_back_to_prompt_only() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if "response_format" in json.loads(req.content):
            return httpx.Response(400, text="response_format unsupported")
        return httpx.Response(200, json=chat('{"n": 1}'))

    m, _ = provider(handler)
    assert m.complete([{"role": "user", "content": "x"}], schema=SCHEMA).parsed == {"n": 1}
    assert m.response_format is False


def test_invalid_replies_are_repaired_with_the_problems_shown() -> None:
    replies = ["not json", '{"n": "two"}', '{"n": 2}']
    sent: list[list[dict]] = []

    def respond(messages, schema, images):
        sent.append(messages)
        return replies[len(sent) - 1]

    r = FakeModel(respond).complete([{"role": "user", "content": "x"}], schema=SCHEMA, validate=need_n)
    assert r.parsed == {"n": 2} and r.attempts == 3 and r.provider == "fake"
    assert "no JSON" in sent[1][-1]["content"] and "`n` must be an integer" in sent[2][-1]["content"]
    assert sent[2][-2] == {"role": "assistant", "content": '{"n": "two"}'}
    assert r.tokens_in > 0 and r.tokens_out > 0  # summed over the three attempts

    with pytest.raises(InvalidReply) as e:
        FakeModel(scripted(["nope"]), repair_tries=1).complete([{"role": "user", "content": "x"}], schema=SCHEMA)
    assert e.value.reply.attempts == 2


def test_make_model_reads_env_only_and_names_what_is_missing() -> None:
    with pytest.raises(MissingConfig, match="BASETEN_API_KEY"):
        make_model("baseten", env={})
    m = make_model("baseten", env={"BASETEN_API_KEY": "secret"})
    assert m.model == MUSE_OPEN_ID and m.base_url == "https://inference.baseten.co/v1" and "secret" not in repr(m)
    with pytest.raises(MissingConfig, match="META_MODEL_API_URL"):
        make_model("meta", env={"META_API_KEY": "k"})
    meta = make_model("meta", env={"META_API_KEY": "k", "META_MODEL_API_URL": "https://x/v1", "META_MUSE_MODEL": "m"})
    assert meta.verified is False
    assert meta.response_format is False
    priced = make_model("modal", env={"MODAL_MUSE_URL": "https://m/v1", "MODAL_API_KEY": "k", "NOMANUAL_PRICE_MODAL": "0.5,1.5"})
    assert (priced.price_in_per_mtok, priced.price_out_per_mtok) == (0.5, 1.5)
    assert available({"FIREWORKS_API_KEY": "k", "FIREWORKS_MUSE_MODEL": "f"}) == ["fireworks"]
    with pytest.raises(ValueError):
        make_model("openai", env={})


def test_call_log_records_requests_without_image_bytes(tmp_path) -> None:
    log = CallLog(tmp_path / "model_calls.jsonl")
    r = FakeModel(scripted(["ok"])).complete([{"role": "user", "content": "hi"}])
    log.write("propose", [{"role": "user", "content": "hi"}], r, n_images=2)
    log.write("propose", [{"role": "user", "content": "hi"}], None, error="timeout")
    rows = [json.loads(line) for line in (tmp_path / "model_calls.jsonl").read_text().splitlines()]
    assert rows[0]["text"] == "ok" and rows[0]["images"] == 2 and rows[0]["provider"] == "fake"
    assert rows[1]["error"] == "timeout" and "text" not in rows[1]


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("BASETEN_API_KEY"), reason="BASETEN_API_KEY not set")
def test_live_baseten_structured_reply() -> None:
    r = make_model("baseten").complete([{"role": "user", "content": "Reply with n = 3."}], schema=SCHEMA, validate=need_n)
    assert r.parsed["n"] == 3 and r.tokens_in > 0


def test_every_muse_provider_gets_the_reply_budget_and_effort() -> None:
    env = {"BASETEN_API_KEY": "k", "FIREWORKS_API_KEY": "k", "FIREWORKS_MUSE_MODEL": "m", "META_API_KEY": "k",
           "META_MODEL_API_URL": "https://example.invalid/v1", "META_MUSE_MODEL": "m",
           "MODAL_MUSE_URL": "https://example.invalid/v1", "MODAL_API_KEY": "k"}
    for provider in ("baseten", "fireworks", "meta", "modal"):
        m = make_model(provider, env=env)
        assert m.max_tokens == 16384 and m.reasoning_effort == "medium", provider
