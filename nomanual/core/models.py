"""Reasoning-model providers (CONTRACTS §7).

One interface, `ReasoningModel.complete(messages, schema=None, images=None, validate=None)`,
returning a `ModelReply` with the text, the parsed JSON (when a schema was given), the model's
reasoning (when the provider returns it) and accounting: tokens, latency, cost, attempts.

Providers:
- `OpenAICompatible`: chat completions over httpx, used for Baseten (primary), Modal, Fireworks
  and the Meta Model API. Transport errors, timeouts, HTTP 429 and 5xx are retried with
  exponential backoff; other HTTP errors fail at once.
- `FakeModel`: in-process and deterministic, for tests and the fake-model benchmark. Its replies
  come from a Python function, never from Muse; every reply it gives says so (`provider="fake"`).

Structured output: with a `schema`, the request asks for JSON matching it (as `response_format`
when the provider supports it, and always in the system message), the reply text is parsed as
JSON (code fences and prose around the object are tolerated), and `validate` (if given) lists
problems with the parsed value. On a parse or validation failure the model is shown its reply and
the problems and asked again, at most `repair_tries` times. Accounting sums every attempt.

Keys come only from environment variables. `make_model` raises `MissingConfig` naming the missing
variable when a provider is not configured, so callers can skip cleanly.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

Message = dict[str, Any]
Validator = Callable[[Any], list[str]]

BASETEN_URL = "https://inference.baseten.co/v1"
MUSE_OPEN_ID = "meta-models/muse-spark-open-1026"
"""The hackathon's model id on Baseten. The public docs say `meta-models/muse-spark-open`; the
event organisers gave this id, so it is the one used (CONTRACTS §7)."""
MUSE_MAX_TOKENS = 16384
MUSE_EFFORT = "medium"
"""Muse Spark Open on Baseten: room for reasoning plus the JSON answer, and an effort below the
provider default (xhigh). `NOMANUAL_REASONING_EFFORT` overrides the effort."""

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ModelError(RuntimeError):
    """A provider call failed for good (after retries), or never could be made."""


class MissingConfig(ModelError):
    """A provider's environment variables are not set."""


class InvalidReply(ModelError):
    """The model's reply stayed unparsable or invalid after every repair attempt."""

    def __init__(self, message: str, reply: ModelReply) -> None:
        super().__init__(message)
        self.reply = reply


@dataclass
class ModelReply:
    text: str
    parsed: Any = None
    reasoning: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    cost_known: bool = True
    """False when the provider's prices are not configured (cost_usd is then 0, not a real 0)."""
    provider: str = ""
    model: str = ""
    attempts: int = 1
    problems: list[str] = field(default_factory=list)
    """Parse/validation problems of the last attempt (empty when the reply is valid)."""


class ReasoningModel(Protocol):
    provider: str
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        schema: dict[str, Any] | None = None,
        images: Sequence[bytes | Path] | None = None,
        validate: Validator | None = None,
    ) -> ModelReply: ...


# --- JSON handling -----------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """The JSON value in a model reply: the whole text, a fenced block, or the first balanced
    object/array in it. Raises ValueError when there is none."""
    def clean_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        return {key.strip(): value for key, value in pairs}

    candidates = [text.strip(), *(m.strip() for m in _FENCE.findall(text))]
    for c in candidates:
        try:
            return json.loads(c, object_pairs_hook=clean_keys)
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder(object_pairs_hook=clean_keys)
    # Some replies have complete proposals followed by a malformed last item. Preserve only
    # complete JSON objects in the hypotheses array; do not invent braces, values or rules.
    # This precedes generic inner-object extraction, which would lose the proposal envelope.
    match = re.search(r'(?<!\\)"hypotheses\s*"\s*:\s*\[', text)
    if match:
        pos = match.end()
        complete = []
        while pos < len(text):
            while pos < len(text) and text[pos] in " \t\r\n,":
                pos += 1
            try:
                value, end = decoder.raw_decode(text, pos)
            except json.JSONDecodeError:
                break
            if not isinstance(value, dict):
                break
            complete.append(value)
            pos = end
        if complete:
            return {"hypotheses": complete}
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text, i)
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in the reply")


def schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "Reply with one JSON value only, no prose and no code fences. It must match this JSON "
        "schema:\n" + json.dumps(schema, separators=(",", ":"))
    )


def repair_message(problems: list[str]) -> Message:
    listed = "\n".join(f"- {p}" for p in problems[:20])
    return {
        "role": "user",
        "content": "Your reply could not be used:\n" + listed
        + "\nReply again with corrected JSON only, matching the schema.",
    }


def _image_part(image: bytes | Path) -> dict[str, Any]:
    data = image.read_bytes() if isinstance(image, Path) else image
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": url}}


def with_images(messages: Sequence[Message], images: Sequence[bytes | Path] | None) -> list[Message]:
    """Attach images (whole frames, no annotations) to the last user message as content parts."""
    out = [dict(m) for m in messages]
    if not images:
        return out
    last = max(i for i, m in enumerate(out) if m["role"] == "user")
    text = out[last]["content"]
    parts = text if isinstance(text, list) else [{"type": "text", "text": text}]
    out[last]["content"] = [*parts, *(_image_part(im) for im in images)]
    return out


def with_schema(messages: Sequence[Message], schema: dict[str, Any] | None) -> list[Message]:
    out = [dict(m) for m in messages]
    if schema is None:
        return out
    note = schema_instruction(schema)
    if out and out[0]["role"] == "system":
        out[0]["content"] = f"{out[0]['content']}\n\n{note}"
    else:
        out.insert(0, {"role": "system", "content": note})
    return out


def check_reply(text: str, schema: dict[str, Any] | None, validate: Validator | None) -> tuple[Any, list[str]]:
    """(parsed value, problems). Without a schema the text itself is the value."""
    if schema is None:
        return text, (validate(text) if validate else [])
    try:
        value = extract_json(text)
    except ValueError as e:
        return None, [str(e)]
    return value, (validate(value) if validate else [])


# --- the structured-output loop shared by every provider ---------------------------------------


def structured(
    call: Callable[[list[Message], dict[str, Any] | None], ModelReply],
    messages: Sequence[Message],
    schema: dict[str, Any] | None,
    validate: Validator | None,
    repair_tries: int,
) -> ModelReply:
    """Run `call` and repair its reply up to `repair_tries` times; accounting sums all attempts.
    Raises InvalidReply (carrying the last reply) when no attempt was usable."""
    convo = with_schema(messages, schema)
    total: ModelReply | None = None
    for attempt in range(repair_tries + 1):
        reply = call(convo, schema)
        reply.parsed, reply.problems = check_reply(reply.text, schema, validate)
        if total is None:
            total = reply
        else:
            total = _accumulate(total, reply)
        total.attempts = attempt + 1
        if not total.problems:
            return total
        convo = [*convo, {"role": "assistant", "content": reply.text}, repair_message(reply.problems)]
    assert total is not None
    raise InvalidReply(f"{total.provider}: no valid reply after {total.attempts} attempts: {total.problems[:3]}", total)


def _accumulate(total: ModelReply, latest: ModelReply) -> ModelReply:
    return ModelReply(
        text=latest.text,
        parsed=latest.parsed,
        reasoning=latest.reasoning,
        tokens_in=total.tokens_in + latest.tokens_in,
        tokens_out=total.tokens_out + latest.tokens_out,
        latency_s=total.latency_s + latest.latency_s,
        cost_usd=total.cost_usd + latest.cost_usd,
        cost_known=total.cost_known and latest.cost_known,
        provider=latest.provider,
        model=latest.model,
        problems=latest.problems,
    )


# --- OpenAI-compatible providers ---------------------------------------------------------------


@dataclass
class OpenAICompatible:
    """Chat completions at `base_url` (`POST {base_url}/chat/completions`)."""

    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None
    timeout_s: float = 120.0
    retries: int = 3
    backoff_s: float = 2.0
    repair_tries: int = 2
    max_tokens: int = 4096
    """Caps the whole completion, reasoning included (Baseten's reasoning docs)."""
    reasoning_effort: str | None = None
    """Top-level `reasoning_effort` (low | medium | high | xhigh), sent when set. Muse Spark Open
    always reasons and defaults to xhigh on Baseten; at that effort a 4096-token cap was all
    reasoning and no answer in the first live call (2026-10-27)."""
    temperature: float = 0.2
    response_format: bool = True
    """Send `response_format: json_schema` with a schema (turned off automatically if the provider
    rejects it with HTTP 400)."""
    verified: bool = True
    """False for an endpoint whose details were not checked against a live call (Meta API)."""
    client: httpx.Client | None = None
    sleep: Callable[[float], None] = time.sleep

    def complete(
        self,
        messages: Sequence[Message],
        schema: dict[str, Any] | None = None,
        images: Sequence[bytes | Path] | None = None,
        validate: Validator | None = None,
    ) -> ModelReply:
        return structured(self._call, with_images(messages, images), schema, validate, self.repair_tries)

    def _payload(self, messages: list[Message], schema: dict[str, Any] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if schema is not None and self.response_format:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
        return body

    def _call(self, messages: list[Message], schema: dict[str, Any] | None) -> ModelReply:
        client = self.client or httpx.Client(timeout=self.timeout_s)
        try:
            data, latency = self._post(client, messages, schema)
        finally:
            if self.client is None:
                client.close()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        t_in, t_out = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        known = self.price_in_per_mtok is not None and self.price_out_per_mtok is not None
        cost = (t_in * (self.price_in_per_mtok or 0) + t_out * (self.price_out_per_mtok or 0)) / 1e6
        return ModelReply(
            text=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
            tokens_in=t_in,
            tokens_out=t_out,
            latency_s=latency,
            cost_usd=cost,
            cost_known=known,
            provider=self.provider,
            model=self.model,
        )

    def _post(self, client: httpx.Client, messages: list[Message], schema: dict[str, Any] | None) -> tuple[dict[str, Any], float]:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last = ""
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                r = client.post(url, json=self._payload(messages, schema), headers=headers, timeout=self.timeout_s)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = f"{type(e).__name__}: {e}"
            else:
                latency = time.perf_counter() - t0
                if r.status_code == 200:
                    try:
                        return r.json(), latency
                    except ValueError:
                        last = "response body is not JSON"
                elif r.status_code == 400 and schema is not None and self.response_format:
                    # Some OpenAI-compatible servers reject json_schema; the schema is also in the
                    # system message, so retry once without it (not counted as a retry).
                    self.response_format = False
                    return self._post(client, messages, schema)
                elif r.status_code in RETRY_STATUS:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                else:
                    raise ModelError(f"{self.provider}: HTTP {r.status_code}: {r.text[:300]}")
            if attempt < self.retries:
                self.sleep(self.backoff_s * 2**attempt)
        raise ModelError(f"{self.provider}: failed after {self.retries + 1} tries: {last}")


# --- fake --------------------------------------------------------------------------------------


Responder = Callable[[list[Message], dict[str, Any] | None, list[bytes | Path]], str]
"""(messages, schema, images) -> reply text. The fake vision path looks at the images."""


@dataclass
class FakeModel:
    """Deterministic in-process model for tests. Not Muse: `provider` is "fake" and every caller
    that reports results must say so. Tokens are counted as characters / 4; latency and cost 0."""

    responder: Responder
    model: str = "fake"
    provider: str = "fake"
    repair_tries: int = 2
    calls: int = 0

    def complete(
        self,
        messages: Sequence[Message],
        schema: dict[str, Any] | None = None,
        images: Sequence[bytes | Path] | None = None,
        validate: Validator | None = None,
    ) -> ModelReply:
        pictures = list(images or ())

        def call(convo: list[Message], sch: dict[str, Any] | None) -> ModelReply:
            self.calls += 1
            text = self.responder(convo, sch, pictures)
            chars_in = sum(len(m["content"]) if isinstance(m["content"], str) else 0 for m in convo)
            return ModelReply(text=text, tokens_in=chars_in // 4, tokens_out=len(text) // 4,
                              provider=self.provider, model=self.model)

        return structured(call, messages, schema, validate, self.repair_tries)


def scripted(replies: Sequence[str]) -> Responder:
    """A responder that returns `replies` in order (the last one repeats)."""
    state = {"i": 0}

    def respond(messages: list[Message], schema: dict[str, Any] | None, images: list[bytes | Path]) -> str:
        i = min(state["i"], len(replies) - 1)
        state["i"] += 1
        return replies[i]

    return respond


# --- provider table ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    needs: tuple[str, ...]
    """Environment variables that must be set."""
    verified: bool
    note: str


PROVIDERS: dict[str, ProviderSpec] = {
    "baseten": ProviderSpec("baseten", ("BASETEN_API_KEY",), True, f"Muse Spark Open, {MUSE_OPEN_ID}"),
    "modal": ProviderSpec("modal", ("MODAL_MUSE_URL", "MODAL_API_KEY"), True, "Muse Spark Open on Modal"),
    "fireworks": ProviderSpec("fireworks", ("FIREWORKS_API_KEY", "FIREWORKS_MUSE_MODEL"), True, "Muse Spark Open on Fireworks"),
    "meta": ProviderSpec(
        "meta", ("META_API_KEY", "META_MODEL_API_URL", "META_MUSE_MODEL"), False,
        "Muse Spark (API) via the Meta Model API: endpoint and model id not verified, set both from Meta's docs",
    ),
}

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"


def missing_env(provider: str, env: dict[str, str] | None = None) -> list[str]:
    env = dict(os.environ) if env is None else env
    return [v for v in PROVIDERS[provider].needs if not env.get(v)]


def available(env: dict[str, str] | None = None) -> list[str]:
    """Providers whose environment variables are all set."""
    return [p for p in PROVIDERS if not missing_env(p, env)]


def _price(env: dict[str, str], provider: str) -> tuple[float | None, float | None]:
    """Prices in USD per million tokens from `NOMANUAL_PRICE_<PROVIDER>="in,out"` (none built in:
    the hackathon's prices were not known when this was written)."""
    raw = env.get(f"NOMANUAL_PRICE_{provider.upper()}")
    if not raw:
        return None, None
    a, b = (float(x) for x in raw.split(","))
    return a, b


def make_model(provider: str, env: dict[str, str] | None = None, **kwargs: Any) -> OpenAICompatible:
    """Build a live provider from environment variables; MissingConfig names what is missing."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; one of {sorted(PROVIDERS)} (or use FakeModel)")
    env = dict(os.environ) if env is None else env
    missing = missing_env(provider, env)
    if missing:
        raise MissingConfig(f"{provider}: set {', '.join(missing)}")
    p_in, p_out = _price(env, provider)
    common: dict[str, Any] = {
        "price_in_per_mtok": p_in, "price_out_per_mtok": p_out,
        "max_tokens": MUSE_MAX_TOKENS,
        "reasoning_effort": env.get("NOMANUAL_REASONING_EFFORT") or MUSE_EFFORT,
        **kwargs,
    }
    # Muse Spark always reasons; at the provider default effort it can spend the whole reply on
    # reasoning (the first Baseten smoke test, 2026-10-27; Fireworks needed a retry and Meta
    # returned no JSON in the 2026-10-30 smoke test). Every Muse provider gets the same settings.
    if provider == "baseten":
        return OpenAICompatible("baseten", BASETEN_URL, MUSE_OPEN_ID, env["BASETEN_API_KEY"], **common)
    if provider == "modal":
        model = env.get("MODAL_MUSE_MODEL") or MUSE_OPEN_ID
        return OpenAICompatible("modal", env["MODAL_MUSE_URL"], model, env["MODAL_API_KEY"], **common)
    if provider == "fireworks":
        return OpenAICompatible("fireworks", FIREWORKS_URL, env["FIREWORKS_MUSE_MODEL"], env["FIREWORKS_API_KEY"],
                                **common)
    return OpenAICompatible("meta", env["META_MODEL_API_URL"], env["META_MUSE_MODEL"], env["META_API_KEY"],
                            verified=False, **{"response_format": False, **common})


# --- call log ----------------------------------------------------------------------------------


@dataclass
class CallLog:
    """Every model request and response of a run (SPEC §6), one JSON line each, next to the
    event log. Image bytes are not stored, only their count."""

    path: Path

    def write(self, purpose: str, messages: Sequence[Message], reply: ModelReply | None, error: str = "",
              n_images: int = 0) -> None:
        record = {
            "t": time.time(),
            "purpose": purpose,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "images": n_images,
            "error": error,
        }
        if reply is not None:
            record |= {
                "provider": reply.provider, "model": reply.model, "text": reply.text,
                "reasoning": reply.reasoning, "tokens_in": reply.tokens_in, "tokens_out": reply.tokens_out,
                "latency_s": round(reply.latency_s, 3), "cost_usd": reply.cost_usd,
                "cost_known": reply.cost_known, "attempts": reply.attempts, "problems": reply.problems,
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
