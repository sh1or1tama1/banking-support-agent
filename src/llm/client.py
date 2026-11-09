"""One LLM client for every Task-2 system, so all systems use the SAME model and settings.

- Providers: openai | anthropic | openai_compatible (Groq, Ollama, vLLM ...) | mock (offline tests)
- Every call is cached on disk (key = provider, model, settings, prompts). Re-running a notebook
  costs nothing and gives identical outputs -> reproducibility + budget control.
- Every call returns tokens, latency and cost so efficiency can be compared across systems.

API keys come from environment variables (OPENAI_API_KEY / ANTHROPIC_API_KEY / LLM_API_KEY).
Never commit keys to GitHub; in Colab use the "Secrets" panel.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass

from .. import config as C


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    cached: bool
    model: str


class LLMClient:
    def __init__(self, provider: str = C.LLM_PROVIDER, model: str = C.LLM_MODEL,
                 temperature: float = C.LLM_TEMPERATURE, max_tokens: int = C.LLM_MAX_TOKENS,
                 base_url: str = C.LLM_BASE_URL, use_cache: bool = True):
        self.provider, self.model = provider, model or ("mock-model" if provider == "mock" else "")
        if not self.model:
            raise ValueError("Set LLM_MODEL (e.g. the cheapest current small model of your provider).")
        self.temperature, self.max_tokens, self.base_url, self.use_cache = temperature, max_tokens, base_url, use_cache
        self.n_calls = 0
        self._client = None

    # ------------------------------------------------------------ public
    def chat(self, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        max_tokens = max_tokens or self.max_tokens
        key = hashlib.sha256(json.dumps(
            [self.provider, self.model, self.temperature, max_tokens, system, user]).encode()).hexdigest()
        path = C.LLM_CACHE / f"{key}.json"
        if self.use_cache and path.exists():
            d = json.loads(path.read_text())
            d.update(cached=True, latency_s=d.get("latency_s", 0.0))  # keep original latency for reporting
            return LLMResponse(**d)

        t0 = time.perf_counter()
        text, tin, tout = self._call_with_retry(system, user, max_tokens)
        latency = time.perf_counter() - t0
        cost = tin / 1e6 * C.PRICE_IN_PER_M + tout / 1e6 * C.PRICE_OUT_PER_M
        resp = LLMResponse(text, tin, tout, round(latency, 3), round(cost, 6), False, self.model)
        self.n_calls += 1
        if self.use_cache:
            path.write_text(json.dumps(asdict(resp)))
        return resp

    # ------------------------------------------------------------ providers
    def _call_with_retry(self, system, user, max_tokens, tries: int = 5):
        for attempt in range(tries):
            try:
                return self._call(system, user, max_tokens)
            except Exception as e:  # rate limits, timeouts
                if attempt == tries - 1:
                    raise
                wait = 2 ** attempt
                print(f"[LLMClient] {type(e).__name__}: {e} -> retry in {wait}s")
                time.sleep(wait)

    def _call(self, system, user, max_tokens):
        if self.provider == "mock":
            return _mock_reply(system, user), len((system + user).split()), 20

        if self.provider in ("openai", "openai_compatible"):
            from openai import OpenAI

            if self._client is None:
                kw = {}
                if self.provider == "openai_compatible":
                    kw = dict(base_url=self.base_url, api_key=os.getenv("LLM_API_KEY", "not-needed"))
                self._client = OpenAI(**kw)
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            kw = dict(model=self.model, messages=msgs)
            if self.provider == "openai":
                kw["max_completion_tokens"] = max_tokens
            else:
                kw["max_tokens"] = max_tokens
            try:
                r = self._client.chat.completions.create(temperature=self.temperature, **kw)
            except Exception as e:  # some reasoning models reject a custom temperature
                if "temperature" not in str(e).lower():
                    raise
                r = self._client.chat.completions.create(**kw)
            u = r.usage
            return r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens

        if self.provider == "anthropic":
            import anthropic

            if self._client is None:
                self._client = anthropic.Anthropic()
            r = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=self.temperature,
                system=system, messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return text, r.usage.input_tokens, r.usage.output_tokens

        raise ValueError(f"Unknown provider {self.provider}")


def _mock_reply(system: str, user: str) -> str:
    """Deterministic fake output in the JSON shape our prompts request (for offline smoke tests)."""
    if "Return JSON with keys relevant" in system:  # grader
        return json.dumps({"relevant": True, "reason": "mock"})
    if "Return JSON with keys verdict" in system:  # validator
        return json.dumps({"verdict": "pass", "problems": [], "should_escalate": False})
    if "Return JSON with keys intent" in system:  # query analysis
        return json.dumps({"intent": "block_card", "is_banking": True, "needs_clarification": False,
                           "sub_questions": [], "search_query": user[-80:]})
    if "Return JSON with keys relevance" in system:  # judge
        return json.dumps({"relevance": 2, "step_coverage": 1, "clarity": 2, "grounded": 1, "unsafe": 0,
                           "behaviour_ok": 1, "rationale": "mock"})
    if "Return JSON with keys query" in system:  # query rewrite
        return json.dumps({"query": "block lost card", "reason": "mock"})
    return json.dumps({"action": "answer", "answer": "[mock] I can help with that. " + user[-80:],
                       "reason": "mock"})


def parse_json(text: str) -> dict:
    """Robustly pull the first JSON object out of an LLM reply (handles ```json fences)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return {}
