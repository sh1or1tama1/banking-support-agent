"""Task-2 systems share ONE interface so the evaluation harness can run any of them:

    system.answer(query: str) -> dict with at least
        action ('answer'|'clarify'|'escalate'), answer, contexts (list of doc_ids),
        n_llm_calls, input_tokens, output_tokens, cost_usd, latency_s, trace (list of steps)

Systems
  DirectLLM  - reference lower bound: same LLM + same policy, no retrieval.
  NaiveRAG   - BASELINE / champion: retrieve top-k once -> generate once.
  AgenticRAG - CHALLENGER (src/rag/agentic_rag.py, owned by member B).
"""
from __future__ import annotations

import time

from ..llm.client import LLMClient, parse_json
from ..llm.prompts import load_prompt
from .retriever import format_context

VALID_ACTIONS = {"answer", "clarify", "escalate"}


def finalize(raw_text: str) -> tuple[str, str, bool]:
    """Parse the model's JSON. Unparseable output is kept as an answer but flagged."""
    d = parse_json(raw_text)
    action = str(d.get("action", "")).lower()
    ok = action in VALID_ACTIONS and bool(d.get("answer"))
    return (action if ok else "answer"), (d.get("answer") or raw_text), ok


class _Base:
    name = "base"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _empty(self) -> dict:
        return {"system": self.name, "contexts": [], "context_scores": [], "n_llm_calls": 0,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_latency_s": 0.0, "trace": []}

    def _add_call(self, out: dict, step: str, resp) -> None:
        out["n_llm_calls"] += 1
        out["input_tokens"] += resp.input_tokens
        out["output_tokens"] += resp.output_tokens
        out["cost_usd"] += resp.cost_usd
        out["llm_latency_s"] += resp.latency_s
        out["trace"].append({"step": step, "output": resp.text[:500], "cached": resp.cached})


class DirectLLM(_Base):
    name = "direct_llm"

    def __init__(self, llm: LLMClient, prompt: str = "direct_llm_v1"):
        super().__init__(llm)
        self.prompt = load_prompt(prompt)

    def answer(self, query: str) -> dict:
        t0 = time.perf_counter()
        out = self._empty()
        sys_p, user_p = self.prompt.render(query=query)
        resp = self.llm.chat(sys_p, user_p)
        self._add_call(out, "generate", resp)
        out["action"], out["answer"], out["parse_ok"] = finalize(resp.text)
        out["prompt_version"] = self.prompt.version
        out["wall_time_s"] = round(time.perf_counter() - t0, 3)
        return out


class NaiveRAG(_Base):
    name = "naive_rag"

    def __init__(self, llm: LLMClient, retriever, k: int = 3, prompt: str = "naive_rag_v1"):
        super().__init__(llm)
        self.retriever, self.k = retriever, k
        self.prompt = load_prompt(prompt)

    def answer(self, query: str) -> dict:
        t0 = time.perf_counter()
        out = self._empty()
        hits = self.retriever.search(query, self.k)
        out["contexts"] = [h["doc_id"] for h in hits]
        out["context_scores"] = [round(h["score"], 4) for h in hits]
        out["context_intents"] = [h["intent"] for h in hits]
        out["trace"].append({"step": "retrieve", "output": out["contexts"]})
        sys_p, user_p = self.prompt.render(query=query, context=format_context(hits))
        resp = self.llm.chat(sys_p, user_p)
        self._add_call(out, "generate", resp)
        out["action"], out["answer"], out["parse_ok"] = finalize(resp.text)
        out["prompt_version"] = self.prompt.version
        out["wall_time_s"] = round(time.perf_counter() - t0, 3)
        return out
