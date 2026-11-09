"""CHALLENGER: Agentic RAG (LangGraph state machine).  Owner: member B.

v0 scaffold - runs end-to-end; member B tunes prompts, adds re-ranking, and runs the ablations.

              +-----------+   not banking / injection
   query ---> |  analyze  |------------------------------------------------+
              +-----------+                                                 |
                    | intent (from Task-1 classifier tool or LLM), sub-questions,
                    v search_query, needs_clarification                    |
              +-----------+        +---------+  none relevant & rounds left |
              | retrieve  |------->|  grade  |--------> rewrite --+         |
              +-----------+        +---------+                    |         |
                    ^                   | >=1 relevant            |         |
                    +-------------------|-------------------------+         |
                                        v            none relevant, no rounds left
                                  +----------+ ---------------------------> escalate
                                  | generate |<-----+                       |
                                  +----------+      | fail & 0 revisions    |
                                        v           |                       |
                                  +----------+------+                       |
                                  | validate |--- fail after 1 revision --> escalate
                                  +----------+                              |
                                        | pass                              v
                                       END <--------------------------------+

Design choices worth defending in Q&A
- Bounded loops (max 2 retrieval rounds, max 1 revision) -> predictable cost/latency, no infinite loops.
- Intent from the Task-1 classifier is used as a METADATA FILTER for retrieval (tool use) when
  confident; low confidence -> unfiltered search (avoids error propagation from a wrong intent).
- Hybrid retrieval (dense + BM25 via Reciprocal Rank Fusion) helps typos / exact banking terms.
- Grading (CRAG-style) and validation (Self-RAG / Reflexion-style) are separate LLM calls so
  each can be switched off for ablation.
References: LangGraph docs; Yan et al. 2024 (Corrective RAG); Asai et al. 2023 (Self-RAG);
Yao et al. 2023 (ReAct); Shinn et al. 2023 (Reflexion).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..llm.client import LLMClient, parse_json
from ..llm.prompts import load_prompt
from .retriever import format_context, hybrid_search
from .systems import _Base, finalize


@dataclass
class AgenticConfig:
    use_classifier_tool: bool = True   # use Task-1 classifier as a tool (else LLM-predicted intent)
    intent_conf_threshold: float = 0.6  # below this, do not filter retrieval by intent
    use_hybrid: bool = True            # dense + BM25 (RRF) instead of dense only
    use_grading: bool = True           # relevance grading + query rewrite loop
    use_validation: bool = True        # validate -> revise once -> escalate
    max_retrieval_rounds: int = 2
    max_revisions: int = 1
    k: int = 3


class State(TypedDict, total=False):
    query: str
    intent: Optional[str]
    intent_conf: float
    is_banking: bool
    needs_clarification: bool
    sub_questions: list
    search_query: str
    hits: list
    relevant_hits: list
    rounds: int
    draft: dict
    feedback: str
    revisions: int
    final_action: str
    final_answer: str
    escalation_reason: str
    out: dict


class AgenticRAG(_Base):
    name = "agentic_rag"

    def __init__(self, llm: LLMClient, dense, bm25=None, intents: list[str] | None = None,
                 classifier: Callable[[str], tuple[str, float]] | None = None,
                 cfg: AgenticConfig | None = None, name: str | None = None):
        super().__init__(llm)
        self.dense, self.bm25, self.classifier = dense, bm25, classifier
        self.cfg = cfg or AgenticConfig()
        self.intents = intents or sorted(dense.kb["intent"].unique())
        if name:
            self.name = name
        self.p = {k: load_prompt(f"agentic_{k}_v1") for k in ("analyze", "grade", "rewrite", "generate", "validate")}
        self.graph = self._build()

    # ------------------------------------------------------------ helpers
    def _llm(self, state: State, step: str, prompt: str, **kw) -> dict:
        sys_p, user_p = self.p[prompt].render(**kw)
        resp = self.llm.chat(sys_p, user_p)
        self._add_call(state["out"], step, resp)
        return parse_json(resp.text)

    def _search(self, q: str, intent: str | None) -> list[dict]:
        if self.cfg.use_hybrid and self.bm25 is not None:
            return hybrid_search([self.dense, self.bm25], q, self.cfg.k, intent)
        return self.dense.search(q, self.cfg.k, intent)

    # ------------------------------------------------------------ nodes
    def analyze(self, s: State) -> State:
        a = self._llm(s, "analyze", "analyze", query=s["query"], intents=", ".join(self.intents))
        intent, conf = a.get("intent"), 0.5
        if self.cfg.use_classifier_tool and self.classifier is not None:
            intent, conf = self.classifier(s["query"])  # TOOL CALL: Task-1 model
            s["out"]["trace"].append({"step": "tool:intent_classifier", "output": [intent, round(conf, 3)]})
        return {"intent": intent if intent in self.intents else None, "intent_conf": float(conf),
                "is_banking": bool(a.get("is_banking", True)),
                "needs_clarification": bool(a.get("needs_clarification", False)),
                "sub_questions": a.get("sub_questions") or [],
                "search_query": a.get("search_query") or s["query"], "rounds": 0, "revisions": 0, "feedback": ""}

    def retrieve(self, s: State) -> State:
        use_filter = s.get("intent") and s.get("intent_conf", 0) >= self.cfg.intent_conf_threshold
        # Multi-part question: search each part, merge (dedup by doc_id)
        queries = s["sub_questions"] if len(s.get("sub_questions", [])) > 1 else [s["search_query"]]
        hits, seen = [], set()
        for q in queries:
            for h in self._search(q, s["intent"] if (use_filter and len(queries) == 1) else None):
                if h["doc_id"] not in seen:
                    seen.add(h["doc_id"])
                    hits.append(h)
        s["out"]["trace"].append({"step": f"retrieve(round {s['rounds'] + 1}, filter={bool(use_filter)})",
                                  "output": [h["doc_id"] for h in hits]})
        return {"hits": hits, "rounds": s["rounds"] + 1}

    def grade(self, s: State) -> State:
        if not self.cfg.use_grading:
            return {"relevant_hits": s["hits"]}
        g = self._llm(s, "grade", "grade", query=s["query"], context=format_context(s["hits"]))
        rel = g.get("relevant", [])
        if rel is True:  # tolerate {"relevant": true}
            rel = list(range(1, len(s["hits"]) + 1))
        keep = [s["hits"][i - 1] for i in rel if isinstance(i, int) and 1 <= i <= len(s["hits"])]
        return {"relevant_hits": keep}

    def rewrite(self, s: State) -> State:
        r = self._llm(s, "rewrite", "rewrite", query=s["query"], previous=s["search_query"])
        # second round searches without the intent filter (the filter may have been the problem)
        return {"search_query": r.get("query") or s["query"], "intent_conf": 0.0, "sub_questions": []}

    def generate(self, s: State) -> State:
        notes = (f"predicted intent={s.get('intent')} (conf {s.get('intent_conf', 0):.2f}); "
                 f"sub-questions={s.get('sub_questions')}; needs_clarification={s.get('needs_clarification')}")
        sys_p, user_p = self.p["generate"].render(query=s["query"], notes=notes,
                                                  context=format_context(s["relevant_hits"]), feedback=s.get("feedback", ""))
        resp = self.llm.chat(sys_p, user_p)
        self._add_call(s["out"], "generate" if not s.get("feedback") else "revise", resp)
        action, answer, ok = finalize(resp.text)
        return {"draft": {"action": action, "answer": answer, "parse_ok": ok}}

    def validate(self, s: State) -> State:
        if not self.cfg.use_validation or s["draft"]["action"] != "answer":
            return {"final_action": s["draft"]["action"], "final_answer": s["draft"]["answer"]}
        v = self._llm(s, "validate", "validate", query=s["query"], context=format_context(s["relevant_hits"]),
                      draft=str(s["draft"]))
        if str(v.get("verdict", "pass")).lower() == "pass" and not v.get("should_escalate"):
            return {"final_action": "answer", "final_answer": s["draft"]["answer"]}
        if v.get("should_escalate") or s["revisions"] >= self.cfg.max_revisions:
            return {"final_action": "escalate_pending", "escalation_reason": "; ".join(map(str, v.get("problems", [])))}
        return {"feedback": "; ".join(map(str, v.get("problems", []))) or "Improve grounding.",
                "revisions": s["revisions"] + 1, "final_action": "revise"}

    def escalate(self, s: State) -> State:
        reason = s.get("escalation_reason") or ("out of scope" if not s.get("is_banking", True) else "no supporting evidence")
        msg = ("I'm sorry, I can't help with that here. I'm connecting you with one of our human agents, "
               "who can assist you further.")
        return {"final_action": "escalate", "final_answer": msg, "escalation_reason": reason}

    # ------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(State)
        for n in ("analyze", "retrieve", "grade", "rewrite", "generate", "validate", "escalate"):
            g.add_node(n, getattr(self, n))
        g.set_entry_point("analyze")
        g.add_conditional_edges("analyze", lambda s: "retrieve" if s["is_banking"] else "escalate",
                                {"retrieve": "retrieve", "escalate": "escalate"})
        g.add_edge("retrieve", "grade")

        def after_grade(s):
            if s["relevant_hits"] or s.get("needs_clarification"):
                return "generate"
            return "rewrite" if s["rounds"] < self.cfg.max_retrieval_rounds else "escalate"

        g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite", "escalate": "escalate"})
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", "validate")

        def after_validate(s):
            return {"revise": "generate", "escalate_pending": "escalate"}.get(s["final_action"], END)

        g.add_conditional_edges("validate", after_validate, {"generate": "generate", "escalate": "escalate", END: END})
        g.add_edge("escalate", END)
        return g.compile()

    # ------------------------------------------------------------ public interface (same as other systems)
    def answer(self, query: str) -> dict:
        t0 = time.perf_counter()
        out: dict[str, Any] = self._empty()
        final = self.graph.invoke({"query": query, "out": out}, {"recursion_limit": 25})
        hits = final.get("relevant_hits") or final.get("hits") or []
        out.update(
            action=final.get("final_action", "escalate"), answer=final.get("final_answer", ""),
            contexts=[h["doc_id"] for h in hits], context_intents=[h["intent"] for h in hits],
            context_scores=[round(h["score"], 4) for h in hits],
            predicted_intent=final.get("intent"), retrieval_rounds=final.get("rounds", 0),
            revisions=final.get("revisions", 0), escalation_reason=final.get("escalation_reason"),
            parse_ok=(final.get("draft") or {}).get("parse_ok", True),
            prompt_version="+".join(p.version for p in self.p.values()),
            config=self.cfg.__dict__.copy(), wall_time_s=round(time.perf_counter() - t0, 3),
        )
        return out
