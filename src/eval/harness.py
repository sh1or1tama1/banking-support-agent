"""Run any Task-2 system over the frozen evaluation table and save one JSONL line per query.

    from src.eval.harness import run_system, load_run, summarize
    run_system(NaiveRAG(llm, retriever), eval_df, tag="naive_rag")

Runs are resumable (already-answered eval_ids are skipped), so a crashed Colab session or an
API rate-limit never loses work. Combined with the LLM cache, re-runs are free.
"""
from __future__ import annotations

import json
import re

import pandas as pd
from tqdm.auto import tqdm

from .. import config as C
from ..utils import sample_per_group


def run_path(tag: str):
    return C.RUNS / f"{tag}.jsonl"


def run_system(system, eval_df: pd.DataFrame, tag: str | None = None, limit: int | None = None) -> pd.DataFrame:
    tag = tag or system.name
    path = run_path(tag)
    done = set()
    if path.exists():
        done = {json.loads(l)["eval_id"] for l in path.read_text().splitlines() if l.strip()}
    todo = eval_df[~eval_df["eval_id"].isin(done)]
    if limit:  # development run: a small sample spread over every eval set (not just the first rows)
        per_set = max(1, limit // max(1, eval_df["eval_set"].nunique()))
        todo = sample_per_group(todo, "eval_set", per_set)
    with path.open("a") as f:
        for r in tqdm(todo.itertuples(), total=len(todo), desc=tag):
            try:
                out = system.answer(r.query)
            except Exception as e:  # keep going; record the failure
                out = {"system": tag, "action": "error", "answer": f"{type(e).__name__}: {e}", "trace": []}
            out.update(eval_id=r.eval_id, eval_set=r.eval_set, query=r.query, run_tag=tag)
            f.write(json.dumps(out, default=str) + "\n")
    return load_run(tag)


def load_run(tag: str) -> pd.DataFrame:
    return pd.read_json(run_path(tag), lines=True)


# --------------------------------------------------------------- automatic checks (no LLM needed)
SPECIFIC_FACT_RE = {
    "phone_number": re.compile(r"\+?\d[\d\-\s().]{7,}\d"),
    "url": re.compile(r"(?:https?://|www\.)\S+", re.I),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "money_amount": re.compile(r"[$€£]\s?\d"),
    "percentage": re.compile(r"\d+(?:\.\d+)?\s?%"),
    "clock_time": re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", re.I),
}
ACTION_CLAIM_RE = re.compile(r"\b(?:I have|I've|we have|we've)\s+(?:blocked|cancell?ed|closed|transferred|sent|activated|reset)\b", re.I)


def auto_checks(run: pd.DataFrame) -> pd.DataFrame:
    """Heuristic red flags: invented specifics (policy rule 2) and claimed account actions (rule 1).
    Text inside {{placeholders}} is ignored. These are proxies; the human rubric is the ground truth."""
    run = run.copy()
    text = run["answer"].fillna("").str.replace(r"\{\{[^{}]*\}\}", " ", regex=True)
    for k, rx in SPECIFIC_FACT_RE.items():
        run[f"invented_{k}"] = text.str.contains(rx)
    run["invented_any_specific"] = run[[f"invented_{k}" for k in SPECIFIC_FACT_RE]].any(axis=1)
    run["claims_account_action"] = text.str.contains(ACTION_CLAIM_RE)
    return run


def behaviour_correct(action: str, expected: str) -> bool:
    """answer-sets expect 'answer'; unsupported/OOD expect 'escalate'; ambiguous accepts clarify OR escalate."""
    if expected == "clarify":
        return action in ("clarify", "escalate")
    return action == expected


def summarize(run: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Per eval_set: action accuracy, escalation rates, red-flag rates, cost/latency/tokens."""
    d = auto_checks(run).merge(eval_df[["eval_id", "expected_action", "gold_intent"]], on="eval_id", how="left")
    d["behaviour_ok"] = [behaviour_correct(a, e) for a, e in zip(d["action"], d["expected_action"])]
    if "context_intents" in d:
        d["top1_context_intent_ok"] = [
            (isinstance(ci, list) and len(ci) > 0 and ci[0] == g) if isinstance(g, str) else None
            for ci, g in zip(d["context_intents"], d["gold_intent"])
        ]
    agg = d.groupby("eval_set").agg(
        n=("eval_id", "count"),
        behaviour_accuracy=("behaviour_ok", "mean"),
        answer_rate=("action", lambda s: (s == "answer").mean()),
        escalate_rate=("action", lambda s: (s == "escalate").mean()),
        clarify_rate=("action", lambda s: (s == "clarify").mean()),
        invented_specific_rate=("invented_any_specific", "mean"),
        action_claim_rate=("claims_account_action", "mean"),
        json_parse_ok=("parse_ok", "mean"),
        mean_llm_calls=("n_llm_calls", "mean"),
        mean_tokens=("input_tokens", lambda s: (s + d.loc[s.index, "output_tokens"]).mean()),
        total_cost_usd=("cost_usd", "sum"),
        median_wall_s=("wall_time_s", "median"),
    )
    return agg.round(4)


def escalation_confusion(run: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Should-escalate (unsupported + OOD) vs did-escalate -> precision / recall / F1 of escalation."""
    d = run.merge(eval_df[["eval_id", "expected_action"]], on="eval_id")
    d = d[d["expected_action"] != "clarify"]
    should, did = d["expected_action"] == "escalate", d["action"] == "escalate"
    tp, fp, fn = int((should & did).sum()), int((~should & did).sum()), int((should & ~did).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return pd.DataFrame([{"tp": tp, "fp": fp, "fn": fn, "tn": int((~should & ~did).sum()),
                          "precision": round(p, 4), "recall": round(r, 4),
                          "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}])
