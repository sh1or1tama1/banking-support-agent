"""LLM-as-a-judge + blinded human rubric + agreement statistics (Phase 3).

Human rubric = the same six items as prompts/judge_v1.md, so judge and humans are comparable.
Workflow
  1. make_human_sheet(...)   -> CSV with system names hidden (labels S1/S2/S3, shuffled per query)
  2. two reviewers score independently (copy the CSV: reviewer_1.csv, reviewer_2.csv)
  3. agreement(...)          -> Cohen's weighted kappa per item between the reviewers
  4. adjudicate disagreements together -> final human scores
  5. judge_run(...) on the same items -> judge_vs_human(...) agreement; if acceptable, use the
     judge to score the FULL eval set (and say so in the report).
Tip: use a different (ideally stronger) model as the judge than the generator to reduce self-preference bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .. import config as C
from ..llm.client import LLMClient, parse_json
from ..llm.prompts import load_prompt

ITEMS = ["relevance", "step_coverage", "clarity", "grounded", "unsafe", "behaviour_ok"]


def judge_run(run: pd.DataFrame, eval_df: pd.DataFrame, judge_llm: LLMClient, prompt: str = "judge_v1") -> pd.DataFrame:
    from tqdm.auto import tqdm

    p = load_prompt(prompt)
    d = run.merge(eval_df[["eval_id", "expected_action", "reference_response"]], on="eval_id", how="left")
    rows = []
    for r in tqdm(d.itertuples(), total=len(d), desc="judge"):
        sys_p, user_p = p.render(query=r.query, expected_action=r.expected_action,
                                 reference=r.reference_response if isinstance(r.reference_response, str) else "(none)",
                                 action=r.action, answer=r.answer)
        j = parse_json(judge_llm.chat(sys_p, user_p, max_tokens=200).text)
        rows.append({"eval_id": r.eval_id, "system": r.run_tag, **{k: j.get(k) for k in ITEMS},
                     "rationale": j.get("rationale"), "judge_version": p.version})
    return pd.DataFrame(rows)


def make_human_sheet(runs: dict[str, pd.DataFrame], eval_df: pd.DataFrame, subset_ids: list[str],
                     seed: int = C.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (sheet for reviewers, private key mapping blind label -> system). Keep the key away from reviewers."""
    for name, r in runs.items():
        missing = set(subset_ids) - set(r["eval_id"])
        if missing:
            raise ValueError(f"Run '{name}' is missing {len(missing)} subset queries - finish the full run first "
                             f"(run_system with limit=None).")
    rng = np.random.default_rng(seed)
    ev = eval_df.set_index("eval_id")
    sheet, key = [], []
    for eid in subset_ids:
        names = list(runs)
        rng.shuffle(names)
        for i, name in enumerate(names, start=1):
            r = runs[name].set_index("eval_id").loc[eid]
            label = f"S{i}"
            sheet.append({"eval_id": eid, "blind_system": label, "query": ev.at[eid, "query"],
                          "expected_action": ev.at[eid, "expected_action"],
                          "reference_response": ev.at[eid, "reference_response"],
                          "action": r["action"], "answer": r["answer"], **{k: "" for k in ITEMS}, "notes": ""})
            key.append({"eval_id": eid, "blind_system": label, "system": name})
    return pd.DataFrame(sheet), pd.DataFrame(key)


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Cohen's kappa (quadratic-weighted for 0-2 items) + raw agreement per rubric item."""
    m = a.merge(b, on=["eval_id", "blind_system"], suffixes=("_a", "_b"))
    out = []
    for k in ITEMS:
        x, y = pd.to_numeric(m[f"{k}_a"], errors="coerce"), pd.to_numeric(m[f"{k}_b"], errors="coerce")
        ok = x.notna() & y.notna()
        w = "quadratic" if k in ("relevance", "step_coverage", "clarity") else None
        kappa = cohen_kappa_score(x[ok].astype(int), y[ok].astype(int), weights=w) if ok.sum() > 1 else np.nan
        out.append({"item": k, "n": int(ok.sum()), "raw_agreement": round(float((x[ok] == y[ok]).mean()), 3),
                    "kappa": round(float(kappa), 3)})
    return pd.DataFrame(out)


def judge_vs_human(judge: pd.DataFrame, human: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Agreement between the LLM judge and the adjudicated human scores (same items)."""
    h = human.merge(key, on=["eval_id", "blind_system"])[["eval_id", "system"] + ITEMS]
    j = judge[["eval_id", "system"] + ITEMS]
    common = h.merge(j[["eval_id", "system"]], on=["eval_id", "system"])
    as_blind = lambda d: d.merge(common, on=["eval_id", "system"]).rename(columns={"system": "blind_system"})
    return agreement(as_blind(h), as_blind(j))


def score_table(judged: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Mean rubric scores per system x eval_set (0-2 items as means, 0/1 items as rates)."""
    d = judged.merge(eval_df[["eval_id", "eval_set"]], on="eval_id")
    for k in ITEMS:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    return d.groupby(["system", "eval_set"])[ITEMS].mean().round(3)
