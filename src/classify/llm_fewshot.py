"""Task 1 CHALLENGER #2: prompt-based LLM intent classification (no training).  Owner: member C.

Few-shot prompt = list of the 26 intents + k labelled examples per intent taken from the TRAIN split.
Output is constrained to JSON {"intent": "..."}; anything outside the label set counts as an error
(reported separately as 'invalid_label_rate').
"""
from __future__ import annotations

import pandas as pd

from .. import config as C
from ..llm.client import LLMClient, parse_json
from ..utils import sample_per_group

SYSTEM = """You classify customer messages for a retail bank into exactly one intent.
Allowed intents (category: intent):
{label_list}

Examples:
{examples}

Return JSON with keys intent (one of the allowed intent names, copied exactly)."""

USER = 'Customer message: """{query}"""'


class LLMFewShotClassifier:
    def __init__(self, llm: LLMClient, train: pd.DataFrame, k_per_intent: int = 2, seed: int = C.SEED):
        self.llm = llm
        self.labels = sorted(train["intent"].unique())
        cats = train.drop_duplicates("intent").set_index("intent")["category"]
        label_list = "\n".join(f"- {cats[i]}: {i}" for i in self.labels)
        shots = sample_per_group(train, "intent", k_per_intent, seed)
        examples = "\n".join(f'"{r.instruction}" -> {r.intent}' for r in shots.itertuples())
        self.system = SYSTEM.format(label_list=label_list, examples=examples)
        self.version = f"llm_fewshot@v1(k={k_per_intent})"

    def predict_one(self, query: str) -> dict:
        r = self.llm.chat(self.system, USER.format(query=query), max_tokens=30)
        intent = str(parse_json(r.text).get("intent", "")).strip()
        return {"pred": intent if intent in self.labels else "INVALID", "raw": r.text,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd, "latency_s": r.latency_s}

    def predict(self, queries: list[str]) -> pd.DataFrame:
        from tqdm.auto import tqdm

        return pd.DataFrame([self.predict_one(q) for q in tqdm(queries, desc="llm-fewshot")])
