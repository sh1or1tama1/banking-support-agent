"""Task 1 CHALLENGER #1: fine-tuned BERT (bert-base-uncased, 26-way head).  Owner: member C.

Runs on a single NVIDIA T4 (Colab) in roughly 5-15 minutes with fp16. Settings are deliberately
standard so they are easy to defend:
  max_length 64 (covers >99% of queries; check in EDA), lr 2e-5, batch 32, up to 4 epochs,
  weight decay 0.01, warm-up 10%, best epoch chosen on VALIDATION macro-F1, seed 42.

NOTE: not executed in the offline build environment (no GPU / model download there);
run it in Colab and record the environment (GPU, versions) in the report.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import config as C


def train_bert(train: pd.DataFrame, val: pd.DataFrame, out_dir=None, epochs: int = 4, lr: float = 2e-5,
               batch_size: int = 32, max_length: int = 64):
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    set_seed(C.SEED)
    out_dir = out_dir or str(C.OUTPUTS / "bert_intent")
    labels = sorted(train["intent"].unique())
    l2i = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(C.BERT_MODEL)

    def to_ds(d: pd.DataFrame):
        ds = Dataset.from_dict({"text": d["instruction"].tolist(), "label": [l2i[x] for x in d["intent"]]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_length), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        C.BERT_MODEL, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i)

    def compute_metrics(p):
        pred = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, pred), "macro_f1": f1_score(p.label_ids, pred, average="macro")}

    warmup = int(0.1 * epochs * -(-len(train) // batch_size))  # 10% of optimisation steps
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr, weight_decay=0.01, warmup_steps=warmup,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=128,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1,
        fp16=torch.cuda.is_available(), seed=C.SEED, report_to="none", logging_steps=50,
    )
    trainer = Trainer(model=model, args=args, train_dataset=to_ds(train), eval_dataset=to_ds(val),
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    history = pd.DataFrame([h for h in trainer.state.log_history if "eval_macro_f1" in h])
    return trainer.model, tok, labels, history


class BertIntent:
    """Inference wrapper: .predict(texts) and .as_tool() for the Agentic RAG."""

    def __init__(self, model_dir=None, max_length: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = model_dir or str(C.OUTPUTS / "bert_intent")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_length = max_length
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict_proba(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt").to(self.device)
                out.append(torch.softmax(self.model(**enc).logits, -1).cpu().numpy())
        return np.vstack(out)

    def predict(self, texts: list[str]) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        p = self.predict_proba(texts)
        ms = (time.perf_counter() - t0) / max(1, len(texts)) * 1000
        return np.array(self.labels)[p.argmax(1)], ms

    def as_tool(self):
        def tool(query: str) -> tuple[str, float]:
            p = self.predict_proba([query])[0]
            return self.labels[int(p.argmax())], float(p.max())
        return tool
