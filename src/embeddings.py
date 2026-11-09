"""Sentence embeddings with an offline fallback.

Primary : sentence-transformers (config.EMBED_MODEL, default all-MiniLM-L6-v2) - use in Colab.
Fallback: TF-IDF (word 1-2 gram) + TruncatedSVD, L2-normalised. Only for offline smoke tests;
          never report fallback numbers as the SBERT results.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from . import config as C


class Embedder:
    def __init__(self, model_name: str = C.EMBED_MODEL, allow_fallback: bool = True):
        self.model_name = model_name
        self.backend = None
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.backend = f"sbert:{model_name}"
        except Exception as e:  # no package / no internet
            if not allow_fallback:
                raise
            warnings.warn(f"sentence-transformers unavailable ({type(e).__name__}); using TF-IDF+SVD fallback.")
            self.backend = "tfidf_svd_fallback"

    def fit(self, corpus: list[str]) -> "Embedder":
        """Only the fallback needs fitting (on the TRAIN corpus, to avoid leakage)."""
        if self.backend == "tfidf_svd_fallback":
            n_comp = max(2, min(256, len(corpus) - 1))
            self._model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                TruncatedSVD(n_components=n_comp, random_state=C.SEED),
                Normalizer(),
            ).fit(corpus)
        return self

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        texts = [str(t) for t in texts]
        if self.backend == "tfidf_svd_fallback":
            if self._model is None:
                raise RuntimeError("Call .fit(train_corpus) first when using the fallback backend.")
            return self._model.transform(texts).astype(np.float32)
        return self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 2000
        ).astype(np.float32)
