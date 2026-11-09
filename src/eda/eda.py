"""EDA building blocks. Each function returns a DataFrame and/or saves a figure to outputs/figures.

Covers every item the course rubric lists for EDA (15 pts):
  sources/record counts/missing values/schema  -> schema_table, quality_report
  token distribution, vocabulary, sequence len -> length_table, plot_lengths, vocab_stats
  charts: word cloud, frequency plots, embedding map, similarity heatmap
  label / topic distribution                   -> plot_label_distribution, plot_flags
  linguistic features: n-grams, POS, entities  -> top_ngrams, pos_entity_profile
  generated (synthetic) data: quality, diversity, bias, anomalies, edge cases
                                               -> template_diversity, placeholder_table, anomalies,
                                                  synthetic_vs_real
"""
from __future__ import annotations

import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

from .. import config as C
from ..utils import FLAG_MEANING, PLACEHOLDER_RE, parse_flags, placeholders, savefig, style_axes

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def words(text: str) -> list[str]:
    return TOKEN_RE.findall(PLACEHOLDER_RE.sub(" ", str(text)).lower())


# ------------------------------------------------------------------ 1. audit
def schema_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "null": df.isna().sum(),
            "empty_string": [(df[c].astype(str).str.strip() == "").sum() for c in df.columns],
            "n_unique": df.nunique(),
            "example": [str(df[c].dropna().iloc[0])[:60] if df[c].notna().any() else "" for c in df.columns],
        }
    )


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    norm = df["instruction"].str.lower().str.strip()
    resp_ph = df["response"].map(lambda t: len(placeholders(t)))
    items = {
        "rows": len(df),
        "intents": df["intent"].nunique(),
        "categories": df["category"].nunique(),
        "exact duplicate instructions": int(df["instruction"].duplicated().sum()),
        "duplicate instructions (case/space-insensitive)": int(norm.duplicated().sum()),
        "duplicate (instruction, response) pairs": int(df.duplicated(["instruction", "response"]).sum()),
        "duplicate responses": int(df["response"].duplicated().sum()),
        "instructions mapped to >1 intent": int((df.groupby(norm)["intent"].nunique() > 1).sum()),
        "responses containing {{placeholders}}": int((resp_ph > 0).sum()),
        "instructions containing {{placeholders}}": int(df["instruction"].map(lambda t: len(placeholders(t)) > 0).sum()),
        "rows with no flags": int(df["flags"].map(lambda f: len(parse_flags(f)) == 0).sum()),
    }
    return pd.DataFrame({"check": list(items), "value": list(items.values())})


# ------------------------------------------------------------------ 2. labels
def plot_label_distribution(df: pd.DataFrame, name: str = "eda_intent_distribution"):
    counts = df.groupby(["category", "intent"]).size().reset_index(name="n").sort_values(["category", "n"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(counts["intent"] + "  [" + counts["category"] + "]", counts["n"], color=C.PALETTE[0], height=0.7)
    style_axes(ax, "Rows per intent (grouped by category)", "rows", None)
    ax.grid(axis="x", color=C.GRID)
    ax.grid(axis="y", visible=False)
    savefig(fig, name)
    plt.close(fig)
    return counts


def plot_flags(df: pd.DataFrame, name: str = "eda_flag_distribution"):
    c = Counter(f for fl in df["flags"] for f in parse_flags(fl))
    t = pd.DataFrame(
        [(k, FLAG_MEANING[k], v, v / len(df)) for k, v in c.most_common()],
        columns=["flag", "meaning", "rows", "share"],
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(t["flag"] + "\n" + t["meaning"].str.replace(" ", "\n", n=1), t["share"], color=C.PALETTE[0], width=0.6)
    style_axes(ax, "Share of rows carrying each language-variation flag", None, "share of rows")
    ax.tick_params(axis="x", labelsize=7)
    savefig(fig, name)
    plt.close(fig)
    return t


# ------------------------------------------------------------------ 3. lengths & vocabulary
def length_table(df: pd.DataFrame, tokenizer=None) -> pd.DataFrame:
    """Word counts (+ subword counts if a HF tokenizer is passed, e.g. bert-base-uncased)."""
    out = pd.DataFrame(
        {
            "instruction_words": df["instruction"].map(lambda t: len(words(t))),
            "response_words": df["response"].map(lambda t: len(words(t))),
            "instruction_chars": df["instruction"].str.len(),
            "response_chars": df["response"].str.len(),
        }
    )
    if tokenizer is not None:
        out["instruction_subwords"] = [len(x) for x in tokenizer(list(df["instruction"]), add_special_tokens=False)["input_ids"]]
        out["response_subwords"] = [len(x) for x in tokenizer(list(df["response"]), add_special_tokens=False)["input_ids"]]
    return out


def plot_lengths(lengths: pd.DataFrame, name: str = "eda_length_histograms"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, color, title in (
        (axes[0], "instruction_words", C.PALETTE[0], "Customer query length"),
        (axes[1], "response_words", C.PALETTE[1], "Reference response length"),
    ):
        v = lengths[col]
        ax.hist(v, bins=40, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(v.median(), color=C.INK, linestyle="--", linewidth=1)
        ax.text(v.median(), ax.get_ylim()[1] * 0.92, f" median {v.median():.0f}", color=C.INK, fontsize=8)
        style_axes(ax, title, "words", "rows")
    savefig(fig, name)
    plt.close(fig)


def plot_response_length_by_intent(df: pd.DataFrame, name: str = "eda_response_length_by_intent"):
    d = df.assign(n=df["response"].map(lambda t: len(words(t))))
    order = d.groupby("intent")["n"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.boxplot([d.loc[d["intent"] == i, "n"] for i in order], vert=False, tick_labels=list(order),
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#dbe8f8", color=C.PALETTE[0]), medianprops=dict(color=C.INK))
    style_axes(ax, "Response length by intent (words)", "words", None)
    ax.grid(axis="x", color=C.GRID)
    ax.grid(axis="y", visible=False)
    savefig(fig, name)
    plt.close(fig)


def vocab_stats(texts: pd.Series, label: str) -> dict:
    toks = [w for t in texts for w in words(t)]
    c = Counter(toks)
    return {
        "corpus": label,
        "documents": len(texts),
        "tokens": len(toks),
        "vocabulary": len(c),
        "type_token_ratio": round(len(c) / max(1, len(toks)), 4),
        "hapax_legomena": sum(1 for v in c.values() if v == 1),
        "mean_words_per_doc": round(len(toks) / max(1, len(texts)), 2),
    }


def top_ngrams(texts: pd.Series, n: int = 1, k: int = 20, remove_stop: bool = True) -> pd.DataFrame:
    vec = CountVectorizer(
        ngram_range=(n, n), stop_words=list(ENGLISH_STOP_WORDS) if remove_stop else None,
        token_pattern=r"[A-Za-z0-9']{2,}", preprocessor=lambda t: PLACEHOLDER_RE.sub(" ", str(t)).lower(),
    )
    X = vec.fit_transform(texts)
    freq = np.asarray(X.sum(axis=0)).ravel()
    idx = freq.argsort()[::-1][:k]
    return pd.DataFrame({"ngram": vec.get_feature_names_out()[idx], "count": freq[idx]})


def plot_top_ngrams(texts: pd.Series, name: str = "eda_top_ngrams", k: int = 15):
    uni, bi = top_ngrams(texts, 1, k), top_ngrams(texts, 2, k)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, t, title, color in ((axes[0], uni, "Top unigrams", C.PALETTE[0]), (axes[1], bi, "Top bigrams", C.PALETTE[1])):
        ax.barh(t["ngram"][::-1], t["count"][::-1], color=color, height=0.7)
        style_axes(ax, title + " (queries, stop words removed)", "count", None)
        ax.grid(axis="x", color=C.GRID)
        ax.grid(axis="y", visible=False)
    savefig(fig, name)
    plt.close(fig)
    return uni, bi


def distinctive_terms_per_intent(df: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Top TF-IDF terms of each intent's concatenated queries (a 'topic' view of the labels)."""
    docs = df.groupby("intent")["instruction"].apply(lambda s: " ".join(s))
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z]{3,}", sublinear_tf=True)
    X = vec.fit_transform(docs).toarray()
    vocab = vec.get_feature_names_out()
    return pd.DataFrame({"intent": docs.index, "top_terms": [", ".join(vocab[r.argsort()[::-1][:k]]) for r in X]})


def plot_wordcloud(texts: pd.Series, name: str = "eda_wordcloud"):
    from wordcloud import WordCloud

    def blue(*_, **__):
        return C.PALETTE[0]

    wc = WordCloud(width=1200, height=500, background_color="white", stopwords=set(ENGLISH_STOP_WORDS),
                   color_func=blue, random_state=C.SEED, collocations=False)
    wc.generate(" ".join(PLACEHOLDER_RE.sub(" ", str(t)) for t in texts))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    savefig(fig, name)
    plt.close(fig)


# ------------------------------------------------------------------ 4. linguistic features
def pos_entity_profile(texts: pd.Series, sample: int = 3000, seed: int = C.SEED):
    """spaCy POS-tag distribution and named-entity label counts (on a sample, for speed)."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    s = texts.sample(min(sample, len(texts)), random_state=seed)
    pos, ents, ent_examples = Counter(), Counter(), {}
    for doc in nlp.pipe((PLACEHOLDER_RE.sub("X", str(t)) for t in s), batch_size=256):
        pos.update(t.pos_ for t in doc if not t.is_space)
        for e in doc.ents:
            ents[e.label_] += 1
            ent_examples.setdefault(e.label_, e.text)
    total = sum(pos.values())
    pos_df = pd.DataFrame([(k, v, v / total) for k, v in pos.most_common()], columns=["pos", "count", "share"])
    ent_df = pd.DataFrame([(k, v, ent_examples[k]) for k, v in ents.most_common()], columns=["entity", "count", "example"])
    return pos_df, ent_df


def plot_pos(pos_df: pd.DataFrame, name: str = "eda_pos_distribution"):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(pos_df["pos"], pos_df["share"], color=C.PALETTE[0], width=0.6)
    style_axes(ax, "Part-of-speech distribution (queries, spaCy)", None, "share of tokens")
    savefig(fig, name)
    plt.close(fig)


# ------------------------------------------------------------------ 5. embeddings
def plot_embedding_map(emb: np.ndarray, labels: pd.Series, name: str = "eda_embedding_map", method: str = "umap"):
    """2-D projection coloured by category. 8 palette slots; any 9th+ category is drawn grey."""
    if method == "umap":
        try:
            import umap

            xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=C.SEED).fit_transform(emb)
        except Exception:
            method = "pca"
    if method == "pca":
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=C.SEED).fit_transform(emb)
    cats = labels.value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for i, cat in enumerate(cats):
        m = (labels == cat).to_numpy()
        color = C.PALETTE[i] if i < len(C.PALETTE) else "#9a9893"
        ax.scatter(xy[m, 0], xy[m, 1], s=8, color=color, alpha=0.75, label=cat, linewidths=0)
    style_axes(ax, f"Query embeddings ({method.upper()}), coloured by category", None, None)
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, markerscale=2, loc="center left", bbox_to_anchor=(1, 0.5))
    savefig(fig, name)
    plt.close(fig)
    return xy


def plot_intent_similarity(emb: np.ndarray, intents: pd.Series, name: str = "eda_intent_similarity_heatmap"):
    """Cosine similarity between intent centroids - shows which intents are easy to confuse."""
    labs = sorted(intents.unique())
    cent = np.vstack([emb[(intents == l).to_numpy()].mean(axis=0) for l in labs])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    sim = cent @ cent.T
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(sim, cmap="Blues", vmin=max(0, sim.min()), vmax=1)
    ax.set_xticks(range(len(labs)), labs, rotation=90, fontsize=7, color=C.INK_2)
    ax.set_yticks(range(len(labs)), labs, fontsize=7, color=C.INK_2)
    ax.set_title("Intent centroid cosine similarity", loc="left", fontsize=12, fontweight="bold", color=C.INK)
    fig.colorbar(im, ax=ax, fraction=0.04)
    savefig(fig, name)
    plt.close(fig)
    s = pd.DataFrame(sim, index=labs, columns=labs)
    pairs = [(a, b, s.loc[a, b]) for i, a in enumerate(labs) for b in labs[i + 1:]]
    return pd.DataFrame(pairs, columns=["intent_a", "intent_b", "cosine"]).sort_values("cosine", ascending=False)


# ------------------------------------------------------------------ 6. synthetic-data checks
def template_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """How templated is the generated text? Share of rows sharing the most common opening/closing."""
    rows = []
    for col in ("instruction", "response"):
        first5 = df[col].map(lambda t: " ".join(words(t)[:5]))
        last5 = df[col].map(lambda t: " ".join(words(t)[-5:]))
        rows.append({
            "field": col,
            "distinct_texts_%": round(100 * df[col].nunique() / len(df), 2),
            "distinct_opening_5grams_%": round(100 * first5.nunique() / len(df), 2),
            "most_common_opening": first5.value_counts().index[0],
            "rows_with_that_opening_%": round(100 * first5.value_counts().iloc[0] / len(df), 2),
            "most_common_closing": last5.value_counts().index[0],
            "rows_with_that_closing_%": round(100 * last5.value_counts().iloc[0] / len(df), 2),
        })
    return pd.DataFrame(rows)


def placeholder_table(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    c = Counter(p for t in df["response"] for p in placeholders(t))
    return pd.DataFrame(c.most_common(k), columns=["placeholder", "occurrences"])


def anomalies(df: pd.DataFrame, k: int = 5) -> dict[str, pd.DataFrame]:
    """Edge cases worth showing in the report (and in Task-2 testing)."""
    w_ins = df["instruction"].map(lambda t: len(words(t)))
    w_res = df["response"].map(lambda t: len(words(t)))
    flagset = df["flags"].map(parse_flags)
    unbalanced = df["response"].map(lambda t: t.count("{{") != t.count("}}"))
    cols = ["row_id", "intent", "flags", "instruction", "response"]
    return {
        "shortest_queries": df.loc[w_ins.nsmallest(k).index, cols],
        "longest_queries": df.loc[w_ins.nlargest(k).index, cols],
        "shortest_responses": df.loc[w_res.nsmallest(k).index, cols],
        "offensive_flag_W": df.loc[flagset.map(lambda s: "W" in s), cols].head(k),
        "typo_flag_Z": df.loc[flagset.map(lambda s: "Z" in s), cols].head(k),
        "unbalanced_placeholder_braces": df.loc[unbalanced, cols].head(k),
    }


def synthetic_vs_real(bitext: pd.DataFrame, b77: pd.DataFrame, name: str = "eda_synthetic_vs_real_length"):
    """Compare Bitext (synthetic) queries with BANKING77 (real customer) queries."""
    a = bitext["instruction"].map(lambda t: len(words(t)))
    b = b77["text"].map(lambda t: len(words(t)))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.arange(0, max(a.quantile(0.99), b.quantile(0.99)) + 2)
    ax.hist(a, bins=bins, density=True, alpha=0.65, color=C.PALETTE[0], label="Bitext (synthetic)")
    ax.hist(b, bins=bins, density=True, alpha=0.65, color=C.PALETTE[1], label="BANKING77 (real)")
    style_axes(ax, "Query length: synthetic vs real customers", "words", "density")
    ax.legend(frameon=False)
    savefig(fig, name)
    plt.close(fig)
    return pd.DataFrame([vocab_stats(bitext["instruction"], "Bitext queries"), vocab_stats(b77["text"], "BANKING77 queries")])
