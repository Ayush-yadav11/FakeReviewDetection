"""Train and evaluate the fake review detector, then export the browser model.

Reproduces the pipeline in main.ipynb end to end:
    punctuation stripped -> English stopwords dropped -> word counts -> TF-IDF -> classifier
on a fixed 75/25 split, for the notebook's three classifiers (logistic regression,
SVC and random forest). Results go to results/metrics.json.

The logistic regression is also exported to site/model.json, so the project page can
score a review in the browser with the same weights (see site/detector.js).

Labels in the dataset: CG = computer-generated (fake), OR = original (written by a person).

Usage:
    pip install -r requirements.txt
    python train.py            # all three models (SVC and the forest take a few minutes)
    python train.py --fast     # logistic regression only
"""

import argparse
import json
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "fake reviews dataset.csv"
SEED = 42

try:
    from nltk.corpus import stopwords

    STOPWORDS = frozenset(stopwords.words("english"))
except LookupError:
    import nltk

    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    STOPWORDS = frozenset(stopwords.words("english"))

PUNCT = frozenset(string.punctuation)


def tokenize(review: str) -> list[str]:
    """The notebook's convertmyTxt: drop punctuation, split on whitespace, drop stopwords.

    Case is kept (as in the notebook); stopwords are matched in lower case.
    A set lookup replaces the notebook's stopwords.words() call per word, which made
    training take hours.
    """
    text = "".join(c for c in review if c not in PUNCT)
    return [w for w in text.split() if w.lower() not in STOPWORDS]


def make_pipeline(classifier, min_df: int = 1) -> Pipeline:
    return Pipeline(
        [
            ("bow", CountVectorizer(analyzer=tokenize, min_df=min_df)),
            ("tfidf", TfidfTransformer()),
            ("classifier", classifier),
        ]
    )


def evaluate(name, pipe, x_train, x_test, y_train, y_test):
    start = time.perf_counter()
    pipe.fit(x_train, y_train)
    fit_s = time.perf_counter() - start
    pred = pipe.predict(x_test)
    report = classification_report(y_test, pred, output_dict=True, digits=4)
    cm = confusion_matrix(y_test, pred, labels=["CG", "OR"]).tolist()
    acc = accuracy_score(y_test, pred)
    print(f"{name:<20} accuracy {acc * 100:.2f}%  (fit {fit_s:.0f}s)")
    return {
        "accuracy": round(acc, 4),
        "fit_seconds": round(fit_s, 1),
        "confusion_matrix": {"labels": ["CG", "OR"], "rows_true_cols_pred": cm},
        "per_class": {
            k: {m: round(v, 4) for m, v in report[k].items()} for k in ("CG", "OR")
        },
    }


def export_browser_model(pipe: Pipeline, path: Path, metrics: dict) -> None:
    """Write vocabulary, idf and coefficients so detector.js can score a review."""
    bow = pipe.named_steps["bow"]
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["classifier"]
    vocab = bow.vocabulary_
    terms = sorted(vocab, key=vocab.get)
    idf = tfidf.idf_
    coef = clf.coef_[0]  # positive -> clf.classes_[1]
    model = {
        "about": "Logistic regression on TF-IDF word weights, trained by train.py on 75% of the fake reviews dataset.",
        "positive_class": str(clf.classes_[1]),
        "negative_class": str(clf.classes_[0]),
        "intercept": round(float(clf.intercept_[0]), 6),
        "stopwords": sorted(STOPWORDS),
        "terms": terms,
        "idf": [round(float(idf[vocab[t]]), 4) for t in terms],
        "coef": [round(float(coef[vocab[t]]), 4) for t in terms],
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(f"browser model: {len(terms):,} terms -> {path} ({path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="logistic regression only")
    args = parser.parse_args()

    df = pd.read_csv(DATA)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")]).dropna()
    print(f"{len(df):,} reviews  {df['label'].value_counts().to_dict()}")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text_"], df["label"], test_size=0.25, random_state=SEED, stratify=df["label"]
    )

    results = {
        "dataset": {
            "reviews": int(len(df)),
            "labels": {k: int(v) for k, v in df["label"].value_counts().items()},
            "categories": {k: int(v) for k, v in df["category"].value_counts().items()},
            "train": int(len(x_train)),
            "test": int(len(x_test)),
            "split_seed": SEED,
        },
        "models": {},
    }

    models = {"LogisticRegression": LogisticRegression(max_iter=2000)}
    if not args.fast:
        models["SVC"] = SVC()
        models["RandomForest"] = RandomForestClassifier(n_jobs=-1, random_state=SEED)

    for name, clf in models.items():
        results["models"][name] = evaluate(name, make_pipeline(clf), x_train, x_test, y_train, y_test)

    # The page's model: the same pipeline with terms seen in at least 3 training reviews,
    # so the file stays small enough to load in a browser. Scored on the same test quarter.
    browser = make_pipeline(LogisticRegression(max_iter=2000), min_df=3)
    results["browser_model"] = evaluate("Browser model", browser, x_train, x_test, y_train, y_test)
    results["browser_model"]["terms"] = len(browser.named_steps["bow"].vocabulary_)

    out = ROOT / "results" / "metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"metrics -> {out}")

    export_browser_model(browser, ROOT / "site" / "model.json", results["browser_model"])


if __name__ == "__main__":
    main()
