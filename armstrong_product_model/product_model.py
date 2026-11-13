"""Product recommendation model (step 2 of the authentication system).

Steps:
    1. Clean customer_transactions.csv and customer_social_profiles.csv
    2. Align the two customer IDs (151 and A151 -> 151)
    3. Aggregate social profiles to one row per customer, then left-join onto transactions
    4. Engineer purchase-history features (only from purchases made BEFORE each transaction)
    5. Compare Logistic Regression and Random Forest with customer-grouped cross-validation
    6. Save merged_dataset.csv, the trained model, metrics and a confusion matrix

Usage:
    python product_model.py train            # rebuild the data, evaluate, save the model
    python product_model.py predict 188      # recommended product for a customer

The CLI (main.py) calls predict(customer_id).
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             log_loss)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
TRANSACTIONS_FILE = ROOT / "customer_transactions.csv"
SOCIAL_FILE = ROOT / "customer_social_profiles.csv"
MERGED_FILE = ROOT / "merged_dataset.csv"
MODEL_FILE = ROOT / "product_recommendation_model.joblib"
METRICS_FILE = ROOT / "product_metrics.json"
CONFUSION_FILE = ROOT / "product_confusion_matrix.png"

TARGET = "product_category"
NUMERIC_FEATURES = [
    "engagement_score_mean",
    "purchase_interest_score_mean",
    "social_platform_count",
    "prior_purchase_count",
    "prior_avg_purchase_amount",
    "prior_avg_customer_rating",
]
CATEGORICAL_FEATURES = [
    "social_media_platform_most_common_value",
    "review_sentiment_most_common_value",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
N_FOLDS = 5
SEED = 42


# 1. Cleaning
def load_clean():
    """Return the cleaned transactions and social profiles, each with a numeric customer_id."""
    transactions = pd.read_csv(TRANSACTIONS_FILE).drop_duplicates()
    social = pd.read_csv(SOCIAL_FILE).drop_duplicates()

    transactions["purchase_date"] = pd.to_datetime(transactions["purchase_date"], errors="coerce")
    for col in ["purchase_amount", "customer_rating"]:
        transactions[col] = pd.to_numeric(transactions[col], errors="coerce")
        transactions[col] = transactions[col].fillna(transactions[col].median())
    for col in ["engagement_score", "purchase_interest_score"]:
        social[col] = pd.to_numeric(social[col], errors="coerce")

    # 2. Align the IDs: transactions use 151, social profiles use A151
    transactions["customer_id"] = pd.to_numeric(transactions["customer_id_legacy"], errors="coerce")
    social["customer_id"] = pd.to_numeric(
        social["customer_id_new"].astype(str).str.replace("A", "", regex=False), errors="coerce")
    return transactions, social


def most_common_value(series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else "Unknown"


def classify_engagement(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


# 3. Aggregate + merge
def aggregate_social(social):
    """One row per customer, so the join cannot duplicate transactions."""
    agg = social.groupby("customer_id").agg({
        "social_media_platform": ["nunique", most_common_value],
        "engagement_score": "mean",
        "purchase_interest_score": "mean",
        "review_sentiment": most_common_value,
    }).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg["social_platform_count"] = agg["social_media_platform_nunique"]
    agg["engagement_level"] = agg["engagement_score_mean"].apply(classify_engagement)
    return agg


# 4. Feature engineering
def add_history_features(transactions):
    """Features computed only from each customer's EARLIER purchases (no future leakage)."""
    t = transactions.sort_values(["customer_id", "purchase_date"]).copy()
    grouped = t.groupby("customer_id")
    t["prior_purchase_count"] = grouped.cumcount()
    t["prior_avg_purchase_amount"] = grouped["purchase_amount"].transform(lambda x: x.shift(1).expanding().mean())
    t["prior_avg_customer_rating"] = grouped["customer_rating"].transform(lambda x: x.shift(1).expanding().mean())
    return t


def build_dataset(save=True):
    transactions, social = load_clean()
    merged = add_history_features(transactions).merge(aggregate_social(social), on="customer_id", how="left")

    # Post-merge checks
    assert len(merged) == len(transactions), "the join changed the number of transactions"
    assert not merged["transaction_id"].duplicated().any(), "duplicate transactions after the merge"

    if save:
        merged.to_csv(MERGED_FILE, index=False)
    return merged


# 5. Models and evaluation
def make_pipeline(classifier):
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
         NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


# Strong regularisation on purpose: with 150 transactions and weak signal (see the EDA notebook),
# flexible models become overconfident and their log-loss gets worse than random guessing.
CANDIDATES = {
    "LogisticRegression": lambda: LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                                   class_weight="balanced", random_state=SEED),
}


def cross_validate(data, make_classifier):
    """Customer-grouped K-fold: all purchases of a customer stay on the same side of each split."""
    X, y, groups = data[FEATURES], data[TARGET].astype(str), data["customer_id"]
    classes = sorted(y.unique())
    y_true, y_pred, proba = [], [], []
    for train_idx, test_idx in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        model = make_pipeline(make_classifier()).fit(X.iloc[train_idx], y.iloc[train_idx])
        p = pd.DataFrame(model.predict_proba(X.iloc[test_idx]), columns=model.classes_)
        p = p.reindex(columns=classes, fill_value=0.0)
        y_true += list(y.iloc[test_idx])
        y_pred += list(model.predict(X.iloc[test_idx]))
        proba.append(p.to_numpy())
    proba = np.clip(np.vstack(proba), 1e-15, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=classes),
    }, y_true, y_pred, classes


def save_confusion_matrix(y_true, y_pred, classes, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted product")
    ax.set_ylabel("Actual product")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(CONFUSION_FILE, dpi=120)
    plt.close(fig)


def train():
    data = build_dataset()
    n_classes = data[TARGET].nunique()

    results = {name: cross_validate(data, make) for name, make in CANDIDATES.items()}
    best = min(results, key=lambda n: (-results[n][0]["f1_macro"], results[n][0]["log_loss"]))
    scores, y_true, y_pred, classes = results[best]
    save_confusion_matrix(y_true, y_pred, classes, f"{best}: {N_FOLDS}-fold customer-grouped CV")

    final = make_pipeline(CANDIDATES[best]()).fit(data[FEATURES], data[TARGET].astype(str))
    joblib.dump({"model": final, "features": FEATURES, "classes": list(final.classes_), "name": best}, MODEL_FILE)

    metrics = {
        "evaluation": f"{N_FOLDS}-fold GroupKFold by customer ({len(data)} transactions, "
                      f"{data['customer_id'].nunique()} customers)",
        "chance_accuracy": round(1 / n_classes, 3),
        "chance_log_loss": round(float(np.log(n_classes)), 3),
        "selected_model": best,
        "models": {n: {k: round(float(v), 3) for k, v in r[0].items()} for n, r in results.items()},
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    return metrics


# 6. Prediction (used by the CLI app)
_bundle = None


def customer_profile(customer_id, data=None):
    """Feature row describing a customer's NEXT purchase: their full history so far + social profile."""
    data = build_dataset(save=False) if data is None else data
    history = data[data["customer_id"] == int(customer_id)]
    if history.empty:
        raise KeyError(f"customer {customer_id} not found in the transactions")
    row = history.sort_values("purchase_date").iloc[[-1]].copy()
    row["prior_purchase_count"] = len(history)
    row["prior_avg_purchase_amount"] = history["purchase_amount"].mean()
    row["prior_avg_customer_rating"] = history["customer_rating"].mean()
    return row[FEATURES]


def predict(customer_id, top_k=1):
    """Recommended product category for a customer (top_k > 1 returns a ranked list with probabilities)."""
    global _bundle
    if _bundle is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"{MODEL_FILE.name} not found; run: python product_model.py train")
        _bundle = joblib.load(MODEL_FILE)
    model = _bundle["model"]
    proba = model.predict_proba(customer_profile(customer_id))[0]
    ranked = sorted(zip(model.classes_, proba), key=lambda x: -x[1])
    return ranked[0][0] if top_k == 1 else [(c, round(float(p), 3)) for c, p in ranked[:top_k]]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        m = train()
        print(m["evaluation"])
        print(f"Chance level: accuracy {m['chance_accuracy']}, log-loss {m['chance_log_loss']}")
        for name, s in m["models"].items():
            print(f"{name:<20} accuracy {s['accuracy']:.3f} | macro F1 {s['f1_macro']:.3f} | "
                  f"weighted F1 {s['f1_weighted']:.3f} | log-loss {s['log_loss']:.3f}")
        print("Selected:", m["selected_model"], "->", MODEL_FILE.name)
    elif len(sys.argv) == 3 and sys.argv[1] == "predict":
        for product, p in predict(sys.argv[2], top_k=3):
            print(f"{product:<12} {p:.0%}")
    else:
        print(__doc__)
