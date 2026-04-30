"""
train.py  –  Moral Compass Classifier  –  Training Pipeline
=============================================================
Run from the project root:
    python src/train.py

Steps
-----
1. Load data/moral_dataset.csv  (or synthetic fallback)
2. Clean text
3. TF-IDF vectorisation (unigrams + bigrams)
4. Train Logistic Regression and linear SVM
5. Evaluate with accuracy, classification report, confusion matrix
6. Save best model + vectoriser to models/
"""

from __future__ import annotations

import os
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  (relative to the project root, not this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "moral_dataset.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase and strip punctuation."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """
    Load the moral dataset CSV.
    Supports both column layouts:
      • 'text' + 'label'       (synthetic-only output from data_generation.py)
      • 'text' + 'true_label'  (merged / kaggle-merged dataset)
    """
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at '{DATA_PATH}'.")
        print("        Run:  python src/data_generation.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO]  Loaded {len(df):,} rows from {DATA_PATH}")
    print(f"[INFO]  Columns: {df.columns.tolist()}")

    # Normalise label column name
    if "true_label" in df.columns:
        df = df.rename(columns={"true_label": "label"})
    elif "label" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'label' or 'true_label' column. "
            f"Found: {df.columns.tolist()}"
        )

    # Drop rows with missing text or label
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip()

    print("[INFO]  Label distribution:")
    print(df["label"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train_models() -> None:
    df = load_dataset()

    # Clean text
    df["cleaned_text"] = df["text"].apply(clean_text)

    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO]  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # -----------------------------------------------------------------------
    # TF-IDF vectorisation (unigrams + bigrams, max 5 000 features)
    # -----------------------------------------------------------------------
    print("\n[INFO]  Fitting TF-IDF vectoriser …")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5_000,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=42,
        ),
        "SVM (linear)": SVC(
            kernel="linear",
            class_weight="balanced",
            probability=True,   # enables predict_proba → confidence scores
            random_state=42,
        ),
    }

    best_model      = None
    best_acc        = 0.0
    best_model_name = ""

    for name, model in models.items():
        print(f"\n[TRAIN] {name} …")
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        acc    = accuracy_score(y_test, y_pred)

        print(f"  Accuracy  : {acc:.4f}")
        print("  Classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("  Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring="accuracy")
        print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.4f}")

        if acc > best_acc:
            best_acc        = acc
            best_model      = model
            best_model_name = name

    # -----------------------------------------------------------------------
    # Save artefacts
    # -----------------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model,  os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(vectorizer,  os.path.join(MODELS_DIR, "vectorizer.pkl"))

    print(f"\n[DONE]  Best model: {best_model_name}  (test acc = {best_acc:.4f})")
    print(f"[DONE]  Saved to '{MODELS_DIR}/'")


if __name__ == "__main__":
    train_models()
