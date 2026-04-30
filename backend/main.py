"""
main.py  –  Moral Compass Classifier  –  FastAPI Backend
=========================================================
Run with:  uvicorn backend.main:app --reload --port 8000
  OR       python backend/main.py

Endpoints
---------
  POST /predict   – classify a moral scenario
  GET  /health    – liveness probe
  GET  /classes   – list supported label classes
"""

from __future__ import annotations

import os
import re
import sys

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Resolve paths relative to this file so the API works from any cwd
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
VEC_PATH   = os.path.join(MODELS_DIR, "vectorizer.pkl")

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Moral Compass Classifier API",
    description=(
        "Classifies a moral scenario into one of three categories: "
        "**Utilitarian**, **Ethical**, or **Selfish**."
    ),
    version="1.0.0",
)

# Allow all origins during development – tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-load ML artifacts (loaded once on first request)
# ---------------------------------------------------------------------------
_MODEL: object | None      = None
_VECTORIZER: object | None = None


def load_artifacts() -> None:
    """Load model + vectorizer from disk (once)."""
    global _MODEL, _VECTORIZER
    if _MODEL is not None:
        return  # already loaded
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run `python src/train.py` from the project root first."
        )
    _MODEL      = joblib.load(MODEL_PATH)
    _VECTORIZER = joblib.load(VEC_PATH)


# ---------------------------------------------------------------------------
# Text cleaning  (must match src/train.py)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase and remove punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Explainability helper
# ---------------------------------------------------------------------------
def get_top_features(vec_row, predicted_class: str, top_n: int = 5) -> list[dict]:
    """
    Return the top *top_n* TF-IDF features (words/n-grams) that pushed the
    model toward *predicted_class*, along with signed contribution scores.
    Falls back to an empty list for kernel SVMs (no coef_).
    """
    if not hasattr(_MODEL, "coef_"):
        return []

    feature_names = np.array(_VECTORIZER.get_feature_names_out())
    class_idx     = int(np.where(_MODEL.classes_ == predicted_class)[0][0])

    # coef_ shape: (n_classes, n_features) for OvR; (1, n_features) for binary
    coefs = (
        _MODEL.coef_[class_idx]
        if _MODEL.coef_.ndim > 1
        else _MODEL.coef_[0]
    )

    # Ensure coefs is a dense 1D array (SVC returns a sparse matrix)
    if hasattr(coefs, "toarray"):
        coefs = coefs.toarray()[0]
    else:
        coefs = np.asarray(coefs).flatten()

    contributions = vec_row.toarray()[0] * coefs
    top_idxs      = np.argsort(contributions)[-top_n:][::-1]

    features = []
    for idx in top_idxs:
        score = float(contributions[idx])
        if score > 0:
            features.append({"word": str(feature_names[idx]), "score": round(score, 4)})

    return features


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class FeatureItem(BaseModel):
    word:  str
    score: float


class PredictResponse(BaseModel):
    prediction:      str
    confidence:      float          # 0.0 – 1.0
    confidence_pct:  str            # e.g. "87.3%"
    probabilities:   dict[str, float]  # per-class probabilities
    top_features:    list[FeatureItem]  # words driving the prediction


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Meta"])
def health():
    """Simple liveness probe."""
    return {"status": "ok", "model_loaded": _MODEL is not None}


@app.get("/classes", tags=["Meta"])
def classes():
    """Return the label classes the model was trained on."""
    try:
        load_artifacts()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"classes": _MODEL.classes_.tolist()}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """
    Classify a moral scenario and return the predicted label,
    confidence score, per-class probabilities, and top explanatory keywords.

    **Example request**
    ```json
    { "text": "I donated my kidney to a stranger to save their life." }
    ```

    **Example response**
    ```json
    {
      "prediction":     "Utilitarian",
      "confidence":     0.9123,
      "confidence_pct": "91.2%",
      "probabilities":  { "Ethical": 0.04, "Selfish": 0.05, "Utilitarian": 0.91 },
      "top_features":   [{ "word": "donated kidney", "score": 0.32 }]
    }
    ```
    """
    try:
        load_artifacts()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    cleaned = clean_text(req.text)
    vec_row = _VECTORIZER.transform([cleaned])

    # Per-class probabilities
    if hasattr(_MODEL, "predict_proba"):
        proba_arr  = _MODEL.predict_proba(vec_row)[0]
        proba_dict = {
            cls: round(float(p), 4)
            for cls, p in zip(_MODEL.classes_, proba_arr)
        }
        # Always use argmax of probabilities as the verdict so the label
        # matches the displayed probability bars (predict() can disagree
        # with predict_proba() in calibrated models).
        best_idx   = int(np.argmax(proba_arr))
        prediction = str(_MODEL.classes_[best_idx])
        confidence = float(proba_arr[best_idx])
    else:
        # SVM without probability=True — fall back to raw predict()
        prediction = str(_MODEL.predict(vec_row)[0])
        proba_dict = {cls: 0.0 for cls in _MODEL.classes_}
        proba_dict[prediction] = 1.0
        confidence = 1.0

    top_feats = get_top_features(vec_row, prediction)

    return PredictResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        confidence_pct=f"{confidence * 100:.1f}%",
        probabilities=proba_dict,
        top_features=[FeatureItem(**f) for f in top_feats],
    )


# ---------------------------------------------------------------------------
# Entry point  (python backend/main.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
