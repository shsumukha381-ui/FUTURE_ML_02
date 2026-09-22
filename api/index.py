"""
Vercel Python Serverless Function — Support Ticket Classifier API.

This module wraps the ML prediction pipeline as a single FastAPI ASGI app
served by Vercel's Python runtime. All routes are prefixed with /api/.

Model artifacts are loaded ONCE per cold start at module level to minimize
latency. Subsequent requests reuse the loaded models until the serverless
function is recycled (typically after ~5–15 minutes of inactivity).

Cold start tradeoff:
    First request after idle: ~3-5s (loads scikit-learn + model files)
    Subsequent requests:      ~50-200ms (model already in memory)
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Text Cleaning — lightweight, no NLTK dependency
#
# The training pipeline uses NLTK for stopwords + lemmatization, but for
# serverless deployment we use a hardcoded stopword list and skip lemmatization.
# This avoids bundling ~15MB of NLTK data and eliminates a cold-start penalty.
# The TF-IDF vectorizer is robust to this: unrecognized word forms are simply
# ignored, and the vocabulary learned during training covers the common forms.
# ---------------------------------------------------------------------------
STOPWORDS = frozenset({
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
    'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
    'also', 'could', 'would', 'may', 'might', 'shall', 'need', 'must', 'let',
    'like', 'get', 'got', 'go', 'going', 'went', 'come', 'came', 'take', 'took',
    'make', 'made', 'know', 'knew', 'think', 'thought', 'say', 'said', 'tell',
    'told', 'use', 'used', 'find', 'found', 'give', 'gave', 'want', 'wanted',
    'see', 'seen', 'look', 'looked', 'put', 'set', 'keep', 'kept', 'try', 'tried',
    'still', 'since', 'back', 'well', 'even', 'much', 'many', 'way', 'long',
    'please', 'thank', 'thanks', 'hi', 'hello', 'dear', 'regards', 'sir', 'madam',
    'new', 'old', 'one', 'two', 'first', 'last', 'time', 'day', 'work', 'able',
})

_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_EMAIL_RE = re.compile(r'\S+@\S+')
_NONALPHA_RE = re.compile(r'[^a-z\s]')


def clean_text(text: str) -> str:
    """
    Clean ticket text for ML prediction (serverless-friendly, no NLTK).

    Matches the training pipeline's cleaning as closely as possible:
    lowercase → remove URLs → remove emails → remove special chars →
    remove stopwords. Lemmatization is skipped to avoid NLTK dependency.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub('', text)
    text = _EMAIL_RE.sub('', text)
    text = _NONALPHA_RE.sub(' ', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return ' '.join(words)


# ---------------------------------------------------------------------------
# Model Loading — once per cold start (module-level)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

_models = {
    "vectorizer": None,
    "category_model": None,
    "priority_model": None,
    "metadata": None,
    "loaded": False,
}


def _load_models():
    """Load serialized model artifacts from disk. Called once per cold start."""
    if _models["loaded"]:
        return
    try:
        logger.info("Loading model artifacts from %s ...", MODELS_DIR)
        _models["vectorizer"] = joblib.load(MODELS_DIR / "vectorizer.joblib")
        _models["category_model"] = joblib.load(MODELS_DIR / "category_model.joblib")
        _models["priority_model"] = joblib.load(MODELS_DIR / "priority_model.joblib")
        with open(MODELS_DIR / "metadata.json") as f:
            _models["metadata"] = json.load(f)
        _models["loaded"] = True
        logger.info(
            "Models loaded (version %s, trained %s)",
            _models["metadata"].get("version"),
            _models["metadata"].get("trained_at"),
        )
    except Exception as e:
        logger.error("Failed to load models: %s", e)
        _models["loaded"] = False


# Load at import time → Vercel cold start
_load_models()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class TicketInput(BaseModel):
    """Single ticket prediction request."""
    text: str = Field(..., min_length=3, max_length=50000)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Ticket text cannot be empty.")
        return v.strip()


class BatchTicketInput(BaseModel):
    """Batch prediction request."""
    tickets: List[TicketInput] = Field(..., min_length=1, max_length=1000)


class PredictionResult(BaseModel):
    """Prediction response for a single ticket."""
    text: str
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    category_probabilities: Dict[str, float]
    priority_probabilities: Dict[str, float]


class BatchPredictionResult(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResult]
    total: int


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Support Ticket Classifier API",
    description="Classifies IT support tickets into categories and priority levels.",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# CORS — permissive for same-origin Vercel + local dev + file:// origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Prediction Helper
# ---------------------------------------------------------------------------
def _ensure_loaded():
    """Retry model loading and raise 503 if still unavailable."""
    if not _models["loaded"]:
        _load_models()
    if not _models["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. The serverless function could not find model artifacts.",
        )


def _predict_single(text: str) -> PredictionResult:
    """Run category + priority prediction on a single ticket."""
    cleaned = clean_text(text)
    if not cleaned:
        cleaned = text.lower().strip()

    X = _models["vectorizer"].transform([cleaned])

    # Category
    cat_model = _models["category_model"]
    cat_pred = cat_model.predict(X)[0]
    cat_proba = cat_model.predict_proba(X)[0]
    cat_classes = cat_model.classes_
    cat_conf = float(np.max(cat_proba))
    cat_probs = {str(c): round(float(p), 4) for c, p in zip(cat_classes, cat_proba)}

    # Priority
    pri_model = _models["priority_model"]
    pri_pred = pri_model.predict(X)[0]
    pri_proba = pri_model.predict_proba(X)[0]
    pri_classes = pri_model.classes_
    pri_conf = float(np.max(pri_proba))
    pri_probs = {str(c): round(float(p), 4) for c, p in zip(pri_classes, pri_proba)}

    display_text = text[:500] + ("..." if len(text) > 500 else "")

    return PredictionResult(
        text=display_text,
        category=str(cat_pred),
        category_confidence=round(cat_conf, 4),
        priority=str(pri_pred),
        priority_confidence=round(pri_conf, 4),
        category_probabilities=cat_probs,
        priority_probabilities=pri_probs,
    )


# ---------------------------------------------------------------------------
# Endpoints (all prefixed with /api/ for Vercel routing)
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Health check — returns model loading status."""
    return {
        "status": "healthy" if _models["loaded"] else "degraded",
        "model_loaded": _models["loaded"],
    }


@app.get("/api/model-info")
async def model_info():
    """Return model metadata and evaluation metrics."""
    _ensure_loaded()
    meta = _models["metadata"]
    eval_data = meta.get("evaluation", {})
    return {
        "version": meta.get("version", "1.0.0"),
        "trained_at": meta.get("trained_at"),
        "category_model": meta.get("category_model"),
        "priority_model": meta.get("priority_model"),
        "category_metrics": eval_data.get("category", {}),
        "priority_metrics": eval_data.get("priority", {}),
    }


@app.post("/api/predict", response_model=PredictionResult)
async def predict(ticket: TicketInput):
    """Classify a single support ticket."""
    _ensure_loaded()
    logger.info("Prediction request — %d chars", len(ticket.text))
    result = _predict_single(ticket.text)
    logger.info(
        "Prediction: category=%s (%.2f), priority=%s (%.2f)",
        result.category, result.category_confidence,
        result.priority, result.priority_confidence,
    )
    return result


@app.post("/api/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(batch: BatchTicketInput):
    """Classify multiple tickets from a JSON list (max 1000)."""
    _ensure_loaded()
    texts = [t.text for t in batch.tickets]
    if len(texts) > 1000:
        raise HTTPException(400, f"Batch size {len(texts)} exceeds maximum of 1000.")
    logger.info("Batch prediction — %d tickets", len(texts))
    predictions = [_predict_single(t) for t in texts]
    logger.info("Batch complete — %d processed", len(predictions))
    return BatchPredictionResult(predictions=predictions, total=len(predictions))
