"""
FastAPI backend for the Support Ticket Classification & Prioritization system.

Endpoints:
    POST /predict        — Classify a single ticket
    POST /predict/batch  — Classify multiple tickets (JSON list or CSV upload)
    GET  /health         — Health check
    GET  /model-info     — Model version, training date, and metrics summary

Usage:
    cd support-ticket-classifier
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import sys
import csv
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Ensure the project root is on sys.path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import (
    TicketInput,
    BatchTicketInput,
    PredictionResult,
    BatchPredictionResult,
    HealthResponse,
    ModelInfoResponse,
)
from src.preprocessing import clean_text
from src.train import load_artifacts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
_model_state = {
    "vectorizer": None,
    "category_model": None,
    "priority_model": None,
    "metadata": None,
    "loaded": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup; cleanup on shutdown."""
    try:
        logger.info("Loading model artifacts...")
        vectorizer, cat_model, pri_model, metadata = load_artifacts()
        _model_state["vectorizer"] = vectorizer
        _model_state["category_model"] = cat_model
        _model_state["priority_model"] = pri_model
        _model_state["metadata"] = metadata
        _model_state["loaded"] = True
        logger.info(
            "Models loaded successfully (version %s, trained %s)",
            metadata["version"],
            metadata["trained_at"],
        )
    except FileNotFoundError as e:
        logger.error("Model artifacts not found: %s", e)
        logger.error("Run 'python -m src.run_training' to train models first.")
        _model_state["loaded"] = False
    yield
    logger.info("Shutting down API...")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Support Ticket Classifier API",
    description=(
        "Classifies support tickets into categories and priority levels "
        "using machine learning. Trained on IT service ticket data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend origin (default: localhost:5500, any localhost port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ensure_models_loaded():
    """Raise 503 if models aren't loaded."""
    if not _model_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models not loaded. Train models first with: "
                "python -m src.run_training"
            ),
        )


def _predict_single(text: str) -> PredictionResult:
    """Run prediction on a single ticket text."""
    # Clean text using the same pipeline as training
    cleaned = clean_text(text)
    if not cleaned.strip():
        # If cleaning removes everything, use original text lowered
        cleaned = text.lower().strip()

    # Vectorize
    X = _model_state["vectorizer"].transform([cleaned])

    # Category prediction
    cat_model = _model_state["category_model"]
    cat_pred = cat_model.predict(X)[0]
    cat_proba = cat_model.predict_proba(X)[0]
    cat_classes = cat_model.classes_
    cat_confidence = float(np.max(cat_proba))
    cat_probs_dict = {
        str(cls): round(float(prob), 4) for cls, prob in zip(cat_classes, cat_proba)
    }

    # Priority prediction
    pri_model = _model_state["priority_model"]
    pri_pred = pri_model.predict(X)[0]
    pri_proba = pri_model.predict_proba(X)[0]
    pri_classes = pri_model.classes_
    pri_confidence = float(np.max(pri_proba))
    pri_probs_dict = {
        str(cls): round(float(prob), 4) for cls, prob in zip(pri_classes, pri_proba)
    }

    # Truncate text for response (keep first 500 chars)
    display_text = text[:500] + ("..." if len(text) > 500 else "")

    return PredictionResult(
        text=display_text,
        category=str(cat_pred),
        category_confidence=round(cat_confidence, 4),
        priority=str(pri_pred),
        priority_confidence=round(pri_confidence, 4),
        category_probabilities=cat_probs_dict,
        priority_probabilities=pri_probs_dict,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy" if _model_state["loaded"] else "degraded",
        model_loaded=_model_state["loaded"],
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return model version, training date, and evaluation metrics."""
    _ensure_models_loaded()

    metadata = _model_state["metadata"]
    eval_data = metadata.get("evaluation", {})

    return ModelInfoResponse(
        version=metadata.get("version", "unknown"),
        trained_at=metadata.get("trained_at", "unknown"),
        category_model=type(_model_state["category_model"]).__name__,
        priority_model=type(_model_state["priority_model"]).__name__,
        category_metrics=eval_data.get("category"),
        priority_metrics=eval_data.get("priority"),
        cross_validation=metadata.get("cross_validation"),
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(ticket: TicketInput):
    """
    Classify a single support ticket.

    Returns the predicted category and priority with confidence scores
    and full probability distributions across all classes.
    """
    _ensure_models_loaded()

    logger.info(
        "Prediction request — text length: %d chars",
        len(ticket.text),
    )

    result = _predict_single(ticket.text)

    logger.info(
        "Prediction: category=%s (%.2f), priority=%s (%.2f)",
        result.category, result.category_confidence,
        result.priority, result.priority_confidence,
    )

    return result


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch(batch: BatchTicketInput):
    """
    Classify multiple support tickets from a JSON list.

    Accepts a JSON body: `{"tickets": [{"text": "..."}, ...]}`
    Returns predictions for all tickets (max 1000).
    """
    _ensure_models_loaded()

    texts = [t.text for t in batch.tickets]

    if len(texts) > 1000:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(texts)} exceeds maximum of 1000."
        )

    logger.info("Batch prediction request — %d tickets", len(texts))

    predictions = [_predict_single(text) for text in texts]

    logger.info("Batch prediction complete — %d tickets processed", len(predictions))

    return BatchPredictionResult(
        predictions=predictions,
        total=len(predictions),
    )


@app.post("/predict/upload", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_upload(file: UploadFile = File(...)):
    """
    Classify tickets from a CSV file upload.

    The CSV must contain a text column named one of:
    'text', 'Document', 'ticket_text', 'description', 'Ticket Description'.
    Returns predictions for all rows (max 1000).
    """
    _ensure_models_loaded()

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        content = await file.read()
        decoded = content.decode("utf-8")
        df = pd.read_csv(io.StringIO(decoded))

        # Auto-detect text column
        text_col = None
        for candidate in ["text", "Document", "ticket_text", "description", "Ticket Description"]:
            if candidate in df.columns:
                text_col = candidate
                break
        if text_col is None:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must have a text column. Found columns: {list(df.columns)}. "
                       f"Expected one of: text, Document, ticket_text, description."
            )

        texts = df[text_col].dropna().astype(str).tolist()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if not texts:
        raise HTTPException(status_code=400, detail="No ticket texts found in CSV.")

    if len(texts) > 1000:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(texts)} exceeds maximum of 1000."
        )

    logger.info("CSV upload prediction — %d tickets from %s", len(texts), file.filename)

    predictions = [_predict_single(text) for text in texts]

    logger.info("CSV prediction complete — %d tickets processed", len(predictions))

    return BatchPredictionResult(
        predictions=predictions,
        total=len(predictions),
    )

